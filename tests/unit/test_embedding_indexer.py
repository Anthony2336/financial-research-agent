"""Unit coverage for persistent, provider-versioned filing embeddings."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier, BrokenBarrierError

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from fra.retrieval.hybrid import HybridRetriever
from fra.retrieval.indexing import EmbeddingIndexer
from fra.storage.database import create_schema
from fra.storage.models import Chunk
from fra.storage.repositories import ChunkToStore, FilingRepository


class RecordingEmbeddings:
    """Deterministic provider that exposes document batches without external assets."""

    def __init__(self, version: str = "hash-1024-v1") -> None:
        self.version = version
        self.dimensions = 1024
        self.document_batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[float(index + 1), *([0.0] * 1023)] for index, _ in enumerate(texts)]


class InvalidEmbeddings(RecordingEmbeddings):
    def __init__(self, vectors: list[list[float]]) -> None:
        super().__init__()
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return self._vectors


@pytest.fixture
def repository() -> FilingRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    repository = FilingRepository(engine)
    repository.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000001",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/nvda.htm",
        raw_text="first chunk\nsecond chunk",
        content_hash="a" * 64,
        chunks=[
            ChunkToStore("MD&A", 0, "first chunk", 2, 0, 11),
            ChunkToStore("Risk Factors", 1, "second chunk", 2, 12, 24),
        ],
    )
    return repository


def _stored_embedding_state(repository: FilingRepository) -> list[tuple[str | None, object]]:
    with Session(repository.engine) as session:
        return list(
            session.execute(
                select(Chunk.embedding_model, Chunk.embedding).order_by(Chunk.chunk_index)
            ).all()
        )


def test_indexer_embeds_documents_once(repository: FilingRepository) -> None:
    """A second index pass for the same provider version must not embed documents again."""
    embeddings = RecordingEmbeddings()
    indexer = EmbeddingIndexer(repository, embeddings)

    assert indexer.ensure_indexed("NVDA", "NVDA-v1") == 2
    assert indexer.ensure_indexed("NVDA", "NVDA-v1") == 0

    assert embeddings.document_batches == [["first chunk", "second chunk"]]
    assert {version for version, _ in _stored_embedding_state(repository)} == {
        "hash-1024-v1"
    }


def test_indexer_replaces_a_previous_provider_version_once(
    repository: FilingRepository,
) -> None:
    """Changing provider version must refresh the corpus exactly once."""
    first = RecordingEmbeddings("hash-1024-v1")
    second = RecordingEmbeddings("hash-1024-v2")
    EmbeddingIndexer(repository, first).ensure_indexed("NVDA", "NVDA-v1")

    indexer = EmbeddingIndexer(repository, second)
    assert indexer.ensure_indexed("NVDA", "NVDA-v1") == 2
    assert indexer.ensure_indexed("NVDA", "NVDA-v1") == 0

    assert second.document_batches == [["first chunk", "second chunk"]]
    assert {version for version, _ in _stored_embedding_state(repository)} == {
        "hash-1024-v2"
    }


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0] * 1024], "count"),
        ([[1.0] * 1024, [1.0] * 3], "1024 dimensions"),
    ],
)
def test_indexer_rejects_invalid_batches_before_any_vector_is_written(
    repository: FilingRepository,
    vectors: list[list[float]],
    message: str,
) -> None:
    """Partial or malformed provider output must leave the whole corpus unindexed."""
    with pytest.raises(ValueError, match=message):
        EmbeddingIndexer(repository, InvalidEmbeddings(vectors)).ensure_indexed(
            "NVDA", "NVDA-v1"
        )

    assert _stored_embedding_state(repository) == [(None, None), (None, None)]


def test_sqlite_dense_search_reads_only_the_selected_corpus(
    repository: FilingRepository,
) -> None:
    """The deterministic fallback must not rank rows from another ticker or corpus."""
    EmbeddingIndexer(repository, RecordingEmbeddings()).ensure_indexed("NVDA", "NVDA-v1")
    repository.store_filing(
        ticker="AMD",
        form="10-Q",
        accession_no="0000002488-26-000001",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/amd.htm",
        raw_text="amd chunk",
        content_hash="b" * 64,
        chunks=[ChunkToStore("MD&A", 0, "amd chunk", 2, 0, 9)],
    )
    EmbeddingIndexer(repository, RecordingEmbeddings()).ensure_indexed("AMD", "AMD-v1")

    hits = repository.dense_search(
        ticker="NVDA",
        corpus_version="NVDA-v1",
        embedding_model="hash-1024-v1",
        query_embedding=[1.0, *([0.0] * 1023)],
        limit=1,
    )

    assert len(hits) == 1
    assert hits[0].ticker == "NVDA"
    assert hits[0].corpus_version == "NVDA-v1"


def test_dense_search_excludes_vectors_from_another_provider_version(
    repository: FilingRepository,
) -> None:
    """A query vector must never be compared with a different embedding space."""
    EmbeddingIndexer(repository, RecordingEmbeddings("hash-1024-v1")).ensure_indexed(
        "NVDA", "NVDA-v1"
    )
    query = [1.0, *([0.0] * 1023)]

    assert repository.dense_search(
        ticker="NVDA",
        corpus_version="NVDA-v1",
        embedding_model="sentence-transformers:BAAI/bge-m3",
        query_embedding=query,
        limit=2,
    ) == []
    assert repository.dense_search(
        ticker="NVDA",
        corpus_version="NVDA-v1",
        embedding_model="hash-1024-v1",
        query_embedding=query,
        limit=2,
    )


def test_concurrent_indexers_embed_one_document_batch(tmp_path) -> None:
    """Provider-version locking must make the read/embed/write sequence once-only."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'concurrent.sqlite3'}")
    create_schema(engine)
    repository = FilingRepository(engine)
    repository.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000001",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/nvda.htm",
        raw_text="first chunk\nsecond chunk",
        content_hash="a" * 64,
        chunks=[
            ChunkToStore("MD&A", 0, "first chunk", 2, 0, 11),
            ChunkToStore("Risk Factors", 1, "second chunk", 2, 12, 24),
        ],
    )
    provider = RecordingEmbeddings()
    indexer = EmbeddingIndexer(repository, provider)
    start = Barrier(2)

    def index() -> int:
        start.wait(timeout=5)
        return indexer.ensure_indexed("NVDA", "NVDA-v1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(index) for _ in range(2)]
    errors = [future.exception() for future in futures]

    assert errors == [None, None]
    assert provider.document_batches == [["first chunk", "second chunk"]]


def test_concurrent_cross_provider_queries_keep_their_matching_vector_space(tmp_path) -> None:
    """One provider cannot overwrite the corpus between another provider's index and query."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'provider-race.sqlite3'}")
    create_schema(engine)
    repository = FilingRepository(engine)
    repository.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000001",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/nvda.htm",
        raw_text="first chunk\nsecond chunk",
        content_hash="a" * 64,
        chunks=[
            ChunkToStore("MD&A", 0, "first chunk", 2, 0, 11),
            ChunkToStore("Risk Factors", 1, "second chunk", 2, 12, 24),
        ],
    )
    providers = [
        RecordingEmbeddings("hash-1024-v1"),
        RecordingEmbeddings("sentence-transformers:BAAI/bge-m3"),
    ]
    stores = Barrier(2)
    original_store = repository.store_embeddings

    def coordinated_store(chunks, vectors, embedding_model):
        original_store(chunks, vectors, embedding_model)
        try:
            stores.wait(timeout=0.2)
        except BrokenBarrierError:
            pass

    # Different provider locks currently allow both stores to reach the barrier. A
    # correct corpus lock times out the first store while the other call waits outside.
    repository.store_embeddings = coordinated_store  # type: ignore[method-assign]
    start = Barrier(2)

    def search(provider: RecordingEmbeddings):
        start.wait(timeout=5)
        return HybridRetriever(repository, provider).search_with_metrics(
            "NVDA", "first", "NVDA-v1", k=2
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(search, providers))

    assert [result.metrics.dense_candidates for result in results] == [2, 2]
    assert all(result.evidence for result in results)
