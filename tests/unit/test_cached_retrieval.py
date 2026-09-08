from datetime import date
from hashlib import sha256

import pytest

from financial_evidence_agent.domain import EvidenceChunk
from financial_evidence_agent.retrieval.hybrid import HybridRetriever


def _chunk(
    chunk_id: str = "chunk-1",
    *,
    ticker: str = "NVDA",
    corpus_version: str = "NVDA-v1",
) -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker=ticker,
        corpus_version=corpus_version,
        content="data center revenue increased",
        source_url="https://example.test/filing",
        form="10-Q",
        filed_at=date(2026, 5, 28),
        accession_no="0001045810-26-000041",
        section="MD&A",
        raw_start=0,
        raw_end=29,
    )


class RecordingSyncCache:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, object, int]] = []

    def get_json_sync(self, key: str):
        self.get_calls.append(key)
        return self.values.get(key)

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.set_calls.append((key, value, ttl_seconds))
        self.values[key] = value


class RecordingRepository:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []
        self.id_calls: list[list[str]] = []

    def list_chunks(self, ticker: str, corpus_version: str) -> list[EvidenceChunk]:
        self.calls.append((ticker, corpus_version))
        return self.chunks

    def list_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        self.id_calls.append(chunk_ids)
        by_id = {chunk.id: chunk for chunk in self.chunks}
        return [by_id[source_id] for source_id in chunk_ids if source_id in by_id]

    def dense_search(
        self,
        *,
        ticker: str,
        corpus_version: str,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[EvidenceChunk]:
        del embedding_model, query_embedding
        return [
            chunk
            for chunk in self.chunks
            if chunk.ticker == ticker and chunk.corpus_version == corpus_version
        ][:limit]


class RecordingEmbeddingProvider:
    def __init__(self, version: str = "embedding-v1") -> None:
        self.version = version
        self.dimensions = 1
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0] for _ in texts]


class RecordingReranker:
    def __init__(self, version: str = "rank-v1") -> None:
        self.version = version
        self.calls = 0

    def rerank(self, *, query: str, evidence, limit: int) -> list[EvidenceChunk]:
        self.calls += 1
        return list(evidence[:limit])


def _key(
    *,
    corpus_version: str = "NVDA-v1",
    ticker: str = "NVDA",
    query: str = "revenue",
    limit: int = 1,
    reranker_version: str = "rank-v1",
    embedding_version: str = "embedding-v1",
) -> str:
    query_hash = sha256(query.encode("utf-8")).hexdigest()
    return (
        f"retrieval:{corpus_version}:{ticker}:{query_hash}:"
        f"{limit}:{reranker_version}:{embedding_version}"
    )


def test_cache_hit_skips_repository_dense_sparse_and_reranker_work(monkeypatch) -> None:
    """A valid hit must bypass every retrieval computation, including BM25."""
    expected = _chunk()
    cache = RecordingSyncCache({_key(): [expected.model_dump(mode="json")]})
    repository = RecordingRepository([expected])
    embeddings = RecordingEmbeddingProvider()
    reranker = RecordingReranker()
    metrics = []

    def fail_sparse(*args, **kwargs):
        raise AssertionError("BM25 must not run on a cache hit")

    monkeypatch.setattr("financial_evidence_agent.retrieval.hybrid._sparse_ranking", fail_sparse)
    retriever = HybridRetriever(
        repository,
        embeddings,
        reranker=reranker,
        cache=cache,
        cache_ttl_seconds=90,
        metrics_sink=metrics.append,
    )

    assert retriever.search("nvda", "revenue", "NVDA-v1", k=1) == [expected]
    assert repository.calls == []
    assert repository.id_calls == [[expected.id]]
    assert embeddings.calls == 0
    assert reranker.calls == 0
    assert cache.set_calls == []
    assert metrics[0].cache_hit is True


def test_cache_identity_includes_the_exact_filing_subset() -> None:
    first = _chunk("chunk-first")
    second = _chunk("chunk-second")

    class ScopedRepository(RecordingRepository):
        filing_chunks = {"filing-first": first, "filing-second": second}

        def list_chunks(
            self,
            ticker: str,
            corpus_version: str,
            *,
            filing_ids: tuple[str, ...] | None = None,
        ) -> list[EvidenceChunk]:
            self.calls.append((ticker, corpus_version))
            if filing_ids is None:
                return [first, second]
            return [self.filing_chunks[filing_id] for filing_id in filing_ids]

        def dense_search(self, **kwargs) -> list[EvidenceChunk]:
            filing_ids = kwargs.get("filing_ids")
            return self.list_chunks(
                kwargs["ticker"],
                kwargs["corpus_version"],
                filing_ids=filing_ids,
            )

    cache = RecordingSyncCache()
    retriever = HybridRetriever(
        ScopedRepository([first, second]),
        RecordingEmbeddingProvider(),
        cache=cache,
    )

    assert retriever.search("NVDA", "revenue", "NVDA-v1", k=1, filing_ids=["filing-first"]) == [
        first
    ]
    assert retriever.search("NVDA", "revenue", "NVDA-v1", k=1, filing_ids=["filing-second"]) == [
        second
    ]
    assert len(cache.get_calls) == 2
    assert cache.get_calls[0] != cache.get_calls[1]


def test_cache_hit_reloads_and_rejects_tampered_chunk_content() -> None:
    """A cache entry must not become a second, unaudited evidence repository."""
    canonical = _chunk()
    tampered = canonical.model_copy(update={"content": "Forged cached claim."})
    repository = RecordingRepository([canonical])
    retriever = HybridRetriever(
        repository,
        RecordingEmbeddingProvider(),
        reranker=RecordingReranker(),
        cache=RecordingSyncCache({_key(): [tampered.model_dump(mode="json")]}),
    )

    result = retriever.search("NVDA", "revenue", "NVDA-v1", k=1)

    assert result == [canonical]
    assert result[0].content != "Forged cached claim."
    assert repository.id_calls == [[canonical.id]]
    assert repository.calls == [("NVDA", "NVDA-v1")]


@pytest.mark.parametrize(
    (
        "ticker",
        "query",
        "corpus_version",
        "limit",
        "reranker_version",
        "embedding_version",
        "expected_key",
    ),
    [
        (
            "nvda",
            "revenue",
            "NVDA-v2",
            1,
            "rank-v1",
            "embedding-v1",
            _key(corpus_version="NVDA-v2"),
        ),
        (
            "amd",
            "revenue",
            "AMD-v1",
            1,
            "rank-v1",
            "embedding-v1",
            _key(ticker="AMD", corpus_version="AMD-v1"),
        ),
        (
            "nvda",
            "margin",
            "NVDA-v1",
            1,
            "rank-v1",
            "embedding-v1",
            _key(query="margin"),
        ),
        (
            "nvda",
            "revenue",
            "NVDA-v1",
            2,
            "rank-v1",
            "embedding-v1",
            _key(limit=2),
        ),
        (
            "nvda",
            "revenue",
            "NVDA-v1",
            1,
            "rank-v2",
            "embedding-v1",
            _key(reranker_version="rank-v2"),
        ),
        (
            "nvda",
            "revenue",
            "NVDA-v1",
            1,
            "rank-v1",
            "embedding-v2",
            _key(embedding_version="embedding-v2"),
        ),
    ],
)
def test_cache_key_separates_every_retrieval_input(
    ticker: str,
    query: str,
    corpus_version: str,
    limit: int,
    reranker_version: str,
    embedding_version: str,
    expected_key: str,
) -> None:
    """Dropping any invalidation dimension could serve evidence from the wrong retrieval run."""
    cache = RecordingSyncCache()
    chunk = _chunk(ticker=ticker.upper(), corpus_version=corpus_version)
    retriever = HybridRetriever(
        RecordingRepository([chunk]),
        RecordingEmbeddingProvider(embedding_version),
        reranker=RecordingReranker(reranker_version),
        cache=cache,
        cache_ttl_seconds=90,
    )

    assert retriever.search(ticker, query, corpus_version, k=limit) == [chunk]
    assert cache.get_calls == [expected_key]
    assert cache.set_calls == [(expected_key, [chunk.model_dump(mode="json")], 90)]


def test_empty_or_corrupted_cached_retrieval_is_a_miss() -> None:
    """An empty or invalid cache value must not suppress authoritative corpus retrieval."""
    chunk = _chunk()
    for cached in ([], [{"id": "not-an-evidence-chunk"}]):
        repository = RecordingRepository([chunk])
        cache = RecordingSyncCache({_key(): cached})
        retriever = HybridRetriever(
            repository,
            RecordingEmbeddingProvider(),
            reranker=RecordingReranker(),
            cache=cache,
        )

        assert retriever.search("NVDA", "revenue", "NVDA-v1", k=1) == [chunk]
        assert repository.calls == [("NVDA", "NVDA-v1")]


def test_cache_key_keeps_the_exact_requested_limit_before_result_capping() -> None:
    """Two distinct requested limits must not alias even when both return at most eight."""
    chunk = _chunk()
    cache = RecordingSyncCache()
    retriever = HybridRetriever(
        RecordingRepository([chunk]),
        RecordingEmbeddingProvider(),
        reranker=RecordingReranker(),
        cache=cache,
    )

    assert retriever.search("NVDA", "revenue", "NVDA-v1", k=99) == [chunk]
    assert cache.get_calls == [_key(limit=99)]
