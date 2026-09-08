"""Opt-in verification of scoped pgvector retrieval in an isolated database."""

import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from financial_evidence_agent.config import Settings
from financial_evidence_agent.domain import SourceRef, SourceRefKind
from financial_evidence_agent.memory.research import ResearchMemoryKind
from financial_evidence_agent.retrieval.hybrid import HybridRetriever
from financial_evidence_agent.retrieval.indexing import EmbeddingIndexer
from financial_evidence_agent.storage.memory_repositories import ResearchMemoryRepository
from financial_evidence_agent.storage.models import ResearchMemoryRecord
from financial_evidence_agent.storage.repositories import ChunkToStore, FilingRepository
from financial_evidence_agent.storage.run_repositories import (
    PersistedClaim,
    ResearchRunRepository,
    RunFinish,
    RunStart,
)


class ContentEmbeddings:
    version = "integration-1024-v1"
    dimensions = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            ([1.0, 0.0] if "accelerator" in value else [0.0, 1.0])
            + ([0.0] * 1022)
            for value in texts
        ]


class RecordingContentEmbeddings(ContentEmbeddings):
    def __init__(self, version: str = "integration-1024-v1") -> None:
        self.version = version
        self.batches: list[list[str]] = []
        self._lock = Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.batches.append(texts)
        return super().embed(texts)


@pytest.fixture
def postgres_engine(monkeypatch: pytest.MonkeyPatch) -> Generator[Engine, None, None]:
    """Create and remove only this test's dedicated PostgreSQL database."""
    if os.environ.get("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("set RUN_POSTGRES_INTEGRATION=1 to run Postgres integration tests")

    configured = make_url(Settings().database_url)
    database_name = f"financial_evidence_vector_test_{uuid4().hex}"
    admin_url = configured.set(database="postgres")
    test_url = configured.set(database=database_name)
    admin = create_engine(
        admin_url,
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 2},
    )
    database_created = False
    engine: Engine | None = None
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
        database_created = True
    except OperationalError:
        admin.dispose()
        pytest.skip("Postgres is unavailable or the test user cannot create databases")

    try:
        config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
        monkeypatch.setenv(
            "DATABASE_URL",
            test_url.render_as_string(hide_password=False),
        )
        command.upgrade(config, "head")
        engine = create_engine(test_url)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        if database_created:
            with admin.connect() as connection:
                connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                    ),
                    {"database_name": database_name},
                )
                connection.execute(text(f'DROP DATABASE "{database_name}"'))
        admin.dispose()


@pytest.fixture
def postgres_repository(postgres_engine: Engine) -> FilingRepository:
    return FilingRepository(postgres_engine)


def _store_unindexed_corpus(repository: FilingRepository, ticker: str, marker: str) -> str:
    contents = [f"{marker} accelerator demand", f"{marker} customer risk"]
    return repository.store_filing(
        ticker=ticker,
        form="10-Q",
        accession_no=f"0000000000-26-{marker}",
        filed_at=date(2026, 5, 20),
        source_url=f"https://www.sec.gov/Archives/{marker}.htm",
        raw_text="\n".join(contents),
        content_hash=("a" if ticker == "NVDA" else "b") * 64,
        chunks=[
            ChunkToStore("MD&A", 0, contents[0], 3, 0, len(contents[0])),
            ChunkToStore(
                "Risk Factors",
                1,
                contents[1],
                3,
                len(contents[0]) + 1,
                len(contents[0]) + 1 + len(contents[1]),
            ),
        ],
    )


def _store_corpus(repository: FilingRepository, ticker: str, marker: str) -> str:
    version = _store_unindexed_corpus(repository, ticker, marker)
    EmbeddingIndexer(repository, ContentEmbeddings()).ensure_indexed(ticker, version)
    return version


def test_postgres_dense_search_is_scoped_and_uses_stored_vectors(
    postgres_engine: Engine,
    postgres_repository: FilingRepository,
) -> None:
    """SQL filtering and LIMIT must exclude a closer vector from another issuer."""
    _store_corpus(postgres_repository, "NVDA", "nvda")
    _store_corpus(postgres_repository, "AMD", "amd")

    hits = postgres_repository.dense_search(
        ticker="NVDA",
        corpus_version="NVDA-v1",
        embedding_model="integration-1024-v1",
        query_embedding=[1.0, *([0.0] * 1023)],
        limit=1,
    )

    with postgres_engine.connect() as connection:
        null_embeddings = connection.execute(
            text(
                "SELECT count(*) FROM chunks AS ch "
                "JOIN filings AS f ON f.id = ch.filing_id "
                "JOIN companies AS c ON c.id = f.company_id "
                "WHERE c.ticker = 'NVDA' AND f.corpus_version = 'NVDA-v1' "
                "AND ch.embedding IS NULL"
            )
        ).scalar_one()

    assert len(hits) == 1
    assert {hit.ticker for hit in hits} == {"NVDA"}
    assert hits[0].content == "nvda accelerator demand"
    assert null_embeddings == 0
    assert postgres_repository.dense_search(
        ticker="NVDA",
        corpus_version="NVDA-v1",
        embedding_model="another-provider-v1",
        query_embedding=[1.0, *([0.0] * 1023)],
        limit=1,
    ) == []


def test_postgres_provider_lock_embeds_one_concurrent_batch(
    postgres_engine: Engine,
    postgres_repository: FilingRepository,
) -> None:
    """The advisory lock must serialize independent repository instances."""
    _store_unindexed_corpus(postgres_repository, "NVDA", "nvda")
    second_repository = FilingRepository(postgres_engine)
    provider = RecordingContentEmbeddings()
    repositories = [postgres_repository, second_repository]
    start = Barrier(2)

    def index(repository: FilingRepository) -> int:
        start.wait(timeout=5)
        return EmbeddingIndexer(repository, provider).ensure_indexed(
            "NVDA",
            "NVDA-v1",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(index, repository) for repository in repositories]

    assert [future.exception() for future in futures] == [None, None]
    assert provider.batches == [
        ["nvda accelerator demand", "nvda customer risk"]
    ]


def test_postgres_ticker_embedding_context_does_not_starve_single_connection_pool(
    postgres_engine: Engine,
) -> None:
    """The advisory xact lock and protected helpers must share one pooled connection."""
    small_pool = create_engine(
        postgres_engine.url,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    try:
        repositories = [FilingRepository(small_pool), FilingRepository(small_pool)]
        _store_unindexed_corpus(repositories[0], "NVDA", "small-pool")
        provider = RecordingContentEmbeddings("small-pool-model")
        start = Barrier(2)

        def index(repository: FilingRepository) -> int:
            start.wait(timeout=5)
            return EmbeddingIndexer(repository, provider).ensure_indexed(
                "NVDA",
                "NVDA-v1",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(index, repository) for repository in repositories]
            results = [future.result(timeout=10) for future in futures]

        assert sorted(results) == [0, 2]
        assert provider.batches == [
            ["small-pool accelerator demand", "small-pool customer risk"]
        ]
        chunks = repositories[0].list_chunks("NVDA", "NVDA-v1")
        with Session(small_pool) as session:
            models = session.scalars(
                text(
                    "SELECT embedding_model FROM chunks "
                    "WHERE id IN (:first_id, :second_id) ORDER BY id"
                ).bindparams(
                    first_id=chunks[0].id,
                    second_id=chunks[1].id,
                )
            ).all()
        assert models == ["small-pool-model", "small-pool-model"]
    finally:
        small_pool.dispose()


def test_postgres_cross_provider_queries_keep_matching_vector_space(
    postgres_engine: Engine,
    postgres_repository: FilingRepository,
) -> None:
    """The corpus advisory lock must cover provider reindex through pgvector query."""
    _store_unindexed_corpus(postgres_repository, "NVDA", "nvda")
    repositories = [postgres_repository, FilingRepository(postgres_engine)]
    providers = [
        RecordingContentEmbeddings("hash-1024-v1"),
        RecordingContentEmbeddings("sentence-transformers:BAAI/bge-m3"),
    ]
    stores = Barrier(2)

    for repository in repositories:
        original_store = repository.store_embeddings

        def coordinated_store(
            chunks,
            vectors,
            embedding_model,
            *,
            _original=original_store,
        ):
            _original(chunks, vectors, embedding_model)
            try:
                stores.wait(timeout=0.3)
            except BrokenBarrierError:
                pass

        repository.store_embeddings = coordinated_store  # type: ignore[method-assign]

    start = Barrier(2)

    def search(values):
        repository, provider = values
        start.wait(timeout=5)
        return HybridRetriever(repository, provider).search_with_metrics(
            "NVDA", "accelerator", "NVDA-v1", k=2
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(search, zip(repositories, providers, strict=True)))

    assert [result.metrics.dense_candidates for result in results] == [2, 2]


def test_postgres_research_memory_uses_cosine_top_three_and_matching_model(
    postgres_engine: Engine,
    postgres_repository: FilingRepository,
) -> None:
    """The HNSW cosine path must stay ticker/model scoped and return at most three."""
    corpus_version = _store_unindexed_corpus(postgres_repository, "NVDA", "nvda")
    chunk = postgres_repository.list_chunks("NVDA", corpus_version)[0]
    source_ref = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.FILING,
        source_id=chunk.id,
    )
    summaries = [f"Guarded memory summary {index}." for index in range(4)]
    runs = ResearchRunRepository(postgres_engine)
    runs.start(RunStart(run_id="memory-vector-run", ticker="NVDA", request="Research"))
    runs.finish(
        RunFinish(
            run_id="memory-vector-run",
            effective_intent="research_request",
            status="completed",
            corpus_scope=[corpus_version],
            prompt_version="research-v1",
            trace_id="trace-memory-vector",
            report_markdown="# Guarded",
            claims=[
                PersistedClaim(
                    kind="verified_fact",
                    text=summary,
                    confidence="high",
                    source_refs=[source_ref],
                    guard_status="retained",
                )
                for summary in summaries
            ],
        )
    )
    now = datetime(2026, 9, 1, tzinfo=UTC)
    memories = ResearchMemoryRepository(postgres_engine, clock=lambda: now)
    vectors = (
        [1.0, 0.0, *([0.0] * 1022)],
        [0.9, 0.1, *([0.0] * 1022)],
        [0.8, 0.2, *([0.0] * 1022)],
        [-1.0, 0.0, *([0.0] * 1022)],
    )
    for summary, vector in zip(summaries, vectors, strict=True):
        memories.store_guarded(
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary=summary,
            source_run_id="memory-vector-run",
            evidence_source_refs=(source_ref,),
            corpus_version=corpus_version,
            embedding=vector,
            embedding_model="integration-memory-1024-v1",
            importance=0.8,
            created_at=now,
            expires_at=now + timedelta(days=90),
        )
    with Session(postgres_engine) as session, session.begin():
        session.add_all(
            ResearchMemoryRecord(
                id=f"{index + 100:064x}",
                scope_key="ticker:NVDA",
                ticker="NVDA",
                memory_kind="research_summary",
                summary=f"Invalid leading memory {index}.",
                source_run_id="memory-vector-run",
                evidence_source_refs=[f"NVDA:filing:missing-{index}"],
                corpus_version=corpus_version,
                embedding=[1.0, *([0.0] * 1023)],
                embedding_model="integration-memory-1024-v1",
                importance=1.0,
                created_at=now,
                expires_at=now + timedelta(days=90),
            )
            for index in range(20)
        )

    hits = memories.search(
        "NVDA",
        [1.0, *([0.0] * 1023)],
        limit=99,
        embedding_model="integration-memory-1024-v1",
        current_corpus_version=corpus_version,
    )

    assert [memory.summary for memory in hits] == summaries[:3]
    assert all(not memory.stale for memory in hits)
    assert memories.search(
        "NVDA",
        [1.0, *([0.0] * 1023)],
        embedding_model="another-model",
    ) == []
