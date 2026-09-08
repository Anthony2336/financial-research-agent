"""Integration coverage for fixture-backed hybrid retrieval."""

from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from financial_evidence_agent.domain import EvidenceChunk
from financial_evidence_agent.retrieval.hybrid import (
    HashEmbeddingProvider,
    HybridRetriever,
)
from financial_evidence_agent.retrieval.ingest import ingest_fixture
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.models import Filing
from financial_evidence_agent.storage.repositories import ChunkToStore, FilingRepository


class _RepositoryWithDistractors:
    """Adds same-scope low-relevance chunks without changing fixture persistence."""

    def __init__(self, repository: FilingRepository, distractors: list[EvidenceChunk]) -> None:
        self._repository = repository
        self._distractors = distractors

    def list_chunks(self, ticker: str, corpus_version: str) -> list[EvidenceChunk]:
        return [
            *self._repository.list_chunks(ticker, corpus_version),
            *self._distractors,
        ]

    def dense_search(self, **kwargs) -> list[EvidenceChunk]:
        return self._repository.dense_search(**kwargs)


class _TailReranker:
    """Selects the last candidate to verify injected rerankers affect integration retrieval."""

    version = "tail-v1"
    selected_id: str | None = None

    def rerank(
        self, *, query: str, evidence: list[EvidenceChunk], limit: int
    ) -> list[EvidenceChunk]:
        self.selected_id = evidence[-1].id
        return evidence[-1:]


def _nvda_distractors() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            id=f"nvda-v1-distractor-{index}",
            ticker="NVDA",
            corpus_version="NVDA-v1",
            content=f"unrelated disclosure marker {index}",
            source_url="https://example.test/nvda-distractor",
            form="10-Q",
            filed_at=date(2025, 5, 28),
            accession_no="0001045810-25-000041",
            section="Other Disclosure",
            raw_start=index,
            raw_end=index + 1,
        )
        for index in range(8)
    ]


@pytest.fixture
def repository() -> FilingRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    return FilingRepository(engine)


@pytest.fixture
def hybrid_retriever(repository: FilingRepository) -> HybridRetriever:
    fixture = Path("tests/fixtures/nvda_10q.html")
    assert ingest_fixture(fixture, "NVDA", "10-Q", repository) == "NVDA-v1"
    assert ingest_fixture(fixture, "AMD", "10-Q", repository) == "AMD-v1"
    wrapped_repository = _RepositoryWithDistractors(repository, _nvda_distractors())
    return HybridRetriever(wrapped_repository, HashEmbeddingProvider())


def test_fixture_retrieves_mda_for_semantic_data_center_query(
    hybrid_retriever: HybridRetriever,
) -> None:
    """A ranking regression would hide the fixture's MD&A adoption evidence."""
    result = hybrid_retriever.search("NVDA", "accelerated computing adoption", "NVDA-v1", k=1)

    assert len(result) == 1
    assert result[0].section == "MD&A"
    assert all(chunk.ticker == "NVDA" for chunk in result)
    assert all(chunk.corpus_version == "NVDA-v1" for chunk in result)


def test_challenge_assignment_reserves_risk_factors_at_limit_one(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Risk reservation applies to typed challenge queries, not generic support lookup."""
    result = hybrid_retriever.search(
        "NVDA",
        "accelerated computing adoption",
        "NVDA-v1",
        k=1,
        evidence_side="challenge",
    )

    assert len(result) == 1
    assert result[0].section == "Risk Factors"


def test_fixture_retrieves_risk_factors_for_risk_query(
    hybrid_retriever: HybridRetriever,
) -> None:
    """A ranking regression would hide the filing's customer-concentration risks."""
    result = hybrid_retriever.search("NVDA", "risks from customer concentration", "NVDA-v1")

    assert result
    assert result[0].section == "Risk Factors"
    assert all(chunk.corpus_version == "NVDA-v1" for chunk in result)


def test_search_never_returns_other_ticker_chunks(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Ticker-scoping failures would expose another issuer's evidence."""
    result = hybrid_retriever.search("NVDA", "data center demand", "NVDA-v1")

    assert result
    assert all(chunk.ticker == "NVDA" for chunk in result)


def test_fixture_retrieval_uses_an_injected_reranker(
    repository: FilingRepository,
) -> None:
    """The hybrid flow must apply the injected adapter after fixture candidate fusion."""
    fixture = Path("tests/fixtures/nvda_10q.html")
    assert ingest_fixture(fixture, "NVDA", "10-Q", repository) == "NVDA-v1"
    reranker = _TailReranker()
    retriever = HybridRetriever(repository, HashEmbeddingProvider(), reranker=reranker)

    result = retriever.search("NVDA", "accelerated computing adoption", "NVDA-v1", k=1)

    assert [chunk.id for chunk in result] == [reranker.selected_id]


def test_hybrid_retrieval_ranks_multiple_filings_only_inside_the_selected_snapshot(
    repository: FilingRepository,
) -> None:
    """Sparse and dense candidates may span forms but never ticker or snapshot boundaries."""

    def store(form: str, accession: str, content: str, filed_at: date) -> str:
        repository.store_filing(
            ticker="NVDA",
            form=form,
            accession_no=accession,
            filed_at=filed_at,
            source_url=f"https://www.sec.gov/{accession}",
            raw_text=content,
            content_hash=sha256(content.encode()).hexdigest(),
            chunks=[ChunkToStore("MD&A", 0, content, 3, 0, len(content))],
        )
        with Session(repository.engine) as session:
            return session.scalar(select(Filing.id).where(Filing.accession_no == accession))

    quarterly_id = store(
        "10-Q",
        "nvda-quarterly",
        "quarterly revenue accelerator growth",
        date(2025, 5, 28),
    )
    current_id = store(
        "8-K", "nvda-current", "current revenue accelerator outlook", date(2025, 6, 2)
    )
    version = repository.create_corpus("NVDA", [quarterly_id, current_id], date(2025, 6, 2))
    retriever = HybridRetriever(repository, HashEmbeddingProvider())

    results = retriever.search("NVDA", "revenue accelerator", version, k=8)

    assert {chunk.form for chunk in results} == {"10-Q", "8-K"}
    assert {chunk.corpus_version for chunk in results} == {version}
    assert retriever.search("AMD", "revenue accelerator", version, k=8) == []


def test_hybrid_retrieval_filters_the_same_filing_subset_before_sparse_and_dense_ranking(
    repository: FilingRepository,
) -> None:
    """A user-selected form subset must constrain both candidate generators before ranking."""

    def store(form: str, accession: str, content: str, filed_at: date) -> str:
        repository.store_filing(
            ticker="NVDA",
            form=form,
            accession_no=accession,
            filed_at=filed_at,
            source_url=f"https://www.sec.gov/{accession}",
            raw_text=content,
            content_hash=sha256(content.encode()).hexdigest(),
            chunks=[ChunkToStore("MD&A", 0, content, 3, 0, len(content))],
        )
        with Session(repository.engine) as session:
            filing_id = session.scalar(select(Filing.id).where(Filing.accession_no == accession))
        assert filing_id is not None
        return filing_id

    quarterly_id = store(
        "10-Q",
        "nvda-subset-quarterly",
        "shared scope quarterly evidence",
        date(2025, 5, 28),
    )
    current_id = store(
        "8-K", "nvda-subset-current", "shared scope current evidence", date(2025, 6, 2)
    )
    version = repository.create_corpus("NVDA", [quarterly_id, current_id], date(2025, 6, 2))
    retriever = HybridRetriever(repository, HashEmbeddingProvider())

    results = retriever.search(
        "NVDA",
        "shared scope evidence",
        version,
        k=8,
        filing_ids=[quarterly_id],
    )

    assert results
    assert {chunk.form for chunk in results} == {"10-Q"}
    assert {chunk.accession_no for chunk in results} == {"nvda-subset-quarterly"}
    assert {
        chunk.accession_no
        for chunk in repository.list_chunks("NVDA", version, filing_ids=[quarterly_id])
    } == {"nvda-subset-quarterly"}
    provider = HashEmbeddingProvider()
    query_vector = provider.embed(["shared scope evidence"])[0]
    assert {
        chunk.accession_no
        for chunk in repository.dense_search(
            ticker="NVDA",
            corpus_version=version,
            filing_ids=[quarterly_id],
            embedding_model=provider.version,
            query_embedding=query_vector,
            limit=100,
        )
    } == {"nvda-subset-quarterly"}
