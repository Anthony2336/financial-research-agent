"""Offline SEC contracts plus opt-in live SEC/OpenAI/Tavily connectivity smokes."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread

import httpx
import pytest
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

import financial_evidence_agent.retrieval.ingest as ingest_module
import financial_evidence_agent.storage.repositories as repositories_module
from financial_evidence_agent.cli import app
from financial_evidence_agent.retrieval.indexing import EmbeddingIndexer
from financial_evidence_agent.retrieval.ingest import IngestSummary, ingest_sec
from financial_evidence_agent.retrieval.sec import (
    HttpSecGateway,
    SecFilingCandidate,
    SecFilingDocument,
    SecProviderError,
    SecProviderErrorCode,
    SecRequestRateLimiter,
)
from financial_evidence_agent.retrieval.xbrl import (
    HttpCompanyFactsGateway,
    SecCompanyFactsDocument,
    XbrlError,
    company_facts_url,
    normalize_company_facts,
)
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.fact_repositories import CompanyFactRepository
from financial_evidence_agent.storage.models import (
    Chunk,
    Company,
    CompanyFactRecord,
    CorpusFiling,
    Filing,
    ResearchCorpus,
    ResearchRun,
    SkillRun,
    SourceFetchRecord,
)
from financial_evidence_agent.storage.repositories import (
    EmbeddingStateConflictError,
    FilingRepository,
    FilingToStore,
)
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.gateway import AllowlistedWebGateway
from financial_evidence_agent.web_evidence.providers import (
    HttpxRedirectResolver,
    TavilySearchProvider,
)
from financial_evidence_agent.web_evidence.source_policy import build_standard_source_policy

runner = CliRunner()


@pytest.fixture
def engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    return engine


@pytest.fixture
def repository(engine: Engine) -> FilingRepository:
    return FilingRepository(engine)


def _candidate(
    accession_no: str,
    form: str,
    filed_at: date,
    primary_document: str = "nvda-20250528.htm",
) -> SecFilingCandidate:
    return SecFilingCandidate(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        legal_name="NVIDIA CORPORATION",
        accession_no=accession_no,
        form=form,
        filed_at=filed_at,
        primary_document=primary_document,
    )


LIVE_HTML = b"""<!doctype html><html><body>
<h2>MD&amp;A</h2><p>Data center revenue grew because customer demand increased.</p>
<h2>Risk Factors</h2><p>Customer concentration may cause results to fluctuate.</p>
</body></html>"""


class RecordingGateway:
    def __init__(
        self,
        candidates: list[SecFilingCandidate] | None = None,
        document: SecFilingDocument | None = None,
        documents: dict[str, SecFilingDocument] | None = None,
    ) -> None:
        self.candidates = candidates or []
        self.document = document
        self.documents = documents or {}
        self.list_calls: list[tuple[str, str]] = []
        self.fetch_calls: list[tuple[SecFilingCandidate, str]] = []

    def list_filings(self, ticker: str, user_agent: str) -> list[SecFilingCandidate]:
        self.list_calls.append((ticker, user_agent))
        return self.candidates

    def fetch_document(
        self,
        filing: SecFilingCandidate,
        user_agent: str,
    ) -> SecFilingDocument:
        self.fetch_calls.append((filing, user_agent))
        if filing.accession_no in self.documents:
            return self.documents[filing.accession_no]
        if self.document is None:
            self.document = SecFilingDocument(
                source_url=(
                    f"https://www.sec.gov/Archives/edgar/data/{int(filing.cik)}/"
                    f"{filing.accession_no.replace('-', '')}/{filing.primary_document}"
                ),
                raw_bytes=LIVE_HTML,
            )
        return self.document


class RecordingSyncCache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, object, int]] = []
        self.delete_calls: list[str] = []

    def get_json_sync(self, key: str) -> object | None:
        self.get_calls.append(key)
        return self.values.get(key)

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.set_calls.append((key, value, ttl_seconds))
        self.values[key] = value

    def delete_json_sync(self, key: str) -> None:
        self.delete_calls.append(key)
        self.values.pop(key, None)


class FailingSyncCache:
    def get_json_sync(self, key: str) -> object | None:
        del key
        raise RuntimeError("private-cache-secret")

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        del key, value, ttl_seconds
        raise RuntimeError("private-cache-secret")

    def delete_json_sync(self, key: str) -> None:
        del key
        raise RuntimeError("private-cache-secret")


class RecordingCompanyFactsGateway:
    def __init__(self, document: SecCompanyFactsDocument) -> None:
        self.document = document
        self.calls: list[tuple[str, str, str]] = []

    def fetch_company_facts(
        self,
        ticker: str,
        cik: str,
        user_agent: str,
    ) -> SecCompanyFactsDocument:
        self.calls.append((ticker, cik, user_agent))
        return self.document


class FailOnFetchGateway(RecordingGateway):
    def __init__(
        self,
        *,
        candidates: list[SecFilingCandidate],
        documents: dict[str, SecFilingDocument],
        fail_at: int,
    ) -> None:
        super().__init__(candidates=candidates, documents=documents)
        self.fail_at = fail_at

    def fetch_document(
        self,
        filing: SecFilingCandidate,
        user_agent: str,
    ) -> SecFilingDocument:
        self.fetch_calls.append((filing, user_agent))
        if len(self.fetch_calls) == self.fail_at:
            raise RuntimeError("injected filing fetch failure")
        return self.documents[filing.accession_no]


class FailingEmbeddingProvider:
    version = "failing-1024-v1"
    dimensions = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("injected embedding failure")


class RecordingEmbeddingProvider:
    version = "recording-1024-v1"
    dimensions = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index == 0), *([0.0] * 1023)] for index, _ in enumerate(texts)]


class CorpusFailingRepository(FilingRepository):
    def _create_corpus_in_session(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected corpus creation failure")


def _document(candidate: SecFilingCandidate) -> SecFilingDocument:
    marker = candidate.accession_no.encode()
    return SecFilingDocument(
        source_url=(
            f"https://www.sec.gov/Archives/edgar/data/{int(candidate.cik)}/"
            f"{candidate.accession_no.replace('-', '')}/{candidate.primary_document}"
        ),
        raw_bytes=(
            b"<!doctype html><html><body><h2>MD&amp;A</h2><p>"
            + marker
            + b" revenue evidence.</p></body></html>"
        ),
    )


def _document_with_bytes(
    candidate: SecFilingCandidate, raw_bytes: bytes
) -> SecFilingDocument:
    document = _document(candidate)
    return document.model_copy(update={"raw_bytes": raw_bytes})


def _company_facts_document(raw_bytes: bytes | None = None) -> SecCompanyFactsDocument:
    return SecCompanyFactsDocument(
        source_url=company_facts_url("1045810"),
        raw_bytes=(
            raw_bytes
            if raw_bytes is not None
            else Path("tests/fixtures/sec/companyfacts_nvda.json").read_bytes()
        ),
        fetched_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )


def _evidence_row_counts(engine: Engine) -> tuple[int, int, int, int, int, int]:
    with Session(engine) as session:
        return tuple(
            int(session.scalar(select(func.count()).select_from(model)) or 0)
            for model in (
                Company,
                CompanyFactRecord,
                Filing,
                Chunk,
                ResearchCorpus,
                CorpusFiling,
            )
        )  # type: ignore[return-value]


def _require_live_values(*names: str) -> list[str]:
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail(
            "live smoke requires environment variables: " + ", ".join(sorted(missing))
        )
    return [os.environ[name].strip() for name in names]


@pytest.mark.parametrize(
    ("ticker", "forms", "as_of_date", "user_agent"),
    [
        ("../NVDA", ["10-Q"], None, "Example Research Operator research-operator@example.com"),
        ("NVDA", ["S-1"], None, "Example Research Operator research-operator@example.com"),
        ("NVDA", [], None, "Example Research Operator research-operator@example.com"),
        ("NVDA", ["10-Q"], None, "not-an-identifying-agent"),
        ("NVDA", ["10-Q"], None, "Example\x00Operator research-operator@example.com"),
        ("NVDA", ["10-Q"], None, "Example\tOperator research-operator@example.com"),
        ("NVDA", ["10-Q"], None, "Example\x7fOperator research-operator@example.com"),
        ("NVDA", ["10-Q"], "2025-05-28", "Example Research Operator research-operator@example.com"),
    ],
)
def test_invalid_live_inputs_make_zero_gateway_calls(
    repository: FilingRepository,
    ticker: str,
    forms: list[str],
    as_of_date: object,
    user_agent: str,
) -> None:
    gateway = RecordingGateway()

    with pytest.raises(ValueError):
        ingest_sec(
            ticker,
            forms,
            as_of_date,  # type: ignore[arg-type]
            repository=repository,
            user_agent=user_agent,
            gateway=gateway,
        )

    assert gateway.list_calls == []
    assert gateway.fetch_calls == []


def test_live_ingest_selects_newest_requested_filing_at_cutoff_and_is_idempotent(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    selected = _candidate("0001045810-25-000041", "8-K", date(2025, 5, 20), "nvda-8k.htm")
    quarterly = _candidate("0001045810-25-000039", "10-Q", date(2025, 5, 15))
    eligible = [selected, quarterly]
    gateway = RecordingGateway(
        candidates=[
            _candidate("0001045810-25-000050", "10-Q", date(2025, 6, 1)),
            _candidate("0001045810-25-000045", "10-K", date(2025, 5, 25)),
            *eligible,
        ],
        documents={item.accession_no: _document(item) for item in eligible},
    )

    first = ingest_sec(
        " nvda ",
        ["10-q", "8-K", "10-Q"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )
    second = ingest_sec(
        "NVDA",
        ["10-Q", "8-K"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )

    assert first == second
    assert first.ticker == "NVDA"
    assert first.requested_forms == ["10-Q", "8-K"]
    assert first.selected_form == "8-K"
    assert first.accession_no == selected.accession_no
    assert first.filed_at == selected.filed_at
    assert first.corpus_version == "NVDA-v1"
    assert first.chunk_count == 2
    assert first.content_hash == sha256(_document(selected).raw_bytes).hexdigest()
    assert [call[0].accession_no for call in gateway.fetch_calls] == [
        selected.accession_no,
        quarterly.accession_no,
        selected.accession_no,
        quarterly.accession_no,
    ]

    chunks = repository.list_chunks("NVDA", first.corpus_version)
    assert {chunk.section for chunk in chunks} == {"MD&A"}
    assert {chunk.accession_no for chunk in chunks} == {
        selected.accession_no,
        quarterly.accession_no,
    }
    with Session(engine) as session:
        filing = session.scalar(
            select(Filing).where(Filing.accession_no == selected.accession_no)
        )
        assert filing is not None
        assert filing.raw_text.startswith(_document(selected).raw_bytes.decode("utf-8"))
        assert filing.content_hash == sha256(_document(selected).raw_bytes).hexdigest()
        assert all(
            filing.raw_text[chunk.raw_start : chunk.raw_end] == chunk.content
            for chunk in chunks
            if chunk.accession_no == selected.accession_no
        )


def test_live_ingest_persists_validated_sec_company_metadata_and_one_configured_ir_domain(
    repository: FilingRepository,
) -> None:
    candidate = SecFilingCandidate(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        legal_name="NVIDIA CORPORATION",
        accession_no="0001045810-25-000041",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        primary_document="nvda-20250528.htm",
    )

    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[candidate]),
        configured_issuer_domains={
            "NVDA": frozenset({"investor.nvidia.com"})
        },
    )

    company = repository.get_company("NVDA")
    assert company is not None
    assert company.cik == "0001045810"
    assert company.legal_name == "NVIDIA CORPORATION"
    assert company.ir_domain == "investor.nvidia.com"


def test_live_ingest_explicitly_fetches_and_persists_normalized_company_facts(
    repository: FilingRepository,
) -> None:
    candidate = SecFilingCandidate(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        legal_name="NVIDIA CORPORATION",
        accession_no="0001045810-25-000041",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        primary_document="nvda-20250528.htm",
    )
    facts_gateway = RecordingCompanyFactsGateway(
        SecCompanyFactsDocument(
            source_url=company_facts_url("1045810"),
            raw_bytes=Path("tests/fixtures/sec/companyfacts_nvda.json").read_bytes(),
            fetched_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
        )
    )
    fact_repository = CompanyFactRepository(repository.engine)

    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[candidate]),
        fact_repository=fact_repository,
        company_facts_gateway=facts_gateway,
    )

    assert facts_gateway.calls == [
        ("NVDA", "0001045810", "Example Research Operator research-operator@example.com")
    ]
    facts = fact_repository.list_facts("NVDA", limit=10)
    assert len(facts) == 3
    assert all(fact.ticker == "NVDA" and fact.cik == "0001045810" for fact in facts)


def test_atomic_ingest_rolls_back_company_when_companyfacts_are_malformed(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))

    with pytest.raises(XbrlError):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=RecordingGateway(candidates=[candidate]),
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document(b"{}")
            ),
        )

    assert _evidence_row_counts(engine) == (0, 0, 0, 0, 0, 0)


def test_atomic_ingest_rolls_back_facts_and_first_filing_when_second_fetch_fails(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    first = _candidate("0001045810-25-000042", "10-Q", date(2025, 5, 28), "first.htm")
    second = _candidate("0001045810-25-000041", "8-K", date(2025, 5, 27), "second.htm")
    gateway = FailOnFetchGateway(
        candidates=[first, second],
        documents={item.accession_no: _document(item) for item in (first, second)},
        fail_at=2,
    )

    with pytest.raises(RuntimeError, match="filing fetch"):
        ingest_sec(
            "NVDA",
            ["10-Q", "8-K"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == (0, 0, 0, 0, 0, 0)


def test_atomic_ingest_rolls_back_when_second_filing_parse_fails(
    engine: Engine,
    repository: FilingRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _candidate("0001045810-25-000042", "10-Q", date(2025, 5, 28), "first.htm")
    second = _candidate("0001045810-25-000041", "8-K", date(2025, 5, 27), "second.htm")
    gateway = RecordingGateway(
        candidates=[first, second],
        documents={item.accession_no: _document(item) for item in (first, second)},
    )
    real_parse = ingest_module.parse_supported_sections
    calls = 0

    def fail_second_parse(raw_html: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected filing parse failure")
        return real_parse(raw_html)

    monkeypatch.setattr(ingest_module, "parse_supported_sections", fail_second_parse)

    with pytest.raises(ValueError, match="filing parse"):
        ingest_sec(
            "NVDA",
            ["10-Q", "8-K"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == (0, 0, 0, 0, 0, 0)


def test_atomic_ingest_computes_embeddings_before_any_durable_write(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))

    with pytest.raises(RuntimeError, match="embedding"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=RecordingGateway(candidates=[candidate]),
            embedding_provider=FailingEmbeddingProvider(),
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == (0, 0, 0, 0, 0, 0)


def test_atomic_ingest_rolls_back_every_row_when_corpus_creation_fails(
    engine: Engine,
) -> None:
    repository = CorpusFailingRepository(engine)
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))

    with pytest.raises(RuntimeError, match="corpus creation"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=RecordingGateway(candidates=[candidate]),
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == (0, 0, 0, 0, 0, 0)


def test_failed_atomic_refresh_preserves_all_preexisting_rows(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    existing = _candidate("0001045810-25-000040", "10-Q", date(2025, 5, 20), "old.htm")
    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[existing]),
        fact_repository=CompanyFactRepository(engine),
        company_facts_gateway=RecordingCompanyFactsGateway(_company_facts_document()),
    )
    before = _evidence_row_counts(engine)
    company_before = repository.get_company("NVDA")
    first = _candidate("0001045810-25-000042", "10-Q", date(2025, 5, 28), "first.htm")
    second = _candidate("0001045810-25-000041", "8-K", date(2025, 5, 27), "second.htm")

    with pytest.raises(RuntimeError, match="filing fetch"):
        ingest_sec(
            "NVDA",
            ["10-Q", "8-K"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=FailOnFetchGateway(
                candidates=[first, second],
                documents={item.accession_no: _document(item) for item in (first, second)},
                fail_at=2,
            ),
            fact_repository=CompanyFactRepository(engine),
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == before
    assert repository.get_company("NVDA") == company_before


def test_atomic_ingest_is_idempotent_and_skips_current_embeddings(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    gateway = RecordingGateway(candidates=[candidate])
    facts_gateway = RecordingCompanyFactsGateway(_company_facts_document())
    embedding_provider = RecordingEmbeddingProvider()
    fact_repository = CompanyFactRepository(engine)

    first = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
        embedding_provider=embedding_provider,
        fact_repository=fact_repository,
        company_facts_gateway=facts_gateway,
    )
    before = _evidence_row_counts(engine)
    repeated = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
        embedding_provider=embedding_provider,
        fact_repository=fact_repository,
        company_facts_gateway=facts_gateway,
    )

    assert repeated == first
    assert _evidence_row_counts(engine) == before == (1, 3, 1, 2, 1, 1)
    assert len(embedding_provider.calls) == 1
    with Session(engine) as session:
        chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert chunks
    assert all(chunk.embedding_model == embedding_provider.version for chunk in chunks)
    assert all(len(chunk.embedding or []) == 1024 for chunk in chunks)


def test_embedding_write_context_is_ticker_scoped_across_corpus_versions(
    repository: FilingRepository,
) -> None:
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first_writer() -> None:
        with repository.embedding_write_context("NVDA"):
            first_entered.set()
            assert release_first.wait(1)

    def second_writer() -> None:
        with repository.embedding_write_context("NVDA"):
            second_entered.set()

    first = Thread(target=first_writer)
    second = Thread(target=second_writer)
    first.start()
    assert first_entered.wait(1)
    second.start()
    try:
        assert not second_entered.wait(0.05)
    finally:
        release_first.set()
        first.join(1)
        second.join(1)

    assert second_entered.is_set()


def test_embedding_write_context_uses_one_session_for_all_protected_helpers(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = FilingRepository(engine)
    corpus_version = repository.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-25-000041",
        filed_at=date(2025, 5, 28),
        source_url="https://www.sec.gov/Archives/nvda.htm",
        raw_text="Atomic embedding evidence.",
        content_hash="a" * 64,
        chunks=[
            ingest_module.ChunkToStore(
                section="MD&A",
                chunk_index=0,
                content="Atomic embedding evidence.",
                token_count=3,
                raw_start=0,
                raw_end=26,
            )
        ],
    )
    real_session = Session
    session_count = 0

    def counting_session(*args, **kwargs):
        nonlocal session_count
        session_count += 1
        return real_session(*args, **kwargs)

    monkeypatch.setattr(repositories_module, "Session", counting_session)

    with repository.embedding_write_context("NVDA") as session:
        missing = repository.list_chunks_requiring_embedding_in_session(
            session,
            "NVDA",
            corpus_version,
            "model-a",
        )
        repository.store_embeddings_in_session(
            session,
            missing,
            [[1.0, *([0.0] * 1023)]],
            "model-a",
        )
        listed = repository.list_chunks_in_session(
            session,
            "NVDA",
            corpus_version,
        )
        dense = repository.dense_search_in_session(
            session,
            ticker="NVDA",
            corpus_version=corpus_version,
            embedding_model="model-a",
            query_embedding=[1.0, *([0.0] * 1023)],
            limit=1,
        )
        committed = repository.commit_ingest_batch_in_session(
            session,
            ticker="NVDA",
            cik="0001045810",
            legal_name="NVIDIA CORPORATION",
            ir_domain=None,
            filings=(
                FilingToStore(
                    form="10-Q",
                    accession_no="0001045810-25-000041",
                    filed_at=date(2025, 5, 28),
                    source_url="https://www.sec.gov/Archives/nvda.htm",
                    raw_text="Atomic embedding evidence.",
                    content_hash="a" * 64,
                    chunks=(
                        ingest_module.ChunkToStore(
                            section="MD&A",
                            chunk_index=0,
                            content="Atomic embedding evidence.",
                            token_count=3,
                            raw_start=0,
                            raw_end=26,
                        ),
                    ),
                    embedding_model="model-a",
                ),
            ),
            facts=(),
            fact_repository=CompanyFactRepository(engine),
            as_of_date=date(2025, 5, 28),
        )

    assert session_count == 1
    assert committed.corpus_version == corpus_version
    assert [chunk.id for chunk in listed] == [missing[0].id]
    assert [chunk.id for chunk in dense] == [missing[0].id]


def test_atomic_ingest_holds_embedding_context_only_after_network_and_through_commit(
    engine: Engine,
) -> None:
    class AuditedRepository(FilingRepository):
        lock_held = False
        state_checked = False
        committed = False

        @contextmanager
        def embedding_write_context(self, ticker: str):
            with super().embedding_write_context(ticker) as session:
                self.lock_held = True
                try:
                    yield session
                finally:
                    self.lock_held = False

        def filing_requires_embedding_in_session(self, session, **kwargs):
            assert self.lock_held
            self.state_checked = True
            return super().filing_requires_embedding_in_session(session, **kwargs)

        def commit_ingest_batch_in_session(self, session, **kwargs):
            assert self.lock_held
            self.committed = True
            return super().commit_ingest_batch_in_session(session, **kwargs)

    repository = AuditedRepository(engine)
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))

    class AuditedGateway(RecordingGateway):
        def list_filings(self, ticker: str, user_agent: str):
            assert not repository.lock_held
            return super().list_filings(ticker, user_agent)

        def fetch_document(self, filing: SecFilingCandidate, user_agent: str):
            assert not repository.lock_held
            return super().fetch_document(filing, user_agent)

    class AuditedFactsGateway(RecordingCompanyFactsGateway):
        def fetch_company_facts(self, ticker: str, cik: str, user_agent: str):
            assert not repository.lock_held
            return super().fetch_company_facts(ticker, cik, user_agent)

    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=AuditedGateway(candidates=[candidate]),
        embedding_provider=RecordingEmbeddingProvider(),
        fact_repository=CompanyFactRepository(engine),
        company_facts_gateway=AuditedFactsGateway(_company_facts_document()),
    )

    assert repository.state_checked is True
    assert repository.committed is True
    assert repository.lock_held is False


def test_atomic_ingest_rejects_embedding_model_change_after_current_state_check(
    engine: Engine,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    provider = RecordingEmbeddingProvider()
    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=FilingRepository(engine),
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[candidate]),
        embedding_provider=provider,
    )

    class ConflictRepository(FilingRepository):
        def filing_requires_embedding_in_session(self, session, **kwargs):
            required = super().filing_requires_embedding_in_session(session, **kwargs)
            assert required is False
            for chunk in session.scalars(select(Chunk)).all():
                chunk.embedding_model = "model-b"
            session.flush()
            return False

    repository = ConflictRepository(engine)

    with pytest.raises(EmbeddingStateConflictError, match="embedding state changed"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=RecordingGateway(candidates=[candidate]),
            embedding_provider=provider,
        )


def test_sqlite_model_writers_serialize_and_atomic_ingest_finishes_with_model_a() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = FilingRepository(engine)
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    seeded = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[candidate]),
    )
    model_b_entered = Event()
    release_model_b = Event()
    atomic_network_done = Event()
    errors: list[BaseException] = []

    class BlockingModelB:
        version = "model-b"
        dimensions = 1024

        def embed(self, texts: list[str]) -> list[list[float]]:
            model_b_entered.set()
            assert release_model_b.wait(1)
            return [[0.0, 1.0, *([0.0] * 1022)] for _ in texts]

    class AtomicGateway(RecordingGateway):
        def fetch_document(self, filing: SecFilingCandidate, user_agent: str):
            document = super().fetch_document(filing, user_agent)
            atomic_network_done.set()
            return document

    model_a = RecordingEmbeddingProvider()

    def write_model_b() -> None:
        try:
            EmbeddingIndexer(repository, BlockingModelB()).ensure_indexed(
                "NVDA",
                seeded.corpus_version,
            )
        except BaseException as error:
            errors.append(error)

    def run_atomic_model_a() -> None:
        try:
            ingest_sec(
                "NVDA",
                ["10-Q"],
                None,
                repository=repository,
                user_agent="Example Research Operator research-operator@example.com",
                gateway=AtomicGateway(candidates=[candidate]),
                embedding_provider=model_a,
            )
        except BaseException as error:
            errors.append(error)

    writer_b = Thread(target=write_model_b)
    writer_a = Thread(target=run_atomic_model_a)
    writer_b.start()
    assert model_b_entered.wait(1)
    writer_a.start()
    assert atomic_network_done.wait(1)
    assert model_a.calls == []
    release_model_b.set()
    writer_b.join(2)
    writer_a.join(2)

    assert errors == []
    assert not writer_b.is_alive()
    assert not writer_a.is_alive()
    assert len(model_a.calls) == 1
    with Session(engine) as session:
        chunks = session.scalars(select(Chunk)).all()
    assert chunks
    assert all(chunk.embedding_model == model_a.version for chunk in chunks)


def test_atomic_fact_conflict_rolls_back_metadata_and_new_filing(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    existing = _candidate("0001045810-25-000040", "10-Q", date(2025, 5, 20), "old.htm")
    fact_repository = CompanyFactRepository(engine)
    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=RecordingGateway(candidates=[existing]),
        fact_repository=fact_repository,
        company_facts_gateway=RecordingCompanyFactsGateway(_company_facts_document()),
    )
    with Session(engine) as session, session.begin():
        row = session.scalar(select(CompanyFactRecord).limit(1))
        assert row is not None
        row.value += 1
    before = _evidence_row_counts(engine)
    company_before = repository.get_company("NVDA")
    newer = _candidate("0001045810-25-000041", "8-K", date(2025, 5, 28), "new.htm")

    with pytest.raises(ValueError, match="conflicts with persisted exact fact"):
        ingest_sec(
            "NVDA",
            ["8-K"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=RecordingGateway(candidates=[newer]),
            configured_issuer_domains={
                "NVDA": frozenset({"new.investor.nvidia.com"})
            },
            fact_repository=fact_repository,
            company_facts_gateway=RecordingCompanyFactsGateway(
                _company_facts_document()
            ),
        )

    assert _evidence_row_counts(engine) == before
    assert repository.get_company("NVDA") == company_before


def test_live_ingest_rejects_ambiguous_configured_ir_domains_before_sec_calls(
    repository: FilingRepository,
) -> None:
    gateway = RecordingGateway()

    with pytest.raises(ValueError, match="exactly one"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
            configured_issuer_domains={
                "NVDA": frozenset({"investor.nvidia.com", "ir.nvidia.com"})
            },
        )

    assert gateway.list_calls == []


def test_sec_raw_cache_uses_accession_key_exact_ttl_and_skips_repeat_fetch(
    repository: FilingRepository,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    gateway = RecordingGateway(candidates=[candidate])
    cache = RecordingSyncCache()

    first = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
        cache=cache,
        sec_cache_ttl_seconds=86_400,
    )
    repeated = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
        cache=cache,
        sec_cache_ttl_seconds=86_400,
    )

    key = f"sec:{candidate.accession_no}"
    assert repeated == first
    assert [call[0].accession_no for call in gateway.fetch_calls] == [
        candidate.accession_no
    ]
    assert cache.get_calls == [key, key]
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0][0] == key
    assert cache.set_calls[0][2] == 86_400
    assert set(cache.set_calls[0][1]) == {
        "accession_no",
        "ticker",
        "cik",
        "legal_name",
        "form",
        "filed_at",
        "source_url",
        "content_hash",
        "raw_html",
    }


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("accession_no", "0001045810-25-000099"),
        ("ticker", "AMD"),
        ("cik", "0000320193"),
        ("legal_name", "Advanced Micro Devices, Inc."),
        ("form", "8-K"),
        ("filed_at", "2025-05-27"),
        ("source_url", "https://www.sec.gov/Archives/edgar/data/1045810/forged.htm"),
        ("content_hash", "f" * 64),
        ("raw_html", "<html>forged</html>"),
        ("unexpected", "malformed-cache-envelope"),
    ],
)
def test_sec_raw_cache_tamper_is_a_miss_and_cannot_change_citation_identity(
    repository: FilingRepository,
    field: str,
    tampered_value: str,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    cache = RecordingSyncCache()
    prime_gateway = RecordingGateway(candidates=[candidate])
    first = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=prime_gateway,
        cache=cache,
    )
    key = f"sec:{candidate.accession_no}"
    payload = dict(cache.values[key])
    payload[field] = tampered_value
    cache.values[key] = payload
    fallback_gateway = RecordingGateway(candidates=[candidate])

    summary = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=fallback_gateway,
        cache=cache,
    )

    assert len(fallback_gateway.fetch_calls) == 1
    assert summary.source_url == first.source_url
    assert summary.content_hash == first.content_hash
    assert key in cache.delete_calls


def test_sec_cache_outage_degrades_to_fetch_without_leaking_raw_cache_error(
    repository: FilingRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    gateway = RecordingGateway(candidates=[candidate])

    with caplog.at_level(logging.WARNING):
        summary = ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
            cache=FailingSyncCache(),
        )

    assert summary.ticker == "NVDA"
    assert len(gateway.fetch_calls) == 1
    assert "private-cache-secret" not in caplog.text


def test_all_zero_sec_cik_is_rejected_before_document_fetch(
    repository: FilingRepository,
) -> None:
    zero_cik = SecFilingCandidate.model_construct(
        ticker="ZERO",
        resolved_cik="0",
        cik="0",
        legal_name="Zero CIK Corporation",
        accession_no="0000000000-25-000001",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        primary_document="zero-20250528.htm",
    )
    gateway = RecordingGateway(candidates=[zero_cik])

    with pytest.raises(ValueError, match="SEC filing candidate"):
        ingest_sec(
            "ZERO",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert gateway.fetch_calls == []


def test_live_ingest_builds_deterministic_four_filing_snapshots_and_preserves_history(
    repository: FilingRepository,
) -> None:
    """Each requested form gets its newest filing before remaining slots are filled by date."""
    ten_k = _candidate("0001045810-25-000045", "10-K", date(2025, 5, 25), "10k.htm")
    eight_k = _candidate("0001045810-25-000044", "8-K", date(2025, 5, 24), "8k.htm")
    ten_q = _candidate("0001045810-25-000043", "10-Q", date(2025, 5, 23), "10q.htm")
    extra = _candidate("0001045810-25-000042", "8-K", date(2025, 5, 22), "8k-old.htm")
    excluded = _candidate("0001045810-25-000050", "10-Q", date(2025, 6, 1), "future.htm")
    candidates = [excluded, extra, ten_q, eight_k, ten_k]
    gateway = RecordingGateway(
        candidates=candidates,
        documents={item.accession_no: _document(item) for item in candidates},
    )

    first = ingest_sec(
        "NVDA",
        ["10-Q", "10-K", "8-K"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )
    repeated = ingest_sec(
        "NVDA",
        ["10-Q", "10-K", "8-K"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )

    expected_accessions = {
        ten_k.accession_no,
        eight_k.accession_no,
        ten_q.accession_no,
        extra.accession_no,
    }
    assert repeated.corpus_version == first.corpus_version == "NVDA-v1"
    assert first.selected_form == "10-K"
    assert {chunk.accession_no for chunk in repository.list_chunks("NVDA", "NVDA-v1")} == (
        expected_accessions
    )
    assert [call[0].accession_no for call in gateway.fetch_calls[:4]] == [
        ten_k.accession_no,
        eight_k.accession_no,
        ten_q.accession_no,
        extra.accession_no,
    ]
    recent = repository.list_recent_filings("NVDA", forms=[], limit=4)
    assert {filing.accession_no for filing in recent} == expected_accessions
    assert {filing.corpus_version for filing in recent} == {first.corpus_version}
    assert (
        repository.corpus_version_for_filings(
            "NVDA", [filing.id for filing in recent]
        )
        == first.corpus_version
    )

    newer_q = _candidate("0001045810-25-000046", "10-Q", date(2025, 5, 26), "10q-new.htm")
    gateway.candidates = [newer_q, *candidates]
    gateway.documents[newer_q.accession_no] = _document(newer_q)
    current = ingest_sec(
        "NVDA",
        ["10-Q", "10-K", "8-K"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )

    assert current.corpus_version == "NVDA-v2"
    assert newer_q.accession_no in {
        chunk.accession_no for chunk in repository.list_chunks("NVDA", "NVDA-v2")
    }
    assert {chunk.accession_no for chunk in repository.list_chunks("NVDA", "NVDA-v1")} == (
        expected_accessions
    )


def test_live_ingest_skips_duplicate_content_and_fills_from_ranked_candidates(
    repository: FilingRepository,
) -> None:
    """Distinct accessions with identical bytes occupy only one snapshot slot."""
    ten_k = _candidate("0001045810-25-000045", "10-K", date(2025, 5, 25), "10k.htm")
    duplicate = _candidate("0001045810-25-000044", "8-K", date(2025, 5, 24), "8k.htm")
    ten_q = _candidate("0001045810-25-000043", "10-Q", date(2025, 5, 23), "10q.htm")
    older_eight_k = _candidate(
        "0001045810-25-000042", "8-K", date(2025, 5, 22), "8k-old.htm"
    )
    older_ten_q = _candidate(
        "0001045810-25-000041", "10-Q", date(2025, 5, 21), "10q-old.htm"
    )
    candidates = [older_eight_k, ten_q, older_ten_q, duplicate, ten_k]
    shared_bytes = b"<html><body><h2>MD&amp;A</h2><p>shared evidence</p></body></html>"
    documents = {item.accession_no: _document(item) for item in candidates}
    documents[ten_k.accession_no] = _document_with_bytes(ten_k, shared_bytes)
    documents[duplicate.accession_no] = _document_with_bytes(duplicate, shared_bytes)
    gateway = RecordingGateway(candidates=candidates, documents=documents)

    first = ingest_sec(
        "NVDA",
        ["10-Q", "10-K", "8-K"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )
    gateway.candidates = list(reversed(candidates))
    repeated = ingest_sec(
        "NVDA",
        ["8-K", "10-K", "10-Q"],
        date(2025, 5, 28),
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )

    expected = {
        ten_k.accession_no,
        ten_q.accession_no,
        older_eight_k.accession_no,
        older_ten_q.accession_no,
    }
    assert repeated.corpus_version == first.corpus_version
    assert {
        chunk.accession_no
        for chunk in repository.list_chunks("NVDA", first.corpus_version)
    } == expected
    assert duplicate.accession_no not in expected
    expected_fetch_order = [
        ten_k.accession_no,
        duplicate.accession_no,
        ten_q.accession_no,
        older_eight_k.accession_no,
        older_ten_q.accession_no,
    ]
    assert [call[0].accession_no for call in gateway.fetch_calls[:5]] == expected_fetch_order
    assert [call[0].accession_no for call in gateway.fetch_calls[5:]] == expected_fetch_order


def test_malformed_sec_candidate_is_rejected_before_document_fetch(
    repository: FilingRepository,
) -> None:
    malformed = SecFilingCandidate.model_construct(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        legal_name="NVIDIA CORPORATION",
        accession_no="0001045810-25-000041",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        primary_document="../attacker.example/filing.htm",
    )
    gateway = RecordingGateway(candidates=[malformed])

    with pytest.raises(ValueError, match="SEC filing candidate"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert len(gateway.list_calls) == 1
    assert gateway.fetch_calls == []


def test_sec_candidate_missing_legal_name_is_rejected_before_document_fetch(
    repository: FilingRepository,
) -> None:
    missing_name = SecFilingCandidate.model_construct(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        accession_no="0001045810-25-000041",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        primary_document="nvda-20250528.htm",
    )
    gateway = RecordingGateway(candidates=[missing_name])

    with pytest.raises(ValueError, match="SEC filing candidate"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert gateway.fetch_calls == []


def test_accession_must_belong_to_candidate_cik_before_document_fetch(
    repository: FilingRepository,
) -> None:
    mismatched = _candidate("0000320193-25-000041", "10-Q", date(2025, 5, 28))
    gateway = RecordingGateway(candidates=[mismatched])

    with pytest.raises(ValueError, match="SEC filing candidate"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert gateway.fetch_calls == []


def test_cross_issuer_candidate_is_rejected_without_fetch_or_database_rows(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    apple_candidate = SecFilingCandidate.model_construct(
        ticker="AAPL",
        resolved_cik="320193",
        cik="320193",
        legal_name="APPLE INC.",
        accession_no="0000320193-25-000001",
        form="10-Q",
        filed_at=date(2025, 5, 2),
        primary_document="aapl-20250329.htm",
    )
    gateway = RecordingGateway(candidates=[apple_candidate])

    with pytest.raises(ValueError, match="issuer"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert gateway.fetch_calls == []
    assert repository.get_company("NVDA") is None
    with Session(engine) as session:
        assert session.scalars(select(Filing)).all() == []


def test_claimed_ticker_cannot_hide_candidate_cik_from_another_issuer(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    spoofed_candidate = SecFilingCandidate.model_construct(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="320193",
        legal_name="APPLE INC.",
        accession_no="0000320193-25-000001",
        form="10-Q",
        filed_at=date(2025, 5, 2),
        primary_document="aapl-20250329.htm",
    )
    gateway = RecordingGateway(candidates=[spoofed_candidate])

    with pytest.raises(ValueError, match="issuer"):
        ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=gateway,
        )

    assert gateway.fetch_calls == []
    assert repository.get_company("NVDA") is None
    with Session(engine) as session:
        assert session.scalars(select(Filing)).all() == []


def test_live_ingest_reports_zero_chunks_without_claiming_researchability(
    repository: FilingRepository,
) -> None:
    candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 28))
    gateway = RecordingGateway(
        candidates=[candidate],
        document=SecFilingDocument(
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581025000041/nvda-20250528.htm"
            ),
            raw_bytes=b"<html><body><p>No supported section heading.</p></body></html>",
        ),
    )

    summary = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=gateway,
    )

    assert summary.chunk_count == 0
    assert repository.list_chunks("NVDA", summary.corpus_version) == []


def test_same_hash_with_different_filing_metadata_fails_instead_of_lying(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    raw_bytes = b"<html><body><p>No supported section heading.</p></body></html>"
    first_candidate = _candidate("0001045810-25-000041", "10-Q", date(2025, 5, 20))
    second_candidate = _candidate(
        "0001045810-25-000042",
        "8-K",
        date(2025, 5, 21),
        "nvda-20250521.htm",
    )
    first_gateway = RecordingGateway(
        candidates=[first_candidate],
        document=SecFilingDocument(
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581025000041/nvda-20250528.htm"
            ),
            raw_bytes=raw_bytes,
        ),
    )
    second_gateway = RecordingGateway(
        candidates=[second_candidate],
        document=SecFilingDocument(
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581025000042/nvda-20250521.htm"
            ),
            raw_bytes=raw_bytes,
        ),
    )

    first = ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=first_gateway,
    )
    with pytest.raises(ValueError, match="different filing metadata"):
        ingest_sec(
            "NVDA",
            ["8-K"],
            None,
            repository=repository,
            user_agent="Example Research Operator research-operator@example.com",
            gateway=second_gateway,
        )

    assert first.chunk_count == 0
    persisted = repository.get_filing("NVDA", first.corpus_version)
    assert persisted is not None
    assert persisted.accession_no == first_candidate.accession_no
    assert persisted.form == first_candidate.form
    assert persisted.filed_at == first_candidate.filed_at
    with Session(engine) as session:
        assert len(session.scalars(select(Filing)).all()) == 1


def test_http_gateway_uses_fixed_hosts_identity_timeout_and_sub_ten_rps_rate() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(
                200,
                json={"0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"}},
            )
        if request.url.path == "/submissions/CIK0001045810.json":
            return httpx.Response(
                200,
                json={
                    "cik": "0001045810",
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0001045810-25-000041"],
                            "filingDate": ["2025-05-28"],
                            "form": ["10-Q"],
                            "primaryDocument": ["nvda-20250528.htm"],
                        }
                    },
                },
            )
        return httpx.Response(200, content=LIVE_HTML)

    current_time = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    gateway = HttpSecGateway(
        transport=httpx.MockTransport(handler),
        clock=lambda: current_time[0],
        sleep=sleep,
    )
    user_agent = "Example Research Operator research-operator@example.com"

    candidates = gateway.list_filings("NVDA", user_agent)
    document = gateway.fetch_document(candidates[0], user_agent)

    assert document.raw_bytes == LIVE_HTML
    assert candidates[0].legal_name == "NVIDIA CORP"
    assert [request.url.host for request in requests] == [
        "www.sec.gov",
        "data.sec.gov",
        "www.sec.gov",
    ]
    assert all(request.url.scheme == "https" for request in requests)
    assert all(request.headers["user-agent"] == user_agent for request in requests)
    assert all(request.extensions["timeout"]["connect"] == 5.0 for request in requests)
    assert all(request.extensions["timeout"]["read"] == 30.0 for request in requests)
    assert len(sleeps) == 2
    assert all(seconds > 0.1 for seconds in sleeps)


def test_combined_ingest_shares_one_rate_schedule_across_sec_and_companyfacts(
    engine: Engine,
    repository: FilingRepository,
) -> None:
    current_time = [0.0]
    request_times: list[float] = []

    def sleep(seconds: float) -> None:
        current_time[0] += seconds

    limiter = SecRequestRateLimiter(clock=lambda: current_time[0], sleep=sleep)

    def sec_handler(request: httpx.Request) -> httpx.Response:
        request_times.append(current_time[0])
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(
                200,
                json={
                    "0": {
                        "ticker": "NVDA",
                        "cik_str": 1045810,
                        "title": "NVIDIA CORPORATION",
                    }
                },
                request=request,
            )
        if request.url.path == "/submissions/CIK0001045810.json":
            return httpx.Response(
                200,
                json={
                    "cik": "0001045810",
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0001045810-25-000041"],
                            "filingDate": ["2025-05-28"],
                            "form": ["10-Q"],
                            "primaryDocument": ["nvda-20250528.htm"],
                        }
                    },
                },
                request=request,
            )
        return httpx.Response(200, content=LIVE_HTML, request=request)

    def facts_handler(request: httpx.Request) -> httpx.Response:
        request_times.append(current_time[0])
        return httpx.Response(
            200,
            content=Path("tests/fixtures/sec/companyfacts_nvda.json").read_bytes(),
            request=request,
        )

    ingest_sec(
        "NVDA",
        ["10-Q"],
        None,
        repository=repository,
        user_agent="Example Research Operator research-operator@example.com",
        gateway=HttpSecGateway(
            transport=httpx.MockTransport(sec_handler),
            rate_limiter=limiter,
        ),
        fact_repository=CompanyFactRepository(engine),
        company_facts_gateway=HttpCompanyFactsGateway(
            transport=httpx.MockTransport(facts_handler),
            clock=lambda: datetime(2026, 9, 1, 12, tzinfo=UTC),
            rate_limiter=limiter,
        ),
    )

    assert request_times == pytest.approx([0.0, 0.11, 0.22, 0.33])


def test_http_gateway_rejects_submissions_cik_that_does_not_match_ticker_map() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(
                200,
                json={
                    "0": {
                        "ticker": "NVDA",
                        "cik_str": 1045810,
                        "title": "NVIDIA CORP",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "cik": "0000320193",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0001045810-25-000041"],
                        "filingDate": ["2025-05-28"],
                        "form": ["10-Q"],
                        "primaryDocument": ["nvda-20250528.htm"],
                    }
                },
            },
        )

    gateway = HttpSecGateway(transport=httpx.MockTransport(handler))

    with pytest.raises(SecProviderError, match="CIK") as raised:
        gateway.list_filings("NVDA", "Example Research Operator research-operator@example.com")

    assert raised.value.code is SecProviderErrorCode.SCOPE_MISMATCH
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [
        (429, SecProviderErrorCode.RATE_LIMITED),
        (503, SecProviderErrorCode.UNAVAILABLE),
        (httpx.ReadTimeout("private-provider-token"), SecProviderErrorCode.UNAVAILABLE),
        (RuntimeError("private-provider-token"), SecProviderErrorCode.UNAVAILABLE),
    ],
)
def test_http_sec_provider_errors_are_typed_and_sanitized(
    outcome: int | Exception,
    expected_code: SecProviderErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, text="private-provider-token", request=request)

    gateway = HttpSecGateway(transport=httpx.MockTransport(handler))

    with pytest.raises(SecProviderError) as raised:
        gateway.list_filings("NVDA", "Example Research Operator research-operator@example.com")

    assert raised.value.code is expected_code
    assert "private-provider-token" not in str(raised.value)


def test_http_sec_malformed_json_error_is_typed_and_sanitized() -> None:
    gateway = HttpSecGateway(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"private-provider-token-not-json",
                request=request,
            )
        )
    )

    with pytest.raises(SecProviderError) as raised:
        gateway.list_filings("NVDA", "Example Research Operator research-operator@example.com")

    assert raised.value.code is SecProviderErrorCode.MALFORMED_RESPONSE
    assert "private-provider-token" not in str(raised.value)


def test_fixture_cli_is_offline_idempotent_and_rejects_live_option_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'ingest.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    arguments = [
        "ingest",
        "--fixture",
        "tests/fixtures/nvda_10q.html",
        "--ticker",
        "NVDA",
        "--form",
        "10-Q",
    ]

    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)
    conflict = runner.invoke(app, [*arguments, "--forms", "10-Q,8-K"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_summary = json.loads(first.output)
    second_summary = json.loads(second.output)
    assert first_summary == second_summary
    assert first_summary["ticker"] == "NVDA"
    assert first_summary["requested_forms"] == ["10-Q"]
    assert first_summary["selected_form"] == "10-Q"
    assert first_summary["corpus_version"] == "NVDA-v1"
    assert first_summary["chunk_count"] > 0
    assert conflict.exit_code == 2
    assert "mutually exclusive" in conflict.output


def test_live_cli_wires_forms_date_and_resolved_identity_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI parsing must reach the injectable ingest boundary without opening SEC transport."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'live-wiring.sqlite3'}"
    user_agent = "Example Research Operator research-operator@example.com"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("SEC_USER_AGENT", user_agent)
    monkeypatch.setenv("SEC_CACHE_TTL_SECONDS", "43200")
    monkeypatch.setenv("XBRL_MAX_RESPONSE_BYTES", "12345")
    monkeypatch.setenv(
        "WEB_ISSUER_DOMAINS",
        '{"NVDA":["investor.nvidia.com"]}',
    )
    calls: list[tuple[str, list[str], date | None, str]] = []
    wired: dict[str, object] = {}

    def fake_ingest_sec(
        ticker: str,
        forms: list[str],
        as_of_date: date | None,
        *,
        repository: FilingRepository,
        user_agent: str,
        gateway: object | None = None,
        embedding_provider: object,
        cache: object,
        sec_cache_ttl_seconds: int,
        configured_issuer_domains: object,
        fact_repository: object,
        company_facts_gateway: object,
        rate_limiter: object,
    ) -> IngestSummary:
        del gateway
        assert isinstance(repository, FilingRepository)
        assert getattr(embedding_provider, "version") == (
            "sentence-transformers:BAAI/bge-m3"
        )
        wired.update(
            cache=cache,
            sec_cache_ttl_seconds=sec_cache_ttl_seconds,
            configured_issuer_domains=configured_issuer_domains,
            fact_repository=fact_repository,
            company_facts_gateway=company_facts_gateway,
            rate_limiter=rate_limiter,
        )
        calls.append((ticker, forms, as_of_date, user_agent))
        return IngestSummary(
            ticker="NVDA",
            requested_forms=["10-Q", "8-K"],
            selected_form="10-Q",
            accession_no="0001045810-25-000041",
            filed_at=date(2025, 5, 28),
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581025000041/nvda-20250528.htm"
            ),
            content_hash="a" * 64,
            corpus_version="NVDA-v1",
            chunk_count=2,
        )

    monkeypatch.setattr(ingest_module, "ingest_sec", fake_ingest_sec)

    result = runner.invoke(
        app,
        [
            "ingest",
            "--ticker",
            "nvda",
            "--forms",
            "10-q,8-K",
            "--as-of-date",
            "2025-05-28",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("nvda", ["10-q", "8-K"], date(2025, 5, 28), user_agent)]
    assert wired["sec_cache_ttl_seconds"] == 43_200
    assert wired["configured_issuer_domains"] == {
        "NVDA": frozenset({"investor.nvidia.com"})
    }
    assert isinstance(wired["fact_repository"], CompanyFactRepository)
    assert callable(getattr(wired["company_facts_gateway"], "fetch_company_facts"))
    assert getattr(wired["company_facts_gateway"], "_max_response_bytes") == 12_345
    assert getattr(wired["company_facts_gateway"], "_rate_limiter") is wired["rate_limiter"]
    assert callable(getattr(wired["cache"], "get_json_sync"))
    assert json.loads(result.output)["selected_form"] == "10-Q"


@pytest.mark.live_sec
def test_real_nvda_sec_ingest_smoke(tmp_path: Path) -> None:
    """Explicit opt-in smoke test; never selected by the default pytest command."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'live.sqlite3'}")
    create_schema(engine)

    summary = ingest_sec(
        "NVDA",
        ["10-Q"],
        date.today(),
        repository=FilingRepository(engine),
        user_agent="Example Research Operator research-operator@example.com",
    )

    assert summary.ticker == "NVDA"
    assert summary.selected_form == "10-Q"
    assert summary.source_url.startswith("https://www.sec.gov/Archives/")
    assert len(summary.content_hash) == 64


@pytest.mark.live_sec
def test_real_nvda_companyfacts_smoke() -> None:
    """Explicit opt-in Company Facts smoke; default tests remain fixture-only."""
    document = HttpCompanyFactsGateway().fetch_company_facts(
        "NVDA",
        "0001045810",
        "Example Research Operator research-operator@example.com",
    )
    snapshot = normalize_company_facts(
        document,
        ticker="NVDA",
        cik="0001045810",
    )

    assert snapshot.ticker == "NVDA"
    assert snapshot.cik == "0001045810"
    assert snapshot.facts
    assert all(fact.form in {"10-K", "10-Q", "8-K"} for fact in snapshot.facts)


@pytest.mark.live_provider
def test_live_p1_application_smoke(tmp_path: Path) -> None:
    api_key, fast_model, analyst_model, embedding_cache_dir = _require_live_values(
        "OPENAI_API_KEY",
        "FAST_MODEL",
        "ANALYST_MODEL",
        "EMBEDDING_CACHE_DIR",
    )
    database_url = f"sqlite+pysqlite:///{tmp_path / 'live-p1.sqlite3'}"
    env = {
        "DATABASE_URL": database_url,
        "REDIS_URL": "",
        "OPENAI_API_KEY": api_key,
        "FAST_MODEL": fast_model,
        "ANALYST_MODEL": analyst_model,
        "EMBEDDING_CACHE_DIR": embedding_cache_dir,
        "TAVILY_API_KEY": "",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "LANGFUSE_HOST": "",
        "ALPACA_API_KEY_ID": "",
        "ALPACA_API_SECRET_KEY": "",
    }
    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            "--fixture",
            "tests/fixtures/nvda_10q.html",
            "--ticker",
            "NVDA",
            "--form",
            "10-Q",
        ],
        env=env,
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Analyze the latest earnings evidence and risks.",
            "--mode",
            "earnings-review",
        ],
        env=env,
    )

    assert ingest_result.exit_code == 0, ingest_result.output
    assert result.exit_code == 0, result.output
    assert (
        "> Information gap: allowlisted web fallback unavailable; local evidence only."
        in result.output
    )
    assert r"Recipe: earnings\_review" in result.output
    assert r"Recipe: financial\_data\_verification" in result.output
    assert "Research assistance only; not investment advice." in result.output

    engine = create_engine(database_url)
    with Session(engine) as session:
        research_runs = session.scalars(select(ResearchRun)).all()
        skill_runs = session.scalars(select(SkillRun)).all()
        source_fetches = session.scalars(select(SourceFetchRecord)).all()
        chunks = session.scalars(select(Chunk)).all()

    assert len(research_runs) == 1
    assert research_runs[0].status in {"completed", "partial"}
    assert research_runs[0].effective_intent == "earnings_review_request"
    assert research_runs[0].report_markdown == result.output
    assert [run.recipe_name for run in skill_runs] == [
        "earnings_review",
        "financial_data_verification",
    ]
    assert all(run.status in {"completed", "partial"} for run in skill_runs)
    assert source_fetches
    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)
    assert {chunk.embedding_model for chunk in chunks} == {"sentence-transformers:BAAI/bge-m3"}


@pytest.mark.live_provider
@pytest.mark.asyncio
async def test_real_tavily_provider_smoke() -> None:
    api_key = _require_live_values("TAVILY_API_KEY")[0]
    try:
        provider = TavilySearchProvider(api_key)
    except RuntimeError as error:
        raise AssertionError(
            "live Tavily smoke requires the optional dependency set; run uv sync --frozen "
            "--all-groups --extra web-search before selecting this check"
        ) from error

    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    policy = build_standard_source_policy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )
    evidence = await AllowlistedWebGateway(
        provider=provider,
        redirect_resolver=HttpxRedirectResolver(timeout_seconds=10),
        source_policy=policy,
        repository=WebEvidenceRepository(engine),
    ).search(
        ticker="NVDA",
        query="NVIDIA quarterly results",
        max_results=3,
    )

    assert evidence
    assert len(evidence) <= 3
    assert all(item.published_at is not None for item in evidence)
    assert all(item.published_at <= item.fetched_at for item in evidence)
    assert all(policy.classify(ticker="NVDA", url=item.source_url) for item in evidence)
    assert all(WebEvidenceRepository(engine).get_many([item.id]) == [item] for item in evidence)
