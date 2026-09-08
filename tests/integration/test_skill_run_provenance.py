"""Integration coverage for skill-run provenance across SEC and web evidence."""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import financial_evidence_agent.domain as domain
from financial_evidence_agent.domain import (
    SourceKind,
    SourceTier,
    WebEvidence,
)
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.models import Chunk, Filing, SkillRun, WebEvidenceRecord
from financial_evidence_agent.storage.repositories import ChunkToStore, FilingRepository
from financial_evidence_agent.storage.run_repositories import ResearchRunRepository, RunStart
from financial_evidence_agent.storage.web_repositories import (
    SkillRunRepository,
    WebEvidenceRepository,
)


@pytest.fixture
def engine():
    database_engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(database_engine)
    return database_engine


def _start_application_run(engine, run_id: str) -> str:
    ResearchRunRepository(engine).start(
        RunStart(
            run_id=run_id,
            ticker="NVDA",
            request="Review public evidence.",
            requested_intent="company-profile",
        )
    )
    return run_id


def test_skill_run_records_exact_recipe_and_mixed_sources_without_polluting_filings(
    engine,
) -> None:
    """Web snapshots must never be represented as filing or chunk records."""
    filing_repository = FilingRepository(engine)
    corpus_version = filing_repository.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000001",
        filed_at=date(2026, 5, 28),
        source_url="https://www.sec.gov/Archives/nvda-10q.htm",
        raw_text="Revenue increased.",
        content_hash="sec-content-hash",
        chunks=[
            ChunkToStore(
                section="MD&A",
                chunk_index=0,
                content="Revenue increased.",
                token_count=2,
                raw_start=0,
                raw_end=18,
            )
        ],
    )
    sec_source_id = filing_repository.list_chunks("NVDA", corpus_version)[0].id
    with Session(engine) as session:
        filing_count_before = session.scalar(select(func.count()).select_from(Filing))
        chunk_count_before = session.scalar(select(func.count()).select_from(Chunk))

    web_source = WebEvidenceRepository(engine).upsert(
        WebEvidence(
            id="caller-supplied-id",
            ticker="NVDA",
            title="NVIDIA quarterly results",
            content="Revenue increased year over year.",
            source_url="https://investor.nvidia.com/results",
            source_kind=SourceKind.ISSUER_IR,
            source_tier=SourceTier.PRIMARY,
            published_at=datetime(2026, 5, 28, tzinfo=UTC),
            fetched_at=datetime(2026, 5, 29, 9, 30, tzinfo=UTC),
            content_hash="sha256:web-snapshot",
        )
    )
    recipe_snapshot = {
        "name": "deep_research",
        "version": "1.0.0",
        "budget": {"max_rounds": 2, "max_web_calls": 1},
        "allowed_tools": ["search_filings", "search_allowlisted_web"],
    }
    run_repository = SkillRunRepository(engine)
    application_run_id = _start_application_run(engine, "application-run-mixed")
    run_id = run_repository.start(
        application_run_id=application_run_id,
        ticker="nvda",
        recipe_name="deep_research",
        recipe_version="1.0.0",
        recipe_snapshot=recipe_snapshot,
    )
    run_repository.finish(
        run_id,
        status="completed",
        source_ids=[
            domain.SourceRef(
                ticker="NVDA",
                kind=domain.SourceRefKind.FILING,
                source_id=sec_source_id,
            ),
            domain.SourceRef(
                ticker="NVDA",
                kind=domain.SourceRefKind.WEB,
                source_id=web_source.id,
            ),
        ],
        errors=[],
    )

    with Session(engine) as session:
        run = session.get(SkillRun, run_id)
        filing_count_after = session.scalar(select(func.count()).select_from(Filing))
        chunk_count_after = session.scalar(select(func.count()).select_from(Chunk))
        web_count = session.scalar(select(func.count()).select_from(WebEvidenceRecord))

    assert run is not None
    assert run.ticker == "NVDA"
    assert run.recipe_name == "deep_research"
    assert run.recipe_version == "1.0.0"
    assert run.recipe_snapshot == recipe_snapshot
    assert run.status == "completed"
    assert run.source_ids == [
        f"NVDA:filing:{sec_source_id}",
        f"NVDA:web:{web_source.id}",
    ]
    assert run.errors == []
    assert run.completed_at is not None
    assert (filing_count_after, chunk_count_after, web_count) == (
        filing_count_before,
        chunk_count_before,
        1,
    )


def test_skill_run_requires_and_persists_owning_application_run_id(engine) -> None:
    """A recipe execution cannot exist without its owning application lifecycle."""
    ResearchRunRepository(engine).start(
        RunStart(
            run_id="application-run-1",
            ticker="NVDA",
            request="Review public earnings evidence.",
            requested_intent="earnings-review",
        )
    )
    repository = SkillRunRepository(engine)

    skill_run_id = repository.start(
        application_run_id="application-run-1",
        ticker="NVDA",
        recipe_name="earnings_review",
        recipe_version="1.0.0",
        recipe_snapshot={"name": "earnings_review", "version": "1.0.0"},
    )

    with Session(engine) as session:
        skill_run = session.get(SkillRun, skill_run_id)
    assert skill_run is not None
    assert skill_run.run_id == "application-run-1"

    with pytest.raises(ValueError, match="application run"):
        repository.start(
            application_run_id="missing-application-run",
            ticker="NVDA",
            recipe_name="earnings_review",
            recipe_version="1.0.0",
            recipe_snapshot={"name": "earnings_review", "version": "1.0.0"},
        )


def test_skill_run_persists_partial_as_a_distinct_final_status(engine) -> None:
    """A useful but incomplete guarded result must not be recorded as completed."""
    repository = SkillRunRepository(engine)
    run_id = repository.start(
        application_run_id=_start_application_run(engine, "application-run-partial"),
        ticker="NVDA",
        recipe_name="financial_data_verification",
        recipe_version="1.0.0",
        recipe_snapshot={"name": "financial_data_verification", "version": "1.0.0"},
    )

    repository.finish(
        run_id,
        status="partial",
        source_ids=[
            domain.SourceRef(
                ticker="NVDA",
                kind=domain.SourceRefKind.FILING,
                source_id="sec-source",
            )
        ],
        errors=["FINANCIAL_DATA_DISCREPANCY: Revenue"],
    )

    with Session(engine) as session:
        run = session.get(SkillRun, run_id)

    assert run is not None
    assert run.status == "partial"
    assert run.source_ids == ["NVDA:filing:sec-source"]


def test_skill_run_rejects_cross_ticker_source_reference(engine) -> None:
    """A source already namespaced to AMD cannot be attached to an NVDA skill run."""
    repository = SkillRunRepository(engine)
    run_id = repository.start(
        application_run_id=_start_application_run(engine, "application-run-cross-ticker"),
        ticker="NVDA",
        recipe_name="earnings_review",
        recipe_version="1.0.0",
        recipe_snapshot={"name": "earnings_review", "version": "1.0.0"},
    )

    with pytest.raises(ValueError, match="ticker"):
        repository.finish(
            run_id,
            status="completed",
            source_ids=[
                domain.SourceRef(
                    ticker="AMD",
                    kind=domain.SourceRefKind.FILING,
                    source_id="amd-source",
                )
            ],
            errors=[],
        )
