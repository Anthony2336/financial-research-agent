"""Persistence contracts for research-run provenance."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import fra.domain as domain
from fra.storage.database import create_schema
from fra.storage.models import ClaimRecord, ResearchRun
from fra.storage.run_repositories import (
    PersistedClaim,
    ResearchRunRepository,
    RunFinish,
    RunStart,
)


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    return engine


def test_run_repository_persists_completed_guarded_claims(sqlite_engine) -> None:
    """Dropping source references would make retained output unauditable."""
    repository = ResearchRunRepository(sqlite_engine)
    repository.start(RunStart(run_id="run-1", ticker="NVDA", request="question"))
    repository.finish(
        RunFinish(
            run_id="run-1",
            effective_intent="research_request",
            status="completed",
            corpus_scope=["NVDA-v1"],
            prompt_version="research-v1",
            trace_id="trace-1",
            report_markdown="# report",
            claims=[
                PersistedClaim(
                    kind="verified_fact",
                    text="Revenue increased.",
                    confidence="high",
                    source_refs=[
                        domain.SourceRef(
                            ticker="NVDA",
                            kind=domain.SourceRefKind.FILING,
                            source_id="chunk-1",
                        )
                    ],
                    guard_status="retained",
                )
            ],
        )
    )

    stored = repository.get("run-1")

    assert stored.status == "completed"
    assert stored.claims[0].source_refs == [
        domain.SourceRef(
            ticker="NVDA", kind=domain.SourceRefKind.FILING, source_id="chunk-1"
        )
    ]


def test_run_repository_reads_pre_backfill_claim_refs_as_typed_fallbacks(
    sqlite_engine,
) -> None:
    """Legacy filing IDs stay filing-typed while unknown naked IDs stay unresolved."""
    repository = ResearchRunRepository(sqlite_engine)
    repository.start(RunStart(run_id="run-legacy", ticker="NVDA", request="question"))
    with Session(sqlite_engine) as session, session.begin():
        run = session.scalar(select(ResearchRun).where(ResearchRun.run_id == "run-legacy"))
        assert run is not None
        session.add(
            ClaimRecord(
                id="legacy-claim",
                run_id=run.id,
                kind="verified_fact",
                text="Legacy claim.",
                confidence="high",
                evidence_chunk_ids=["legacy-filing"],
                source_refs=["legacy-unknown"],
                guard_status="retained",
            )
        )

    stored = repository.get("run-legacy")

    assert stored.claims[0].source_refs == [
        domain.SourceRef(
            ticker="NVDA",
            kind=domain.SourceRefKind.FILING,
            source_id="legacy-filing",
        ),
        domain.UnresolvedSourceRef(
            ticker="NVDA",
            source_id="legacy-unknown",
            reason="legacy_unresolved",
        ),
    ]
