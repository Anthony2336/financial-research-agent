"""Persistence for completed research-run provenance."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from fra.domain import (
    SourceRef,
    SourceReference,
    SourceRefKind,
    StrictModel,
    decode_stored_source_ref,
)
from fra.storage.models import ClaimRecord, ResearchRun, SourceFetchRecord


class RunStart(StrictModel):
    """Immutable input recorded when a research run begins."""

    run_id: str
    ticker: str
    request: str
    requested_intent: str | None = None


class PersistedClaim(StrictModel):
    """A guarded claim eligible for final persisted output."""

    kind: str
    text: str
    confidence: str
    source_refs: list[SourceRef]
    guard_status: str


class SourceFetchWrite(StrictModel):
    """A source retrieval attempt associated with a research run."""

    run_id: str
    source_kind: str
    source_ref: str
    requested_at: datetime
    fetched_at: datetime | None
    status: str
    error_code: str | None = None


class RunFinish(StrictModel):
    """Final, guarded output written when a research run completes."""

    run_id: str
    effective_intent: str
    status: str
    corpus_scope: list[str]
    prompt_version: str | None
    trace_id: str | None
    report_markdown: str
    claims: list[PersistedClaim]


class StoredPersistedClaim(StrictModel):
    """A stored claim whose legacy references may remain explicitly unresolved."""

    kind: str
    text: str
    confidence: str
    source_refs: list[SourceReference]
    guard_status: str


class StoredResearchRun(StrictModel):
    """Completed run provenance materialized for callers."""

    run_id: str
    ticker: str
    request: str
    requested_intent: str | None
    effective_intent: str
    status: str
    corpus_scope: list[str]
    prompt_version: str | None
    trace_id: str | None
    report_markdown: str
    claims: list[StoredPersistedClaim]
    source_fetches: list[SourceFetchWrite]


class ResearchRunRepository:
    """Write research provenance only at guarded run boundaries."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def start(self, value: RunStart) -> None:
        """Record the original user request before research begins."""
        with Session(self._engine) as session, session.begin():
            session.add(_new_run_record(value))

    def record_fetch(self, value: SourceFetchWrite) -> None:
        """Record a source retrieval without changing completed claims."""
        with Session(self._engine) as session, session.begin():
            session.add(_new_source_fetch_record(value))

    def finish(self, value: RunFinish) -> None:
        """Persist final guarded claims and their completion provenance."""
        with Session(self._engine) as session, session.begin():
            run = _required_run(session, value.run_id)
            _apply_finish(run, value)
            session.add_all(_claim_records(run.id, value))

    def get(self, run_id: str) -> StoredResearchRun:
        """Read one run with detached claim and fetch provenance."""
        with Session(self._engine) as session:
            return _stored_run(_required_run(session, run_id))


def _new_run_record(value: RunStart) -> ResearchRun:
    return ResearchRun(
        id=str(uuid4()),
        run_id=value.run_id,
        ticker=value.ticker.upper(),
        thesis=value.request,
        status="running",
        corpus_version="",
        requested_intent=value.requested_intent,
        effective_intent=None,
        corpus_scope=[],
        prompt_version=None,
        trace_id=None,
        report_markdown=None,
        created_at=datetime.now(UTC),
        completed_at=None,
    )


def _new_source_fetch_record(value: SourceFetchWrite) -> SourceFetchRecord:
    return SourceFetchRecord(
        id=str(uuid4()),
        run_id=value.run_id,
        source_kind=value.source_kind,
        source_ref=value.source_ref,
        requested_at=value.requested_at,
        fetched_at=value.fetched_at,
        status=value.status,
        error_code=value.error_code,
    )


def _required_run(session: Session, run_id: str) -> ResearchRun:
    run = session.scalar(select(ResearchRun).where(ResearchRun.run_id == run_id))
    if run is None:
        raise ValueError(f"unknown research run: {run_id}")
    return run


def _apply_finish(run: ResearchRun, value: RunFinish) -> None:
    run.status = value.status
    run.effective_intent = value.effective_intent
    run.corpus_scope = list(value.corpus_scope)
    run.corpus_version = value.corpus_scope[-1] if value.corpus_scope else ""
    run.prompt_version = value.prompt_version
    run.trace_id = value.trace_id
    run.report_markdown = value.report_markdown
    run.completed_at = datetime.now(UTC)


def _claim_records(run_id: str, value: RunFinish) -> list[ClaimRecord]:
    return [
        ClaimRecord(
            id=str(uuid4()),
            run_id=run_id,
            kind=claim.kind,
            text=claim.text,
            confidence=claim.confidence,
            evidence_chunk_ids=[
                reference.source_id
                for reference in claim.source_refs
                if reference.kind is SourceRefKind.FILING
            ],
            source_refs=[reference.encode() for reference in claim.source_refs],
            guard_status=claim.guard_status,
        )
        for claim in value.claims
    ]


def _stored_run(run: ResearchRun) -> StoredResearchRun:
    return StoredResearchRun(
        run_id=run.run_id,
        ticker=run.ticker,
        request=run.thesis,
        requested_intent=run.requested_intent,
        effective_intent=run.effective_intent or "",
        status=run.status,
        corpus_scope=list(run.corpus_scope or []),
        prompt_version=run.prompt_version,
        trace_id=run.trace_id,
        report_markdown=run.report_markdown or "",
        claims=[
            StoredPersistedClaim(
                kind=claim.kind,
                text=claim.text,
                confidence=claim.confidence,
                source_refs=_stored_source_refs(run.ticker, claim),
                guard_status=claim.guard_status,
            )
            for claim in run.claims
        ],
        source_fetches=[
            SourceFetchWrite(
                run_id=run.run_id,
                source_kind=fetch.source_kind,
                source_ref=fetch.source_ref,
                requested_at=fetch.requested_at,
                fetched_at=fetch.fetched_at,
                status=fetch.status,
                error_code=fetch.error_code,
            )
            for fetch in run.source_fetches
        ],
    )


def _stored_source_refs(ticker: str, claim: ClaimRecord) -> list[SourceReference]:
    values: list[SourceReference] = [
        SourceRef(ticker=ticker, kind=SourceRefKind.FILING, source_id=source_id)
        for source_id in claim.evidence_chunk_ids
    ]
    values.extend(
        decode_stored_source_ref(source_ref, ticker=ticker)
        for source_ref in claim.source_refs or []
    )
    deduplicated: list[SourceReference] = []
    seen: set[str] = set()
    for value in values:
        key = value.encode() if isinstance(value, SourceRef) else value.storage_value()
        if key not in seen:
            deduplicated.append(value)
            seen.add(key)
    return deduplicated
