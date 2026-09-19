"""Repositories for web snapshots and skill-run provenance."""

from collections.abc import Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fra.domain import (
    SourceRef,
    WebEvidence,
    canonicalize_source_url,
    content_addressed_web_evidence_id,
)
from fra.storage.models import ResearchRun, SkillRun, WebEvidenceRecord


class WebEvidenceRepository:
    """Persist versioned web snapshots outside the filing corpus."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, evidence: WebEvidence) -> WebEvidence | None:
        """Persist a snapshot once per ticker, normalized URL, and content hash."""
        published_at, fetched_at = _validated_input_times(evidence)
        ticker = evidence.ticker.upper()
        source_url = str(canonicalize_source_url(evidence.source_url))
        evidence_id = content_addressed_web_evidence_id(
            ticker,
            source_url,
            evidence.content_hash,
        )
        try:
            with Session(self._engine) as session, session.begin():
                existing = session.scalar(
                    select(WebEvidenceRecord).where(
                        WebEvidenceRecord.ticker == ticker,
                        WebEvidenceRecord.source_url == source_url,
                        WebEvidenceRecord.content_hash == evidence.content_hash,
                    )
                )
                if existing is not None:
                    return self._validated_evidence(existing)

                record = WebEvidenceRecord(
                    id=evidence_id,
                    ticker=ticker,
                    title=evidence.title,
                    content=evidence.content,
                    source_url=source_url,
                    source_kind=evidence.source_kind.value,
                    source_tier=evidence.source_tier.value,
                    published_at=published_at,
                    fetched_at=fetched_at,
                    content_hash=evidence.content_hash,
                    time_metadata_validated=True,
                )
                session.add(record)
                session.flush()
                return self._validated_evidence(record)
        except IntegrityError:
            with Session(self._engine) as session:
                existing = session.scalar(
                    select(WebEvidenceRecord).where(
                        WebEvidenceRecord.ticker == ticker,
                        WebEvidenceRecord.source_url == source_url,
                        WebEvidenceRecord.content_hash == evidence.content_hash,
                    )
                )
                if existing is None:
                    raise
                return self._validated_evidence(existing)

    def get_many(self, evidence_ids: Sequence[str]) -> list[WebEvidence]:
        """Return known snapshots in caller order, omitting unknown IDs."""
        if not evidence_ids:
            return []
        with Session(self._engine) as session:
            rows = session.scalars(
                select(WebEvidenceRecord).where(
                    WebEvidenceRecord.id.in_(evidence_ids),
                    WebEvidenceRecord.time_metadata_validated.is_(True),
                )
            ).all()
            records = {
                row.id: evidence
                for row in rows
                if (evidence := self._validated_evidence(row)) is not None
            }
            return [records[evidence_id] for evidence_id in evidence_ids if evidence_id in records]

    @staticmethod
    def _validated_evidence(record: WebEvidenceRecord) -> WebEvidence | None:
        if not record.time_metadata_validated:
            return None
        published_at = _utc_datetime(record.published_at)
        fetched_at = _utc_datetime(record.fetched_at)
        if published_at is None or fetched_at is None or published_at > fetched_at:
            return None
        return WebEvidence(
            id=record.id,
            ticker=record.ticker,
            title=record.title,
            content=record.content,
            source_url=record.source_url,
            source_kind=record.source_kind,
            source_tier=record.source_tier,
            published_at=published_at,
            fetched_at=fetched_at,
            content_hash=record.content_hash,
        )


class SkillRunRepository:
    """Record immutable recipe inputs and their final source provenance."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def start(
        self,
        *,
        application_run_id: str,
        ticker: str,
        recipe_name: str,
        recipe_version: str,
        recipe_snapshot: dict[str, object],
    ) -> str:
        """Start a skill run with a detached copy of its recipe snapshot."""
        run_id = str(uuid4())
        with Session(self._engine) as session, session.begin():
            parent_id = session.scalar(
                select(ResearchRun.id).where(ResearchRun.run_id == application_run_id)
            )
            if parent_id is None:
                raise ValueError("owning application run was not persisted")
            session.add(
                SkillRun(
                    id=run_id,
                    run_id=application_run_id,
                    ticker=ticker.upper(),
                    recipe_name=recipe_name,
                    recipe_version=recipe_version,
                    recipe_snapshot=deepcopy(recipe_snapshot),
                    status="running",
                    source_ids=[],
                    errors=[],
                    started_at=datetime.now(UTC),
                    completed_at=None,
                )
            )
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: Sequence[SourceRef],
        errors: Sequence[str],
    ) -> None:
        """Finish an existing run without modifying its recipe snapshot."""
        with Session(self._engine) as session, session.begin():
            run = session.get(SkillRun, run_id)
            if run is None:
                raise ValueError(f"unknown skill run: {run_id}")
            if any(reference.ticker != run.ticker for reference in source_ids):
                raise ValueError("source reference ticker does not match skill run ticker")
            run.status = status
            run.source_ids = [reference.encode() for reference in source_ids]
            run.errors = list(errors)
            run.completed_at = datetime.now(UTC)


def _utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validated_input_times(evidence: WebEvidence) -> tuple[datetime, datetime]:
    published_at = evidence.published_at
    fetched_at = evidence.fetched_at
    if (
        published_at is None
        or published_at.tzinfo is None
        or published_at.utcoffset() is None
        or fetched_at.tzinfo is None
        or fetched_at.utcoffset() is None
    ):
        raise ValueError("web evidence timestamps must be aware and complete")
    normalized_published = published_at.astimezone(UTC)
    normalized_fetched = fetched_at.astimezone(UTC)
    if normalized_published > normalized_fetched:
        raise ValueError("web evidence publication timestamp must not follow fetched_at")
    return normalized_published, normalized_fetched
