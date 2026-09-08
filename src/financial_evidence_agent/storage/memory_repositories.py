"""PostgreSQL/SQLite repository for citation-bound research memories."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from math import sqrt
from typing import Protocol

from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from financial_evidence_agent.domain import SourceRef, SourceRefKind, WebEvidence
from financial_evidence_agent.memory.research import ResearchMemory, ResearchMemoryKind
from financial_evidence_agent.retrieval.indexing import PERSISTED_EMBEDDING_DIMENSIONS
from financial_evidence_agent.storage.models import (
    Chunk,
    ClaimRecord,
    Company,
    CorpusFiling,
    Filing,
    ResearchCorpus,
    ResearchMemoryRecord,
    ResearchRun,
    WebEvidenceRecord,
)

logger = logging.getLogger(__name__)
_ALLOWED_KINDS = tuple(kind.value for kind in ResearchMemoryKind)
_FINAL_RESEARCH_STATUSES = frozenset({"completed", "partial"})
_MAX_RESULTS = 3
_SEARCH_PAGE_SIZE = 8


class CurrentWebEvidenceValidator(Protocol):
    """Narrow current-policy validator required for web-backed memory."""

    def validate(self, *, ticker: str, evidence: WebEvidence) -> object:
        """Reject any noncanonical row or source outside the active policy."""


class ResearchMemoryRepository:
    """Persist/query memories only after revalidating durable citation ownership."""

    def __init__(
        self,
        engine: Engine,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        web_evidence_validator: CurrentWebEvidenceValidator | None = None,
    ) -> None:
        self._engine = engine
        self._clock = clock
        self._web_evidence_validator = web_evidence_validator

    def store_guarded(
        self,
        *,
        ticker: str,
        memory_kind: ResearchMemoryKind,
        summary: str,
        source_run_id: str,
        evidence_source_refs: tuple[SourceRef, ...],
        corpus_version: str,
        embedding: list[float],
        embedding_model: str,
        importance: float,
        created_at: datetime,
        expires_at: datetime,
    ) -> ResearchMemory:
        """Idempotently store one valid post-guard memory and never mutate it."""
        normalized_ticker = ticker.strip().upper()
        encoded_refs = tuple(sorted(reference.encode() for reference in evidence_source_refs))
        memory_id = _stable_memory_id(
            normalized_ticker,
            memory_kind,
            summary,
            source_run_id,
            encoded_refs,
        )
        memory = ResearchMemory(
            id=memory_id,
            scope_key=f"ticker:{normalized_ticker}",
            ticker=normalized_ticker,
            memory_kind=memory_kind,
            summary=summary,
            source_run_id=source_run_id,
            evidence_source_refs=tuple(
                SourceRef.decode(value, expected_ticker=normalized_ticker)
                for value in encoded_refs
            ),
            corpus_version=corpus_version,
            embedding=tuple(embedding),
            embedding_model=embedding_model,
            importance=importance,
            created_at=created_at,
            expires_at=expires_at,
        )
        if memory.expires_at <= self._aware_now():
            raise ValueError("research memory expires_at must be in the future")

        with Session(self._engine) as session:
            self._validate_binding(session, memory)
            existing = session.get(ResearchMemoryRecord, memory.id)
            if existing is not None:
                return self._validated_memory(session, existing)

        try:
            with Session(self._engine) as session, session.begin():
                self._validate_binding(session, memory)
                session.add(_memory_record(memory))
        except IntegrityError:
            with Session(self._engine) as session:
                existing = session.get(ResearchMemoryRecord, memory.id)
                if existing is None:
                    raise
                return self._validated_memory(session, existing)
        return memory

    def search(
        self,
        ticker: str,
        query_embedding: list[float],
        limit: int = 3,
        *,
        embedding_model: str,
        current_corpus_version: str | None = None,
    ) -> list[ResearchMemory]:
        """Return valid semantic Top-3 hints with deterministic tie-breaking."""
        if limit <= 0:
            raise ValueError("research memory search limit must be positive")
        if len(query_embedding) != PERSISTED_EMBEDDING_DIMENSIONS:
            raise ValueError("research memory query embedding must have 1024 dimensions")
        if not embedding_model.strip():
            raise ValueError("research memory embedding model must not be blank")
        bounded_limit = min(limit, _MAX_RESULTS)
        normalized_ticker = ticker.strip().upper()
        now = self._aware_now()
        with Session(self._engine) as session:
            statement = select(ResearchMemoryRecord).where(
                ResearchMemoryRecord.ticker == normalized_ticker,
                ResearchMemoryRecord.memory_kind.in_(_ALLOWED_KINDS),
                ResearchMemoryRecord.expires_at > now,
            )
            statement = statement.where(
                ResearchMemoryRecord.embedding_model == embedding_model
            )
            memories: list[ResearchMemory] = []

            def validate_batch(records: Sequence[ResearchMemoryRecord]) -> bool:
                for record in records:
                    try:
                        memory = self._validated_memory(
                            session,
                            record,
                            current_corpus_version=current_corpus_version,
                        )
                    except (TypeError, ValidationError, ValueError):
                        logger.warning("citation-invalid research memory excluded")
                        continue
                    memories.append(memory)
                    if len(memories) == bounded_limit:
                        return True
                return False

            if session.get_bind().dialect.name == "postgresql":
                distance = ResearchMemoryRecord.embedding.cosine_distance(query_embedding)
                ordered = statement.order_by(
                    distance,
                    ResearchMemoryRecord.importance.desc(),
                    ResearchMemoryRecord.id,
                )
                offset = 0
                while len(memories) < bounded_limit:
                    records = session.scalars(
                        ordered.limit(_SEARCH_PAGE_SIZE).offset(offset)
                    ).all()
                    if not records or validate_batch(records):
                        break
                    offset += len(records)
                    if len(records) < _SEARCH_PAGE_SIZE:
                        break
            else:
                records = sorted(
                    session.scalars(statement).all(),
                    key=lambda record: (
                        -_cosine_similarity(query_embedding, record.embedding),
                        -record.importance,
                        record.id,
                    ),
                )
                for offset in range(0, len(records), _SEARCH_PAGE_SIZE):
                    if validate_batch(records[offset : offset + _SEARCH_PAGE_SIZE]):
                        break
            return memories

    def _validated_memory(
        self,
        session: Session,
        record: ResearchMemoryRecord,
        *,
        current_corpus_version: str | None = None,
    ) -> ResearchMemory:
        memory = _research_memory(
            record,
            stale=(
                current_corpus_version is not None
                and record.corpus_version != current_corpus_version
            ),
        )
        if memory.expires_at <= self._aware_now():
            raise ValueError("research memory is expired")
        self._validate_binding(session, memory)
        return memory

    def _validate_binding(self, session: Session, memory: ResearchMemory) -> None:
        run = session.scalar(
            select(ResearchRun).where(ResearchRun.run_id == memory.source_run_id)
        )
        if (
            run is None
            or run.completed_at is None
            or run.status not in _FINAL_RESEARCH_STATUSES
        ):
            raise ValueError("source run must exist and be durably completed")
        if run.ticker != memory.ticker:
            raise ValueError("source run ticker does not match research memory ticker")
        if memory.corpus_version not in set(run.corpus_scope or []):
            raise ValueError("research memory corpus is absent from source run scope")

        corpus = session.scalar(
            select(ResearchCorpus)
            .join(ResearchCorpus.company)
            .where(
                Company.ticker == memory.ticker,
                ResearchCorpus.version == memory.corpus_version,
            )
        )
        if corpus is None:
            raise ValueError("research memory corpus does not belong to ticker")

        claims = session.scalars(
            select(ClaimRecord).where(
                ClaimRecord.run_id == run.id,
                ClaimRecord.guard_status == "retained",
            )
        ).all()
        authorized_by_claim = [
            (_normalized_text(claim.text), _claim_source_refs(memory.ticker, claim))
            for claim in claims
        ]
        requested = {reference.encode() for reference in memory.evidence_source_refs}
        if not any(
            requested <= claim_refs
            and normalized_claim == memory.summary
            for normalized_claim, claim_refs in authorized_by_claim
        ):
            raise ValueError("every research memory source must be a retained citation")

        filing_ids = {
            reference.source_id
            for reference in memory.evidence_source_refs
            if reference.kind is SourceRefKind.FILING
        }
        if filing_ids:
            valid_filing_ids = set(
                session.scalars(
                    select(Chunk.id)
                    .join(Filing, Filing.id == Chunk.filing_id)
                    .join(CorpusFiling, CorpusFiling.filing_id == Filing.id)
                    .join(ResearchCorpus, ResearchCorpus.id == CorpusFiling.corpus_id)
                    .join(Company, Company.id == ResearchCorpus.company_id)
                    .where(
                        Company.ticker == memory.ticker,
                        ResearchCorpus.version == memory.corpus_version,
                        Filing.company_id == ResearchCorpus.company_id,
                        Chunk.id.in_(filing_ids),
                    )
                ).all()
            )
            if valid_filing_ids != filing_ids:
                raise ValueError("filing citation does not belong to ticker/corpus")

        web_ids = {
            reference.source_id
            for reference in memory.evidence_source_refs
            if reference.kind is SourceRefKind.WEB
        }
        if web_ids:
            records = session.scalars(
                select(WebEvidenceRecord).where(WebEvidenceRecord.id.in_(web_ids))
            ).all()
            if {record.id for record in records} != web_ids:
                raise ValueError("web citation does not belong to memory ticker")
            if self._web_evidence_validator is None:
                raise ValueError("web citation requires a current source-policy validator")
            for record in records:
                self._web_evidence_validator.validate(
                    ticker=memory.ticker,
                    evidence=_web_evidence(record),
                )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("research memory clock must return an aware datetime")
        return value.astimezone(UTC)


def _stable_memory_id(
    ticker: str,
    memory_kind: ResearchMemoryKind,
    summary: str,
    source_run_id: str,
    encoded_refs: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "ticker": ticker,
            "memory_kind": memory_kind.value,
            "summary": _normalized_text(summary),
            "source_run_id": source_run_id,
            "evidence_source_refs": list(encoded_refs),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _memory_record(memory: ResearchMemory) -> ResearchMemoryRecord:
    return ResearchMemoryRecord(
        id=memory.id,
        scope_key=memory.scope_key,
        ticker=memory.ticker,
        memory_kind=memory.memory_kind.value,
        summary=memory.summary,
        source_run_id=memory.source_run_id,
        evidence_source_refs=[
            reference.encode() for reference in memory.evidence_source_refs
        ],
        corpus_version=memory.corpus_version,
        embedding=list(memory.embedding),
        embedding_model=memory.embedding_model,
        importance=memory.importance,
        created_at=memory.created_at,
        expires_at=memory.expires_at,
    )


def _research_memory(record: ResearchMemoryRecord, *, stale: bool) -> ResearchMemory:
    return ResearchMemory(
        id=record.id,
        scope_key=record.scope_key,
        ticker=record.ticker,
        memory_kind=record.memory_kind,
        summary=record.summary,
        source_run_id=record.source_run_id,
        evidence_source_refs=tuple(
            SourceRef.decode(value, expected_ticker=record.ticker)
            for value in record.evidence_source_refs
        ),
        corpus_version=record.corpus_version,
        embedding=tuple(float(value) for value in record.embedding),
        embedding_model=record.embedding_model,
        importance=record.importance,
        created_at=_as_aware(record.created_at),
        expires_at=_as_aware(record.expires_at),
        stale=stale,
    )


def _claim_source_refs(ticker: str, claim: ClaimRecord) -> set[str]:
    values = {
        SourceRef(
            ticker=ticker,
            kind=SourceRefKind.FILING,
            source_id=source_id,
        ).encode()
        for source_id in claim.evidence_chunk_ids
    }
    for value in claim.source_refs or []:
        try:
            values.add(SourceRef.decode(value, expected_ticker=ticker).encode())
        except ValueError:
            continue
    return values


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _web_evidence(record: WebEvidenceRecord) -> WebEvidence:
    return WebEvidence(
        id=record.id,
        ticker=record.ticker,
        title=record.title,
        content=record.content,
        source_url=record.source_url,
        source_kind=record.source_kind,
        source_tier=record.source_tier,
        published_at=(
            None if record.published_at is None else _as_aware(record.published_at)
        ),
        fetched_at=_as_aware(record.fetched_at),
        content_hash=record.content_hash,
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have matching dimensions")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
