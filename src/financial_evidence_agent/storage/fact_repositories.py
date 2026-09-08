"""Narrow immutable persistence boundary for normalized SEC Company Facts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from financial_evidence_agent.retrieval.xbrl import CompanyFact
from financial_evidence_agent.storage.models import Company, CompanyFactRecord


class CompanyFactRepository:
    """Persist exact SEC facts and expose only ticker/concept-scoped reads."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Return the shared engine required by the atomic ingestion unit of work."""
        return self._engine

    def save_facts(self, facts: Sequence[CompanyFact]) -> list[CompanyFact]:
        """Idempotently persist a single issuer's validated immutable fact batch."""
        ticker, cik, unique = _validated_fact_batch(facts)
        if not unique:
            return []

        try:
            with Session(self._engine) as session, session.begin():
                return self._save_validated_facts_in_session(
                    session,
                    ticker=ticker,
                    cik=cik,
                    unique=unique,
                )
        except IntegrityError:
            canonical = self._get_many(list(unique))
            if len(canonical) != len(unique) or any(
                not _same_business_fact(stored, unique[stored.id]) for stored in canonical
            ):
                raise
            return canonical

    def save_facts_in_session(
        self,
        session: Session,
        facts: Sequence[CompanyFact],
    ) -> list[CompanyFact]:
        """Save a validated batch without owning or committing the transaction."""
        if session.get_bind() is not self._engine:
            raise ValueError("fact session must use the repository engine")
        ticker, cik, unique = _validated_fact_batch(facts)
        if not unique:
            return []
        return self._save_validated_facts_in_session(
            session,
            ticker=ticker,
            cik=cik,
            unique=unique,
        )

    def _save_validated_facts_in_session(
        self,
        session: Session,
        *,
        ticker: str,
        cik: str,
        unique: dict[str, CompanyFact],
    ) -> list[CompanyFact]:
        self._validate_company_scope(session, ticker=ticker, cik=cik)
        stored = {
            record.id: record
            for record in session.scalars(
                select(CompanyFactRecord).where(CompanyFactRecord.id.in_(list(unique)))
            ).all()
        }
        result: list[CompanyFact] = []
        for fact in unique.values():
            record = stored.get(fact.id)
            if record is not None:
                canonical = _fact(record)
                if not _same_business_fact(canonical, fact):
                    raise ValueError("company fact id conflicts with persisted exact fact")
                result.append(canonical)
                continue
            record = _record(fact)
            session.add(record)
            session.flush()
            result.append(_fact(record))
        return result

    def list_facts(
        self,
        ticker: str,
        *,
        concepts: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[CompanyFact]:
        """Return a bounded deterministic read for Task 10 financial verification."""
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker or not 1 <= limit <= 1_000:
            raise ValueError("company fact read scope is invalid")
        normalized_concepts: list[str] | None = None
        if concepts is not None:
            if isinstance(concepts, (str, bytes)):
                raise ValueError("concepts must be a sequence")
            normalized_concepts = [concept.strip() for concept in concepts]
            if not normalized_concepts or any(not concept for concept in normalized_concepts):
                raise ValueError("concepts must contain non-empty SEC concept names")
        with Session(self._engine) as session:
            statement = (
                select(CompanyFactRecord)
                .where(CompanyFactRecord.ticker == normalized_ticker)
                .order_by(
                    CompanyFactRecord.filed_at.desc(),
                    CompanyFactRecord.taxonomy,
                    CompanyFactRecord.concept,
                    CompanyFactRecord.id,
                )
                .limit(limit)
            )
            if normalized_concepts is not None:
                statement = statement.where(CompanyFactRecord.concept.in_(normalized_concepts))
            facts = [_fact(record) for record in session.scalars(statement).all()]
            ciks = {fact.cik for fact in facts}
            if len(ciks) > 1:
                raise ValueError("company fact read contains conflicting CIK scope")
            if ciks:
                self._validate_company_scope(
                    session,
                    ticker=normalized_ticker,
                    cik=next(iter(ciks)),
                )
            return facts

    @staticmethod
    def _validate_company_scope(session: Session, *, ticker: str, cik: str) -> None:
        company = session.scalar(select(Company).where(Company.ticker == ticker))
        if company is None or company.cik is None:
            raise ValueError("company fact ticker has no validated company CIK")
        try:
            matches = int(company.cik) == int(cik)
        except ValueError as error:
            raise ValueError("persisted company CIK is malformed") from error
        if not matches:
            raise ValueError("company fact CIK conflicts with persisted ticker ownership")

    def _get_many(self, fact_ids: Sequence[str]) -> list[CompanyFact]:
        with Session(self._engine) as session:
            records = session.scalars(
                select(CompanyFactRecord).where(CompanyFactRecord.id.in_(fact_ids))
            ).all()
            by_id = {record.id: _fact(record) for record in records}
            return [by_id[fact_id] for fact_id in fact_ids if fact_id in by_id]


def _record(fact: CompanyFact) -> CompanyFactRecord:
    return CompanyFactRecord(
        id=fact.id,
        ticker=fact.ticker,
        cik=fact.cik,
        taxonomy=fact.taxonomy,
        concept=fact.concept,
        period_start=fact.period_start,
        period_end=fact.period_end,
        instant=fact.instant,
        unit=fact.unit,
        currency=fact.currency,
        value=fact.value,
        form=fact.form,
        filed_at=fact.filed_at,
        accession_no=fact.accession_no,
        source_url=fact.source_url,
        frame=fact.frame,
        raw_content_hash=fact.raw_content_hash,
        fetched_at=fact.fetched_at,
    )


def _fact(record: CompanyFactRecord) -> CompanyFact:
    return CompanyFact(
        id=record.id,
        ticker=record.ticker,
        cik=record.cik,
        taxonomy=record.taxonomy,
        concept=record.concept,
        period_start=record.period_start,
        period_end=record.period_end,
        instant=record.instant,
        unit=record.unit,
        currency=record.currency,
        value=record.value,
        form=record.form,
        filed_at=record.filed_at,
        accession_no=record.accession_no,
        source_url=record.source_url,
        frame=record.frame,
        raw_content_hash=record.raw_content_hash,
        fetched_at=_utc(record.fetched_at),
    )


def _same_business_fact(left: CompanyFact, right: CompanyFact) -> bool:
    left_values = left.model_dump(mode="python", exclude={"fetched_at", "raw_content_hash"})
    right_values = right.model_dump(mode="python", exclude={"fetched_at", "raw_content_hash"})
    return left_values == right_values


def _validated_fact_batch(
    facts: Sequence[CompanyFact],
) -> tuple[str, str, dict[str, CompanyFact]]:
    if isinstance(facts, (str, bytes)):
        raise ValueError("facts must be a sequence of CompanyFact values")
    validated: list[CompanyFact] = []
    for fact in facts:
        try:
            validated.append(CompanyFact.model_validate(fact.model_dump(mode="python")))
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise ValueError("company fact is malformed or has invalid identity") from error
    if not validated:
        return "", "", {}
    scopes = {(fact.ticker, fact.cik) for fact in validated}
    if len(scopes) != 1:
        raise ValueError("company fact batch contains conflicting ticker or CIK scope")
    ticker, cik = next(iter(scopes))
    unique: dict[str, CompanyFact] = {}
    for fact in validated:
        existing = unique.get(fact.id)
        if existing is not None and not _same_business_fact(existing, fact):
            raise ValueError("company fact batch contains conflicting duplicate facts")
        unique[fact.id] = existing or fact
    return ticker, cik, unique


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
