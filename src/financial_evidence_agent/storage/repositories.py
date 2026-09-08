"""Repository methods that preserve filing provenance independently of retrieval rank."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from math import sqrt
from threading import Lock
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from sqlalchemy import Engine, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from financial_evidence_agent.domain import EvidenceChunk
from financial_evidence_agent.storage.models import (
    Chunk,
    Company,
    CorpusFiling,
    Filing,
    ResearchCorpus,
)

if TYPE_CHECKING:
    from financial_evidence_agent.retrieval.xbrl import CompanyFact

_ALLOWED_FORMS = frozenset({"10-K", "10-Q", "8-K"})
_EMBEDDING_LOCKS: dict[tuple[str, str], Lock] = {}
_EMBEDDING_LOCKS_GUARD = Lock()


@dataclass(frozen=True)
class ChunkToStore:
    """A parsed filing span ready to persist without an embedding."""

    section: str
    chunk_index: int
    content: str
    token_count: int
    raw_start: int
    raw_end: int


@dataclass(frozen=True)
class FilingToStore:
    """One fully fetched and parsed filing ready for one atomic ingest commit."""

    form: str
    accession_no: str
    filed_at: date
    source_url: str
    raw_text: str
    content_hash: str
    chunks: tuple[ChunkToStore, ...]
    embedding_model: str | None = None
    embedding_vectors: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        vectors = self.embedding_vectors
        if vectors is None:
            if self.embedding_model is not None and not self.embedding_model.strip():
                raise ValueError("embedding model must not be blank")
            return
        if self.embedding_model is None or not self.embedding_model.strip():
            raise ValueError("prepared vectors require an embedding model")
        if len(vectors) != len(self.chunks) or any(len(vector) != 1024 for vector in vectors):
            raise ValueError("prepared filing embeddings must match 1024-dimensional chunks")


@dataclass(frozen=True)
class AtomicIngestResult:
    """Identifiers produced by one committed explicit-ingestion transaction."""

    corpus_version: str
    filing_ids: tuple[str, ...]


class EmbeddingStateConflictError(RuntimeError):
    """The desired embedding state changed before the atomic commit."""


class AtomicFactWriter(Protocol):
    """Session-bound fact writer used by the explicit ingestion unit of work."""

    @property
    def engine(self) -> Engine:
        """Return the exact engine shared by the filing unit of work."""

    def save_facts_in_session(
        self,
        session: Session,
        facts: Sequence[CompanyFact],
    ) -> list[CompanyFact]:
        """Validate and save facts without committing the caller's transaction."""


@dataclass(frozen=True)
class CompanyRead:
    """Public company metadata available to read-only consumers."""

    ticker: str
    cik: str | None
    legal_name: str | None
    ir_domain: str | None


@dataclass(frozen=True)
class FilingRead:
    """Public filing metadata available to read-only consumers."""

    id: str
    ticker: str
    form: str
    filed_at: date
    source_url: str
    accession_no: str
    corpus_version: str
    content_hash: str


class FilingRepository:
    """Persist filings idempotently and return their citable chunks."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Return the shared persistence engine for adjacent repositories."""
        return self._engine

    def upsert_company_metadata(
        self,
        *,
        ticker: str,
        cik: str,
        legal_name: str,
        ir_domain: str | None,
    ) -> CompanyRead:
        """Persist validated SEC issuer identity and one configured IR host."""
        normalized_ticker, normalized_cik, normalized_name, normalized_domain = (
            _normalized_company_metadata(
                ticker=ticker,
                cik=cik,
                legal_name=legal_name,
                ir_domain=ir_domain,
            )
        )

        with Session(self._engine) as session, session.begin():
            self._lock_ticker(session, normalized_ticker)
            company = self._upsert_company_metadata_in_session(
                session,
                ticker=normalized_ticker,
                cik=normalized_cik,
                legal_name=normalized_name,
                ir_domain=normalized_domain,
            )
            return CompanyRead(
                ticker=company.ticker,
                cik=company.cik,
                legal_name=company.legal_name,
                ir_domain=company.ir_domain,
            )

    @staticmethod
    def _upsert_company_metadata_in_session(
        session: Session,
        *,
        ticker: str,
        cik: str,
        legal_name: str,
        ir_domain: str | None,
    ) -> Company:
        company = session.scalar(select(Company).where(Company.ticker == ticker))
        if company is None:
            company = Company(id=str(uuid4()), ticker=ticker)
            session.add(company)
        elif company.cik is not None:
            try:
                same_cik = int(company.cik) == int(cik)
            except ValueError as error:
                raise ValueError("persisted company CIK metadata is invalid") from error
            if not same_cik:
                raise ValueError("company CIK metadata conflicts with persisted issuer")
        company.cik = cik
        company.legal_name = legal_name
        company.ir_domain = ir_domain
        session.flush()
        return company

    @staticmethod
    def _lock_ticker(session: Session, ticker: str) -> None:
        if session.get_bind().dialect.name == "postgresql":
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:ticker))"),
                {"ticker": ticker},
            )

    def store_filing(
        self,
        *,
        ticker: str,
        form: str,
        accession_no: str,
        filed_at: date,
        source_url: str,
        raw_text: str,
        content_hash: str,
        chunks: Iterable[ChunkToStore],
    ) -> str:
        """Store one filing and expose it as a backward-compatible one-filing snapshot."""
        filing_id = self.store_filing_record(
            ticker=ticker,
            form=form,
            accession_no=accession_no,
            filed_at=filed_at,
            source_url=source_url,
            raw_text=raw_text,
            content_hash=content_hash,
            chunks=chunks,
        )
        return self.create_corpus(ticker, [filing_id], filed_at)

    def store_filing_record(
        self,
        *,
        ticker: str,
        form: str,
        accession_no: str,
        filed_at: date,
        source_url: str,
        raw_text: str,
        content_hash: str,
        chunks: Iterable[ChunkToStore],
    ) -> str:
        """Store one immutable filing record without selecting corpus membership."""
        normalized_ticker = ticker.upper()
        with Session(self._engine) as session, session.begin():
            if session.get_bind().dialect.name == "postgresql":
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:ticker))"),
                    {"ticker": normalized_ticker},
                )
            company = session.scalar(select(Company).where(Company.ticker == normalized_ticker))
            if company is None:
                company = Company(id=str(uuid4()), ticker=normalized_ticker)
                session.add(company)
                session.flush()

            existing = session.scalar(
                select(Filing).where(
                    Filing.company_id == company.id,
                    Filing.content_hash == content_hash,
                )
            )
            if existing is not None:
                metadata = (
                    existing.form,
                    existing.accession_no,
                    existing.filed_at,
                    existing.source_url,
                )
                if metadata != (form, accession_no, filed_at, source_url):
                    raise ValueError(
                        "content hash is already associated with different filing metadata"
                    )
                return existing.id

            legacy_version = self._next_legacy_filing_version(
                session, company.id, normalized_ticker
            )
            filing = Filing(
                id=str(uuid4()),
                company_id=company.id,
                accession_no=accession_no,
                form=form,
                filed_at=filed_at,
                source_url=source_url,
                raw_text=raw_text,
                content_hash=content_hash,
                corpus_version=legacy_version,
            )
            session.add(filing)
            session.flush()
            session.add_all(
                Chunk(
                    id=str(uuid4()),
                    filing_id=filing.id,
                    section=chunk.section,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    raw_start=chunk.raw_start,
                    raw_end=chunk.raw_end,
                )
                for chunk in chunks
            )
            return filing.id

    def filing_requires_embedding(
        self,
        *,
        ticker: str,
        content_hash: str,
        embedding_model: str,
    ) -> bool:
        """Report whether a fetched filing is new or has stale/missing chunk vectors."""
        with Session(self._engine) as session:
            return self.filing_requires_embedding_in_session(
                session,
                ticker=ticker,
                content_hash=content_hash,
                embedding_model=embedding_model,
            )

    def filing_requires_embedding_in_session(
        self,
        session: Session,
        *,
        ticker: str,
        content_hash: str,
        embedding_model: str,
    ) -> bool:
        """Session-bound current-model check for atomic ingestion."""
        filing = session.scalar(
            select(Filing)
            .join(Filing.company)
            .where(
                Company.ticker == ticker.strip().upper(),
                Filing.content_hash == content_hash,
            )
        )
        if filing is None:
            return True
        chunks = session.scalars(
            select(Chunk).where(Chunk.filing_id == filing.id)
        ).all()
        return any(
            chunk.embedding is None or chunk.embedding_model != embedding_model
            for chunk in chunks
        )

    def commit_ingest_batch(
        self,
        *,
        ticker: str,
        cik: str,
        legal_name: str,
        ir_domain: str | None,
        filings: Sequence[FilingToStore],
        facts: Sequence[CompanyFact],
        fact_repository: AtomicFactWriter,
        as_of_date: date,
    ) -> AtomicIngestResult:
        """Commit issuer metadata, facts, filings, chunks, vectors, and corpus once."""
        with self.embedding_write_context(ticker) as session:
            return self.commit_ingest_batch_in_session(
                session,
                ticker=ticker,
                cik=cik,
                legal_name=legal_name,
                ir_domain=ir_domain,
                filings=filings,
                facts=facts,
                fact_repository=fact_repository,
                as_of_date=as_of_date,
            )

    def commit_ingest_batch_in_session(
        self,
        session: Session,
        *,
        ticker: str,
        cik: str,
        legal_name: str,
        ir_domain: str | None,
        filings: Sequence[FilingToStore],
        facts: Sequence[CompanyFact],
        fact_repository: AtomicFactWriter,
        as_of_date: date,
    ) -> AtomicIngestResult:
        """Commit the prepared batch in the caller's protected embedding transaction."""
        normalized_ticker, normalized_cik, normalized_name, normalized_domain = (
            _normalized_company_metadata(
                ticker=ticker,
                cik=cik,
                legal_name=legal_name,
                ir_domain=ir_domain,
            )
        )
        if fact_repository.engine is not self._engine:
            raise ValueError("atomic ingest repositories must share one database engine")
        if not 1 <= len(filings) <= 4:
            raise ValueError("atomic ingest requires one to four prepared filings")
        if len({filing.content_hash for filing in filings}) != len(filings):
            raise ValueError("prepared filing content hashes must be unique")
        if type(as_of_date) is not date:
            raise ValueError("as_of_date must be a date")
        if session.get_bind() is not self._engine:
            raise ValueError("atomic ingest session must use the repository engine")

        self._lock_ticker(session, normalized_ticker)
        company = self._upsert_company_metadata_in_session(
            session,
            ticker=normalized_ticker,
            cik=normalized_cik,
            legal_name=normalized_name,
            ir_domain=normalized_domain,
        )
        fact_repository.save_facts_in_session(session, facts)
        filing_ids = tuple(
            self._store_prepared_filing_in_session(
                session,
                company=company,
                ticker=normalized_ticker,
                filing=filing,
            )
            for filing in filings
        )
        corpus_version = self._create_corpus_in_session(
            session,
            normalized_ticker,
            filing_ids,
            as_of_date,
        )
        return AtomicIngestResult(
            corpus_version=corpus_version,
            filing_ids=filing_ids,
        )

    def _store_prepared_filing_in_session(
        self,
        session: Session,
        *,
        company: Company,
        ticker: str,
        filing: FilingToStore,
    ) -> str:
        existing = session.scalar(
            select(Filing).where(
                Filing.company_id == company.id,
                Filing.content_hash == filing.content_hash,
            )
        )
        if existing is not None:
            metadata = (
                existing.form,
                existing.accession_no,
                existing.filed_at,
                existing.source_url,
            )
            expected = (
                filing.form,
                filing.accession_no,
                filing.filed_at,
                filing.source_url,
            )
            if metadata != expected:
                raise ValueError(
                    "content hash is already associated with different filing metadata"
                )
            self._apply_prepared_embeddings(session, existing, filing)
            return existing.id

        if (
            filing.chunks
            and filing.embedding_model is not None
            and filing.embedding_vectors is None
        ):
            raise EmbeddingStateConflictError(
                "embedding state changed before atomic ingest commit"
            )

        record = Filing(
            id=str(uuid4()),
            company_id=company.id,
            accession_no=filing.accession_no,
            form=filing.form,
            filed_at=filing.filed_at,
            source_url=filing.source_url,
            raw_text=filing.raw_text,
            content_hash=filing.content_hash,
            corpus_version=self._next_legacy_filing_version(session, company.id, ticker),
        )
        session.add(record)
        session.flush()
        vectors = filing.embedding_vectors
        session.add_all(
            Chunk(
                id=str(uuid4()),
                filing_id=record.id,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                raw_start=chunk.raw_start,
                raw_end=chunk.raw_end,
                embedding=(list(vectors[index]) if vectors is not None else None),
                embedding_model=filing.embedding_model,
            )
            for index, chunk in enumerate(filing.chunks)
        )
        return record.id

    @staticmethod
    def _apply_prepared_embeddings(
        session: Session,
        record: Filing,
        filing: FilingToStore,
    ) -> None:
        if filing.embedding_model is None:
            return
        chunks = session.scalars(
            select(Chunk)
            .where(Chunk.filing_id == record.id)
            .order_by(Chunk.chunk_index)
        ).all()
        expected = [
            (
                chunk.chunk_index,
                chunk.section,
                chunk.content,
                chunk.token_count,
                chunk.raw_start,
                chunk.raw_end,
            )
            for chunk in filing.chunks
        ]
        actual = [
            (
                chunk.chunk_index,
                chunk.section,
                chunk.content,
                chunk.token_count,
                chunk.raw_start,
                chunk.raw_end,
            )
            for chunk in chunks
        ]
        if actual != expected:
            raise EmbeddingStateConflictError(
                "embedding state changed before atomic ingest commit"
            )
        if filing.embedding_vectors is None:
            if any(
                chunk.embedding is None
                or chunk.embedding_model != filing.embedding_model
                for chunk in chunks
            ):
                raise EmbeddingStateConflictError(
                    "embedding state changed before atomic ingest commit"
                )
            return
        for chunk, vector in zip(chunks, filing.embedding_vectors, strict=True):
            chunk.embedding = list(vector)
            chunk.embedding_model = filing.embedding_model

    def create_corpus(
        self,
        ticker: str,
        filing_ids: Sequence[str],
        as_of_date: date,
    ) -> str:
        """Create or reuse an immutable one-to-four filing snapshot for one ticker."""
        normalized_ticker = ticker.upper()
        if isinstance(filing_ids, (str, bytes)):
            raise ValueError("filing_ids must be a sequence of ids")
        member_ids = list(filing_ids)
        if not 1 <= len(member_ids) <= 4:
            raise ValueError("a research corpus must contain one to four filings")
        if len(set(member_ids)) != len(member_ids) or any(not item for item in member_ids):
            raise ValueError("corpus filing ids must be non-empty and unique")
        if type(as_of_date) is not date:
            raise ValueError("as_of_date must be a date")
        membership_hash = _membership_hash(member_ids)

        try:
            with Session(self._engine) as session, session.begin():
                self._lock_ticker(session, normalized_ticker)
                return self._create_corpus_in_session(
                    session,
                    normalized_ticker,
                    member_ids,
                    as_of_date,
                )
        except IntegrityError:
            winner = self._corpus_version_for_identity(
                normalized_ticker,
                membership_hash,
                set(member_ids),
            )
            if winner is not None:
                return winner
            raise

    def _create_corpus_in_session(
        self,
        session: Session,
        ticker: str,
        filing_ids: Sequence[str],
        as_of_date: date,
    ) -> str:
        member_ids = list(filing_ids)
        if not 1 <= len(member_ids) <= 4:
            raise ValueError("a research corpus must contain one to four filings")
        if len(set(member_ids)) != len(member_ids) or any(not item for item in member_ids):
            raise ValueError("corpus filing ids must be non-empty and unique")
        if type(as_of_date) is not date:
            raise ValueError("as_of_date must be a date")
        membership_hash = _membership_hash(member_ids)
        company = session.scalar(select(Company).where(Company.ticker == ticker))
        if company is None:
            raise ValueError(f"no company exists for ticker {ticker}")
        filings = session.scalars(select(Filing).where(Filing.id.in_(member_ids))).all()
        if len(filings) != len(member_ids) or any(
            filing.company_id != company.id for filing in filings
        ):
            raise ValueError("all corpus filings must belong to the same company")
        if any(filing.filed_at > as_of_date for filing in filings):
            raise ValueError("corpus filings must not be newer than as_of_date")
        membership_as_of_date = max(filing.filed_at for filing in filings)

        identified = session.scalars(
            select(ResearchCorpus).where(
                ResearchCorpus.company_id == company.id,
                ResearchCorpus.membership_hash == membership_hash,
            )
        ).all()
        if identified:
            matching = self._exact_membership_corpora(session, identified, set(member_ids))
            if len(identified) == len(matching) == 1:
                return matching[0].version
            raise ValueError("corpus membership identity is ambiguous or corrupt")

        version = self._next_corpus_version(session, company.id, ticker)
        corpus = ResearchCorpus(
            id=str(uuid4()),
            company_id=company.id,
            version=version,
            membership_hash=membership_hash,
            as_of_date=membership_as_of_date,
            created_at=datetime.now(UTC),
        )
        session.add(corpus)
        session.flush()
        session.add_all(
            CorpusFiling(corpus_id=corpus.id, filing_id=filing_id)
            for filing_id in member_ids
        )
        return version

    def list_chunks(
        self,
        ticker: str,
        corpus_version: str,
        *,
        filing_ids: Sequence[str] | None = None,
    ) -> list[EvidenceChunk]:
        """List only chunks from the requested ticker and corpus version."""
        with Session(self._engine) as session:
            return self.list_chunks_in_session(
                session,
                ticker,
                corpus_version,
                filing_ids=filing_ids,
            )

    def list_chunks_in_session(
        self,
        session: Session,
        ticker: str,
        corpus_version: str,
        *,
        filing_ids: Sequence[str] | None = None,
    ) -> list[EvidenceChunk]:
        """Session-bound chunk read for one protected embedding transaction."""
        records = session.scalars(
            self._scoped_chunks(ticker, corpus_version, filing_ids=filing_ids).order_by(
                Filing.filed_at,
                Filing.accession_no,
                Chunk.chunk_index,
            )
        ).all()
        return [self._evidence_chunk(chunk, corpus_version) for chunk in records]

    def list_chunks_requiring_embedding(
        self,
        ticker: str,
        corpus_version: str,
        embedding_model: str,
    ) -> list[EvidenceChunk]:
        """List scoped chunks whose stored vector is absent or from another provider."""
        with Session(self._engine) as session:
            return self.list_chunks_requiring_embedding_in_session(
                session,
                ticker,
                corpus_version,
                embedding_model,
            )

    def list_chunks_requiring_embedding_in_session(
        self,
        session: Session,
        ticker: str,
        corpus_version: str,
        embedding_model: str,
    ) -> list[EvidenceChunk]:
        """Session-bound stale-vector read for one protected embedding transaction."""
        records = session.scalars(
            self._scoped_chunks(ticker, corpus_version)
            .where(
                or_(
                    Chunk.embedding.is_(None),
                    Chunk.embedding_model != embedding_model,
                    Chunk.embedding_model.is_(None),
                )
            )
            .order_by(Filing.filed_at, Filing.accession_no, Chunk.chunk_index)
        ).all()
        return [self._evidence_chunk(chunk, corpus_version) for chunk in records]

    @contextmanager
    def embedding_write_context(self, ticker: str) -> Iterator[Session]:
        """Yield one transaction owning the ticker advisory/process lock and all writes."""
        lock_key = f"embedding:{ticker.upper()}"
        if self._engine.dialect.name == "postgresql":
            with Session(self._engine) as session, session.begin():
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
                    {"lock_key": lock_key},
                )
                yield session
            return

        process_key = (self._engine.url.render_as_string(hide_password=False), lock_key)
        with _EMBEDDING_LOCKS_GUARD:
            process_lock = _EMBEDDING_LOCKS.setdefault(process_key, Lock())
        with process_lock:
            with Session(self._engine) as session, session.begin():
                yield session

    def store_embeddings(
        self,
        chunks: list[EvidenceChunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """Validate a complete 1,024-dimensional batch, then write it atomically."""
        with Session(self._engine) as session, session.begin():
            self.store_embeddings_in_session(
                session,
                chunks,
                vectors,
                embedding_model,
            )

    def store_embeddings_in_session(
        self,
        session: Session,
        chunks: list[EvidenceChunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """Session-bound vector write for one protected embedding transaction."""
        if len(vectors) != len(chunks):
            raise ValueError("embedding vector count must match chunk count")
        if any(len(vector) != 1024 for vector in vectors):
            raise ValueError("stored embedding vectors must have 1024 dimensions")
        if not embedding_model.strip():
            raise ValueError("embedding model version must not be blank")
        chunk_ids = [chunk.id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("embedding chunks must be unique")
        records = {
            chunk.id: chunk
            for chunk in session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()
        }
        if set(records) != set(chunk_ids):
            raise ValueError("every embedding chunk must already be persisted")
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = records[chunk.id]
            record.embedding = vector
            record.embedding_model = embedding_model

    def dense_search(
        self,
        *,
        ticker: str,
        corpus_version: str,
        filing_ids: Sequence[str] | None = None,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[EvidenceChunk]:
        """Return cosine Top-K from only the selected ticker and corpus."""
        with Session(self._engine) as session:
            return self.dense_search_in_session(
                session,
                ticker=ticker,
                corpus_version=corpus_version,
                filing_ids=filing_ids,
                embedding_model=embedding_model,
                query_embedding=query_embedding,
                limit=limit,
            )

    def dense_search_in_session(
        self,
        session: Session,
        *,
        ticker: str,
        corpus_version: str,
        filing_ids: Sequence[str] | None = None,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[EvidenceChunk]:
        """Session-bound dense search for one protected embedding transaction."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if len(query_embedding) != 1024:
            raise ValueError("query embedding must have 1024 dimensions")

        statement = self._scoped_chunks(ticker, corpus_version, filing_ids=filing_ids).where(
            Chunk.embedding.is_not(None),
            Chunk.embedding_model == embedding_model,
        )
        if session.get_bind().dialect.name == "postgresql":
            records = session.scalars(
                statement.order_by(
                    Chunk.embedding.cosine_distance(query_embedding), Chunk.id
                ).limit(limit)
            ).all()
        else:
            scoped = session.scalars(statement).all()
            records = sorted(
                scoped,
                key=lambda chunk: (
                    -_cosine_similarity(query_embedding, chunk.embedding or []),
                    chunk.id,
                ),
            )[:limit]
        return [self._evidence_chunk(chunk, corpus_version) for chunk in records]

    def get_company(self, ticker: str) -> CompanyRead | None:
        """Return one normalized ticker's public metadata, if it is supported."""
        with Session(self._engine) as session:
            company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
            if company is None:
                return None
            return CompanyRead(
                ticker=company.ticker,
                cik=company.cik,
                legal_name=company.legal_name,
                ir_domain=company.ir_domain,
            )

    def list_recent_filings(
        self,
        ticker: str,
        forms: list[str],
        limit: int,
        *,
        as_of_date: date | None = None,
        corpus_version: str | None = None,
    ) -> list[FilingRead]:
        """List recent filings from the ticker's latest immutable allowed snapshot."""
        selected_version = corpus_version or self.latest_corpus_version(
            ticker,
            as_of_date=as_of_date,
            forms=forms,
        )
        if selected_version is None:
            return []
        with Session(self._engine) as session:
            statement = (
                select(Filing)
                .join(CorpusFiling, CorpusFiling.filing_id == Filing.id)
                .join(ResearchCorpus, ResearchCorpus.id == CorpusFiling.corpus_id)
                .join(Company, Company.id == ResearchCorpus.company_id)
                .where(
                    Company.ticker == ticker.upper(),
                    ResearchCorpus.version == selected_version,
                    Filing.company_id == ResearchCorpus.company_id,
                    Filing.form.in_(_ALLOWED_FORMS),
                )
                .order_by(Filing.filed_at.desc(), Filing.accession_no.desc())
                .limit(limit)
            )
            if forms:
                statement = statement.where(Filing.form.in_(forms))
            if as_of_date is not None:
                statement = statement.where(
                    ResearchCorpus.as_of_date <= as_of_date,
                    Filing.filed_at <= as_of_date,
                )
            return [
                self._filing_read(filing, selected_version)
                for filing in session.scalars(statement).all()
            ]

    def list_filings_by_ids(self, ticker: str, filing_ids: list[str]) -> list[FilingRead]:
        """Return only requested filing ids belonging to the normalized ticker."""
        if not filing_ids:
            return []
        with Session(self._engine) as session:
            filings = session.scalars(
                select(Filing)
                .join(Filing.company)
                .where(
                    Company.ticker == ticker.upper(),
                    Filing.id.in_(filing_ids),
                    Filing.form.in_(_ALLOWED_FORMS),
                )
            ).all()
            corpus_version = self.corpus_version_for_filings(ticker, filing_ids)
            records = {filing.id: self._filing_read(filing, corpus_version) for filing in filings}
            return [records[filing_id] for filing_id in filing_ids if filing_id in records]

    def corpus_version_for_filings(self, ticker: str, filing_ids: Sequence[str]) -> str | None:
        """Return the snapshot whose exact membership matches the supplied filing ids."""
        member_ids = list(filing_ids)
        if not member_ids or len(member_ids) != len(set(member_ids)):
            return None
        with Session(self._engine) as session:
            company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
            if company is None:
                return None
            allowed_ids = set(
                session.scalars(
                    select(Filing.id).where(
                        Filing.company_id == company.id,
                        Filing.id.in_(member_ids),
                        Filing.form.in_(_ALLOWED_FORMS),
                    )
                ).all()
            )
            if allowed_ids != set(member_ids):
                return None
            corpora = session.scalars(
                select(ResearchCorpus).where(ResearchCorpus.company_id == company.id)
            ).all()
            requested = set(member_ids)
            matching: list[ResearchCorpus] = []
            for corpus in corpora:
                stored = set(
                    session.scalars(
                        select(CorpusFiling.filing_id).where(CorpusFiling.corpus_id == corpus.id)
                    ).all()
                )
                if stored == requested:
                    matching.append(corpus)
            if len(matching) != 1:
                return None
            return (
                matching[0].version
                if matching[0].membership_hash == _membership_hash(member_ids)
                else None
            )

    def get_filing(self, ticker: str, corpus_version: str) -> FilingRead | None:
        """Return the newest filing associated with one exact ticker snapshot."""
        with Session(self._engine) as session:
            filing = session.scalar(
                select(Filing)
                .join(CorpusFiling, CorpusFiling.filing_id == Filing.id)
                .join(ResearchCorpus, ResearchCorpus.id == CorpusFiling.corpus_id)
                .join(Company, Company.id == ResearchCorpus.company_id)
                .where(
                    Company.ticker == ticker.upper(),
                    ResearchCorpus.version == corpus_version,
                    Filing.company_id == ResearchCorpus.company_id,
                    Filing.form.in_(_ALLOWED_FORMS),
                )
                .order_by(Filing.filed_at.desc(), Filing.accession_no.desc())
            )
            return self._filing_read(filing, corpus_version) if filing is not None else None

    def latest_corpus_version(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
        forms: Sequence[str] | None = None,
    ) -> str | None:
        """Return the newest immutable snapshot containing allowed P0 evidence."""
        with Session(self._engine) as session:
            statement = (
                select(ResearchCorpus.version)
                .join(ResearchCorpus.company)
                .join(CorpusFiling, CorpusFiling.corpus_id == ResearchCorpus.id)
                .join(Filing, Filing.id == CorpusFiling.filing_id)
                .where(
                    Company.ticker == ticker.upper(),
                    Filing.company_id == ResearchCorpus.company_id,
                    Filing.form.in_(_ALLOWED_FORMS),
                )
                .distinct()
            )
            if as_of_date is not None:
                statement = statement.where(ResearchCorpus.as_of_date <= as_of_date)
            if forms:
                statement = statement.where(Filing.form.in_(list(forms)))
            versions = session.scalars(statement).all()
            return max(versions, key=self._corpus_sequence, default=None)

    def list_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        """Return persisted chunks in caller order for source-span rendering."""
        if not chunk_ids:
            return []
        with Session(self._engine) as session:
            chunks = session.scalars(
                select(Chunk)
                .join(Chunk.filing)
                .where(Chunk.id.in_(chunk_ids), Filing.form.in_(_ALLOWED_FORMS))
            ).all()
            records = {chunk.id: self._evidence_chunk(chunk) for chunk in chunks}
            return [records[chunk_id] for chunk_id in chunk_ids if chunk_id in records]

    def list_chunks_in_corpus_by_ids(
        self,
        ticker: str,
        corpus_version: str,
        chunk_ids: Sequence[str],
    ) -> list[EvidenceChunk]:
        """Return caller-ordered chunk ids only when associated with the exact snapshot."""
        if not chunk_ids:
            return []
        with Session(self._engine) as session:
            chunks = session.scalars(
                self._scoped_chunks(ticker, corpus_version).where(Chunk.id.in_(chunk_ids))
            ).all()
            records = {chunk.id: self._evidence_chunk(chunk, corpus_version) for chunk in chunks}
            return [records[chunk_id] for chunk_id in chunk_ids if chunk_id in records]

    @staticmethod
    def _filing_read(filing: Filing, corpus_version: str | None = None) -> FilingRead:
        return FilingRead(
            id=filing.id,
            ticker=filing.company.ticker,
            form=filing.form,
            filed_at=filing.filed_at,
            source_url=filing.source_url,
            accession_no=filing.accession_no,
            corpus_version=corpus_version or filing.corpus_version,
            content_hash=filing.content_hash,
        )

    @staticmethod
    def _evidence_chunk(chunk: Chunk, corpus_version: str | None = None) -> EvidenceChunk:
        return EvidenceChunk(
            id=chunk.id,
            ticker=chunk.filing.company.ticker,
            corpus_version=corpus_version or chunk.filing.corpus_version,
            content=chunk.content,
            source_url=chunk.filing.source_url,
            form=chunk.filing.form,
            filed_at=chunk.filing.filed_at,
            accession_no=chunk.filing.accession_no,
            section=chunk.section,
            raw_start=chunk.raw_start,
            raw_end=chunk.raw_end,
        )

    @staticmethod
    def _next_legacy_filing_version(session: Session, company_id: str, ticker: str) -> str:
        versions = session.scalars(
            select(Filing.corpus_version).where(Filing.company_id == company_id)
        ).all()
        latest = max(
            (
                int(version.rsplit("-v", maxsplit=1)[1])
                for version in versions
                if version.startswith(f"{ticker}-v")
                and version.rsplit("-v", maxsplit=1)[1].isdigit()
            ),
            default=0,
        )
        return f"{ticker}-v{latest + 1}"

    @staticmethod
    def _next_corpus_version(session: Session, company_id: str, ticker: str) -> str:
        versions = session.scalars(
            select(ResearchCorpus.version).where(ResearchCorpus.company_id == company_id)
        ).all()
        latest = max(
            (FilingRepository._corpus_sequence(version) for version in versions),
            default=0,
        )
        return f"{ticker}-v{latest + 1}"

    @staticmethod
    def _exact_membership_corpora(
        session: Session,
        corpora: Sequence[ResearchCorpus],
        member_ids: set[str],
    ) -> list[ResearchCorpus]:
        matching: list[ResearchCorpus] = []
        for corpus in corpora:
            stored = set(
                session.scalars(
                    select(CorpusFiling.filing_id).where(CorpusFiling.corpus_id == corpus.id)
                ).all()
            )
            if stored == member_ids:
                matching.append(corpus)
        return matching

    def _corpus_version_for_identity(
        self,
        ticker: str,
        membership_hash: str,
        member_ids: set[str],
    ) -> str | None:
        """Re-read a uniqueness-race winner only when its identity is exact."""
        with Session(self._engine) as session:
            company = session.scalar(select(Company).where(Company.ticker == ticker))
            if company is None:
                return None
            corpora = session.scalars(
                select(ResearchCorpus).where(
                    ResearchCorpus.company_id == company.id,
                    ResearchCorpus.membership_hash == membership_hash,
                )
            ).all()
            matching = self._exact_membership_corpora(session, corpora, member_ids)
            return matching[0].version if len(corpora) == len(matching) == 1 else None

    @staticmethod
    def _scoped_chunks(
        ticker: str,
        corpus_version: str,
        *,
        filing_ids: Sequence[str] | None = None,
    ):
        statement = (
            select(Chunk)
            .join(Filing, Filing.id == Chunk.filing_id)
            .join(CorpusFiling, CorpusFiling.filing_id == Filing.id)
            .join(ResearchCorpus, ResearchCorpus.id == CorpusFiling.corpus_id)
            .join(Company, Company.id == ResearchCorpus.company_id)
            .where(
                Company.ticker == ticker.upper(),
                ResearchCorpus.version == corpus_version,
                Filing.company_id == ResearchCorpus.company_id,
                Filing.form.in_(_ALLOWED_FORMS),
            )
        )
        if filing_ids is not None:
            statement = statement.where(Filing.id.in_(list(filing_ids)))
        return statement

    @staticmethod
    def _corpus_sequence(corpus_version: str) -> int:
        """Extract the monotonic per-ticker sequence written by ``store_filing``."""
        return int(corpus_version.rsplit("-v", maxsplit=1)[1])


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have matching dimensions")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _membership_hash(filing_ids: Sequence[str]) -> str:
    return sha256("\n".join(sorted(filing_ids)).encode()).hexdigest()


def _normalized_company_metadata(
    *,
    ticker: str,
    cik: str,
    legal_name: str,
    ir_domain: str | None,
) -> tuple[str, str, str, str | None]:
    normalized_ticker = ticker.strip().upper()
    normalized_cik = cik.strip().zfill(10)
    normalized_name = " ".join(legal_name.split())
    normalized_domain = ir_domain.strip().lower().rstrip(".") if ir_domain else None
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized_ticker):
        raise ValueError("invalid company ticker metadata")
    if not re.fullmatch(r"\d{10}", normalized_cik) or int(normalized_cik) == 0:
        raise ValueError("invalid company CIK metadata")
    if (
        not normalized_name
        or len(normalized_name) > 255
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized_name)
    ):
        raise ValueError("invalid company legal name metadata")
    if normalized_domain is not None and (
        not normalized_domain
        or "://" in normalized_domain
        or any(character in normalized_domain for character in "/@:")
    ):
        raise ValueError("IR domain must be a validated bare host")
    return normalized_ticker, normalized_cik, normalized_name, normalized_domain
