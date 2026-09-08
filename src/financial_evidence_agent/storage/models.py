"""SQLAlchemy records for evidence and research-run provenance."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    """Base class for persistence records."""


Embedding = Vector(1024).with_variant(JSON(none_as_null=True), "sqlite")
EvidenceSourceRefs = JSONB().with_variant(JSON(none_as_null=True), "sqlite")


class ExactNumeric(TypeDecorator[Decimal]):
    """Keep Decimal text exact in SQLite while using NUMERIC in PostgreSQL."""

    impl = Numeric(38, 18)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(80))
        return dialect.type_descriptor(Numeric(38, 18))

    def process_bind_param(self, value: Decimal | None, dialect):
        if value is None:
            return None
        return str(value) if dialect.name == "sqlite" else value

    def process_result_value(self, value: object, dialect) -> Decimal | None:
        del dialect
        return None if value is None else Decimal(str(value))


class Company(Base):
    """A company whose filings make up an evidence corpus."""

    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    cik: Mapped[str | None] = mapped_column(String(20), nullable=True)
    legal_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ir_domain: Mapped[str | None] = mapped_column(String(255), nullable=True)

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")
    research_corpora: Mapped[list["ResearchCorpus"]] = relationship(
        back_populates="company"
    )


class Filing(Base):
    """One source filing and the corpus version it introduced."""

    __tablename__ = "filings"
    __table_args__ = (
        UniqueConstraint("company_id", "content_hash"),
        UniqueConstraint("company_id", "corpus_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    accession_no: Mapped[str] = mapped_column(String(40), index=True)
    form: Mapped[str] = mapped_column(String(10))
    filed_at: Mapped[date] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(String(2_000))
    raw_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    corpus_version: Mapped[str] = mapped_column(String(32), index=True)

    company: Mapped[Company] = relationship(back_populates="filings")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="filing")
    corpus_associations: Mapped[list["CorpusFiling"]] = relationship(
        back_populates="filing"
    )


class CompanyFactRecord(Base):
    """One immutable exact SEC Company Fact, separate from filing/web/market evidence."""

    __tablename__ = "company_facts"
    __table_args__ = (
        CheckConstraint(
            "form IN ('10-K', '10-Q', '8-K')",
            name="ck_company_facts_form",
        ),
        CheckConstraint(
            "(instant IS NOT NULL AND period_start IS NULL AND period_end IS NULL) OR "
            "(instant IS NULL AND period_start IS NOT NULL AND period_end IS NOT NULL)",
            name="ck_company_facts_period",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    cik: Mapped[str] = mapped_column(String(10), index=True)
    taxonomy: Mapped[str] = mapped_column(String(255), index=True)
    concept: Mapped[str] = mapped_column(String(255), index=True)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    instant: Mapped[date | None] = mapped_column(Date, nullable=True)
    unit: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    value: Mapped[Decimal] = mapped_column(ExactNumeric())
    form: Mapped[str] = mapped_column(String(10))
    filed_at: Mapped[date] = mapped_column(Date, index=True)
    accession_no: Mapped[str] = mapped_column(String(40), index=True)
    source_url: Mapped[str] = mapped_column(String(2_000))
    frame: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_content_hash: Mapped[str] = mapped_column(String(64), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ResearchCorpus(Base):
    """One immutable, versioned filing set for a single company."""

    __tablename__ = "research_corpora"
    __table_args__ = (
        UniqueConstraint("company_id", "version"),
        UniqueConstraint("company_id", "membership_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    version: Mapped[str] = mapped_column(String(32), index=True)
    membership_hash: Mapped[str] = mapped_column(String(64))
    as_of_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    company: Mapped[Company] = relationship(back_populates="research_corpora")
    filing_associations: Mapped[list["CorpusFiling"]] = relationship(
        back_populates="corpus"
    )


class CorpusFiling(Base):
    """Immutable membership of one filing in one research corpus snapshot."""

    __tablename__ = "corpus_filings"

    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id"), primary_key=True
    )
    filing_id: Mapped[str] = mapped_column(
        ForeignKey("filings.id"), primary_key=True, index=True
    )

    corpus: Mapped[ResearchCorpus] = relationship(back_populates="filing_associations")
    filing: Mapped[Filing] = relationship(back_populates="corpus_associations")


class Chunk(Base):
    """A source-addressable child span from one filing."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("filing_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filing_id: Mapped[str] = mapped_column(ForeignKey("filings.id"), index=True)
    section: Mapped[str] = mapped_column(String(100), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    raw_start: Mapped[int] = mapped_column(Integer)
    raw_end: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Embedding, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    filing: Mapped[Filing] = relationship(back_populates="chunks")


class ResearchRun(Base):
    """A reproducible local record of a research run."""

    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    thesis: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50))
    corpus_version: Mapped[str] = mapped_column(String(32))
    requested_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    corpus_scope: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    claims: Mapped[list["ClaimRecord"]] = relationship(back_populates="research_run")
    source_fetches: Mapped[list["SourceFetchRecord"]] = relationship(
        back_populates="research_run"
    )
    research_memories: Mapped[list["ResearchMemoryRecord"]] = relationship(
        back_populates="source_run"
    )
    skill_runs: Mapped[list["SkillRun"]] = relationship(back_populates="application_run")


class ClaimRecord(Base):
    """A guarded claim generated by a research run."""

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(16))
    evidence_chunk_ids: Mapped[list[str]] = mapped_column(JSON)
    source_refs: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    guard_status: Mapped[str] = mapped_column(String(32))

    research_run: Mapped[ResearchRun] = relationship(back_populates="claims")


class SourceFetchRecord(Base):
    """One attempted source retrieval attached to a research run."""

    __tablename__ = "source_fetches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.run_id"), index=True)
    source_kind: Mapped[str] = mapped_column(String(32))
    source_ref: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    research_run: Mapped[ResearchRun] = relationship(back_populates="source_fetches")


class ResearchMemoryRecord(Base):
    """One expiring planner hint bound to a completed guarded research run."""

    __tablename__ = "research_memories"
    __table_args__ = (
        CheckConstraint(
            "memory_kind IN ('research_summary', 'counterevidence', "
            "'open_question', 'source_pointer')",
            name="ck_research_memories_kind",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_research_memories_expiry",
        ),
        CheckConstraint(
            "importance >= 0 AND importance <= 1",
            name="ck_research_memories_importance",
        ),
        CheckConstraint(
            "json_array_length(evidence_source_refs) > 0",
            name="ck_research_memories_evidence_nonempty",
        ).ddl_if(dialect="sqlite"),
        CheckConstraint(
            "jsonb_array_length(evidence_source_refs) > 0",
            name="ck_research_memories_evidence_nonempty",
        ).ddl_if(dialect="postgresql"),
        Index(
            "ix_research_memories_scope_kind_expiry",
            "ticker",
            "memory_kind",
            "expires_at",
        ),
        Index(
            "ix_research_memories_embedding_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ).ddl_if(dialect="postgresql"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(64))
    ticker: Mapped[str] = mapped_column(String(10))
    memory_kind: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(String(1_200))
    source_run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.run_id"), index=True
    )
    evidence_source_refs: Mapped[list[str]] = mapped_column(EvidenceSourceRefs)
    corpus_version: Mapped[str] = mapped_column(String(32), index=True)
    embedding: Mapped[list[float]] = mapped_column(Embedding)
    embedding_model: Mapped[str] = mapped_column(String(255), index=True)
    importance: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    source_run: Mapped[ResearchRun] = relationship(back_populates="research_memories")


class WebEvidenceRecord(Base):
    """One immutable snapshot retrieved from an approved web source."""

    __tablename__ = "web_evidence"
    __table_args__ = (UniqueConstraint("ticker", "source_url", "content_hash"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(String(2_000))
    source_kind: Mapped[str] = mapped_column(String(32))
    source_tier: Mapped[str] = mapped_column(String(32))
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    time_metadata_validated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )


class SkillRun(Base):
    """Execution provenance for one immutable research-recipe snapshot."""

    __tablename__ = "skill_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.run_id"), index=True
    )
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    recipe_name: Mapped[str] = mapped_column(String(100), index=True)
    recipe_version: Mapped[str] = mapped_column(String(32))
    recipe_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[Literal["running", "completed", "partial", "refused", "failed"]] = (
        mapped_column(String(16))
    )
    source_ids: Mapped[list[str]] = mapped_column(JSON)
    errors: Mapped[list[str]] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    application_run: Mapped[ResearchRun] = relationship(back_populates="skill_runs")


class MarketSnapshotRecord(Base):
    """One immutable normalized provider snapshot and its bundle provenance."""

    __tablename__ = "market_snapshots"

    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    feed: Mapped[str] = mapped_column(String(32), index=True)
    coverage: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    exchange: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(String(3))
    price: Mapped[Decimal] = mapped_column(ExactNumeric())
    open: Mapped[Decimal] = mapped_column(ExactNumeric())
    day_high: Mapped[Decimal] = mapped_column(ExactNumeric())
    day_low: Mapped[Decimal] = mapped_column(ExactNumeric())
    previous_close: Mapped[Decimal] = mapped_column(ExactNumeric())
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_status: Mapped[str] = mapped_column(String(16))
    delayed_by_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    bundle_status: Mapped[str] = mapped_column(String(16), default="partial")
    freshness_label: Mapped[str] = mapped_column(
        String(32), default="market-status-unknown"
    )
    errors: Mapped[list[str]] = mapped_column(JSON, default=list)


class MarketBarRecord(Base):
    """One immutable normalized provider daily bar."""

    __tablename__ = "market_bars"

    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    feed: Mapped[str] = mapped_column(String(32), index=True)
    coverage: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    exchange: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(String(3))
    interval: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[Decimal] = mapped_column(ExactNumeric())
    high: Mapped[Decimal] = mapped_column(ExactNumeric())
    low: Mapped[Decimal] = mapped_column(ExactNumeric())
    close: Mapped[Decimal] = mapped_column(ExactNumeric())
    volume: Mapped[int] = mapped_column(BigInteger)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), index=True)


class MarketBundleRecord(Base):
    """One immutable exact bundle assembly over immutable market source records."""

    __tablename__ = "market_bundles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    feed: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("market_snapshots.id"), index=True
    )
    bar_ids: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    freshness_label: Mapped[str] = mapped_column(String(32))
    errors: Mapped[list[str]] = mapped_column(JSON)
    observation_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
