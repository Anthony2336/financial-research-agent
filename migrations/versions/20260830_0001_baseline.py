"""Create the baseline evidence-agent schema.

Revision ID: 20260830_0001
Revises:
Create Date: 2026-08-30 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20260830_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    connection = op.get_bind()
    existing_tables = set(sa.inspect(connection).get_table_names()) - {"alembic_version"}
    if existing_tables:
        raise RuntimeError(
            "DATABASE_UNVERSIONED_SCHEMA_REJECTED: an unversioned database contains "
            "existing tables. Back up/export the database and recreate it before running "
            "uv run alembic upgrade head. No application DDL was applied."
        )
    if connection.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "companies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("cik", sa.String(length=20), nullable=True),
        sa.Column("legal_name", sa.String(length=255), nullable=True),
        sa.Column("ir_domain", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker"),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"], unique=False)
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("thesis", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("corpus_version", sa.String(length=32), nullable=False),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index("ix_research_runs_run_id", "research_runs", ["run_id"], unique=False)
    op.create_index("ix_research_runs_ticker", "research_runs", ["ticker"], unique=False)
    op.create_table(
        "web_evidence",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(length=2000), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_tier", sa.String(length=32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "source_url", "content_hash"),
    )
    op.create_index("ix_web_evidence_content_hash", "web_evidence", ["content_hash"], unique=False)
    op.create_index("ix_web_evidence_ticker", "web_evidence", ["ticker"], unique=False)
    op.create_table(
        "skill_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("recipe_name", sa.String(length=100), nullable=False),
        sa.Column("recipe_version", sa.String(length=32), nullable=False),
        sa.Column("recipe_snapshot", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_ids", sa.JSON(), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_skill_runs_recipe_name", "skill_runs", ["recipe_name"], unique=False)
    op.create_index("ix_skill_runs_ticker", "skill_runs", ["ticker"], unique=False)
    op.create_table(
        "filings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("accession_no", sa.String(length=40), nullable=False),
        sa.Column("form", sa.String(length=10), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column("source_url", sa.String(length=2000), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("corpus_version", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id", "content_hash"),
        sa.UniqueConstraint("company_id", "corpus_version"),
    )
    op.create_index("ix_filings_accession_no", "filings", ["accession_no"], unique=False)
    op.create_index("ix_filings_company_id", "filings", ["company_id"], unique=False)
    op.create_index("ix_filings_content_hash", "filings", ["content_hash"], unique=False)
    op.create_index("ix_filings_corpus_version", "filings", ["corpus_version"], unique=False)
    op.create_table(
        "claims",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("guard_status", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_claims_run_id", "claims", ["run_id"], unique=False)
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("filing_id", sa.String(length=36), nullable=False),
        sa.Column("section", sa.String(length=100), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("raw_start", sa.Integer(), nullable=False),
        sa.Column("raw_end", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(dim=1024), nullable=True),
        sa.ForeignKeyConstraint(["filing_id"], ["filings.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("filing_id", "chunk_index"),
    )
    op.create_index("ix_chunks_filing_id", "chunks", ["filing_id"], unique=False)
    op.create_index("ix_chunks_section", "chunks", ["section"], unique=False)
def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("claims")
    op.drop_table("filings")
    op.drop_table("skill_runs")
    op.drop_table("web_evidence")
    op.drop_table("research_runs")
    op.drop_table("companies")
