"""Add immutable multi-filing research corpus snapshots.

Revision ID: 20260831_0008
Revises: 20260830_0007
Create Date: 2026-08-31 00:00:08
"""

from collections.abc import Sequence
from datetime import UTC, datetime, time
from hashlib import sha256

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0008"
down_revision: str | Sequence[str] | None = "20260830_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_corpora",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("membership_hash", sa.String(length=64), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id", "membership_hash"),
        sa.UniqueConstraint("company_id", "version"),
    )
    op.create_index(
        "ix_research_corpora_company_id",
        "research_corpora",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_corpora_version",
        "research_corpora",
        ["version"],
        unique=False,
    )
    op.create_table(
        "corpus_filings",
        sa.Column("corpus_id", sa.String(length=36), nullable=False),
        sa.Column("filing_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(["corpus_id"], ["research_corpora.id"]),
        sa.ForeignKeyConstraint(["filing_id"], ["filings.id"]),
        sa.PrimaryKeyConstraint("corpus_id", "filing_id"),
    )
    op.create_index(
        "ix_corpus_filings_filing_id",
        "corpus_filings",
        ["filing_id"],
        unique=False,
    )

    connection = op.get_bind()
    filings = connection.execute(
        sa.text(
            "SELECT id, company_id, corpus_version, filed_at "
            "FROM filings ORDER BY company_id, corpus_version, id"
        )
    ).mappings().all()
    for filing in filings:
        filed_at = filing["filed_at"]
        if isinstance(filed_at, str):
            filed_at = datetime.fromisoformat(filed_at).date()
        connection.execute(
            sa.text(
                "INSERT INTO research_corpora "
                "(id, company_id, version, membership_hash, as_of_date, created_at) "
                "VALUES (:id, :company_id, :version, :membership_hash, "
                ":as_of_date, :created_at)"
            ),
            {
                "id": filing["id"],
                "company_id": filing["company_id"],
                "version": filing["corpus_version"],
                "membership_hash": sha256(filing["id"].encode()).hexdigest(),
                "as_of_date": filed_at,
                "created_at": datetime.combine(filed_at, time.min, UTC),
            },
        )
        connection.execute(
            sa.text(
                "INSERT INTO corpus_filings (corpus_id, filing_id) "
                "VALUES (:corpus_id, :filing_id)"
            ),
            {"corpus_id": filing["id"], "filing_id": filing["id"]},
        )


def downgrade() -> None:
    op.drop_index("ix_corpus_filings_filing_id", table_name="corpus_filings")
    op.drop_table("corpus_filings")
    op.drop_index("ix_research_corpora_version", table_name="research_corpora")
    op.drop_index("ix_research_corpora_company_id", table_name="research_corpora")
    op.drop_table("research_corpora")
