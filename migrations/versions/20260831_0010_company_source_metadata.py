"""Add exact SEC Company Facts without historical backfill.

Revision ID: 20260831_0010
Revises: 20260831_0009
Create Date: 2026-08-31 00:00:10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0010"
down_revision: str | Sequence[str] | None = "20260831_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "web_evidence",
        sa.Column(
            "time_metadata_validated",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    exact_numeric = sa.Numeric(38, 18).with_variant(sa.String(length=80), "sqlite")
    op.create_table(
        "company_facts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("taxonomy", sa.String(length=255), nullable=False),
        sa.Column("concept", sa.String(length=255), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("instant", sa.Date(), nullable=True),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=True),
        sa.Column("value", exact_numeric, nullable=False),
        sa.Column("form", sa.String(length=10), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column("accession_no", sa.String(length=40), nullable=False),
        sa.Column("source_url", sa.String(length=2000), nullable=False),
        sa.Column("frame", sa.String(length=64), nullable=True),
        sa.Column("raw_content_hash", sa.String(length=64), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "form IN ('10-K', '10-Q', '8-K')",
            name="ck_company_facts_form",
        ),
        sa.CheckConstraint(
            "(instant IS NOT NULL AND period_start IS NULL AND period_end IS NULL) OR "
            "(instant IS NULL AND period_start IS NOT NULL AND period_end IS NOT NULL)",
            name="ck_company_facts_period",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "ticker",
        "cik",
        "taxonomy",
        "concept",
        "filed_at",
        "accession_no",
        "raw_content_hash",
        "fetched_at",
    ):
        op.create_index(
            f"ix_company_facts_{column}",
            "company_facts",
            [column],
            unique=False,
        )


def downgrade() -> None:
    for column in reversed(
        (
            "ticker",
            "cik",
            "taxonomy",
            "concept",
            "filed_at",
            "accession_no",
            "raw_content_hash",
            "fetched_at",
        )
    ):
        op.drop_index(f"ix_company_facts_{column}", table_name="company_facts")
    op.drop_table("company_facts")
    with op.batch_alter_table("web_evidence") as batch_op:
        batch_op.drop_column("time_metadata_validated")
