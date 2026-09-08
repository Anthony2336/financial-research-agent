"""Add research-run completion and source-fetch provenance.

Revision ID: 20260830_0002
Revises: 20260830_0001
Create Date: 2026-08-30 00:00:01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0002"
down_revision: str | Sequence[str] | None = "20260830_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_runs",
        sa.Column("requested_intent", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "research_runs",
        sa.Column("effective_intent", sa.String(length=64), nullable=True),
    )
    op.add_column("research_runs", sa.Column("corpus_scope", sa.JSON(), nullable=True))
    op.add_column("research_runs", sa.Column("prompt_version", sa.String(length=64), nullable=True))
    op.add_column("research_runs", sa.Column("trace_id", sa.String(length=128), nullable=True))
    op.add_column("research_runs", sa.Column("completed_at", sa.DateTime(), nullable=True))
    op.add_column("claims", sa.Column("source_refs", sa.JSON(), nullable=True))
    op.create_table(
        "source_fetches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.run_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_source_fetches_run_id", "source_fetches", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_table("source_fetches")
    op.drop_column("claims", "source_refs")
    op.drop_column("research_runs", "completed_at")
    op.drop_column("research_runs", "trace_id")
    op.drop_column("research_runs", "prompt_version")
    op.drop_column("research_runs", "corpus_scope")
    op.drop_column("research_runs", "effective_intent")
    op.drop_column("research_runs", "requested_intent")
