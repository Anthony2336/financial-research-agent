"""Add citation-bound long-term research memories.

Revision ID: 20260831_0009
Revises: 20260831_0008
Create Date: 2026-08-31 00:00:09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260831_0009"
down_revision: str | Sequence[str] | None = "20260831_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    evidence_nonempty = (
        "jsonb_array_length(evidence_source_refs) > 0"
        if op.get_bind().dialect.name == "postgresql"
        else "json_array_length(evidence_source_refs) > 0"
    )
    op.create_table(
        "research_memories",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("scope_key", sa.String(length=64), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("memory_kind", sa.String(length=32), nullable=False),
        sa.Column("summary", sa.String(length=1200), nullable=False),
        sa.Column("source_run_id", sa.String(length=64), nullable=False),
        sa.Column(
            "evidence_source_refs",
            JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("corpus_version", sa.String(length=32), nullable=False),
        sa.Column("embedding", Vector(dim=1024), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "memory_kind IN ('research_summary', 'counterevidence', "
            "'open_question', 'source_pointer')",
            name="ck_research_memories_kind",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_research_memories_expiry",
        ),
        sa.CheckConstraint(
            "importance >= 0 AND importance <= 1",
            name="ck_research_memories_importance",
        ),
        sa.CheckConstraint(
            evidence_nonempty,
            name="ck_research_memories_evidence_nonempty",
        ),
        sa.ForeignKeyConstraint(["source_run_id"], ["research_runs.run_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_memories_scope_kind_expiry",
        "research_memories",
        ["ticker", "memory_kind", "expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_research_memories_source_run_id",
        "research_memories",
        ["source_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_memories_corpus_version",
        "research_memories",
        ["corpus_version"],
        unique=False,
    )
    op.create_index(
        "ix_research_memories_embedding_model",
        "research_memories",
        ["embedding_model"],
        unique=False,
    )
    if op.get_bind().dialect.name == "postgresql":
        op.create_index(
            "ix_research_memories_embedding_cosine",
            "research_memories",
            ["embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_index(
            "ix_research_memories_embedding_cosine",
            table_name="research_memories",
        )
    op.drop_index(
        "ix_research_memories_embedding_model",
        table_name="research_memories",
    )
    op.drop_index(
        "ix_research_memories_corpus_version",
        table_name="research_memories",
    )
    op.drop_index(
        "ix_research_memories_source_run_id",
        table_name="research_memories",
    )
    op.drop_index(
        "ix_research_memories_scope_kind_expiry",
        table_name="research_memories",
    )
    op.drop_table("research_memories")
