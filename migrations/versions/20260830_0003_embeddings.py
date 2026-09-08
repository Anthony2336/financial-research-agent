"""Add provider-version metadata for persisted filing embeddings.

Revision ID: 20260830_0003
Revises: 20260830_0002
Create Date: 2026-08-30 00:00:02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0003"
down_revision: str | Sequence[str] | None = "20260830_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_chunks_embedding_model",
        "chunks",
        ["embedding_model"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_model", table_name="chunks")
    op.drop_column("chunks", "embedding_model")
