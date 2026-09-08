"""Derive research-corpus cutoffs from immutable filing membership.

Revision ID: 20260831_0011
Revises: 20260831_0010
Create Date: 2026-09-04 00:00:11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0011"
down_revision: str | Sequence[str] | None = "20260831_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE research_corpora SET as_of_date = ("
            "SELECT MAX(f.filed_at) FROM corpus_filings AS cf "
            "JOIN filings AS f ON f.id = cf.filing_id "
            "WHERE cf.corpus_id = research_corpora.id"
            ") WHERE EXISTS ("
            "SELECT 1 FROM corpus_filings AS cf "
            "WHERE cf.corpus_id = research_corpora.id"
            ")"
        )
    )


def downgrade() -> None:
    # The caller's historical selection cutoff was never immutable corpus identity.
    pass
