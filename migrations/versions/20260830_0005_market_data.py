"""Add immutable normalized market snapshots and daily bars.

Revision ID: 20260830_0005
Revises: 20260830_0004
Create Date: 2026-08-30 00:00:04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0005"
down_revision: str | Sequence[str] | None = "20260830_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_snapshots",
        sa.Column("id", sa.String(length=512), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("feed", sa.String(length=32), nullable=False),
        sa.Column("coverage", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("exchange", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("price", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("open", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("day_high", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("day_low", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("previous_close", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_status", sa.String(length=16), nullable=False),
        sa.Column("delayed_by_seconds", sa.Integer(), nullable=True),
        sa.Column("raw_payload_hash", sa.String(length=64), nullable=False),
        sa.Column("bundle_status", sa.String(length=16), nullable=False),
        sa.Column("freshness_label", sa.String(length=32), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "feed", "symbol", "raw_payload_hash"),
    )
    for column in ("provider", "feed", "symbol", "as_of", "fetched_at", "raw_payload_hash"):
        op.create_index(f"ix_market_snapshots_{column}", "market_snapshots", [column])

    op.create_table(
        "market_bars",
        sa.Column("id", sa.String(length=512), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("feed", sa.String(length=32), nullable=False),
        sa.Column("coverage", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("exchange", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("interval", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("high", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("low", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("close", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider", "feed", "symbol", "interval", "raw_payload_hash"
        ),
    )
    for column in (
        "provider",
        "feed",
        "symbol",
        "timestamp",
        "fetched_at",
        "raw_payload_hash",
    ):
        op.create_index(f"ix_market_bars_{column}", "market_bars", [column])


def downgrade() -> None:
    op.drop_table("market_bars")
    op.drop_table("market_snapshots")
