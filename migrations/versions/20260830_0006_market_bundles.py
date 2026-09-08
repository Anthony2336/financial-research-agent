"""Add immutable market bundles without rewriting stamped 0005 databases.

Revision ID: 20260830_0006
Revises: 20260830_0005
Create Date: 2026-08-30 00:00:05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0006"
down_revision: str | Sequence[str] | None = "20260830_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "provider",
        "feed",
        "symbol",
        "snapshot_id",
        "bar_ids",
        "status",
        "freshness_label",
        "errors",
        "created_at",
    }
)
_OBSERVATION_INDEX = "ix_market_bundles_observation_at"


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "market_bundles" not in inspector.get_table_names():
        _create_market_bundles()
        return

    columns = {column["name"]: column for column in inspector.get_columns("market_bundles")}
    missing = sorted(_REQUIRED_COLUMNS - columns.keys())
    if missing:
        raise RuntimeError(
            "MARKET_BUNDLES_SCHEMA_INVALID: existing market_bundles is missing "
            + ", ".join(missing)
        )

    observation = columns.get("observation_at")
    if observation is None:
        _add_and_backfill_observation_at(connection)
        _ensure_observation_index(connection)
        return

    if observation["nullable"]:
        raise RuntimeError(
            "MARKET_BUNDLES_SCHEMA_INVALID: observation_at must be non-null"
        )
    if _null_observation_count(connection):
        raise RuntimeError(
            "MARKET_BUNDLES_SCHEMA_INVALID: observation_at contains null values"
        )
    indexes = {index["name"] for index in sa.inspect(connection).get_indexes("market_bundles")}
    if _OBSERVATION_INDEX not in indexes:
        raise RuntimeError(
            "MARKET_BUNDLES_SCHEMA_INVALID: observation_at index is missing"
        )


def downgrade() -> None:
    """Forward-only compatibility migration; retained market history is not discarded."""


def _create_market_bundles() -> None:
    op.create_table(
        "market_bundles",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("feed", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("snapshot_id", sa.String(length=512), nullable=False),
        sa.Column("bar_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("freshness_label", sa.String(length=32), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.Column("observation_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["market_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "provider",
        "feed",
        "symbol",
        "snapshot_id",
        "observation_at",
        "created_at",
    ):
        op.create_index(f"ix_market_bundles_{column}", "market_bundles", [column])


def _add_and_backfill_observation_at(connection) -> None:
    op.add_column(
        "market_bundles",
        sa.Column("observation_at", sa.DateTime(timezone=True), nullable=True),
    )
    bundles = sa.table(
        "market_bundles",
        sa.column("snapshot_id", sa.String()),
        sa.column("observation_at", sa.DateTime(timezone=True)),
    )
    snapshots = sa.table(
        "market_snapshots",
        sa.column("id", sa.String()),
        sa.column("fetched_at", sa.DateTime(timezone=True)),
    )
    fetched_at = (
        sa.select(snapshots.c.fetched_at)
        .where(snapshots.c.id == bundles.c.snapshot_id)
        .scalar_subquery()
    )
    connection.execute(
        sa.update(bundles)
        .where(bundles.c.observation_at.is_(None))
        .values(observation_at=fetched_at)
    )
    if _null_observation_count(connection):
        raise RuntimeError(
            "MARKET_BUNDLES_BACKFILL_FAILED: bundle snapshot provenance is missing"
        )
    with op.batch_alter_table("market_bundles") as batch_op:
        batch_op.alter_column(
            "observation_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )


def _ensure_observation_index(connection) -> None:
    indexes = {index["name"] for index in sa.inspect(connection).get_indexes("market_bundles")}
    if _OBSERVATION_INDEX not in indexes:
        op.create_index(
            _OBSERVATION_INDEX,
            "market_bundles",
            ["observation_at"],
        )


def _null_observation_count(connection) -> int:
    bundles = sa.table(
        "market_bundles",
        sa.column("observation_at", sa.DateTime(timezone=True)),
    )
    return int(
        connection.scalar(
            sa.select(sa.func.count()).select_from(bundles).where(
                bundles.c.observation_at.is_(None)
            )
        )
        or 0
    )
