"""Remove raw-payload uniqueness from immutable market observations.

Revision ID: 20260830_0007
Revises: 20260830_0006
Create Date: 2026-08-30 00:00:06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0007"
down_revision: str | Sequence[str] | None = "20260830_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SQLITE_BATCH_NAMING = {"uq": "uq_%(table_name)s_%(column_0_N_name)s"}
_SNAPSHOT_UNIQUE_COLUMNS = ("provider", "feed", "symbol", "raw_payload_hash")
_BAR_UNIQUE_COLUMNS = ("provider", "feed", "symbol", "interval", "raw_payload_hash")


def upgrade() -> None:
    _drop_unique_constraint(
        "market_snapshots",
        _SNAPSHOT_UNIQUE_COLUMNS,
        sqlite_constraint_name="uq_market_snapshots_provider_feed_symbol_raw_payload_hash",
    )
    _drop_unique_constraint(
        "market_bars",
        _BAR_UNIQUE_COLUMNS,
        sqlite_constraint_name="uq_market_bars_provider_feed_symbol_interval_raw_payload_hash",
    )


def downgrade() -> None:
    """Forward-only compatibility migration; immutable observation IDs remain authoritative."""


def _drop_unique_constraint(
    table_name: str,
    column_names: tuple[str, ...],
    *,
    sqlite_constraint_name: str,
) -> None:
    connection = op.get_bind()
    constraints = [
        constraint
        for constraint in sa.inspect(connection).get_unique_constraints(table_name)
        if tuple(constraint.get("column_names") or ()) == column_names
    ]
    if not constraints:
        return
    if connection.dialect.name == "sqlite":
        with op.batch_alter_table(
            table_name,
            recreate="always",
            naming_convention=_SQLITE_BATCH_NAMING,
        ) as batch_op:
            batch_op.drop_constraint(sqlite_constraint_name, type_="unique")
        return
    with op.batch_alter_table(table_name) as batch_op:
        for constraint in constraints:
            constraint_name = constraint.get("name")
            if constraint_name is None:
                raise RuntimeError(
                    f"{table_name} unique constraint name is required for migration"
                )
            batch_op.drop_constraint(constraint_name, type_="unique")
