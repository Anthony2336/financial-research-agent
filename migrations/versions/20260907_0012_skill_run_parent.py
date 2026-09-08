"""Link every skill run to its owning application run.

Revision ID: 20260907_0012
Revises: 20260831_0011
Create Date: 2026-09-07 00:00:12
"""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

import sqlalchemy as sa
from alembic import op

revision: str = "20260907_0012"
down_revision: str | Sequence[str] | None = "20260831_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_PARENT_THESIS = "[legacy skill run parent]"


def upgrade() -> None:
    op.add_column(
        "skill_runs",
        sa.Column("run_id", sa.String(length=64), nullable=True),
    )
    connection = op.get_bind()
    skill_runs = sa.table(
        "skill_runs",
        sa.column("id", sa.String(length=36)),
        sa.column("run_id", sa.String(length=64)),
        sa.column("ticker", sa.String(length=10)),
        sa.column("recipe_name", sa.String(length=100)),
        sa.column("status", sa.String(length=16)),
        sa.column("started_at", sa.DateTime(timezone=True)),
        sa.column("completed_at", sa.DateTime(timezone=True)),
    )
    research_runs = sa.table(
        "research_runs",
        sa.column("id", sa.String(length=36)),
        sa.column("run_id", sa.String(length=64)),
        sa.column("ticker", sa.String(length=10)),
        sa.column("thesis", sa.Text()),
        sa.column("status", sa.String(length=50)),
        sa.column("corpus_version", sa.String(length=32)),
        sa.column("requested_intent", sa.String(length=64)),
        sa.column("effective_intent", sa.String(length=64)),
        sa.column("corpus_scope", sa.JSON()),
        sa.column("prompt_version", sa.String(length=64)),
        sa.column("trace_id", sa.String(length=128)),
        sa.column("report_markdown", sa.Text()),
        sa.column("created_at", sa.DateTime()),
        sa.column("completed_at", sa.DateTime()),
    )
    for row in connection.execute(sa.select(skill_runs)).mappings():
        parent_run_id = f"legacy-skill-{row['id']}"
        parent_exists = connection.scalar(
            sa.select(research_runs.c.id).where(research_runs.c.run_id == parent_run_id)
        )
        if parent_exists is None:
            connection.execute(
                sa.insert(research_runs).values(
                    id=str(uuid5(NAMESPACE_URL, parent_run_id)),
                    run_id=parent_run_id,
                    ticker=row["ticker"],
                    thesis=_LEGACY_PARENT_THESIS,
                    status=row["status"],
                    corpus_version="",
                    requested_intent=row["recipe_name"],
                    effective_intent=None,
                    corpus_scope=[],
                    prompt_version=None,
                    trace_id=None,
                    report_markdown=None,
                    created_at=row["started_at"],
                    completed_at=row["completed_at"],
                )
            )
        connection.execute(
            sa.update(skill_runs)
            .where(skill_runs.c.id == row["id"])
            .values(run_id=parent_run_id)
        )

    with op.batch_alter_table("skill_runs") as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch_op.create_foreign_key(
            "fk_skill_runs_run_id_research_runs",
            "research_runs",
            ["run_id"],
            ["run_id"],
        )
        batch_op.create_index("ix_skill_runs_run_id", ["run_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("skill_runs") as batch_op:
        batch_op.drop_index("ix_skill_runs_run_id")
        batch_op.drop_constraint(
            "fk_skill_runs_run_id_research_runs",
            type_="foreignkey",
        )
        batch_op.drop_column("run_id")
    op.execute(
        sa.text(
            "DELETE FROM research_runs WHERE thesis = :thesis "
            "AND run_id LIKE 'legacy-skill-%'"
        ).bindparams(thesis=_LEGACY_PARENT_THESIS)
    )
