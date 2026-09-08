"""Schema helpers for isolated fixtures and migrated PostgreSQL deployments."""

from pathlib import Path

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from financial_evidence_agent.storage.models import Base


def create_schema(engine: Engine) -> None:
    """Create non-destructive isolated test schema tables from the ORM metadata."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


class DatabaseMigrationRequiredError(RuntimeError):
    """Raised when a deployed PostgreSQL schema is absent or behind Alembic head."""

    command = "uv run alembic upgrade head"

    def __init__(self) -> None:
        super().__init__(
            "DATABASE_MIGRATION_REQUIRED: database schema is absent or behind; "
            f"run {self.command}"
        )


def ensure_migrations_current(engine: Engine) -> None:
    """Require PostgreSQL databases to be at the repository Alembic head revision."""
    if engine.dialect.name != "postgresql":
        return

    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    config = Config(str(_project_root() / "alembic.ini"))
    expected_revision = ScriptDirectory.from_config(config).get_current_head()
    try:
        with engine.connect() as connection:
            actual_revision = MigrationContext.configure(connection).get_current_revision()
    except SQLAlchemyError as error:
        raise DatabaseMigrationRequiredError() from error
    if actual_revision != expected_revision:
        raise DatabaseMigrationRequiredError()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
