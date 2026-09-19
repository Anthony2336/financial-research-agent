"""Bootstrap contracts for deployed database migration state."""

from types import SimpleNamespace

import pytest

from fra import bootstrap
from fra.bootstrap import BootstrapConfigurationError, BootstrapErrorCode
from fra.storage.database import DatabaseMigrationRequiredError


def test_postgres_bootstrap_requires_alembic_head_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling create_schema for an unversioned production database would bypass migrations."""
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    def migration_required(_engine) -> None:
        raise DatabaseMigrationRequiredError()

    def fail_create_schema(_engine) -> None:
        raise AssertionError("PostgreSQL bootstrap must not call create_schema")

    monkeypatch.setattr(bootstrap, "ensure_migrations_current", migration_required)
    monkeypatch.setattr(bootstrap, "create_schema", fail_create_schema)

    with pytest.raises(BootstrapConfigurationError) as error:
        bootstrap._initialize_schema(engine)  # type: ignore[arg-type]

    assert error.value.code is BootstrapErrorCode.DATABASE_MIGRATION_REQUIRED
    assert "uv run alembic upgrade head" in error.value.detail
