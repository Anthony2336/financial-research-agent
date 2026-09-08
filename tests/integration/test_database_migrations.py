"""Alembic contracts for production database schema changes."""

import json
import logging
import os
from collections.abc import Generator
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from financial_evidence_agent.config import Settings
from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketSnapshot,
    market_bar_observation_id,
    market_snapshot_observation_id,
)
from financial_evidence_agent.retrieval.xbrl import (
    SecCompanyFactsDocument,
    company_facts_url,
    normalize_company_facts,
)
from financial_evidence_agent.storage.fact_repositories import CompanyFactRepository
from financial_evidence_agent.storage.market_repositories import MarketDataRepository
from financial_evidence_agent.storage.models import (
    Chunk,
    ClaimRecord,
    Company,
    Filing,
    MarketBarRecord,
    MarketBundleRecord,
    MarketSnapshotRecord,
    ResearchMemoryRecord,
    ResearchRun,
    SkillRun,
)
from financial_evidence_agent.storage.repositories import FilingRepository


@pytest.fixture
def postgres_url() -> Generator[str, None, None]:
    """Create and drop only this test's UUID migration database."""
    if os.environ.get("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("set RUN_POSTGRES_INTEGRATION=1 to run Postgres integration tests")

    configured = make_url(Settings().database_url)
    database_name = f"financial_evidence_migration_test_{uuid4().hex}"
    admin_url = configured.set(database="postgres")
    test_url = configured.set(database=database_name)
    admin = create_engine(
        admin_url,
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 2},
    )
    database_created = False
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
        database_created = True
    except OperationalError:
        admin.dispose()
        pytest.skip("Postgres is unavailable or the test user cannot create databases")

    try:
        yield test_url.render_as_string(hide_password=False)
    finally:
        if database_created:
            with admin.connect() as connection:
                connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                    ),
                    {"database_name": database_name},
                )
                connection.execute(text(f'DROP DATABASE "{database_name}"'))
        admin.dispose()


def _alembic_config(tmp_path: Path, ini_url: str):
    from alembic.config import Config

    project_root = Path(__file__).parents[2]
    source = (project_root / "alembic.ini").read_text()
    configured = source.replace(
        "script_location = %(here)s/migrations",
        f"script_location = {project_root / 'migrations'}",
    )
    configured = configured.replace(
        "sqlalchemy.url = postgresql+psycopg://financial_evidence:financial_evidence@localhost:5432/financial_evidence",
        f"sqlalchemy.url = {ini_url}",
    )
    config_path = tmp_path / f"alembic-{len(list(tmp_path.iterdir()))}.ini"
    config_path.write_text(configured)
    return Config(str(config_path))


def test_skill_run_parent_link_0012_upgrades_legacy_sqlite_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'skill-run-parent.sqlite3'}"
    _exercise_skill_run_parent_link_0012(database_url, tmp_path, monkeypatch)


def test_skill_run_parent_link_0012_upgrades_legacy_postgres_rows(
    postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_skill_run_parent_link_0012(postgres_url, tmp_path, monkeypatch)


def _exercise_skill_run_parent_link_0012(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy recipe rows gain a non-null owning application run and real FK."""
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(tmp_path, database_url)
    command.upgrade(config, "20260831_0011")
    engine = create_engine(database_url)
    started_at = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO skill_runs "
                    "(id, ticker, recipe_name, recipe_version, recipe_snapshot, status, "
                    "source_ids, errors, started_at, completed_at) VALUES "
                    "(:id, :ticker, :recipe_name, :recipe_version, :recipe_snapshot, "
                    ":status, :source_ids, :errors, :started_at, :completed_at)"
                ),
                {
                    "id": "skill-legacy-0012",
                    "ticker": "NVDA",
                    "recipe_name": "earnings_review",
                    "recipe_version": "1.0.0",
                    "recipe_snapshot": json.dumps(
                        {"name": "earnings_review", "version": "1.0.0"}
                    ),
                    "status": "completed",
                    "source_ids": json.dumps([]),
                    "errors": json.dumps([]),
                    "started_at": started_at,
                    "completed_at": started_at,
                },
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        columns = {column["name"]: column for column in inspector.get_columns("skill_runs")}
        foreign_keys = inspector.get_foreign_keys("skill_runs")
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            skill = connection.execute(
                text("SELECT run_id FROM skill_runs WHERE id = 'skill-legacy-0012'")
            ).mappings().one()
            parent = connection.execute(
                text(
                    "SELECT ticker, thesis, status FROM research_runs "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": skill["run_id"]},
            ).mappings().one()
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert columns["run_id"]["nullable"] is False
    assert any(
        foreign_key["constrained_columns"] == ["run_id"]
        and foreign_key["referred_table"] == "research_runs"
        and foreign_key["referred_columns"] == ["run_id"]
        for foreign_key in foreign_keys
    )
    assert skill["run_id"] == "legacy-skill-skill-legacy-0012"
    assert parent == {
        "ticker": "NVDA",
        "thesis": "[legacy skill run parent]",
        "status": "completed",
    }


_MARKET_0005_SHAPES = (
    "original_without_bundles",
    "interim_without_observation",
    "interim_with_observation",
)


@pytest.mark.parametrize("shape", _MARKET_0005_SHAPES)
def test_market_bundle_0006_upgrades_sqlite_0005_shapes(
    shape: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every stamped 0005 shape must upgrade without losing historical market data."""
    database_url = f"sqlite+pysqlite:///{tmp_path / f'market-{shape}.sqlite3'}"
    _exercise_market_bundle_0006_upgrade(
        database_url,
        shape=shape,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("shape", _MARKET_0005_SHAPES)
def test_market_bundle_0006_upgrades_postgres_0005_shapes(
    postgres_url: str,
    shape: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real PostgreSQL must preserve/backfill every previously possible 0005 shape."""
    _exercise_market_bundle_0006_upgrade(
        postgres_url,
        shape=shape,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def _exercise_market_bundle_0006_upgrade(
    database_url: str,
    *,
    shape: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", database_url)
    command.upgrade(_alembic_config(tmp_path, database_url), "20260830_0005")
    engine = create_engine(database_url)
    newer_time = datetime(2026, 8, 31, 14, 6, tzinfo=UTC)
    older_time = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)
    newer = _migration_snapshot(newer_time, "e" * 64)
    older = _migration_snapshot(older_time, "f" * 64)
    try:
        with engine.begin() as connection:
            if "market_bundles" in inspect(connection).get_table_names():
                connection.execute(text("DROP TABLE market_bundles"))
        repository = MarketDataRepository(engine)
        repository.save_snapshot(newer)
        repository.save_snapshot(older)
        if shape != "original_without_bundles":
            _create_interim_market_bundle_table(
                engine,
                with_observation_at=shape == "interim_with_observation",
            )
            _insert_interim_market_bundles(
                engine,
                newer=newer,
                older=older,
                with_observation_at=shape == "interim_with_observation",
            )
    finally:
        engine.dispose()

    command.upgrade(_alembic_config(tmp_path, database_url), "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        columns = {column["name"]: column for column in inspector.get_columns("market_bundles")}
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            rows = connection.execute(
                text(
                    "SELECT id, snapshot_id, observation_at FROM market_bundles "
                    "ORDER BY id"
                )
            ).mappings().all()

        repository = MarketDataRepository(engine)
        latest = repository.latest_bundle("alpaca", "iex", "NVDA")
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert columns["observation_at"]["nullable"] is False
    assert latest is not None
    assert latest.snapshot.id == newer.id
    if shape == "original_without_bundles":
        assert rows == []
    else:
        assert [row["id"] for row in rows] == ["bundle-newer", "bundle-older"]
        observations = {row["snapshot_id"]: _as_utc(row["observation_at"]) for row in rows}
        assert observations == {newer.id: newer_time, older.id: older_time}


def _migration_snapshot(fetched_at: datetime, raw_payload_hash: str) -> MarketSnapshot:
    as_of = fetched_at.replace(minute=fetched_at.minute - 1)
    return MarketSnapshot(
        id=_observation_id(
            "snapshot",
            symbol="NVDA",
            source_timestamp=as_of,
            fetched_at=fetched_at,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        price=Decimal("123.45"),
        open=Decimal("122.00"),
        day_high=Decimal("124.00"),
        day_low=Decimal("121.50"),
        previous_close=Decimal("121.00"),
        as_of=as_of,
        fetched_at=fetched_at,
        market_status="open",
        delayed_by_seconds=60,
        raw_payload_hash=raw_payload_hash,
    )


def _create_interim_market_bundle_table(
    engine,
    *,
    with_observation_at: bool,
) -> None:
    metadata = MetaData()
    Table("market_snapshots", metadata, autoload_with=engine)
    columns = [
        Column("id", String(64), primary_key=True),
        Column("provider", String(32), nullable=False, index=True),
        Column("feed", String(32), nullable=False, index=True),
        Column("symbol", String(10), nullable=False, index=True),
        Column(
            "snapshot_id",
            String(512),
            ForeignKey("market_snapshots.id"),
            nullable=False,
            index=True,
        ),
        Column("bar_ids", JSON, nullable=False),
        Column("status", String(16), nullable=False),
        Column("freshness_label", String(32), nullable=False),
        Column("errors", JSON, nullable=False),
    ]
    if with_observation_at:
        columns.append(
            Column("observation_at", DateTime(timezone=True), nullable=False, index=True)
        )
    columns.append(Column("created_at", DateTime(timezone=True), nullable=False, index=True))
    Table("market_bundles", metadata, *columns)
    metadata.create_all(engine)


def _insert_interim_market_bundles(
    engine,
    *,
    newer: MarketSnapshot,
    older: MarketSnapshot,
    with_observation_at: bool,
) -> None:
    metadata = MetaData()
    bundles = Table("market_bundles", metadata, autoload_with=engine)
    values = [
        {
            "id": "bundle-newer",
            "provider": "alpaca",
            "feed": "iex",
            "symbol": "NVDA",
            "snapshot_id": newer.id,
            "bar_ids": [],
            "status": "partial",
            "freshness_label": "open-iex",
            "errors": [],
            "created_at": datetime(2026, 8, 31, 14, 7, tzinfo=UTC),
        },
        {
            "id": "bundle-older",
            "provider": "alpaca",
            "feed": "iex",
            "symbol": "NVDA",
            "snapshot_id": older.id,
            "bar_ids": [],
            "status": "partial",
            "freshness_label": "open-iex",
            "errors": [],
            "created_at": datetime(2026, 8, 31, 15, tzinfo=UTC),
        },
    ]
    if with_observation_at:
        values[0]["observation_at"] = newer.fetched_at
        values[1]["observation_at"] = older.fetched_at
    with engine.begin() as connection:
        connection.execute(bundles.insert(), values)


def _as_utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _observation_id(
    kind: str,
    *,
    symbol: str,
    source_timestamp: datetime,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "provider": "alpaca",
            "feed": "iex",
            "symbol": symbol,
            "source_timestamp": _iso_z(source_timestamp),
            "fetched_at": _iso_z(fetched_at),
            "raw_payload_hash": raw_payload_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"market-{kind}:{sha256(payload).hexdigest()}"


def test_alembic_upgrade_head_creates_run_provenance(
    postgres_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh production schemas require migrated provenance tables, not create_all."""
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    command.upgrade(_alembic_config(tmp_path, postgres_url), "head")

    engine = create_engine(postgres_url)
    try:
        tables = set(inspect(engine).get_table_names())
        chunk_columns = {column["name"] for column in inspect(engine).get_columns("chunks")}
        memory_indexes = {
            index["name"] for index in inspect(engine).get_indexes("research_memories")
        }
        memory_columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("research_memories")
        }
        fact_columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("company_facts")
        }
        web_columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("web_evidence")
        }
    finally:
        engine.dispose()
    assert {
        "research_runs",
        "research_memories",
        "company_facts",
        "claims",
        "source_fetches",
        "market_snapshots",
        "market_bars",
        "market_bundles",
        "alembic_version",
    } <= tables
    assert "embedding_model" in chunk_columns
    assert "ix_research_memories_embedding_cosine" in memory_indexes
    assert isinstance(memory_columns["evidence_source_refs"]["type"], JSONB)
    assert isinstance(memory_columns["embedding"]["type"], Vector)
    assert memory_columns["embedding"]["type"].dim == 1024
    assert isinstance(fact_columns["value"]["type"], Numeric)
    assert isinstance(web_columns["time_metadata_validated"]["type"], Boolean)
    assert web_columns["time_metadata_validated"]["nullable"] is False


def test_research_corpus_migration_backfills_legacy_filings_without_changing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive corpus migration must preserve every legacy filing and chunk id."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'research-corpus.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(tmp_path, database_url)
    command.upgrade(config, "20260830_0007")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO companies (id, ticker) "
                    "VALUES ('company-nvda', 'NVDA')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO filings "
                    "(id, company_id, accession_no, form, filed_at, source_url, "
                    "raw_text, content_hash, corpus_version) VALUES "
                    "('filing-legacy', 'company-nvda', '0001045810-25-000041', "
                    "'10-Q', '2025-05-28', 'https://www.sec.gov/legacy', "
                    "'legacy evidence', :content_hash, 'NVDA-v1')"
                ),
                {"content_hash": "a" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO chunks "
                    "(id, filing_id, section, chunk_index, content, token_count, "
                    "raw_start, raw_end, embedding, embedding_model) VALUES "
                    "('chunk-legacy', 'filing-legacy', 'MD&A', 0, "
                    "'legacy evidence', 2, 0, 15, NULL, NULL)"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
        corpus_columns = {
            column["name"] for column in inspect(engine).get_columns("research_corpora")
        }
        corpus_uniques = {
            frozenset(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("research_corpora")
        }
        association_indexes = {
            tuple(index["column_names"])
            for index in inspect(engine).get_indexes("corpus_filings")
        }
        with engine.connect() as connection:
            filing_ids = connection.scalars(text("SELECT id FROM filings")).all()
            chunk_ids = connection.scalars(text("SELECT id FROM chunks")).all()
            corpora = connection.execute(
                text(
                    "SELECT rc.version, rc.as_of_date, rc.membership_hash, cf.filing_id "
                    "FROM research_corpora AS rc "
                    "JOIN corpus_filings AS cf ON cf.corpus_id = rc.id"
                )
            ).all()
    finally:
        engine.dispose()

    assert {"research_corpora", "corpus_filings"} <= tables
    assert "membership_hash" in corpus_columns
    assert frozenset({"company_id", "membership_hash"}) in corpus_uniques
    assert ("filing_id",) in association_indexes
    assert filing_ids == ["filing-legacy"]
    assert chunk_ids == ["chunk-legacy"]
    assert [
        (row.version, str(row.as_of_date), row.membership_hash, row.filing_id)
        for row in corpora
    ] == [
        (
            "NVDA-v1",
            "2025-05-28",
            sha256(b"filing-legacy").hexdigest(),
            "filing-legacy",
        )
    ]

    command.downgrade(config, "20260830_0007")
    engine = create_engine(database_url)
    try:
        downgraded_tables = set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            downgraded_filing_ids = connection.scalars(text("SELECT id FROM filings")).all()
            downgraded_chunk_ids = connection.scalars(text("SELECT id FROM chunks")).all()
    finally:
        engine.dispose()

    assert "research_corpora" not in downgraded_tables
    assert "corpus_filings" not in downgraded_tables
    assert downgraded_filing_ids == ["filing-legacy"]
    assert downgraded_chunk_ids == ["chunk-legacy"]


def test_postgres_0007_to_head_preserves_corpus_membership_and_evidence_identity(
    postgres_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production migration chain must preserve the exact legacy evidence boundary."""
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    config = _alembic_config(tmp_path, postgres_url)
    command.upgrade(config, "20260830_0007")
    engine = create_engine(postgres_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO companies (id, ticker) "
                    "VALUES ('company-nvda', 'NVDA')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO filings "
                    "(id, company_id, accession_no, form, filed_at, source_url, "
                    "raw_text, content_hash, corpus_version) VALUES "
                    "('filing-legacy', 'company-nvda', '0001045810-25-000041', "
                    "'10-Q', '2025-05-28', 'https://www.sec.gov/legacy', "
                    "'legacy evidence', :content_hash, 'NVDA-v1')"
                ),
                {"content_hash": "a" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO chunks "
                    "(id, filing_id, section, chunk_index, content, token_count, "
                    "raw_start, raw_end, embedding, embedding_model) VALUES "
                    "('chunk-legacy', 'filing-legacy', 'MD&A', 0, "
                    "'legacy evidence', 2, 0, 15, NULL, NULL)"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(postgres_url)
    try:
        inspector = inspect(engine)
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            preserved = connection.execute(
                text(
                    "SELECT rc.version, rc.membership_hash, cf.filing_id, "
                    "f.accession_no, f.content_hash, c.id, c.content, "
                    "c.raw_start, c.raw_end "
                    "FROM research_corpora AS rc "
                    "JOIN corpus_filings AS cf ON cf.corpus_id = rc.id "
                    "JOIN filings AS f ON f.id = cf.filing_id "
                    "JOIN chunks AS c ON c.filing_id = f.id"
                )
            ).one()
        tables = set(inspector.get_table_names())
        web_columns = {column["name"] for column in inspector.get_columns("web_evidence")}
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert tuple(preserved) == (
        "NVDA-v1",
        sha256(b"filing-legacy").hexdigest(),
        "filing-legacy",
        "0001045810-25-000041",
        "a" * 64,
        "chunk-legacy",
        "legacy evidence",
        0,
        15,
    )
    assert {"research_corpora", "corpus_filings", "research_memories", "company_facts"} <= tables
    assert "time_metadata_validated" in web_columns


def test_research_memory_0009_is_additive_without_historical_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only post-guard application writes may populate the new table."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'research-memory.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(tmp_path, database_url)
    command.upgrade(config, "20260831_0008")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO research_runs "
                    "(id, run_id, ticker, thesis, status, corpus_version, corpus_scope, "
                    "report_markdown, created_at, completed_at) VALUES "
                    "('stored-run-id', 'historical-run', 'NVDA', 'historical request', "
                    "'completed', 'NVDA-v1', '[\"NVDA-v1\"]', '# report', "
                    "'2026-08-31 12:00:00', '2026-08-31 12:01:00')"
                )
            )
        preexisting_tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    command.upgrade(config, "20260831_0009")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        memory_columns = {
            column["name"]: column for column in inspector.get_columns("research_memories")
        }
        memory_checks = {
            constraint["name"]
            for constraint in inspector.get_check_constraints("research_memories")
        }
        memory_indexes = {
            index["name"] for index in inspector.get_indexes("research_memories")
        }
        memory_foreign_keys = inspector.get_foreign_keys("research_memories")
        with Session(engine) as session:
            revision = session.scalar(text("SELECT version_num FROM alembic_version"))
            count = session.scalar(select(func.count()).select_from(ResearchMemoryRecord))
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO research_memories "
                        "(id, scope_key, ticker, memory_kind, summary, source_run_id, "
                        "evidence_source_refs, corpus_version, embedding, embedding_model, "
                        "importance, created_at, expires_at) VALUES "
                        "(:id, 'ticker:NVDA', 'NVDA', 'research_summary', 'summary', "
                        "'historical-run', '[]', 'NVDA-v1', :embedding, 'hash-1024-v1', "
                        "0.5, '2026-09-01 12:00:00', '2026-12-01 12:00:00')"
                    ),
                    {
                        "id": "e" * 64,
                        "embedding": json.dumps([0.0] * 1024),
                    },
                )
    finally:
        engine.dispose()

    assert revision == "20260831_0009"
    assert count == 0
    assert set(memory_columns) == {
        "id",
        "scope_key",
        "ticker",
        "memory_kind",
        "summary",
        "source_run_id",
        "evidence_source_refs",
        "corpus_version",
        "embedding",
        "embedding_model",
        "importance",
        "created_at",
        "expires_at",
    }
    assert memory_columns["embedding"]["nullable"] is False
    assert memory_columns["embedding_model"]["nullable"] is False
    assert {
        "ck_research_memories_kind",
        "ck_research_memories_expiry",
        "ck_research_memories_importance",
        "ck_research_memories_evidence_nonempty",
    } <= memory_checks
    assert {
        "ix_research_memories_scope_kind_expiry",
        "ix_research_memories_source_run_id",
        "ix_research_memories_corpus_version",
        "ix_research_memories_embedding_model",
    } <= memory_indexes
    assert any(
        key["constrained_columns"] == ["source_run_id"]
        and key["referred_table"] == "research_runs"
        and key["referred_columns"] == ["run_id"]
        for key in memory_foreign_keys
    )

    command.downgrade(config, "20260831_0008")
    engine = create_engine(database_url)
    try:
        downgraded_tables = set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            run_ids = connection.scalars(text("SELECT run_id FROM research_runs")).all()
    finally:
        engine.dispose()

    assert downgraded_tables == preexisting_tables
    assert "research_memories" not in downgraded_tables
    assert run_ids == ["historical-run"]


def test_corpus_membership_cutoff_0011_backfills_the_latest_member_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previously stamped request cutoffs must become membership-derived dates."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'corpus-cutoff.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(tmp_path, database_url)
    command.upgrade(config, "20260831_0010")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO companies (id, ticker) VALUES ('company-nvda', 'NVDA')")
            )
            connection.execute(
                text(
                    "INSERT INTO filings "
                    "(id, company_id, accession_no, form, filed_at, source_url, raw_text, "
                    "content_hash, corpus_version) VALUES "
                    "('filing-old', 'company-nvda', 'old', '10-Q', '2025-02-20', "
                    "'https://www.sec.gov/old', 'old', :old_hash, 'NVDA-v1'), "
                    "('filing-new', 'company-nvda', 'new', '10-Q', '2025-05-28', "
                    "'https://www.sec.gov/new', 'new', :new_hash, 'NVDA-v2')"
                ),
                {"old_hash": "a" * 64, "new_hash": "b" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO research_corpora "
                    "(id, company_id, version, membership_hash, as_of_date, created_at) "
                    "VALUES ('corpus-future', 'company-nvda', 'NVDA-v3', :membership_hash, "
                    "'2099-01-01', '2026-09-04 00:00:00')"
                ),
                {"membership_hash": sha256(b"filing-new\nfiling-old").hexdigest()},
            )
            connection.execute(
                text(
                    "INSERT INTO corpus_filings (corpus_id, filing_id) VALUES "
                    "('corpus-future', 'filing-new'), ('corpus-future', 'filing-old')"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            cutoff = connection.scalar(
                text(
                    "SELECT as_of_date FROM research_corpora "
                    "WHERE id = 'corpus-future'"
                )
            )
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert str(cutoff) == "2025-05-28"


def test_company_facts_0010_is_additive_and_downgrade_drops_only_task9_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 9 must not backfill or disturb 0008/0009 data and metadata."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'company-facts.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = _alembic_config(tmp_path, database_url)
    command.upgrade(config, "20260831_0009")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO companies (id, ticker, cik, legal_name, ir_domain) "
                    "VALUES ('company-nvda', 'NVDA', '0001045810', "
                    "'NVIDIA CORPORATION', 'investor.nvidia.com')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO web_evidence "
                    "(id, ticker, title, content, source_url, source_kind, source_tier, "
                    "published_at, fetched_at, content_hash) VALUES "
                    "('legacy-web-time', 'NVDA', 'Legacy', 'Legacy content', "
                    "'https://www.reuters.com/legacy', 'authoritative_web', "
                    "'authoritative_secondary', '2026-08-01 12:00:00', "
                    "'2026-08-02 12:00:00', 'legacy-web-hash')"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        columns = {
            column["name"]: column for column in inspector.get_columns("company_facts")
        }
        checks = {
            constraint["name"]
            for constraint in inspector.get_check_constraints("company_facts")
        }
        indexes = {
            index["name"] for index in inspector.get_indexes("company_facts")
        }
        web_columns = {
            column["name"]: column
            for column in inspector.get_columns("web_evidence")
        }
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            fact_count = connection.scalar(text("SELECT COUNT(*) FROM company_facts"))
            company = connection.execute(
                text(
                    "SELECT ticker, cik, legal_name, ir_domain FROM companies "
                    "WHERE id = 'company-nvda'"
                )
            ).one()
            legacy_time_validated = connection.scalar(
                text(
                    "SELECT time_metadata_validated FROM web_evidence "
                    "WHERE id = 'legacy-web-time'"
                )
            )
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert set(columns) == {
        "id",
        "ticker",
        "cik",
        "taxonomy",
        "concept",
        "period_start",
        "period_end",
        "instant",
        "unit",
        "currency",
        "value",
        "form",
        "filed_at",
        "accession_no",
        "source_url",
        "frame",
        "raw_content_hash",
        "fetched_at",
    }
    assert {"ck_company_facts_form", "ck_company_facts_period"} <= checks
    assert {
        "ix_company_facts_ticker",
        "ix_company_facts_cik",
        "ix_company_facts_concept",
        "ix_company_facts_accession_no",
    } <= indexes
    assert fact_count == 0
    assert web_columns["time_metadata_validated"]["nullable"] is False
    assert legacy_time_validated in {False, 0}
    assert tuple(company) == (
        "NVDA",
        "0001045810",
        "NVIDIA CORPORATION",
        "investor.nvidia.com",
    )

    command.downgrade(config, "20260831_0009")
    engine = create_engine(database_url)
    try:
        downgraded_tables = set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            company_count = connection.scalar(text("SELECT COUNT(*) FROM companies"))
            memory_count = connection.scalar(text("SELECT COUNT(*) FROM research_memories"))
            legacy_web_count = connection.scalar(
                text("SELECT COUNT(*) FROM web_evidence WHERE id = 'legacy-web-time'")
            )
        downgraded_web_columns = {
            column["name"] for column in inspect(engine).get_columns("web_evidence")
        }
    finally:
        engine.dispose()

    assert "company_facts" not in downgraded_tables
    assert "research_memories" in downgraded_tables
    assert company_count == 1
    assert memory_count == 0
    assert legacy_web_count == 1
    assert "time_metadata_validated" not in downgraded_web_columns


def test_company_facts_0010_round_trips_exact_decimal_on_postgres(
    postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real PostgreSQL NUMERIC/timestamptz must match the SQLite/ORM fact contract."""
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    command.upgrade(_alembic_config(tmp_path, postgres_url), "head")
    engine = create_engine(postgres_url)
    try:
        FilingRepository(engine).upsert_company_metadata(
            ticker="NVDA",
            cik="0001045810",
            legal_name="NVIDIA CORPORATION",
            ir_domain="investor.nvidia.com",
        )
        snapshot = normalize_company_facts(
            SecCompanyFactsDocument(
                source_url=company_facts_url("0001045810"),
                raw_bytes=Path("tests/fixtures/sec/companyfacts_nvda.json").read_bytes(),
                fetched_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
            ),
            ticker="NVDA",
            cik="0001045810",
        )
        repository = CompanyFactRepository(engine)
        revenue = next(
            fact
            for fact in snapshot.facts
            if fact.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
        ).model_copy(
            update={"value": Decimal("99999999999999999999.999999999999999999")}
        )
        repository.save_facts([revenue])
        facts = repository.list_facts(
            "NVDA",
            concepts=["RevenueFromContractWithCustomerExcludingAssessedTax"],
            limit=10,
        )
    finally:
        engine.dispose()

    assert len(facts) == 1
    assert facts[0].value == Decimal("99999999999999999999.999999999999999999")
    assert type(facts[0].value) is Decimal
    assert facts[0].fetched_at == datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_market_migration_round_trips_exact_decimal_on_postgres(
    postgres_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Migrated PostgreSQL NUMERIC/timestamptz columns retain exact market provenance."""
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    command.upgrade(_alembic_config(tmp_path, postgres_url), "head")
    engine = create_engine(postgres_url)
    now = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)
    as_of = datetime(2026, 8, 31, 14, tzinfo=UTC)
    snapshot_hash = "a" * 64
    snapshot = MarketSnapshot(
        id=_observation_id(
            "snapshot",
            symbol="NVDA",
            source_timestamp=as_of,
            fetched_at=now,
            raw_payload_hash=snapshot_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        price=Decimal("123.450000000000000001"),
        open=Decimal("122.000000000000000001"),
        day_high=Decimal("124.000000000000000001"),
        day_low=Decimal("121.500000000000000001"),
        previous_close=Decimal("121.000000000000000001"),
        as_of=as_of,
        fetched_at=now,
        market_status="open",
        delayed_by_seconds=60,
        raw_payload_hash=snapshot_hash,
    )
    bar_timestamp = datetime(2026, 8, 28, 4, tzinfo=UTC)
    bar_hash = "b" * 64
    bar = MarketBar(
        id=_observation_id(
            "bar",
            symbol="NVDA",
            source_timestamp=bar_timestamp,
            fetched_at=now,
            raw_payload_hash=bar_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        interval="1Day",
        timestamp=bar_timestamp,
        open=Decimal("120.000000000000000001"),
        high=Decimal("124.000000000000000001"),
        low=Decimal("119.000000000000000001"),
        close=Decimal("123.450000000000000001"),
        volume=9_007_199_254_740_993,
        fetched_at=now,
        raw_payload_hash=bar_hash,
    )
    bundle = MarketDataBundle(
        snapshot=snapshot,
        bars=[bar],
        status="completed",
        freshness_label="open-iex",
    )
    newer_time = datetime(2026, 8, 31, 14, 6, tzinfo=UTC)
    newer = MarketDataBundle(
        snapshot=snapshot.model_copy(
            update={
                "id": market_snapshot_observation_id(
                    provider=snapshot.provider,
                    feed=snapshot.feed,
                    symbol=snapshot.symbol,
                    as_of=datetime(2026, 8, 31, 14, 5, tzinfo=UTC),
                    fetched_at=newer_time,
                    raw_payload_hash="c" * 64,
                ),
                "as_of": datetime(2026, 8, 31, 14, 5, tzinfo=UTC),
                "fetched_at": newer_time,
                "raw_payload_hash": "c" * 64,
            }
        ),
        bars=[
            bar.model_copy(
                update={
                    "id": market_bar_observation_id(
                        provider=bar.provider,
                        feed=bar.feed,
                        symbol=bar.symbol,
                        timestamp=datetime(2026, 8, 29, 4, tzinfo=UTC),
                        fetched_at=newer_time,
                        raw_payload_hash="d" * 64,
                    ),
                    "timestamp": datetime(2026, 8, 29, 4, tzinfo=UTC),
                    "fetched_at": newer_time,
                    "raw_payload_hash": "d" * 64,
                }
            )
        ],
        status="completed",
        freshness_label="open-iex",
    )
    try:
        repository = MarketDataRepository(engine)
        repository.save_bundle(newer)
        repository.save_bundle(bundle)
        stored = repository.latest_bundle("alpaca", "iex", "NVDA")
        snapshot_columns = {
            column["name"]: column for column in inspect(engine).get_columns("market_snapshots")
        }
        bundle_columns = {
            column["name"]: column for column in inspect(engine).get_columns("market_bundles")
        }
    finally:
        engine.dispose()

    assert stored == newer
    assert snapshot_columns["price"]["type"].scale == 18
    assert snapshot_columns["as_of"]["type"].timezone is True
    assert bundle_columns["observation_at"]["type"].timezone is True


def _migration_bar(
    timestamp: datetime,
    *,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> MarketBar:
    return MarketBar(
        id=_observation_id(
            "bar",
            symbol="NVDA",
            source_timestamp=timestamp,
            fetched_at=fetched_at,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        interval="1Day",
        timestamp=timestamp,
        open=Decimal("120.00"),
        high=Decimal("124.00"),
        low=Decimal("119.00"),
        close=Decimal("123.45"),
        volume=9_007_199_254_740_993,
        fetched_at=fetched_at,
        raw_payload_hash=raw_payload_hash,
    )


def _migration_bundle(
    *,
    snapshot_fetched_at: datetime,
    snapshot_raw_payload_hash: str,
    bar_timestamp: datetime,
    bar_fetched_at: datetime,
    bar_raw_payload_hash: str,
) -> MarketDataBundle:
    return MarketDataBundle(
        snapshot=_migration_snapshot(snapshot_fetched_at, snapshot_raw_payload_hash),
        bars=[
            _migration_bar(
                bar_timestamp,
                fetched_at=bar_fetched_at,
                raw_payload_hash=bar_raw_payload_hash,
            )
        ],
        status="completed",
        freshness_label="open-iex",
    )


def test_market_observation_identity_0007_upgrades_sqlite_and_allows_refetches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The August 31, 2026 `0007` migration must remove raw-payload uniqueness without data loss."""
    _exercise_market_observation_identity_0007_upgrade(
        f"sqlite+pysqlite:///{tmp_path / 'market-0007.sqlite3'}",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def test_market_observation_identity_0007_upgrades_postgres_and_allows_refetches(
    postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real PostgreSQL must accept same-payload refetches and corrected rows after `0007`."""
    _exercise_market_observation_identity_0007_upgrade(
        postgres_url,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def _exercise_market_observation_identity_0007_upgrade(
    database_url: str,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alembic import command

    monkeypatch.setenv("DATABASE_URL", database_url)
    command.upgrade(_alembic_config(tmp_path, database_url), "20260830_0006")

    engine = create_engine(database_url)
    original = _migration_bundle(
        snapshot_fetched_at=datetime(2026, 8, 31, 14, 1, tzinfo=UTC),
        snapshot_raw_payload_hash="a" * 64,
        bar_timestamp=datetime(2026, 8, 28, 4, tzinfo=UTC),
        bar_fetched_at=datetime(2026, 8, 31, 14, 1, tzinfo=UTC),
        bar_raw_payload_hash="b" * 64,
    )
    try:
        MarketDataRepository(engine).save_bundle(original)
    finally:
        engine.dispose()

    command.upgrade(_alembic_config(tmp_path, database_url), "head")

    refetched = _migration_bundle(
        snapshot_fetched_at=datetime(2026, 8, 31, 14, 2, tzinfo=UTC),
        snapshot_raw_payload_hash="a" * 64,
        bar_timestamp=datetime(2026, 8, 28, 4, tzinfo=UTC),
        bar_fetched_at=datetime(2026, 8, 31, 14, 2, tzinfo=UTC),
        bar_raw_payload_hash="b" * 64,
    )
    corrected = _migration_bundle(
        snapshot_fetched_at=datetime(2026, 8, 31, 14, 2, tzinfo=UTC),
        snapshot_raw_payload_hash="c" * 64,
        bar_timestamp=datetime(2026, 8, 28, 4, tzinfo=UTC),
        bar_fetched_at=datetime(2026, 8, 31, 14, 2, tzinfo=UTC),
        bar_raw_payload_hash="d" * 64,
    )
    expected_latest = max([refetched, corrected], key=lambda bundle: bundle.snapshot.id)
    engine = create_engine(database_url)
    try:
        repository = MarketDataRepository(engine)
        repository.save_bundle(refetched)
        repository.save_bundle(corrected)
        repository.save_bundle(corrected)
        latest = repository.latest_bundle("alpaca", "iex", "NVDA")
        with Session(engine) as session:
            snapshot_count = session.scalar(select(func.count()).select_from(MarketSnapshotRecord))
            bar_count = session.scalar(select(func.count()).select_from(MarketBarRecord))
            bundle_count = session.scalar(select(func.count()).select_from(MarketBundleRecord))
            revision = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        snapshot_uniques = {
            tuple(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("market_snapshots")
        }
        bar_uniques = {
            tuple(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("market_bars")
        }
    finally:
        engine.dispose()

    assert revision == "20260907_0012"
    assert latest is not None
    assert latest.snapshot.id == expected_latest.snapshot.id
    assert [bar.id for bar in latest.bars] == [bar.id for bar in expected_latest.bars]
    assert latest.status == expected_latest.status
    assert latest.freshness_label == expected_latest.freshness_label
    assert latest.errors == expected_latest.errors
    assert snapshot_count == 3
    assert bar_count == 3
    assert bundle_count == 3
    assert ("provider", "feed", "symbol", "raw_payload_hash") not in snapshot_uniques
    assert ("provider", "feed", "symbol", "interval", "raw_payload_hash") not in bar_uniques


def test_cli_alembic_uses_database_url_without_programmatic_ini_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI config must migrate only Settings.database_url, never the ini placeholder."""
    from alembic import command

    target_path = tmp_path / "configured.sqlite3"
    untouched_path = tmp_path / "not-configured.sqlite3"
    target_url = f"sqlite+pysqlite:///{target_path}"
    untouched_url = f"sqlite+pysqlite:///{untouched_path}"
    monkeypatch.setenv("DATABASE_URL", target_url)

    config = _alembic_config(tmp_path, untouched_url)
    assert config.get_main_option("sqlalchemy.url") != target_url
    command.upgrade(config, "head")

    engine = create_engine(target_url)
    try:
        assert {
            "research_runs",
            "claims",
            "market_snapshots",
            "market_bars",
            "market_bundles",
            "alembic_version",
        } <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert not untouched_path.exists()


def test_alembic_rejects_complete_unversioned_schema_without_changing_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete pre-Alembic schema is still unsafe to adopt by table names alone."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'legacy.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    command.upgrade(_alembic_config(tmp_path, database_url), "20260830_0001")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO companies (id, ticker) VALUES ('legacy-company', 'NVDA')")
            )
            connection.execute(text("DROP TABLE alembic_version"))

        with pytest.raises(
            RuntimeError,
            match="DATABASE_UNVERSIONED_SCHEMA_REJECTED.*[Bb]ack up/export.*recreate",
        ):
            command.upgrade(_alembic_config(tmp_path, database_url), "head")

        with engine.connect() as connection:
            tickers = connection.execute(
                select(text("ticker")).select_from(text("companies"))
            ).all()
        chunk_columns = {column["name"] for column in inspect(engine).get_columns("chunks")}
    finally:
        engine.dispose()

    assert tickers == [("NVDA",)]
    assert "embedding_model" not in chunk_columns


def test_alembic_upgrade_head_rejects_partial_unversioned_schema(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial legacy database must not be mistaken for a safe migration target."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'partial.sqlite3'}"
    metadata = MetaData()
    Table("companies", metadata, Column("id", String(36), primary_key=True))
    engine = create_engine(database_url)
    try:
        metadata.create_all(engine)
        monkeypatch.setenv("DATABASE_URL", database_url)
        with pytest.raises(RuntimeError, match="(?i)back up/export.*recreate"):
            command.upgrade(_alembic_config(tmp_path, database_url), "head")
        assert set(inspect(engine).get_table_names()) == {"alembic_version", "companies"}
    finally:
        engine.dispose()


def test_alembic_rejects_prior_web_uniqueness_shape_without_data_loss(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prior URL/hash-only web schema must be exported or recreated, never stamped."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'legacy-web.sqlite3'}"
    metadata = MetaData()
    legacy_web = Table(
        "web_evidence",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("ticker", String(10), nullable=False),
        Column("source_url", String(2_000), nullable=False),
        Column("content_hash", String(128), nullable=False),
        UniqueConstraint("source_url", "content_hash", name="uq_legacy_web_url_hash"),
    )
    engine = create_engine(database_url)
    try:
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(
                legacy_web.insert().values(
                    id="legacy-web-1",
                    ticker="NVDA",
                    source_url="https://investor.nvidia.com/results",
                    content_hash="legacy-hash",
                )
            )
        monkeypatch.setenv("DATABASE_URL", database_url)

        with pytest.raises(RuntimeError, match="DATABASE_UNVERSIONED_SCHEMA_REJECTED"):
            command.upgrade(_alembic_config(tmp_path, database_url), "head")

        with engine.connect() as connection:
            rows = connection.execute(
                select(
                    legacy_web.c.id,
                    legacy_web.c.ticker,
                    legacy_web.c.source_url,
                    legacy_web.c.content_hash,
                )
            ).all()
        unique_columns = {
            tuple(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("web_evidence")
        }
    finally:
        engine.dispose()

    assert rows == [
        (
            "legacy-web-1",
            "NVDA",
            "https://investor.nvidia.com/results",
            "legacy-hash",
        )
    ]
    assert ("source_url", "content_hash") in unique_columns
    assert ("ticker", "source_url", "content_hash") not in unique_columns


def test_alembic_migration_keeps_existing_application_logger_enabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alembic logging must not suppress cache error logs after one migration invocation."""
    from alembic import command

    logger = logging.getLogger("financial_evidence_agent.storage.cache")
    previous_disabled = logger.disabled
    logger.disabled = False
    try:
        database_url = f"sqlite+pysqlite:///{tmp_path / 'logging.sqlite3'}"
        monkeypatch.setenv("DATABASE_URL", database_url)
        command.upgrade(_alembic_config(tmp_path, database_url), "20260830_0001")

        assert logger.disabled is False
    finally:
        logger.disabled = previous_disabled


def test_source_reference_migration_resolves_only_unambiguous_same_ticker_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy IDs must become filing/web keys only when ticker-scoped type is certain."""
    from alembic import command

    database_url = f"sqlite+pysqlite:///{tmp_path / 'source-refs.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    command.upgrade(_alembic_config(tmp_path, database_url), "20260830_0003")
    engine = create_engine(database_url)
    now = datetime(2026, 8, 30, tzinfo=UTC)
    try:
        with Session(engine) as session, session.begin():
            nvda = Company(id="company-nvda", ticker="NVDA")
            amd = Company(id="company-amd", ticker="AMD")
            session.add_all([nvda, amd])
            session.flush()
            nvda_filing = Filing(
                id="filing-nvda",
                company_id=nvda.id,
                accession_no="0001045810-26-000001",
                form="10-Q",
                filed_at=date(2026, 5, 20),
                source_url="https://www.sec.gov/Archives/nvda.htm",
                raw_text="filing only shared",
                content_hash="a" * 64,
                corpus_version="NVDA-v1",
            )
            amd_filing = Filing(
                id="filing-amd",
                company_id=amd.id,
                accession_no="0000002488-26-000001",
                form="10-Q",
                filed_at=date(2026, 5, 20),
                source_url="https://www.sec.gov/Archives/amd.htm",
                raw_text="other ticker",
                content_hash="b" * 64,
                corpus_version="AMD-v1",
            )
            session.add_all([nvda_filing, amd_filing])
            session.flush()
            session.add_all(
                [
                    Chunk(
                        id="filing-only",
                        filing_id=nvda_filing.id,
                        section="MD&A",
                        chunk_index=0,
                        content="filing only",
                        token_count=2,
                        raw_start=0,
                        raw_end=11,
                    ),
                    Chunk(
                        id="shared-id",
                        filing_id=nvda_filing.id,
                        section="Risk Factors",
                        chunk_index=1,
                        content="shared",
                        token_count=1,
                        raw_start=12,
                        raw_end=18,
                    ),
                    Chunk(
                        id="other-ticker",
                        filing_id=amd_filing.id,
                        section="MD&A",
                        chunk_index=0,
                        content="other",
                        token_count=1,
                        raw_start=0,
                        raw_end=5,
                    ),
                ]
            )
            session.execute(
                text(
                    "INSERT INTO web_evidence "
                    "(id, ticker, title, content, source_url, source_kind, source_tier, "
                    "published_at, fetched_at, content_hash) VALUES "
                    "(:id, 'NVDA', :title, :content, :source_url, 'issuer_ir', "
                    "'primary', :published_at, :fetched_at, :content_hash)"
                ),
                [
                    {
                        "id": "web-only",
                        "title": "Web only",
                        "content": "web only",
                        "source_url": "https://investor.nvidia.com/web-only",
                        "published_at": now,
                        "fetched_at": now,
                        "content_hash": "web-only-hash",
                    },
                    {
                        "id": "shared-id",
                        "title": "Shared",
                        "content": "shared web",
                        "source_url": "https://investor.nvidia.com/shared",
                        "published_at": now,
                        "fetched_at": now,
                        "content_hash": "shared-hash",
                    },
                ],
            )
            run = ResearchRun(
                id="run-internal",
                run_id="run-source-migration",
                ticker="NVDA",
                thesis="question",
                status="completed",
                corpus_version="NVDA-v1",
                requested_intent="company-profile",
                effective_intent="company_profile_request",
                corpus_scope=["NVDA-v1"],
                prompt_version="research-v1",
                trace_id=None,
                report_markdown="# report",
                created_at=now,
                completed_at=now,
            )
            session.add(run)
            session.flush()
            session.add(
                ClaimRecord(
                    id="claim-legacy",
                    run_id=run.id,
                    kind="verified_fact",
                    text="Legacy claim.",
                    confidence="high",
                    evidence_chunk_ids=["filing-only"],
                    source_refs=["web-only", "shared-id", "missing", "other-ticker"],
                    guard_status="retained",
                )
            )
            session.execute(
                text(
                    "INSERT INTO skill_runs "
                    "(id, ticker, recipe_name, recipe_version, recipe_snapshot, status, "
                    "source_ids, errors, started_at, completed_at) VALUES "
                    "(:id, :ticker, :recipe_name, :recipe_version, :recipe_snapshot, "
                    ":status, :source_ids, :errors, :started_at, :completed_at)"
                ),
                {
                    "id": "skill-legacy",
                    "ticker": "NVDA",
                    "recipe_name": "earnings_review",
                    "recipe_version": "1.0.0",
                    "recipe_snapshot": json.dumps({"name": "earnings_review"}),
                    "status": "completed",
                    "source_ids": json.dumps(
                        [
                            "filing-only",
                            "web-only",
                            "shared-id",
                            "missing",
                            "other-ticker",
                        ]
                    ),
                    "errors": json.dumps([]),
                    "started_at": now,
                    "completed_at": now,
                },
            )

        command.upgrade(_alembic_config(tmp_path, database_url), "head")

        with Session(engine) as session:
            claim = session.get(ClaimRecord, "claim-legacy")
            skill = session.get(SkillRun, "skill-legacy")
            assert claim is not None
            assert skill is not None
            claim_refs = claim.source_refs
            skill_refs = skill.source_ids
    finally:
        engine.dispose()

    assert claim_refs == [
        "NVDA:filing:filing-only",
        "NVDA:web:web-only",
        "NVDA:unresolved:shared-id",
        "NVDA:unresolved:missing",
        "NVDA:unresolved:other-ticker",
    ]
    assert skill_refs == [
        "NVDA:filing:filing-only",
        "NVDA:web:web-only",
        "NVDA:unresolved:shared-id",
        "NVDA:unresolved:missing",
        "NVDA:unresolved:other-ticker",
    ]
