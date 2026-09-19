"""Fixture ingestion against a running PostgreSQL/pgvector service."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from fra.config import Settings
from fra.retrieval.ingest import ingest_fixture
from fra.storage.database import create_schema
from fra.storage.repositories import FilingRepository


@pytest.fixture
def postgres_engine():
    """Connect only when explicitly requested, leaving shared database rows intact."""
    if os.environ.get("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("set RUN_POSTGRES_INTEGRATION=1 to run Postgres integration tests")

    engine = create_engine(Settings().database_url, connect_args={"connect_timeout": 2})
    try:
        with engine.connect():
            pass
    except OperationalError:
        engine.dispose()
        pytest.skip("Postgres is unavailable")
    create_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def postgres_repository(postgres_engine) -> FilingRepository:
    return FilingRepository(postgres_engine)


def test_postgres_fixture_ingest_uses_pgvector_schema(
    postgres_engine,
    postgres_repository: FilingRepository,
) -> None:
    """The real pgvector schema must accept an idempotent fixture ingest."""
    ticker = f"PG{uuid4().hex[:8]}"
    fixture_path = Path("tests/fixtures/nvda_10q.html")

    first_version = ingest_fixture(fixture_path, ticker, "10-Q", postgres_repository)
    chunks = postgres_repository.list_chunks(ticker, first_version)
    second_version = ingest_fixture(fixture_path, ticker, "10-Q", postgres_repository)

    with postgres_engine.connect() as connection:
        embedding_type = connection.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute AS a "
                "JOIN pg_class AS c ON c.oid = a.attrelid "
                "WHERE c.relname = 'chunks' AND a.attname = 'embedding'"
            )
        ).scalar_one()

    assert embedding_type == "vector(1024)"
    assert first_version == second_version == f"{ticker.upper()}-v1"
    assert chunks
    assert all(chunk.accession_no and chunk.source_url for chunk in chunks)
