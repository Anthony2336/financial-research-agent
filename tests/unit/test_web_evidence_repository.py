"""Persistence contracts for versioned web evidence snapshots."""

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fra.domain import SourceKind, SourceTier, WebEvidence
from fra.storage import web_repositories
from fra.storage.database import create_schema
from fra.storage.models import WebEvidenceRecord
from fra.storage.web_repositories import WebEvidenceRepository


@pytest.fixture
def engine():
    database_engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(database_engine)
    return database_engine


@pytest.fixture
def repository(engine) -> WebEvidenceRepository:
    return WebEvidenceRepository(engine)


def _evidence(
    *,
    evidence_id: str,
    content: str,
    content_hash: str,
    ticker: str = "nvda",
) -> WebEvidence:
    return WebEvidence(
        id=evidence_id,
        ticker=ticker,
        title="Quarterly results",
        content=content,
        source_url="https://INVESTOR.NVIDIA.COM/results",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 28, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 29, 9, 30, tzinfo=UTC),
        content_hash=content_hash,
    )


def test_upsert_is_idempotent_for_normalized_url_and_content_hash(
    engine,
    repository: WebEvidenceRepository,
) -> None:
    """Trusting caller IDs would create duplicate rows when one snapshot is retried."""
    first = repository.upsert(
        _evidence(evidence_id="caller-id-1", content="Revenue rose.", content_hash="sha256:a")
    )
    second = repository.upsert(
        _evidence(evidence_id="caller-id-2", content="Revenue rose.", content_hash="sha256:a")
    )

    expected_id = sha256(
        b"NVDA\nhttps://investor.nvidia.com/results\nsha256:a"
    ).hexdigest()
    with Session(engine) as session:
        row_count = session.scalar(select(func.count()).select_from(WebEvidenceRecord))
        record = session.get(WebEvidenceRecord, first.id)

    assert record is not None
    assert record.time_metadata_validated is True
    assert first.id == second.id == expected_id
    assert first.ticker == second.ticker == "NVDA"
    assert row_count == 1


@pytest.mark.parametrize(
    "published_at",
    [None, datetime(2026, 5, 28), datetime(2026, 5, 30, tzinfo=UTC)],
)
def test_upsert_rejects_invalid_time_metadata_without_writing(
    engine,
    repository: WebEvidenceRepository,
    published_at: datetime | None,
) -> None:
    evidence = _evidence(
        evidence_id="invalid-time",
        content="Invalid time metadata.",
        content_hash="sha256:invalid-time",
    ).model_copy(update={"published_at": published_at})

    with pytest.raises(ValueError, match="timestamp"):
        repository.upsert(evidence)

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


def test_legacy_unvalidated_row_is_excluded_and_valid_collision_cannot_rewrite_it(
    engine,
    repository: WebEvidenceRepository,
) -> None:
    current = _evidence(
        evidence_id="current",
        content="Immutable collision.",
        content_hash="sha256:collision",
    )
    source_url = "https://investor.nvidia.com/results"
    evidence_id = sha256(f"NVDA\n{source_url}\nsha256:collision".encode()).hexdigest()
    with Session(engine) as session, session.begin():
        session.add(
            WebEvidenceRecord(
                id=evidence_id,
                ticker="NVDA",
                title="Legacy",
                content=current.content,
                source_url=source_url,
                source_kind=SourceKind.ISSUER_IR.value,
                source_tier=SourceTier.PRIMARY.value,
                published_at=datetime(2026, 5, 28),
                fetched_at=datetime(2026, 5, 29, 9, 30),
                content_hash=current.content_hash,
                time_metadata_validated=False,
            )
        )

    assert repository.get_many([evidence_id]) == []
    assert repository.upsert(current) is None
    with Session(engine) as session:
        row = session.get(WebEvidenceRecord, evidence_id)
        assert row is not None
        assert row.title == "Legacy"
        assert row.time_metadata_validated is False


def test_create_schema_web_time_marker_is_nonnullable_and_defaults_false(engine) -> None:
    columns = {column["name"]: column for column in inspect(engine).get_columns("web_evidence")}

    assert columns["time_metadata_validated"]["nullable"] is False
    assert columns["time_metadata_validated"]["default"] is not None


def test_changed_content_hash_creates_a_distinct_evidence_row(
    engine,
    repository: WebEvidenceRepository,
) -> None:
    """Reusing a URL-only ID would overwrite a previously cited evidence snapshot."""
    first = repository.upsert(
        _evidence(evidence_id="ignored", content="Revenue rose.", content_hash="sha256:a")
    )
    second = repository.upsert(
        _evidence(evidence_id="ignored", content="Revenue fell.", content_hash="sha256:b")
    )

    with Session(engine) as session:
        row_count = session.scalar(select(func.count()).select_from(WebEvidenceRecord))

    assert first.id != second.id
    assert first.content == "Revenue rose."
    assert second.content == "Revenue fell."
    assert row_count == 2


def test_same_url_and_content_are_distinct_for_each_ticker(
    engine,
    repository: WebEvidenceRepository,
) -> None:
    """URL/hash-only identity would return another issuer's binding provenance."""
    nvda = repository.upsert(
        _evidence(
            evidence_id="ignored-nvda",
            ticker="NVDA",
            content="Sector update.",
            content_hash="sha256:shared",
        )
    )
    amd = repository.upsert(
        _evidence(
            evidence_id="ignored-amd",
            ticker="AMD",
            content="Sector update.",
            content_hash="sha256:shared",
        )
    )

    with Session(engine) as session:
        row_count = session.scalar(select(func.count()).select_from(WebEvidenceRecord))

    assert nvda.id != amd.id
    assert nvda.ticker == "NVDA"
    assert amd.ticker == "AMD"
    assert row_count == 2


def test_get_many_preserves_request_order_and_omits_unknown_ids(
    repository: WebEvidenceRepository,
) -> None:
    """Database row order or placeholders would misalign citations with requested IDs."""
    first = repository.upsert(
        _evidence(evidence_id="ignored-1", content="First.", content_hash="sha256:first")
    )
    second = repository.upsert(
        _evidence(evidence_id="ignored-2", content="Second.", content_hash="sha256:second")
    )

    found = repository.get_many([second.id, "unknown", first.id])

    assert [evidence.id for evidence in found] == [second.id, first.id]
    assert [evidence.content for evidence in found] == ["Second.", "First."]


def test_web_evidence_enforces_ticker_url_and_content_hash_uniqueness(engine) -> None:
    """Repository checks alone would not prevent concurrent retries from duplicating a snapshot."""
    constraints = inspect(engine).get_unique_constraints("web_evidence")

    assert any(
        set(constraint["column_names"]) == {"ticker", "source_url", "content_hash"}
        for constraint in constraints
    )


class _StaleReadSession(Session):
    """Use a real session while hiding its first scalar result."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hide_next_scalar = True

    def scalar(self, statement, *args, **kwargs):
        if self._hide_next_scalar:
            self._hide_next_scalar = False
            return None
        return super().scalar(statement, *args, **kwargs)


def test_upsert_recovers_when_a_competing_insert_wins_the_unique_constraint(
    engine,
    repository: WebEvidenceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale pre-insert lookup must not make a concurrent retry fail."""
    evidence = _evidence(
        evidence_id="first-attempt",
        content="Revenue rose.",
        content_hash="sha256:concurrent",
    )
    existing = repository.upsert(evidence)
    sessions_created = 0

    def session_factory(bind):
        nonlocal sessions_created
        sessions_created += 1
        if sessions_created == 1:
            return _StaleReadSession(bind)
        return Session(bind)

    monkeypatch.setattr(web_repositories, "Session", session_factory)

    recovered = repository.upsert(evidence.model_copy(update={"id": "retry-attempt"}))

    with Session(engine) as session:
        row_count = session.scalar(select(func.count()).select_from(WebEvidenceRecord))
    assert recovered == existing
    assert row_count == 1


def test_upsert_does_not_swallow_an_unrelated_integrity_error(
    engine,
    repository: WebEvidenceRepository,
) -> None:
    """Conflict recovery must not hide a primary-key collision for different source content."""
    evidence = _evidence(
        evidence_id="ignored",
        content="Revenue rose.",
        content_hash="sha256:unrelated",
    )
    persisted = repository.upsert(evidence)
    with Session(engine) as session, session.begin():
        conflicting_row = session.get(WebEvidenceRecord, persisted.id)
        assert conflicting_row is not None
        conflicting_row.source_url = "https://example.com/different"
        conflicting_row.content_hash = "sha256:different"

    with pytest.raises(IntegrityError):
        repository.upsert(evidence)
