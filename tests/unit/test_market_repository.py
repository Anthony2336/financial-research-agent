"""Exact, scope-safe market-data persistence contracts."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from fra.market_data.models import MarketBar, MarketDataBundle, MarketSnapshot
from fra.storage.database import create_schema
from fra.storage.market_repositories import MarketDataRepository
from fra.storage.models import (
    MarketBarRecord,
    MarketBundleRecord,
    MarketSnapshotRecord,
)

NOW = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    return engine


def _snapshot(**overrides: object) -> MarketSnapshot:
    values: dict[str, object] = {
        "provider": "alpaca",
        "feed": "iex",
        "coverage": "IEX-only",
        "symbol": "NVDA",
        "exchange": "IEX",
        "currency": "USD",
        "price": Decimal("123.450000000000000001"),
        "open": Decimal("122.000000000000000001"),
        "day_high": Decimal("124.000000000000000001"),
        "day_low": Decimal("121.500000000000000001"),
        "previous_close": Decimal("121.000000000000000001"),
        "as_of": NOW - timedelta(minutes=1),
        "fetched_at": NOW,
        "market_status": "open",
        "delayed_by_seconds": 60,
        "raw_payload_hash": "a" * 64,
    }
    values.update(overrides)
    values["id"] = values.get(
        "id",
        _observation_id(
            "snapshot",
            symbol=str(values["symbol"]),
            source_timestamp=_iso_z(values["as_of"]),
            fetched_at=_iso_z(values["fetched_at"]),
            raw_payload_hash=str(values["raw_payload_hash"]),
        ),
    )
    return MarketSnapshot(**values)


def _bar(**overrides: object) -> MarketBar:
    values: dict[str, object] = {
        "provider": "alpaca",
        "feed": "iex",
        "coverage": "IEX-only",
        "symbol": "NVDA",
        "exchange": "IEX",
        "currency": "USD",
        "interval": "1Day",
        "timestamp": datetime(2026, 8, 28, 4, tzinfo=UTC),
        "open": Decimal("120.000000000000000001"),
        "high": Decimal("124.000000000000000001"),
        "low": Decimal("119.000000000000000001"),
        "close": Decimal("123.450000000000000001"),
        "volume": 9_007_199_254_740_993,
        "fetched_at": NOW,
        "raw_payload_hash": "b" * 64,
    }
    values.update(overrides)
    values["id"] = values.get(
        "id",
        _observation_id(
            "bar",
            symbol=str(values["symbol"]),
            source_timestamp=_iso_z(values["timestamp"]),
            fetched_at=_iso_z(values["fetched_at"]),
            raw_payload_hash=str(values["raw_payload_hash"]),
        ),
    )
    return MarketBar(**values)


def _bundle(**overrides: object) -> MarketDataBundle:
    values: dict[str, object] = {
        "snapshot": _snapshot(),
        "bars": [_bar()],
        "status": "completed",
        "freshness_label": "open-iex",
        "errors": [],
    }
    values.update(overrides)
    return MarketDataBundle(**values)


def _iso_z(value: object) -> str:
    assert isinstance(value, datetime)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _observation_id(
    kind: str,
    *,
    symbol: str,
    source_timestamp: str,
    fetched_at: str,
    raw_payload_hash: str,
) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "provider": "alpaca",
            "feed": "iex",
            "symbol": symbol,
            "source_timestamp": source_timestamp,
            "fetched_at": fetched_at,
            "raw_payload_hash": raw_payload_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"market-{kind}:{sha256(payload).hexdigest()}"


def test_market_repository_round_trips_exact_decimal_and_provenance(sqlite_engine) -> None:
    """A float conversion or omitted provenance field would corrupt a retained observation."""
    repository = MarketDataRepository(sqlite_engine)
    expected = _bundle()

    repository.save_bundle(expected)
    stored = repository.latest_bundle("alpaca", "iex", "NVDA")

    assert stored == expected
    assert stored is not None
    assert stored.snapshot.price == Decimal("123.450000000000000001")
    assert stored.bars[0].volume == 9_007_199_254_740_993


def test_market_repository_resolves_typed_ids_without_cross_scope_fallback(sqlite_engine) -> None:
    """A source ID must resolve only to its exact market record and provider scope."""
    repository = MarketDataRepository(sqlite_engine)
    bundle = _bundle()
    repository.save_bundle(bundle)

    assert repository.get_snapshot(bundle.snapshot.id) == bundle.snapshot
    assert repository.get_bar(bundle.bars[0].id) == bundle.bars[0]
    assert repository.latest_bundle("alpaca", "iex", "AMD") is None
    assert repository.latest_bundle("alpaca", "sip", "NVDA") is None


def test_market_repository_deduplicates_same_payload_without_mutating_rows(sqlite_engine) -> None:
    """Saving the identical provider payload twice must not create or update observations."""
    repository = MarketDataRepository(sqlite_engine)
    bundle = _bundle()
    repository.save_bundle(bundle)
    repository.save_bundle(bundle)

    with Session(sqlite_engine) as session:
        snapshot_count = session.scalar(select(func.count()).select_from(MarketSnapshotRecord))
        bar_count = session.scalar(select(func.count()).select_from(MarketBarRecord))
        bundle_count = session.scalar(select(func.count()).select_from(MarketBundleRecord))

    assert snapshot_count == 1
    assert bar_count == 1
    assert bundle_count == 1


def test_market_repository_keeps_same_payload_refetch_as_distinct_observation(
    sqlite_engine,
) -> None:
    """A new fetch time for the same provider payload must remain a new immutable source row."""
    repository = MarketDataRepository(sqlite_engine)
    first = _bundle()
    refetched_at = NOW + timedelta(minutes=1)
    second = _bundle(
        snapshot=_snapshot(
            fetched_at=refetched_at,
            delayed_by_seconds=120,
        ),
        bars=[_bar(fetched_at=refetched_at)],
    )

    repository.save_bundle(first)
    repository.save_bundle(second)

    with Session(sqlite_engine) as session:
        snapshot_count = session.scalar(select(func.count()).select_from(MarketSnapshotRecord))
        bar_count = session.scalar(select(func.count()).select_from(MarketBarRecord))
        bundle_count = session.scalar(select(func.count()).select_from(MarketBundleRecord))

    assert snapshot_count == 2
    assert bar_count == 2
    assert bundle_count == 2
    assert repository.latest_bundle("alpaca", "iex", "NVDA") == second


def test_market_repository_keeps_provider_correction_at_same_source_time(sqlite_engine) -> None:
    """A corrected payload at the same market timestamp must not overwrite the original row."""
    repository = MarketDataRepository(sqlite_engine)
    first = _bundle()
    corrected = _bundle(
        snapshot=_snapshot(
            price=Decimal("123.46"),
            raw_payload_hash="c" * 64,
        ),
        bars=[
            _bar(
                close=Decimal("123.46"),
                raw_payload_hash="d" * 64,
            )
        ],
    )

    repository.save_bundle(first)
    repository.save_bundle(corrected)

    with Session(sqlite_engine) as session:
        snapshot_count = session.scalar(select(func.count()).select_from(MarketSnapshotRecord))
        bar_count = session.scalar(select(func.count()).select_from(MarketBarRecord))

    assert snapshot_count == 2
    assert bar_count == 2
    assert repository.get_snapshot(first.snapshot.id) == first.snapshot
    assert repository.get_snapshot(corrected.snapshot.id) == corrected.snapshot


def test_completed_bundle_then_snapshot_dedupe_preserves_all_original_provenance(
    sqlite_engine,
) -> None:
    """A later partial snapshot save must not rewrite a completed historical observation."""
    repository = MarketDataRepository(sqlite_engine)
    bundle = _bundle(errors=["retained-completed-note"])
    repository.save_bundle(bundle)
    before = _stored_snapshot_fields(sqlite_engine, bundle.snapshot.id)

    repository.save_snapshot(bundle.snapshot)

    assert _stored_snapshot_fields(sqlite_engine, bundle.snapshot.id) == before
    assert repository.latest_bundle("alpaca", "iex", "NVDA") == bundle


def test_snapshot_then_completed_bundle_preserves_snapshot_and_returns_exact_bundle(
    sqlite_engine,
) -> None:
    """Reverse save order must retain immutable snapshot provenance and exact bundle state."""
    repository = MarketDataRepository(sqlite_engine)
    bundle = _bundle(errors=["retained-completed-note"])
    repository.save_snapshot(bundle.snapshot)
    before = _stored_snapshot_fields(sqlite_engine, bundle.snapshot.id)

    repository.save_bundle(bundle)

    assert _stored_snapshot_fields(sqlite_engine, bundle.snapshot.id) == before
    assert repository.latest_bundle("alpaca", "iex", "NVDA") == bundle


def test_market_repository_rejects_changed_payload_reusing_immutable_id(sqlite_engine) -> None:
    """Reusing a source ID for different bytes must not overwrite cited market evidence."""
    repository = MarketDataRepository(sqlite_engine)
    original = _bundle()
    repository.save_bundle(original)
    changed_snapshot = _snapshot(
        id=original.snapshot.id,
        raw_payload_hash="c" * 64,
        price=Decimal("123.46"),
    )

    with pytest.raises(ValueError, match="immutable market snapshot id"):
        repository.save_snapshot(changed_snapshot)

    assert repository.get_snapshot(changed_snapshot.id) == original.snapshot


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("price", Decimal("123.46")),
        ("open", Decimal("122.10")),
        ("day_high", Decimal("124.10")),
        ("day_low", Decimal("121.40")),
        ("previous_close", Decimal("121.10")),
    ],
)
def test_market_repository_rejects_snapshot_fact_drift_reusing_immutable_id(
    sqlite_engine,
    field_name: str,
    field_value: Decimal,
) -> None:
    """A stale snapshot ID must compare every persisted fact field, not only provenance."""
    repository = MarketDataRepository(sqlite_engine)
    original = _bundle()
    repository.save_bundle(original)
    changed_snapshot = _snapshot(
        id=original.snapshot.id,
        raw_payload_hash=original.snapshot.raw_payload_hash,
        **{field_name: field_value},
    )

    with pytest.raises(ValueError, match="immutable market snapshot id"):
        repository.save_snapshot(changed_snapshot)

    assert repository.get_snapshot(changed_snapshot.id) == original.snapshot


def test_market_repository_rejects_changed_bar_reusing_immutable_id(sqlite_engine) -> None:
    """An existing bar source ID must retain every original value and provenance field."""
    repository = MarketDataRepository(sqlite_engine)
    original = _bundle()
    repository.save_bundle(original)
    changed_bar = _bar(
        id=original.bars[0].id,
        raw_payload_hash="c" * 64,
        close=Decimal("122.00"),
    )

    with pytest.raises(ValueError, match="immutable market bar id"):
        repository.save_bars([changed_bar])

    assert repository.get_bar(changed_bar.id) == original.bars[0]


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("open", Decimal("120.10")),
        ("high", Decimal("124.10")),
        ("low", Decimal("118.90")),
        ("close", Decimal("123.40")),
        ("volume", 9_007_199_254_740_992),
    ],
)
def test_market_repository_rejects_bar_fact_drift_reusing_immutable_id(
    sqlite_engine,
    field_name: str,
    field_value: Decimal | int,
) -> None:
    """A stale bar ID must compare every persisted OHLCV field, not only provenance."""
    repository = MarketDataRepository(sqlite_engine)
    original = _bundle()
    repository.save_bundle(original)
    changed_bar = _bar(
        id=original.bars[0].id,
        raw_payload_hash=original.bars[0].raw_payload_hash,
        **{field_name: field_value},
    )

    with pytest.raises(ValueError, match="immutable market bar id"):
        repository.save_bars([changed_bar])

    assert repository.get_bar(changed_bar.id) == original.bars[0]


def test_latest_bundle_uses_only_latest_bounded_bar_fetch(sqlite_engine) -> None:
    """Historical bar batches must not accumulate into a mixed, unbounded latest response."""
    repository = MarketDataRepository(sqlite_engine)
    repository.save_bundle(_bundle())
    later = NOW + timedelta(minutes=5)
    latest_snapshot = _snapshot(
        as_of=later - timedelta(seconds=60),
        fetched_at=later,
        raw_payload_hash="c" * 64,
    )
    latest_bar = _bar(
        timestamp=datetime(2026, 8, 29, 4, tzinfo=UTC),
        fetched_at=later,
        raw_payload_hash="d" * 64,
    )
    repository.save_bundle(_bundle(snapshot=latest_snapshot, bars=[latest_bar]))

    stored = repository.latest_bundle("alpaca", "iex", "NVDA")

    assert stored is not None
    assert stored.snapshot == latest_snapshot
    assert stored.bars == [latest_bar]


def test_latest_bundle_orders_by_market_observation_not_backfill_write_time(
    sqlite_engine,
) -> None:
    """Backfilling an older bundle later must not displace newer fetched evidence."""
    repository = MarketDataRepository(sqlite_engine)
    newer_time = NOW + timedelta(minutes=5)
    newer = _bundle(
        snapshot=_snapshot(
            as_of=newer_time - timedelta(seconds=60),
            fetched_at=newer_time,
            raw_payload_hash="c" * 64,
        ),
        bars=[
            _bar(
                timestamp=datetime(2026, 8, 29, 4, tzinfo=UTC),
                fetched_at=newer_time,
                raw_payload_hash="d" * 64,
            )
        ],
    )

    repository.save_bundle(newer)
    repository.save_bundle(_bundle())

    assert repository.latest_bundle("alpaca", "iex", "NVDA") == newer


def test_latest_bundle_uses_stable_identity_tie_breaker_for_equal_fetch_time(
    sqlite_engine,
) -> None:
    """Equal observation times must produce the same latest result regardless of save order."""
    repository = MarketDataRepository(sqlite_engine)
    tied_time = NOW + timedelta(minutes=10)
    first = _bundle(
        snapshot=_snapshot(
            as_of=tied_time - timedelta(seconds=60),
            fetched_at=tied_time,
            raw_payload_hash="e" * 64,
        ),
        bars=[_bar(fetched_at=tied_time, raw_payload_hash="f" * 64)],
    )
    second = _bundle(
        snapshot=_snapshot(
            as_of=tied_time - timedelta(seconds=60),
            fetched_at=tied_time,
            raw_payload_hash="1" * 64,
        ),
        bars=[_bar(fetched_at=tied_time, raw_payload_hash="2" * 64)],
    )
    higher_identity = max([first, second], key=lambda bundle: bundle.snapshot.id)
    lower_identity = min([first, second], key=lambda bundle: bundle.snapshot.id)

    repository.save_bundle(higher_identity)
    repository.save_bundle(lower_identity)

    assert repository.latest_bundle("alpaca", "iex", "NVDA") == higher_identity


def test_latest_bundle_observation_order_remains_exact_scope(sqlite_engine) -> None:
    """A newer write for another symbol must not affect provider/feed/symbol selection."""
    repository = MarketDataRepository(sqlite_engine)
    nvda = _bundle()
    later = NOW + timedelta(minutes=15)
    amd = _bundle(
        snapshot=_snapshot(
            symbol="AMD",
            as_of=later - timedelta(seconds=60),
            fetched_at=later,
            raw_payload_hash="3" * 64,
        ),
        bars=[
            _bar(
                symbol="AMD",
                fetched_at=later,
                raw_payload_hash="4" * 64,
            )
        ],
    )

    repository.save_bundle(nvda)
    repository.save_bundle(amd)

    assert repository.latest_bundle("alpaca", "iex", "NVDA") == nvda
    assert repository.latest_bundle("alpaca", "iex", "AMD") == amd
    assert repository.latest_bundle("other", "iex", "NVDA") is None
    assert repository.latest_bundle("alpaca", "other", "NVDA") is None


def _stored_snapshot_fields(engine, snapshot_id: str) -> dict[str, object]:
    with Session(engine) as session:
        record = session.get(MarketSnapshotRecord, snapshot_id)
        assert record is not None
        return {
            column.name: getattr(record, column.name)
            for column in MarketSnapshotRecord.__table__.columns
        }
