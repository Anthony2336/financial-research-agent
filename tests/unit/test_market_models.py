"""Contract tests for normalized Alpaca IEX market data."""

import json
from datetime import datetime
from decimal import Decimal
from hashlib import sha256

import pytest
from pydantic import ValidationError

from fra.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
)


def _iso_z(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat().replace(
        "+00:00", "Z"
    )


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
            "source_timestamp": _iso_z(source_timestamp),
            "fetched_at": _iso_z(fetched_at),
            "raw_payload_hash": raw_payload_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"market-{kind}:{sha256(payload).hexdigest()}"


def _snapshot(**overrides: object) -> MarketSnapshot:
    values: dict[str, object] = {
        "provider": "alpaca",
        "feed": "iex",
        "coverage": "IEX-only",
        "symbol": "NVDA",
        "exchange": "NASDAQ",
        "currency": "USD",
        "price": "123.45",
        "open": "122.00",
        "day_high": "124.00",
        "day_low": "121.50",
        "previous_close": "121.00",
        "as_of": "2026-08-30T14:00:00Z",
        "fetched_at": "2026-08-30T14:00:01Z",
        "market_status": "open",
        "delayed_by_seconds": 0,
        "raw_payload_hash": "a" * 64,
    }
    values.update(overrides)
    values["id"] = values.get(
        "id",
        _observation_id(
            "snapshot",
            symbol=str(values["symbol"]),
            source_timestamp=str(values["as_of"]),
            fetched_at=str(values["fetched_at"]),
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
        "exchange": "NASDAQ",
        "currency": "USD",
        "interval": "1Day",
        "timestamp": "2026-08-29T00:00:00Z",
        "open": "120.00",
        "high": "124.00",
        "low": "119.00",
        "close": "123.45",
        "volume": 1_000,
        "fetched_at": "2026-08-30T14:00:01Z",
        "raw_payload_hash": "b" * 64,
    }
    values.update(overrides)
    values["id"] = values.get(
        "id",
        _observation_id(
            "bar",
            symbol=str(values["symbol"]),
            source_timestamp=str(values["timestamp"]),
            fetched_at=str(values["fetched_at"]),
            raw_payload_hash=str(values["raw_payload_hash"]),
        ),
    )
    return MarketBar(**values)


def test_market_snapshot_preserves_decimal_and_iex_scope() -> None:
    snapshot = _snapshot()

    assert snapshot.price == Decimal("123.45")
    assert snapshot.coverage == "IEX-only"


def test_market_observation_ids_are_opaque_digests_of_observation_time_and_payload() -> None:
    snapshot = _snapshot()
    refetched_snapshot = _snapshot(
        fetched_at="2026-08-30T14:00:02Z",
        delayed_by_seconds=1,
    )
    corrected_snapshot = _snapshot(
        price="123.46",
        raw_payload_hash="c" * 64,
    )
    bar = _bar()
    corrected_bar = _bar(close="123.46", raw_payload_hash="d" * 64)

    assert snapshot.id == _observation_id(
        "snapshot",
        symbol="NVDA",
        source_timestamp="2026-08-30T14:00:00Z",
        fetched_at="2026-08-30T14:00:01Z",
        raw_payload_hash="a" * 64,
    )
    assert snapshot.id != refetched_snapshot.id
    assert snapshot.id != corrected_snapshot.id
    assert bar.id != corrected_bar.id
    assert snapshot.id.startswith("market-snapshot:")
    assert bar.id.startswith("market-bar:")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "nvda"),
        ("provider", "other"),
        ("feed", "sip"),
        ("coverage", "US consolidated"),
        ("price", "NaN"),
        ("price", "Infinity"),
        ("day_low", "124.01"),
        ("raw_payload_hash", "not-a-sha256"),
        ("as_of", "2026-08-30T14:00:02Z"),
        ("fetched_at", "2026-08-30T14:00:01+01:00"),
    ],
)
def test_snapshot_rejects_invalid_normalized_market_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _snapshot(**{field: value})


def test_market_models_reject_undocumented_fields() -> None:
    with pytest.raises(ValidationError):
        _snapshot(untrusted_provider_payload={"token": "secret"})


def test_market_models_reject_high_below_open_or_close_values() -> None:
    with pytest.raises(ValidationError, match="day_high"):
        _snapshot(day_low="120.00", day_high="121.00")

    with pytest.raises(ValidationError, match="high"):
        _bar(low="118.00", high="119.00")


def test_market_data_bundle_requires_unique_chronological_bars() -> None:
    newer = _bar(
        id="alpaca:iex:NVDA:1Day:2026-08-30T00:00:00Z",
        timestamp="2026-08-30T00:00:00Z",
    )
    older = _bar()

    with pytest.raises(ValidationError, match="chronological"):
        MarketDataBundle(snapshot=_snapshot(), bars=[newer, older], status="completed")

    with pytest.raises(ValidationError, match="unique"):
        MarketDataBundle(snapshot=_snapshot(), bars=[older, older], status="completed")


def test_market_data_bundle_rejects_bar_scope_mismatch() -> None:
    with pytest.raises(ValidationError, match="scope"):
        MarketDataBundle(
            snapshot=_snapshot(),
            bars=[_bar(symbol="MSFT")],
            status="completed",
        )


def test_market_data_error_exposes_only_stable_code_and_safe_message() -> None:
    error = MarketDataError(MarketDataErrorCode.RATE_LIMITED, "provider request rate limited")

    assert error.code is MarketDataErrorCode.RATE_LIMITED
    assert error.detail == "provider request rate limited"
    assert str(error) == "MARKET_DATA_RATE_LIMITED: provider request rate limited"


def test_market_timestamps_are_utc_datetimes() -> None:
    snapshot = _snapshot()

    assert snapshot.as_of == datetime(2026, 8, 30, 14, 0, tzinfo=snapshot.as_of.tzinfo)
    assert snapshot.as_of.utcoffset() is not None
    assert snapshot.as_of.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "value",
    [True, "1000", 1.5, Decimal("1.5"), -1, 9_223_372_036_854_775_808],
)
def test_market_bar_volume_requires_nonnegative_integral_numeric_values(
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="volume"):
        _bar(volume=value)
