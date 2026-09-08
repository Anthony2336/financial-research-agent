"""Typed JSON cache contracts for normalized market data."""

from datetime import UTC, datetime, timedelta

import pytest

from financial_evidence_agent.market_data.gateway import MarketDataGateway
from financial_evidence_agent.market_data.models import MarketBar, MarketSnapshot
from financial_evidence_agent.storage.cache import MarketDataJsonCache

NOW = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)


def _snapshot(**overrides: object) -> MarketSnapshot:
    values: dict[str, object] = {
        "id": "snapshot",
        "provider": "alpaca",
        "feed": "iex",
        "coverage": "IEX-only",
        "symbol": "NVDA",
        "exchange": "IEX",
        "currency": "USD",
        "price": "123.450000000000000001",
        "open": "122.00",
        "day_high": "124.00",
        "day_low": "121.50",
        "previous_close": "121.00",
        "as_of": NOW - timedelta(seconds=60),
        "fetched_at": NOW,
        "market_status": "open",
        "delayed_by_seconds": 60,
        "raw_payload_hash": "a" * 64,
    }
    values.update(overrides)
    return MarketSnapshot(**values)


def _bar(**overrides: object) -> MarketBar:
    values: dict[str, object] = {
        "id": "bar",
        "provider": "alpaca",
        "feed": "iex",
        "coverage": "IEX-only",
        "symbol": "NVDA",
        "exchange": "IEX",
        "currency": "USD",
        "interval": "1Day",
        "timestamp": datetime(2026, 8, 28, 4, tzinfo=UTC),
        "open": "120.00",
        "high": "124.00",
        "low": "119.00",
        "close": "123.45",
        "volume": 1000,
        "fetched_at": NOW,
        "raw_payload_hash": "b" * 64,
    }
    values.update(overrides)
    return MarketBar(**values)


class MemoryJsonCache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.ttls: dict[str, int] = {}
        self.deleted: list[str] = []

    async def get_json(self, key: str):
        return self.values.get(key)

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.values[key] = value
        self.ttls[key] = ttl_seconds

    async def delete_json(self, key: str) -> None:
        self.deleted.append(key)
        self.values.pop(key, None)


class RecordingProvider:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.bar_calls = 0

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        self.snapshot_calls += 1
        return _snapshot(symbol=symbol)

    async def get_bars(self, symbol: str, *, interval: str, limit: int) -> list[MarketBar]:
        self.bar_calls += 1
        return [_bar(symbol=symbol)]


@pytest.mark.asyncio
async def test_market_cache_uses_fixed_scope_keys_and_preserves_exact_envelopes() -> None:
    """Changing a key segment or dropping timestamp/hash metadata would break safe reuse."""
    raw = MemoryJsonCache()
    cache = MarketDataJsonCache(
        raw,
        max_bars=5,
        max_staleness_seconds=90,
        ttl_seconds=120,
        clock=lambda: NOW,
    )

    await cache.set_snapshot(_snapshot())
    await cache.set_bars([_bar()])

    assert set(raw.values) == {
        "market:alpaca:iex:NVDA:snapshot",
        "market:alpaca:iex:NVDA:bars:1Day:5",
    }
    assert await cache.get_snapshot("alpaca", "iex", "NVDA") == _snapshot()
    assert await cache.get_bars(
        "alpaca", "iex", "NVDA", interval="1Day", limit=5
    ) == [_bar()]
    assert set(raw.ttls.values()) == {120}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        {},
        {"price": "not-a-snapshot"},
        {"snapshot": None},
    ],
)
async def test_market_cache_treats_malformed_snapshot_as_invalidated_miss(value: object) -> None:
    """Malformed Redis data must never escape as normalized market evidence."""
    raw = MemoryJsonCache()
    key = "market:alpaca:iex:NVDA:snapshot"
    raw.values[key] = value
    cache = MarketDataJsonCache(raw, clock=lambda: NOW)

    assert await cache.get_snapshot("alpaca", "iex", "NVDA") is None
    assert raw.deleted == [key]


@pytest.mark.asyncio
async def test_market_cache_invalidates_stale_open_snapshot_and_empty_bars() -> None:
    """Expired or empty entries are misses, never current market data."""
    raw = MemoryJsonCache()
    snapshot_key = "market:alpaca:iex:NVDA:snapshot"
    bars_key = "market:alpaca:iex:NVDA:bars:1Day:5"
    raw.values[snapshot_key] = {
        "snapshot": _snapshot(
            as_of=NOW - timedelta(seconds=91), delayed_by_seconds=91
        ).model_dump(mode="json")
    }
    raw.values[bars_key] = {"bars": []}
    cache = MarketDataJsonCache(
        raw,
        max_bars=5,
        max_staleness_seconds=90,
        clock=lambda: NOW,
    )

    assert await cache.get_snapshot("alpaca", "iex", "NVDA") is None
    assert await cache.get_bars(
        "alpaca", "iex", "NVDA", interval="1Day", limit=5
    ) is None
    assert raw.deleted == [snapshot_key, bars_key]


@pytest.mark.asyncio
async def test_new_gateway_reuses_concrete_cache_without_provider_calls() -> None:
    """A second run gateway should reuse valid Redis data without inheriting run memo state."""
    raw = MemoryJsonCache()
    cache = MarketDataJsonCache(
        raw,
        max_bars=5,
        max_staleness_seconds=90,
        clock=lambda: NOW,
    )
    provider = RecordingProvider()

    await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch("NVDA")
    await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch("NVDA")

    assert provider.snapshot_calls == 1
    assert provider.bar_calls == 1
