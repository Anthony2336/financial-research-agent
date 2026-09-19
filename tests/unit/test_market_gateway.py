"""Consistency, freshness, and operation-budget tests for the market gateway."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from fra.market_data.gateway import MarketDataGateway
from fra.market_data.models import (
    MarketBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
)

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
        "price": "123.45",
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


class FakeProvider:
    def __init__(self, snapshot: MarketSnapshot, bars: list[MarketBar]) -> None:
        self.snapshot = snapshot
        self.bars = bars
        self.snapshot_calls = 0
        self.bars_calls = 0
        self.snapshot_error: MarketDataError | None = None
        self.bars_error: MarketDataError | None = None

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        self.snapshot_calls += 1
        assert symbol == "NVDA"
        await asyncio.sleep(0)
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot

    async def get_bars(
        self, symbol: str, *, interval: str, limit: int
    ) -> list[MarketBar]:
        self.bars_calls += 1
        assert (symbol, interval, limit) == ("NVDA", "1Day", 5)
        await asyncio.sleep(0)
        if self.bars_error is not None:
            raise self.bars_error
        return self.bars


class FakeCache:
    def __init__(
        self,
        *,
        snapshot: MarketSnapshot | None = None,
        bars: list[MarketBar] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.bars = bars
        self.saved_snapshot: MarketSnapshot | None = None
        self.saved_bars: list[MarketBar] | None = None
        self.invalidated_snapshots = 0
        self.invalidated_bars = 0

    async def get_snapshot(self, provider: str, feed: str, symbol: str) -> MarketSnapshot | None:
        return self.snapshot

    async def get_bars(
        self, provider: str, feed: str, symbol: str, *, interval: str, limit: int
    ) -> list[MarketBar] | None:
        return self.bars

    async def set_snapshot(self, snapshot: MarketSnapshot) -> None:
        self.saved_snapshot = snapshot
        self.snapshot = snapshot

    async def set_bars(self, bars: list[MarketBar]) -> None:
        self.saved_bars = bars
        self.bars = bars

    async def invalidate_snapshot(self, provider: str, feed: str, symbol: str) -> None:
        self.invalidated_snapshots += 1
        self.snapshot = None

    async def invalidate_bars(
        self, provider: str, feed: str, symbol: str, *, interval: str
    ) -> None:
        self.invalidated_bars += 1
        self.bars = None


@pytest.mark.asyncio
async def test_gateway_fetches_snapshot_and_bars_once_and_labels_open_data() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])

    bundle = await MarketDataGateway(provider, clock=lambda: NOW).fetch("nvda")

    assert bundle.status == "completed"
    assert bundle.freshness_label == "open-iex"
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_rejects_open_market_stale_quote() -> None:
    provider = FakeProvider(
        _snapshot(as_of=NOW - timedelta(seconds=91), delayed_by_seconds=91),
        [_bar()],
    )

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(
            provider, max_staleness_seconds=90, clock=lambda: NOW
        ).fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.STALE
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 0


@pytest.mark.asyncio
async def test_gateway_fetch_snapshot_rejects_new_stale_open_quote_before_cache() -> None:
    """Snapshot-only callers must not persist stale data before bundle validation runs."""
    provider = FakeProvider(
        _snapshot(as_of=NOW - timedelta(seconds=91), delayed_by_seconds=91),
        [_bar()],
    )
    cache = FakeCache()

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(
            provider,
            cache=cache,
            max_staleness_seconds=90,
            clock=lambda: NOW,
        ).fetch_snapshot("NVDA")

    assert raised.value.code is MarketDataErrorCode.STALE
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 0
    assert cache.saved_snapshot is None


@pytest.mark.asyncio
async def test_gateway_closed_market_uses_latest_bar_without_realtime_claim() -> None:
    timestamp = datetime(2026, 8, 28, 20, tzinfo=UTC)
    provider = FakeProvider(
        _snapshot(as_of=timestamp, fetched_at=NOW, market_status="closed"),
        [_bar(timestamp=timestamp)],
    )

    bundle = await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert bundle.snapshot.market_status.value == "closed"
    assert bundle.freshness_label == "latest-available-iex"
    assert "real-time" not in bundle.freshness_label


@pytest.mark.asyncio
async def test_gateway_rejects_closed_quote_older_than_latest_returned_bar() -> None:
    provider = FakeProvider(
        _snapshot(
            as_of=datetime(2026, 8, 27, 20, tzinfo=UTC),
            fetched_at=NOW,
            market_status="closed",
        ),
        [_bar(timestamp=datetime(2026, 8, 28, 4, tzinfo=UTC))],
    )

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.STALE


@pytest.mark.asyncio
async def test_gateway_labels_unknown_clock_without_realtime_claim() -> None:
    provider = FakeProvider(_snapshot(market_status="unknown"), [_bar()])

    bundle = await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert bundle.freshness_label == "market-status-unknown"
    assert "real-time" not in bundle.freshness_label


@pytest.mark.asyncio
async def test_gateway_returns_partial_snapshot_for_typed_bars_failure() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    provider.bars_error = MarketDataError(
        MarketDataErrorCode.RATE_LIMITED, "market provider rate limited"
    )

    bundle = await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert bundle.status == "partial"
    assert bundle.bars == []
    assert bundle.errors == [MarketDataErrorCode.RATE_LIMITED.value]
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_does_not_degrade_bars_auth_failure_to_partial() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    provider.bars_error = MarketDataError(
        MarketDataErrorCode.CONFIGURATION_MISSING,
        "market provider authentication failed",
    )

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.CONFIGURATION_MISSING
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_raises_typed_scope_mismatch() -> None:
    provider = FakeProvider(_snapshot(), [_bar(symbol="MSFT")])

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.SCOPE_MISMATCH


@pytest.mark.asyncio
async def test_gateway_revalidates_stale_cache_before_falling_back_to_provider() -> None:
    fresh = _snapshot()
    stale = _snapshot(
        id="stale", as_of=NOW - timedelta(seconds=91), delayed_by_seconds=91
    )
    provider = FakeProvider(fresh, [_bar()])
    cache = FakeCache(snapshot=stale, bars=[_bar()])

    snapshot = await MarketDataGateway(
        provider, cache=cache, max_staleness_seconds=90, clock=lambda: NOW
    ).fetch_snapshot("NVDA")

    assert snapshot is fresh
    assert provider.snapshot_calls == 1
    assert cache.saved_snapshot is fresh


@pytest.mark.asyncio
async def test_gateway_uses_scope_valid_cache_without_provider_calls() -> None:
    snapshot = _snapshot()
    bars = [_bar()]
    provider = FakeProvider(snapshot, bars)
    cache = FakeCache(snapshot=snapshot, bars=bars)

    bundle = await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch("NVDA")

    assert bundle.status == "completed"
    assert provider.snapshot_calls == 0
    assert provider.bars_calls == 0


@pytest.mark.asyncio
async def test_gateway_memoizes_snapshot_before_fetch_for_one_run() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    await gateway.fetch_snapshot("NVDA")
    await gateway.fetch("NVDA")

    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_memoizes_bars_before_fetch_for_one_run() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    await gateway.fetch_bars("NVDA")
    await gateway.fetch("NVDA")

    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_repeated_fetch_uses_each_provider_operation_once() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    first = await gateway.fetch("NVDA")
    second = await gateway.fetch("NVDA")

    assert second == first
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_replays_bars_failure_without_another_provider_operation() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    provider.bars_error = MarketDataError(
        MarketDataErrorCode.UNAVAILABLE, "market provider unavailable"
    )
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    with pytest.raises(MarketDataError) as raised:
        await gateway.fetch_bars("NVDA")
    bundle = await gateway.fetch("NVDA")
    again = await gateway.fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.UNAVAILABLE
    assert bundle.errors == again.errors == [MarketDataErrorCode.UNAVAILABLE.value]
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_replays_snapshot_failure_without_another_provider_operation() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    provider.snapshot_error = MarketDataError(
        MarketDataErrorCode.UNAVAILABLE, "market provider unavailable"
    )
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    for _ in range(2):
        with pytest.raises(MarketDataError) as raised:
            await gateway.fetch_snapshot("NVDA")
        assert raised.value.code is MarketDataErrorCode.UNAVAILABLE

    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 0


@pytest.mark.asyncio
async def test_application_composition_needs_fresh_gateway_per_run() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])

    first_run = MarketDataGateway(provider, clock=lambda: NOW)
    await first_run.fetch("NVDA")
    await first_run.fetch("NVDA")
    second_run = MarketDataGateway(provider, clock=lambda: NOW)
    await second_run.fetch("NVDA")

    assert provider.snapshot_calls == 2
    assert provider.bars_calls == 2


@pytest.mark.asyncio
async def test_gateway_rejects_invalid_cached_scope_and_fetches_provider() -> None:
    valid_snapshot = _snapshot()
    valid_bars = [_bar()]
    provider = FakeProvider(valid_snapshot, valid_bars)
    cache = FakeCache(
        snapshot=_snapshot(currency="CAD", exchange="OTC"),
        bars=[_bar(currency="CAD", exchange="OTC")],
    )
    gateway = MarketDataGateway(provider, cache=cache, clock=lambda: NOW)

    bundle = await gateway.fetch("NVDA")

    assert bundle.snapshot is valid_snapshot
    assert bundle.bars == valid_bars
    assert cache.invalidated_snapshots == 1
    assert cache.invalidated_bars == 1
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_validates_provider_scope_before_cache_write() -> None:
    provider = FakeProvider(_snapshot(currency="CAD"), [_bar()])
    cache = FakeCache()

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch_snapshot(
            "NVDA"
        )

    assert raised.value.code is MarketDataErrorCode.SCOPE_MISMATCH
    assert cache.saved_snapshot is None


@pytest.mark.asyncio
async def test_gateway_invalidates_both_cached_values_after_bundle_scope_mismatch() -> None:
    provider = FakeProvider(_snapshot(exchange="NASDAQ"), [_bar(exchange="IEX")])
    cache = FakeCache()

    with pytest.raises(MarketDataError) as raised:
        await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch("NVDA")

    assert raised.value.code is MarketDataErrorCode.SCOPE_MISMATCH
    assert cache.snapshot is None
    assert cache.bars is None
    assert cache.invalidated_snapshots == 1
    assert cache.invalidated_bars == 1


@pytest.mark.asyncio
async def test_gateway_treats_empty_cached_bars_as_miss() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    cache = FakeCache(bars=[])

    bars = await MarketDataGateway(provider, cache=cache, clock=lambda: NOW).fetch_bars(
        "NVDA"
    )

    assert bars == [_bar()]
    assert provider.bars_calls == 1
    assert cache.saved_bars == [_bar()]


@pytest.mark.asyncio
async def test_gateway_maps_empty_provider_bars_to_safe_partial() -> None:
    provider = FakeProvider(_snapshot(), [])

    bundle = await MarketDataGateway(provider, clock=lambda: NOW).fetch("NVDA")

    assert bundle.status == "partial"
    assert bundle.bars == []
    assert bundle.errors == [MarketDataErrorCode.UNAVAILABLE.value]
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_coalesces_concurrent_snapshot_calls() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    first, second = await asyncio.gather(
        gateway.fetch_snapshot("NVDA"),
        gateway.fetch_snapshot("NVDA"),
    )

    assert first == second == _snapshot()
    assert provider.snapshot_calls == 1


@pytest.mark.asyncio
async def test_gateway_coalesces_concurrent_bars_calls() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    first, second = await asyncio.gather(
        gateway.fetch_bars("NVDA"),
        gateway.fetch_bars("NVDA"),
    )

    assert first == second == [_bar()]
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_coalesces_fetch_with_direct_snapshot_and_bars_calls() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    bundle, snapshot, bars = await asyncio.gather(
        gateway.fetch("NVDA"),
        gateway.fetch_snapshot("NVDA"),
        gateway.fetch_bars("NVDA"),
    )

    assert bundle.snapshot == snapshot
    assert bundle.bars == bars
    assert provider.snapshot_calls == 1
    assert provider.bars_calls == 1


@pytest.mark.asyncio
async def test_gateway_coalesces_concurrent_typed_failures() -> None:
    provider = FakeProvider(_snapshot(), [_bar()])
    provider.snapshot_error = MarketDataError(
        MarketDataErrorCode.UNAVAILABLE, "market provider unavailable"
    )
    gateway = MarketDataGateway(provider, clock=lambda: NOW)

    outcomes = await asyncio.gather(
        gateway.fetch_snapshot("NVDA"),
        gateway.fetch_snapshot("NVDA"),
        return_exceptions=True,
    )

    assert all(isinstance(outcome, MarketDataError) for outcome in outcomes)
    assert {
        outcome.code for outcome in outcomes if isinstance(outcome, MarketDataError)
    } == {MarketDataErrorCode.UNAVAILABLE}
    assert provider.snapshot_calls == 1
