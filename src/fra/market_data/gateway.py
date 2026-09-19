"""Guarded cache and provider composition for normalized market data."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Protocol

from fra.domain import StrictModel
from fra.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
    MarketStatus,
)
from fra.market_data.providers import MarketDataProvider

_SUPPORTED_EXCHANGES = frozenset({"AMEX", "ARCA", "IEX", "NASDAQ", "NYSE"})
_PARTIAL_BARS_ERRORS = frozenset(
    {MarketDataErrorCode.UNAVAILABLE, MarketDataErrorCode.RATE_LIMITED}
)


class MarketFetchWrite(StrictModel):
    """Safe metadata for one provider request attempt."""

    provider: Literal["alpaca"]
    feed: Literal["iex"]
    symbol: str
    operation: Literal["snapshot", "clock", "bars"]
    requested_at: datetime
    fetched_at: datetime | None
    status: Literal["completed", "failed"]
    error_code: str | None = None


class MarketFetchWriter(Protocol):
    """Optional persistence boundary for safe market fetch metadata."""

    def record_fetch(self, value: MarketFetchWrite) -> None:
        """Record one request attempt without credentials, query data, or response bodies."""


class NoopMarketFetchWriter:
    """Discard market fetch metadata when no run-scoped writer is composed."""

    def record_fetch(self, value: MarketFetchWrite) -> None:
        del value


class MarketCache(Protocol):
    """Cache interface for normalized market observations."""

    async def get_snapshot(
        self, provider: str, feed: str, symbol: str
    ) -> MarketSnapshot | None: ...

    async def get_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
        limit: int,
    ) -> list[MarketBar] | None: ...

    async def set_snapshot(self, snapshot: MarketSnapshot) -> None: ...

    async def set_bars(self, bars: list[MarketBar]) -> None: ...

    async def invalidate_snapshot(self, provider: str, feed: str, symbol: str) -> None: ...

    async def invalidate_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
    ) -> None: ...


class NoopMarketCache:
    """Deterministic cache miss used until a concrete cache is composed."""

    async def get_snapshot(
        self, provider: str, feed: str, symbol: str
    ) -> None:
        del provider, feed, symbol
        return None

    async def get_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
        limit: int,
    ) -> None:
        del provider, feed, symbol, interval, limit
        return None

    async def set_snapshot(self, snapshot: MarketSnapshot) -> None:
        del snapshot

    async def set_bars(self, bars: list[MarketBar]) -> None:
        del bars

    async def invalidate_snapshot(self, provider: str, feed: str, symbol: str) -> None:
        del provider, feed, symbol

    async def invalidate_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
    ) -> None:
        del provider, feed, symbol, interval


class MarketDataGateway:
    """Run-scoped market gateway; compose a fresh instance per application run."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        max_bars: int = 5,
        max_staleness_seconds: int = 90,
        clock: Callable[[], datetime] | None = None,
        cache: MarketCache | None = None,
    ) -> None:
        if not 1 <= max_bars <= 20:
            raise ValueError("max_bars must be between 1 and 20")
        if max_staleness_seconds <= 0:
            raise ValueError("max_staleness_seconds must be positive")
        self._provider = provider
        self._max_bars = max_bars
        self._max_staleness_seconds = max_staleness_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cache = cache or NoopMarketCache()
        self._snapshot_outcomes: dict[str, MarketSnapshot | MarketDataError] = {}
        self._bars_outcomes: dict[
            str, tuple[MarketBar, ...] | MarketDataError
        ] = {}
        self._snapshot_inflight: dict[str, asyncio.Task[MarketSnapshot]] = {}
        self._bars_inflight: dict[str, asyncio.Task[list[MarketBar]]] = {}

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a scope-valid cached snapshot or make one provider call."""
        normalized_symbol = symbol.strip().upper()
        outcome = self._snapshot_outcomes.get(normalized_symbol)
        if isinstance(outcome, MarketDataError):
            raise outcome
        if outcome is not None:
            return outcome
        task = self._snapshot_inflight.get(normalized_symbol)
        if task is None:
            task = asyncio.create_task(self._resolve_snapshot(normalized_symbol))
            self._snapshot_inflight[normalized_symbol] = task
            task.add_done_callback(
                lambda completed, active_symbol=normalized_symbol: (
                    self._clear_snapshot_inflight(active_symbol, completed)
                )
            )
        return await asyncio.shield(task)

    async def _resolve_snapshot(self, normalized_symbol: str) -> MarketSnapshot:
        try:
            snapshot = await self._get_cached_snapshot(normalized_symbol)
            if snapshot is None:
                snapshot = await self._provider.get_snapshot(normalized_symbol)
                _require_normalized_snapshot_scope(snapshot, normalized_symbol)
                if (
                    snapshot.market_status is MarketStatus.OPEN
                    and not _is_open_snapshot_fresh(
                        snapshot,
                        now=self._clock(),
                        max_staleness_seconds=self._max_staleness_seconds,
                    )
                ):
                    raise MarketDataError(
                        MarketDataErrorCode.STALE,
                        "open-market snapshot is stale",
                    )
                await self._set_cached_snapshot(snapshot)
        except MarketDataError as error:
            self._snapshot_outcomes[normalized_symbol] = error
            raise
        self._snapshot_outcomes[normalized_symbol] = snapshot
        return snapshot

    async def fetch_bars(self, symbol: str) -> list[MarketBar]:
        """Return scope-valid cached daily bars or make one provider call."""
        normalized_symbol = symbol.strip().upper()
        outcome = self._bars_outcomes.get(normalized_symbol)
        if isinstance(outcome, MarketDataError):
            raise outcome
        if outcome is not None:
            return list(outcome)
        task = self._bars_inflight.get(normalized_symbol)
        if task is None:
            task = asyncio.create_task(self._resolve_bars(normalized_symbol))
            self._bars_inflight[normalized_symbol] = task
            task.add_done_callback(
                lambda completed, active_symbol=normalized_symbol: (
                    self._clear_bars_inflight(active_symbol, completed)
                )
            )
        return await asyncio.shield(task)

    async def _resolve_bars(self, normalized_symbol: str) -> list[MarketBar]:
        try:
            bars = await self._get_cached_bars(normalized_symbol)
            if bars is None:
                bars = await self._provider.get_bars(
                    normalized_symbol,
                    interval="1Day",
                    limit=self._max_bars,
                )
                if not bars:
                    raise MarketDataError(
                        MarketDataErrorCode.UNAVAILABLE,
                        "market provider returned no daily bars",
                    )
                _require_normalized_bars_scope(bars, normalized_symbol)
                await self._set_cached_bars(bars)
        except MarketDataError as error:
            self._bars_outcomes[normalized_symbol] = error
            raise
        self._bars_outcomes[normalized_symbol] = tuple(bars)
        return list(bars)

    def _clear_snapshot_inflight(
        self, symbol: str, task: asyncio.Task[MarketSnapshot]
    ) -> None:
        if self._snapshot_inflight.get(symbol) is task:
            del self._snapshot_inflight[symbol]

    def _clear_bars_inflight(
        self, symbol: str, task: asyncio.Task[list[MarketBar]]
    ) -> None:
        if self._bars_inflight.get(symbol) is task:
            del self._bars_inflight[symbol]

    async def fetch(self, symbol: str) -> MarketDataBundle:
        """Fetch one snapshot and one bars operation, allowing a safe bars partial."""
        normalized_symbol = symbol.strip().upper()
        snapshot = await self.fetch_snapshot(normalized_symbol)
        try:
            bars = await self.fetch_bars(normalized_symbol)
        except MarketDataError as error:
            if error.code not in _PARTIAL_BARS_ERRORS:
                raise
            bars = []
            errors = [error.code.value]
            status: Literal["completed", "partial"] = "partial"
        else:
            errors = []
            status = "completed" if bars else "partial"

        try:
            _require_same_scope(snapshot, bars)
        except MarketDataError:
            await self._invalidate_scope(normalized_symbol)
            raise
        _validate_freshness(
            snapshot,
            bars,
            now=self._clock(),
            max_staleness_seconds=self._max_staleness_seconds,
        )
        return MarketDataBundle(
            snapshot=snapshot,
            bars=bars,
            status=status,
            freshness_label=_freshness_label(snapshot.market_status),
            errors=errors,
        )

    async def _get_cached_snapshot(self, symbol: str) -> MarketSnapshot | None:
        try:
            cached = await self._cache.get_snapshot("alpaca", "iex", symbol)
        except Exception:
            return None
        if cached is None:
            return None
        if not _snapshot_matches(cached, symbol):
            await self._invalidate_snapshot(symbol)
            return None
        if cached.market_status is MarketStatus.OPEN and not _is_open_snapshot_fresh(
            cached,
            now=self._clock(),
            max_staleness_seconds=self._max_staleness_seconds,
        ):
            await self._invalidate_snapshot(symbol)
            return None
        return cached

    async def _get_cached_bars(self, symbol: str) -> list[MarketBar] | None:
        try:
            cached = await self._cache.get_bars(
                "alpaca", "iex", symbol, interval="1Day", limit=self._max_bars
            )
        except Exception:
            return None
        if cached is None:
            return None
        if (
            not cached
            or len(cached) > self._max_bars
            or any(not _bar_matches(bar, symbol) for bar in cached)
            or not _bars_are_unique_and_chronological(cached)
        ):
            await self._invalidate_bars(symbol)
            return None
        return cached

    async def _set_cached_snapshot(self, snapshot: MarketSnapshot) -> None:
        try:
            await self._cache.set_snapshot(snapshot)
        except Exception:
            return

    async def _set_cached_bars(self, bars: list[MarketBar]) -> None:
        try:
            await self._cache.set_bars(bars)
        except Exception:
            return

    async def _invalidate_snapshot(self, symbol: str) -> None:
        try:
            await self._cache.invalidate_snapshot("alpaca", "iex", symbol)
        except Exception:
            return

    async def _invalidate_bars(self, symbol: str) -> None:
        try:
            await self._cache.invalidate_bars(
                "alpaca", "iex", symbol, interval="1Day"
            )
        except Exception:
            return

    async def _invalidate_scope(self, symbol: str) -> None:
        await self._invalidate_snapshot(symbol)
        await self._invalidate_bars(symbol)


def _snapshot_matches(snapshot: MarketSnapshot, symbol: str) -> bool:
    return (
        snapshot.provider == "alpaca"
        and snapshot.feed == "iex"
        and snapshot.coverage == "IEX-only"
        and snapshot.symbol == symbol
        and snapshot.currency == "USD"
        and snapshot.exchange in _SUPPORTED_EXCHANGES
    )


def _bar_matches(bar: MarketBar, symbol: str) -> bool:
    return (
        bar.provider == "alpaca"
        and bar.feed == "iex"
        and bar.coverage == "IEX-only"
        and bar.symbol == symbol
        and bar.interval == "1Day"
        and bar.currency == "USD"
        and bar.exchange in _SUPPORTED_EXCHANGES
    )


def _require_normalized_snapshot_scope(snapshot: MarketSnapshot, symbol: str) -> None:
    if not _snapshot_matches(snapshot, symbol):
        raise MarketDataError(
            MarketDataErrorCode.SCOPE_MISMATCH,
            "snapshot is outside the supported market-data scope",
        )


def _require_normalized_bars_scope(bars: list[MarketBar], symbol: str) -> None:
    if (
        any(not _bar_matches(bar, symbol) for bar in bars)
        or not _bars_are_unique_and_chronological(bars)
    ):
        raise MarketDataError(
            MarketDataErrorCode.SCOPE_MISMATCH,
            "bars are outside the supported market-data scope",
        )


def _bars_are_unique_and_chronological(bars: list[MarketBar]) -> bool:
    timestamps = [bar.timestamp for bar in bars]
    return len(timestamps) == len(set(timestamps)) and timestamps == sorted(timestamps)


def _require_same_scope(snapshot: MarketSnapshot, bars: list[MarketBar]) -> None:
    expected = (
        snapshot.provider,
        snapshot.feed,
        snapshot.coverage,
        snapshot.symbol,
        snapshot.exchange,
        snapshot.currency,
    )
    if any(
        (
            bar.provider,
            bar.feed,
            bar.coverage,
            bar.symbol,
            bar.exchange,
            bar.currency,
        )
        != expected
        for bar in bars
    ):
        raise MarketDataError(
            MarketDataErrorCode.SCOPE_MISMATCH,
            "snapshot and bars do not share one market-data scope",
        )


def _is_open_snapshot_fresh(
    snapshot: MarketSnapshot,
    *,
    now: datetime,
    max_staleness_seconds: int,
) -> bool:
    age = (now - snapshot.as_of).total_seconds()
    return 0 <= age <= max_staleness_seconds


def _validate_freshness(
    snapshot: MarketSnapshot,
    bars: list[MarketBar],
    *,
    now: datetime,
    max_staleness_seconds: int,
) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    if snapshot.market_status is MarketStatus.OPEN and not _is_open_snapshot_fresh(
        snapshot,
        now=now,
        max_staleness_seconds=max_staleness_seconds,
    ):
        raise MarketDataError(MarketDataErrorCode.STALE, "open-market snapshot is stale")
    if (
        snapshot.market_status is MarketStatus.CLOSED
        and bars
        and snapshot.as_of < bars[-1].timestamp
    ):
        raise MarketDataError(
            MarketDataErrorCode.STALE,
            "closed-market snapshot predates the latest returned daily bar",
        )


def _freshness_label(status: MarketStatus) -> str:
    if status is MarketStatus.OPEN:
        return "open-iex"
    if status is MarketStatus.CLOSED:
        return "latest-available-iex"
    return "market-status-unknown"
