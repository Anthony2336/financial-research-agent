"""Typed JSON-only Redis caching with deterministic no-cache degradation."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic
from typing import Literal, Protocol, TypeAlias

from pydantic import ValidationError
from redis import Redis as SyncRedis
from redis.asyncio import Redis

from fra.config import Settings
from fra.market_data.models import MarketBar, MarketSnapshot, MarketStatus

logger = logging.getLogger(__name__)

JsonValue: TypeAlias = dict[str, object] | list[object]


class SyncJsonCache(Protocol):
    """Synchronous JSON surface used by the synchronous filing retriever."""

    def get_json_sync(self, key: str) -> JsonValue | None:
        """Return one decoded JSON object or a cache miss."""

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        """Store one JSON-serializable value for a bounded duration."""

    def delete_json_sync(self, key: str) -> None:
        """Invalidate one JSON value without exposing cache failures."""


class JsonCache(Protocol):
    """Matching asynchronous JSON surface used by the web gateway."""

    async def get_json(self, key: str) -> JsonValue | None:
        """Return one decoded JSON object or a cache miss."""

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        """Store one JSON-serializable value for a bounded duration."""

    async def delete_json(self, key: str) -> None:
        """Invalidate one JSON value without exposing cache failures."""


class CompositeJsonCache(JsonCache, SyncJsonCache, Protocol):
    """Cache contract that can serve both synchronous and asynchronous consumers."""


class NoopJsonCache:
    """Deterministic cache miss used when Redis is disabled."""

    async def get_json(self, key: str) -> None:
        del key
        return None

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        del key, value, ttl_seconds

    async def delete_json(self, key: str) -> None:
        del key

    def get_json_sync(self, key: str) -> None:
        del key
        return None

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        del key, value, ttl_seconds

    def delete_json_sync(self, key: str) -> None:
        del key


class InMemoryTtlJsonCache:
    """Small concrete JSON cache used by market application runs without Redis."""

    def __init__(
        self,
        *,
        max_entries: int = 256,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()

    async def get_json(self, key: str) -> JsonValue | None:
        return self.get_json_sync(key)

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.set_json_sync(key, value, ttl_seconds=ttl_seconds)

    async def delete_json(self, key: str) -> None:
        self._entries.pop(key, None)

    def get_json_sync(self, key: str) -> JsonValue | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at <= self._clock():
            return None
        self._entries[key] = entry
        return _decode_json(payload)

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._entries.pop(key, None)
        self._entries[key] = (self._clock() + ttl_seconds, _encode_json(value))
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def delete_json_sync(self, key: str) -> None:
        self._entries.pop(key, None)


class RedisJsonCache:
    """Namespaced Redis adapter that stores canonical JSON and never raises outages."""

    def __init__(
        self,
        client: Redis,
        *,
        sync_client: SyncRedis | None = None,
        namespace: str = "fea",
    ) -> None:
        self._client = client
        self._sync_client = sync_client
        self._namespace = namespace.rstrip(":")

    async def get_json(self, key: str) -> JsonValue | None:
        namespaced_key = self._key(key)
        try:
            return _decode_json(await self._client.get(namespaced_key))
        except Exception:
            logger.warning("cache read failed")
            return None

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        namespaced_key = self._key(key)
        try:
            await self._client.set(
                namespaced_key,
                _encode_json(value),
                ex=ttl_seconds,
            )
        except Exception:
            logger.warning("cache write failed")

    async def delete_json(self, key: str) -> None:
        namespaced_key = self._key(key)
        try:
            await self._client.delete(namespaced_key)
        except Exception:
            logger.warning("cache invalidation failed")

    def get_json_sync(self, key: str) -> JsonValue | None:
        namespaced_key = self._key(key)
        if self._sync_client is None:
            return None
        try:
            return _decode_json(self._sync_client.get(namespaced_key))
        except Exception:
            logger.warning("cache read failed")
            return None

    def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
        namespaced_key = self._key(key)
        if self._sync_client is None:
            return
        try:
            self._sync_client.set(
                namespaced_key,
                _encode_json(value),
                ex=ttl_seconds,
            )
        except Exception:
            logger.warning("cache write failed")

    def delete_json_sync(self, key: str) -> None:
        namespaced_key = self._key(key)
        if self._sync_client is None:
            return
        try:
            self._sync_client.delete(namespaced_key)
        except Exception:
            logger.warning("cache invalidation failed")

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}" if self._namespace else key

    async def aclose(self) -> None:
        """Release the async client on the loop that used market cache operations."""
        await self._client.aclose()


def build_cache(settings: Settings, *, namespace: str = "fea") -> CompositeJsonCache:
    """Build lazy Redis clients, or an inert cache when Redis is disabled."""
    if not settings.redis_url:
        return NoopJsonCache()
    try:
        client = Redis.from_url(settings.redis_url)
        sync_client = SyncRedis.from_url(settings.redis_url)
    except Exception:
        logger.warning("cache construction failed")
        return NoopJsonCache()
    return RedisJsonCache(client, sync_client=sync_client, namespace=namespace)


def build_session_cache(settings: Settings) -> CompositeJsonCache:
    """Build the un-namespaced cache required by exact ``session:{id}`` keys."""
    return build_cache(settings, namespace="")


def build_market_cache(settings: Settings) -> CompositeJsonCache:
    """Prefer Redis while guaranteeing a bounded concrete cache for market runs."""
    cache = build_cache(settings)
    return InMemoryTtlJsonCache() if isinstance(cache, NoopJsonCache) else cache


class MarketDataJsonCache:
    """Typed market cache over the shared Redis/no-op JSON boundary."""

    def __init__(
        self,
        cache: JsonCache,
        *,
        max_bars: int = 5,
        max_staleness_seconds: int = 90,
        ttl_seconds: int = 300,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not 1 <= max_bars <= 20:
            raise ValueError("max_bars must be between 1 and 20")
        if max_staleness_seconds <= 0 or ttl_seconds <= 0:
            raise ValueError("market cache durations must be positive")
        self._cache = cache
        self._max_bars = max_bars
        self._max_staleness_seconds = max_staleness_seconds
        self._ttl_seconds = ttl_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def get_snapshot(
        self, provider: str, feed: str, symbol: str
    ) -> MarketSnapshot | None:
        key = _market_snapshot_key(provider, feed, symbol)
        value = await self._cache.get_json(key)
        try:
            if not isinstance(value, dict) or set(value) != {"snapshot"}:
                raise ValueError("invalid snapshot cache envelope")
            snapshot = MarketSnapshot.model_validate(value["snapshot"])
            if snapshot.market_status is MarketStatus.OPEN and (
                self._clock() - snapshot.as_of
            ).total_seconds() > self._max_staleness_seconds:
                raise ValueError("stale open-market cache entry")
        except (TypeError, ValueError, ValidationError):
            if value is not None:
                await self._cache.delete_json(key)
            return None
        return snapshot

    async def get_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
        limit: int,
    ) -> list[MarketBar] | None:
        key = _market_bars_key(provider, feed, symbol, interval, limit)
        value = await self._cache.get_json(key)
        try:
            if not isinstance(value, dict) or set(value) != {"bars"}:
                raise ValueError("invalid bars cache envelope")
            raw_bars = value["bars"]
            if not isinstance(raw_bars, list) or not raw_bars or len(raw_bars) > limit:
                raise ValueError("invalid cached bars")
            bars = [MarketBar.model_validate(bar) for bar in raw_bars]
        except (TypeError, ValueError, ValidationError):
            if value is not None:
                await self._cache.delete_json(key)
            return None
        return bars

    async def set_snapshot(self, snapshot: MarketSnapshot) -> None:
        await self._cache.set_json(
            _market_snapshot_key(snapshot.provider, snapshot.feed, snapshot.symbol),
            {"snapshot": snapshot.model_dump(mode="json")},
            ttl_seconds=self._ttl_seconds,
        )

    async def set_bars(self, bars: list[MarketBar]) -> None:
        if not bars:
            return
        first = bars[0]
        await self._cache.set_json(
            _market_bars_key(
                first.provider,
                first.feed,
                first.symbol,
                first.interval,
                self._max_bars,
            ),
            {"bars": [bar.model_dump(mode="json") for bar in bars]},
            ttl_seconds=self._ttl_seconds,
        )

    async def invalidate_snapshot(self, provider: str, feed: str, symbol: str) -> None:
        await self._cache.delete_json(_market_snapshot_key(provider, feed, symbol))

    async def invalidate_bars(
        self,
        provider: str,
        feed: str,
        symbol: str,
        *,
        interval: Literal["1Day"],
    ) -> None:
        await self._cache.delete_json(
            _market_bars_key(provider, feed, symbol, interval, self._max_bars)
        )


def _market_snapshot_key(provider: str, feed: str, symbol: str) -> str:
    return f"market:{provider}:{feed}:{symbol}:snapshot"


def _market_bars_key(
    provider: str,
    feed: str,
    symbol: str,
    interval: str,
    limit: int,
) -> str:
    return f"market:{provider}:{feed}:{symbol}:bars:{interval}:{limit}"


def _encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_json(value: object) -> JsonValue | None:
    if value is None:
        return None
    if not isinstance(value, (bytes, str)):
        raise TypeError("cache value must be bytes or text")
    decoded = json.loads(value)
    if not isinstance(decoded, (dict, list)):
        raise TypeError("cache value must contain a JSON object or array")
    return decoded or None
