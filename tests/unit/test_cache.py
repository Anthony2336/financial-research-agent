import logging
from typing import get_type_hints

import pytest

from fra.config import Settings
from fra.storage.cache import NoopJsonCache, RedisJsonCache, build_cache


class FakeAsyncRedis:
    def __init__(self, *, stored: object = None, error: Exception | None = None) -> None:
        self.stored = stored
        self.error = error
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str, int]] = []
        self.delete_calls: list[str] = []

    async def get(self, key: str) -> object:
        self.get_calls.append(key)
        if self.error is not None:
            raise self.error
        return self.stored

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.set_calls.append((key, value, ex))
        if self.error is not None:
            raise self.error

    async def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        if self.error is not None:
            raise self.error


class FakeSyncRedis:
    def __init__(self, *, stored: object = None, error: Exception | None = None) -> None:
        self.stored = stored
        self.error = error
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str, int]] = []
        self.delete_calls: list[str] = []

    def get(self, key: str) -> object:
        self.get_calls.append(key)
        if self.error is not None:
            raise self.error
        return self.stored

    def set(self, key: str, value: str, *, ex: int) -> None:
        self.set_calls.append((key, value, ex))
        if self.error is not None:
            raise self.error

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        if self.error is not None:
            raise self.error


async def test_redis_cache_writes_canonical_namespaced_json_with_ttl() -> None:
    """Changing JSON ordering, namespace, or expiry would break stable cache storage."""
    client = FakeAsyncRedis()
    cache = RedisJsonCache(client, namespace="tenant")

    await cache.set_json("answer", {"z": 1, "a": ["x", 2]}, ttl_seconds=45)

    assert client.set_calls == [("tenant:answer", '{"a":["x",2],"z":1}', 45)]


def test_redis_cache_exposes_matching_synchronous_json_surface() -> None:
    """Running the synchronous retriever through an event loop would violate its API."""
    async_client = FakeAsyncRedis()
    sync_client = FakeSyncRedis(stored=b'{"items":[1]}')
    cache = RedisJsonCache(async_client, sync_client=sync_client, namespace="tenant")

    assert cache.get_json_sync("answer") == {"items": [1]}
    cache.set_json_sync("answer", {"z": 1, "a": 2}, ttl_seconds=30)

    assert sync_client.get_calls == ["tenant:answer"]
    assert sync_client.set_calls == [("tenant:answer", '{"a":2,"z":1}', 30)]


async def test_redis_cache_treats_corrupted_values_as_recorded_misses(caplog) -> None:
    """Malformed Redis data must never become authoritative evidence or escape to callers."""
    cache = RedisJsonCache(FakeAsyncRedis(stored=b"not-json"))

    with caplog.at_level(logging.WARNING):
        result = await cache.get_json("answer")

    assert result is None
    assert "cache read failed" in caplog.text


@pytest.mark.parametrize("stored", [b"{}", b"[]"])
async def test_redis_cache_normalizes_empty_json_containers_to_misses(stored: bytes) -> None:
    """Empty JSON must never become an authoritative hit for either consumer surface."""
    cache = RedisJsonCache(
        FakeAsyncRedis(stored=stored),
        sync_client=FakeSyncRedis(stored=stored),
    )

    assert await cache.get_json("answer") is None
    assert cache.get_json_sync("answer") is None


async def test_redis_cache_degrades_redis_exceptions_to_recorded_misses(caplog) -> None:
    """A Redis outage must preserve deterministic uncached execution."""
    cache = RedisJsonCache(FakeAsyncRedis(error=ConnectionError("redis down")))

    with caplog.at_level(logging.WARNING):
        result = await cache.get_json("answer")
        await cache.set_json("answer", {"ok": True}, ttl_seconds=15)

    assert result is None
    assert "cache read failed" in caplog.text
    assert "cache write failed" in caplog.text
    assert "redis down" not in caplog.text


async def test_redis_cache_failure_logs_are_fixed_for_all_six_operations(caplog) -> None:
    """A valid credential-like session id must never survive a cache outage log."""
    cache_key = "session:sk-live-valid-session-123"
    private_detail = "password=private-redis-password"
    async_client = FakeAsyncRedis(error=RuntimeError(private_detail))
    sync_client = FakeSyncRedis(error=RuntimeError(private_detail))
    cache = RedisJsonCache(async_client, sync_client=sync_client, namespace="")

    with caplog.at_level(logging.WARNING):
        assert await cache.get_json(cache_key) is None
        assert await cache.set_json(cache_key, {"ok": True}, ttl_seconds=15) is None
        assert await cache.delete_json(cache_key) is None
        assert cache.get_json_sync(cache_key) is None
        assert cache.set_json_sync(cache_key, {"ok": True}, ttl_seconds=15) is None
        assert cache.delete_json_sync(cache_key) is None

    records = [
        record
        for record in caplog.records
        if record.name == "fra.storage.cache"
    ]
    assert [record.getMessage() for record in records] == [
        "cache read failed",
        "cache write failed",
        "cache invalidation failed",
        "cache read failed",
        "cache write failed",
        "cache invalidation failed",
    ]
    assert all(record.exc_info is None for record in records)
    assert cache_key not in caplog.text
    assert private_detail not in caplog.text
    assert "Traceback" not in caplog.text


async def test_noop_cache_is_a_miss_on_both_surfaces() -> None:
    """Disabled caching must not retain values or alter synchronous/async callers."""
    cache = NoopJsonCache()

    await cache.set_json("answer", {"ok": True}, ttl_seconds=15)
    cache.set_json_sync("answer", {"ok": True}, ttl_seconds=15)

    assert await cache.get_json("answer") is None
    assert cache.get_json_sync("answer") is None


def test_build_cache_returns_noop_when_redis_is_disabled() -> None:
    """An absent Redis URL must avoid constructing any live dependency."""
    cache = build_cache(Settings(redis_url=None, _env_file=None))

    assert isinstance(cache, NoopJsonCache)


def test_build_cache_return_contract_exposes_sync_and_async_surfaces() -> None:
    """A return type missing either surface cannot be passed safely to both consumers."""
    return_type = get_type_hints(build_cache)["return"]

    assert all(
        hasattr(return_type, method)
        for method in ("get_json", "set_json", "get_json_sync", "set_json_sync")
    )
