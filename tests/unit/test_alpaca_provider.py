"""HTTP-boundary tests for the Alpaca Basic IEX adapter."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from financial_evidence_agent.market_data.alpaca import AlpacaMarketDataProvider
from financial_evidence_agent.market_data.gateway import MarketFetchWrite
from financial_evidence_agent.market_data.models import (
    MarketDataError,
    MarketDataErrorCode,
    MarketStatus,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "alpaca"
NOW = datetime(2026, 8, 31, 14, 0, 2, tzinfo=UTC)


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


class RecordingWriter:
    def __init__(self) -> None:
        self.values: list[MarketFetchWrite] = []

    def record_fetch(self, value: MarketFetchWrite) -> None:
        self.values.append(value)


def _transport(
    *,
    snapshot: object | None = None,
    bars: object | None = None,
    clock: object | None = None,
    statuses: dict[str, int] | None = None,
    timeout_path: str | None = None,
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    payloads = {
        "snapshot": _fixture("snapshot.json") if snapshot is None else snapshot,
        "bars": _fixture("bars.json") if bars is None else bars,
        "clock": _fixture("clock_open.json") if clock is None else clock,
    }
    response_statuses = statuses or {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        operation = (
            "clock"
            if request.url.path == "/v2/clock"
            else "snapshot"
            if request.url.path.endswith("/snapshot")
            else "bars"
        )
        if operation == timeout_path:
            raise httpx.ReadTimeout("secret response body", request=request)
        return httpx.Response(
            response_statuses.get(operation, 200),
            json=payloads[operation],
            request=request,
        )

    return httpx.MockTransport(handler)


def _observation_id(
    kind: str,
    *,
    symbol: str,
    source_timestamp: str,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "provider": "alpaca",
            "feed": "iex",
            "symbol": symbol,
            "source_timestamp": source_timestamp,
            "fetched_at": fetched_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "raw_payload_hash": raw_payload_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"market-{kind}:{sha256(payload).hexdigest()}"


@pytest.mark.asyncio
async def test_alpaca_maps_complete_snapshot_clock_and_bars_fixtures() -> None:
    requests: list[httpx.Request] = []
    writer = RecordingWriter()
    async with httpx.AsyncClient(transport=_transport(requests=requests)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )

        snapshot = await provider.get_snapshot("NVDA")
        bars = await provider.get_bars("NVDA", interval="1Day", limit=5)

    assert snapshot.provider == "alpaca"
    assert snapshot.feed == "iex"
    assert snapshot.coverage == "IEX-only"
    assert snapshot.market_status is MarketStatus.OPEN
    assert snapshot.price == Decimal("123.45")
    assert snapshot.exchange == "IEX"
    assert snapshot.currency == "USD"
    assert snapshot.raw_payload_hash == (
        "8c0a0307a72a8844037559e6d6a763d506ff60e4067b95de827202948e347df8"
    )
    assert snapshot.id == _observation_id(
        "snapshot",
        symbol="NVDA",
        source_timestamp="2026-08-31T14:00:00.123456Z",
        fetched_at=NOW,
        raw_payload_hash=snapshot.raw_payload_hash,
    )
    assert [bar.close for bar in bars] == [Decimal("120.5"), Decimal("121.0")]
    assert [bar.id for bar in bars] == [
        _observation_id(
            "bar",
            symbol="NVDA",
            source_timestamp="2026-08-27T04:00:00Z",
            fetched_at=NOW,
            raw_payload_hash=bars[0].raw_payload_hash,
        ),
        _observation_id(
            "bar",
            symbol="NVDA",
            source_timestamp="2026-08-28T04:00:00Z",
            fetched_at=NOW,
            raw_payload_hash=bars[1].raw_payload_hash,
        ),
    ]
    assert all(bar.feed == "iex" and bar.interval == "1Day" for bar in bars)
    assert len(requests) == 3
    assert {request.url.host for request in requests} == {
        "data.alpaca.markets",
        "paper-api.alpaca.markets",
    }
    assert all(request.headers["APCA-API-KEY-ID"] == "key" for request in requests)
    assert all(request.headers["APCA-API-SECRET-KEY"] == "secret" for request in requests)
    assert [request.url.params.get("feed") for request in requests] == ["iex", None, "iex"]
    assert requests[-1].url.params["timeframe"] == "1Day"
    assert requests[-1].url.params["limit"] == "5"
    assert {value.operation for value in writer.values} == {"snapshot", "clock", "bars"}
    assert all(value.status == "completed" for value in writer.values)
    assert all(
        set(value.model_dump())
        == {
            "provider",
            "feed",
            "symbol",
            "operation",
            "requested_at",
            "fetched_at",
            "status",
            "error_code",
        }
        for value in writer.values
    )


@pytest.mark.asyncio
async def test_alpaca_uses_only_fixed_live_trading_host() -> None:
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(requests=requests)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            trading_environment="live",
            clock=lambda: NOW,
        )
        await provider.get_snapshot("NVDA")

    assert {request.url.host for request in requests} == {
        "data.alpaca.markets",
        "api.alpaca.markets",
    }
    with pytest.raises(ValueError, match="paper or live"):
        AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            trading_environment="https://attacker.example",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, MarketDataErrorCode.CONFIGURATION_MISSING),
        (403, MarketDataErrorCode.CONFIGURATION_MISSING),
        (404, MarketDataErrorCode.UNAVAILABLE),
        (429, MarketDataErrorCode.RATE_LIMITED),
        (503, MarketDataErrorCode.UNAVAILABLE),
    ],
)
async def test_alpaca_maps_snapshot_http_errors_without_retry(
    status: int, expected: MarketDataErrorCode
) -> None:
    requests: list[httpx.Request] = []
    writer = RecordingWriter()
    async with httpx.AsyncClient(
        transport=_transport(statuses={"snapshot": status}, requests=requests)
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )
        with pytest.raises(MarketDataError) as raised:
            await provider.get_snapshot("NVDA")

    assert raised.value.code is expected
    assert sum(request.url.path.endswith("/snapshot") for request in requests) == 1
    assert writer.values[0].error_code == expected.value
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_alpaca_records_clock_attempt_when_required_snapshot_fails() -> None:
    writer = RecordingWriter()
    async with httpx.AsyncClient(
        transport=_transport(statuses={"snapshot": 503})
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )
        with pytest.raises(MarketDataError):
            await provider.get_snapshot("NVDA")

    assert {value.operation for value in writer.values} == {"snapshot", "clock"}
    clock_write = next(value for value in writer.values if value.operation == "clock")
    assert clock_write.status == "completed"


@pytest.mark.asyncio
async def test_alpaca_clock_timeout_degrades_to_unknown_and_records_safe_failure() -> None:
    writer = RecordingWriter()
    async with httpx.AsyncClient(transport=_transport(timeout_path="clock")) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )
        snapshot = await provider.get_snapshot("NVDA")

    assert snapshot.market_status is MarketStatus.UNKNOWN
    clock_write = next(value for value in writer.values if value.operation == "clock")
    assert clock_write.status == "failed"
    assert clock_write.error_code == MarketDataErrorCode.UNAVAILABLE.value


@pytest.mark.asyncio
async def test_alpaca_malformed_clock_degrades_to_unknown_and_records_failure() -> None:
    writer = RecordingWriter()
    async with httpx.AsyncClient(
        transport=_transport(clock={"timestamp": "2026-08-31T14:00:01Z"})
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )
        snapshot = await provider.get_snapshot("NVDA")

    assert snapshot.market_status is MarketStatus.UNKNOWN
    clock_write = next(value for value in writer.values if value.operation == "clock")
    assert clock_write.status == "failed"
    assert clock_write.error_code == MarketDataErrorCode.UNAVAILABLE.value


@pytest.mark.asyncio
async def test_alpaca_maps_closed_clock_fixture() -> None:
    async with httpx.AsyncClient(
        transport=_transport(clock=_fixture("clock_closed.json"))
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        snapshot = await provider.get_snapshot("NVDA")

    assert snapshot.market_status is MarketStatus.CLOSED


@pytest.mark.asyncio
async def test_alpaca_clock_auth_error_fails_typed() -> None:
    async with httpx.AsyncClient(
        transport=_transport(statuses={"clock": 403})
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as raised:
            await provider.get_snapshot("NVDA")

    assert raised.value.code is MarketDataErrorCode.CONFIGURATION_MISSING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "bars"),
    [({}, None), ({"symbol": "NVDA", "latestTrade": {}}, None), (None, {"bars": []})],
)
async def test_alpaca_rejects_empty_or_malformed_payloads(
    snapshot: object | None, bars: object | None
) -> None:
    async with httpx.AsyncClient(
        transport=_transport(snapshot=snapshot, bars=bars)
    ) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as raised:
            if bars is None:
                await provider.get_snapshot("NVDA")
            else:
                await provider.get_bars("NVDA", interval="1Day", limit=5)

    assert raised.value.code is MarketDataErrorCode.UNAVAILABLE


@pytest.mark.asyncio
async def test_alpaca_records_normalization_failure_as_failed_attempt() -> None:
    writer = RecordingWriter()
    payload = _fixture("snapshot.json")
    assert isinstance(payload, dict)
    payload["latestTrade"]["p"] = "not-a-decimal"  # type: ignore[index]
    async with httpx.AsyncClient(transport=_transport(snapshot=payload)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key",
            secret_key="secret",
            client=client,
            clock=lambda: NOW,
            fetch_writer=writer,
        )
        with pytest.raises(MarketDataError):
            await provider.get_snapshot("NVDA")

    snapshot_write = next(value for value in writer.values if value.operation == "snapshot")
    assert snapshot_write.status == "failed"
    assert snapshot_write.fetched_at is None
    assert snapshot_write.error_code == MarketDataErrorCode.UNAVAILABLE.value


@pytest.mark.asyncio
async def test_alpaca_rejects_invalid_interval_limit_and_non_decimal_data() -> None:
    payload = _fixture("snapshot.json")
    assert isinstance(payload, dict)
    payload["latestTrade"]["p"] = "not-a-decimal"  # type: ignore[index]
    async with httpx.AsyncClient(transport=_transport(snapshot=payload)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as raised:
            await provider.get_snapshot("NVDA")
        with pytest.raises(ValueError, match="1Day"):
            await provider.get_bars("NVDA", interval="1Min", limit=5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="between"):
            await provider.get_bars("NVDA", interval="1Day", limit=0)

    assert raised.value.code is MarketDataErrorCode.UNAVAILABLE


@pytest.mark.asyncio
async def test_alpaca_rejects_duplicate_bar_timestamps_as_typed_error() -> None:
    payload = _fixture("bars.json")
    assert isinstance(payload, dict)
    bars = payload["bars"]
    assert isinstance(bars, list)
    bars.append(dict(bars[-1]))
    async with httpx.AsyncClient(transport=_transport(bars=payload)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as raised:
            await provider.get_bars("NVDA", interval="1Day", limit=5)

    assert raised.value.code is MarketDataErrorCode.UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("volume", [True, "1000", -1, 1.5, 9_223_372_036_854_775_808])
async def test_alpaca_rejects_non_integral_or_out_of_range_bar_volume(volume: object) -> None:
    payload = _fixture("bars.json")
    assert isinstance(payload, dict)
    bars = payload["bars"]
    assert isinstance(bars, list)
    bars[0]["v"] = volume  # type: ignore[index]
    async with httpx.AsyncClient(transport=_transport(bars=payload)) as client:
        provider = AlpacaMarketDataProvider(
            key_id="key", secret_key="secret", client=client, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as raised:
            await provider.get_bars("NVDA", interval="1Day", limit=5)

    assert raised.value.code is MarketDataErrorCode.UNAVAILABLE
