"""Real in-process FastMCP contracts for normalized market tools."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

import pytest
from fastmcp import Client
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fra.market_data.gateway import MarketFetchWrite
from fra.market_data.models import (
    MarketBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
)
from fra.mcp_server.market_tools import RunScopedMarketFetchWriter
from fra.mcp_server.server import create_server
from fra.retrieval.hybrid import HashEmbeddingProvider, HybridRetriever
from fra.storage.database import create_schema
from fra.storage.market_repositories import MarketDataRepository
from fra.storage.models import MarketBarRecord, MarketSnapshotRecord
from fra.storage.repositories import FilingRepository
from fra.storage.run_repositories import ResearchRunRepository, RunStart

NOW = datetime.now(UTC).replace(microsecond=0)


def _snapshot() -> MarketSnapshot:
    as_of = NOW - timedelta(seconds=60)
    raw_payload_hash = "a" * 64
    return MarketSnapshot(
        id=_observation_id(
            "snapshot",
            symbol="NVDA",
            source_timestamp=as_of,
            fetched_at=NOW,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        price=Decimal("123.450000000000000001"),
        open=Decimal("122.00"),
        day_high=Decimal("124.00"),
        day_low=Decimal("121.50"),
        previous_close=Decimal("121.00"),
        as_of=as_of,
        fetched_at=NOW,
        market_status="open",
        delayed_by_seconds=60,
        raw_payload_hash=raw_payload_hash,
    )


def _bar() -> MarketBar:
    timestamp = datetime(2026, 8, 28, 4, tzinfo=UTC)
    raw_payload_hash = "b" * 64
    return MarketBar(
        id=_observation_id(
            "bar",
            symbol="NVDA",
            source_timestamp=timestamp,
            fetched_at=NOW,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        interval="1Day",
        timestamp=timestamp,
        open=Decimal("120.00"),
        high=Decimal("124.00"),
        low=Decimal("119.00"),
        close=Decimal("123.45"),
        volume=1000,
        fetched_at=NOW,
        raw_payload_hash=raw_payload_hash,
    )


def _observation_id(
    kind: str,
    *,
    symbol: str,
    source_timestamp: datetime,
    fetched_at: datetime,
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


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class RecordingGateway:
    def __init__(self, error: MarketDataError | None = None) -> None:
        self.error = error
        self.snapshot_calls = 0
        self.bars_calls = 0

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        self.snapshot_calls += 1
        if self.error is not None:
            raise self.error
        assert symbol == "NVDA"
        return _snapshot()

    async def fetch_bars(self, symbol: str) -> list[MarketBar]:
        self.bars_calls += 1
        if self.error is not None:
            raise self.error
        assert symbol == "NVDA"
        return [_bar()]


def _market_server(gateway_factory, *, max_bars: int = 5):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    filing_repository = FilingRepository(engine)
    market_repository = MarketDataRepository(engine)
    server = create_server(
        filing_repository,
        HybridRetriever(filing_repository, HashEmbeddingProvider()),
        market_gateway_factory=gateway_factory,
        market_repository=market_repository,
        market_data_max_bars=max_bars,
    )
    return server, engine, market_repository


async def test_market_tools_return_strict_normalized_scope_and_persist_once() -> None:
    """Each tool permits one gateway operation and retains the returned source record."""
    gateways: list[RecordingGateway] = []

    def factory() -> RecordingGateway:
        gateway = RecordingGateway()
        gateways.append(gateway)
        return gateway

    server, engine, repository = _market_server(factory)
    async with Client(server) as client:
        snapshot = await client.call_tool(
            "get_market_snapshot", {"ticker": "NVDA", "market": "US"}
        )
        bars = await client.call_tool(
            "get_market_bars", {"ticker": "NVDA", "interval": "1Day", "limit": 5}
        )

    assert snapshot.structured_content is not None
    assert snapshot.structured_content["snapshot"]["coverage"] == "IEX-only"
    assert snapshot.structured_content["snapshot"]["price"] == "123.450000000000000001"
    assert bars.structured_content is not None
    assert len(bars.structured_content["bars"]) == 1
    assert [(gateway.snapshot_calls, gateway.bars_calls) for gateway in gateways] == [
        (1, 0),
        (0, 1),
    ]
    assert repository.get_snapshot(_snapshot().id) == _snapshot()
    assert repository.get_bar(_bar().id) == _bar()
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketSnapshotRecord)) == 1
        assert session.scalar(select(func.count()).select_from(MarketBarRecord)) == 1


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_market_snapshot", {"ticker": "nvda", "market": "US"}),
        ("get_market_snapshot", {"ticker": "NVDA", "market": "CA"}),
        ("get_market_snapshot", {"ticker": "NVDA", "market": "US", "extra": True}),
        ("get_market_bars", {"ticker": "NVDA", "interval": "1Hour", "limit": 5}),
        ("get_market_bars", {"ticker": "NVDA", "interval": "1Day", "limit": 6}),
        ("get_market_bars", {"ticker": "NVDA", "interval": "1Day", "limit": "5"}),
    ],
)
async def test_market_tools_reject_non_normalized_or_out_of_scope_inputs(
    name: str, arguments: dict[str, object]
) -> None:
    """Provider execution must remain unreachable for invalid ticker, market, or bounds."""
    gateways: list[RecordingGateway] = []

    def factory() -> RecordingGateway:
        gateway = RecordingGateway()
        gateways.append(gateway)
        return gateway

    server, _, _ = _market_server(factory)
    async with Client(server) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)

    assert result.is_error
    assert gateways == []


@pytest.mark.parametrize(
    ("source_code", "public_code"),
    [
        (MarketDataErrorCode.CONFIGURATION_MISSING, "MARKET_DATA_CONFIGURATION_MISSING"),
        (MarketDataErrorCode.UNAVAILABLE, "MARKET_DATA_UNAVAILABLE"),
        (MarketDataErrorCode.RATE_LIMITED, "MARKET_DATA_RATE_LIMITED"),
        (MarketDataErrorCode.STALE, "STALE_MARKET_DATA"),
        (MarketDataErrorCode.SCOPE_MISMATCH, "MARKET_DATA_SCOPE_MISMATCH"),
    ],
)
async def test_market_tools_map_failures_without_exception_or_provider_text(
    source_code: MarketDataErrorCode, public_code: str
) -> None:
    """External/provider details must not cross the MCP error boundary."""
    provider_detail = "secret provider response token=leak"
    server, _, _ = _market_server(
        lambda: RecordingGateway(MarketDataError(source_code, provider_detail))
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "get_market_snapshot", {"ticker": "NVDA", "market": "US"}
        )

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == public_code
    assert provider_detail not in str(result.structured_content)
    assert "token=leak" not in str(result.structured_content)


async def test_market_bars_tool_preserves_configuration_missing_and_never_persists() -> None:
    """A bars auth/config failure must stay fatal at the MCP boundary and write no local row."""
    server, engine, _ = _market_server(
        lambda: RecordingGateway(
            MarketDataError(
                MarketDataErrorCode.CONFIGURATION_MISSING,
                "provider entitlement failed token=leak",
            )
        )
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "get_market_bars", {"ticker": "NVDA", "interval": "1Day", "limit": 5}
        )

    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == "MARKET_DATA_CONFIGURATION_MISSING"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketBarRecord)) == 0


async def test_market_bars_tool_rejects_future_bars_before_persistence() -> None:
    """Cached or injected future bars must surface a stable stale code and write no rows."""

    class FutureBarsGateway:
        async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
            raise AssertionError(symbol)

        async def fetch_bars(self, symbol: str) -> list[MarketBar]:
            assert symbol == "NVDA"
            future_fetched_at = datetime(2099, 1, 1, tzinfo=UTC)
            return [
                _bar().model_copy(
                    update={
                        "id": _observation_id(
                            "bar",
                            symbol="NVDA",
                            source_timestamp=_bar().timestamp,
                            fetched_at=future_fetched_at,
                            raw_payload_hash="c" * 64,
                        ),
                        "fetched_at": future_fetched_at,
                        "raw_payload_hash": "c" * 64,
                    }
                )
            ]

    server, engine, _ = _market_server(lambda: FutureBarsGateway())
    async with Client(server) as client:
        result = await client.call_tool(
            "get_market_bars", {"ticker": "NVDA", "interval": "1Day", "limit": 5}
        )

    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == "STALE_MARKET_DATA"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketBarRecord)) == 0


@pytest.mark.parametrize(
    ("name", "arguments", "factory_error", "public_code"),
    [
        (
            "get_market_snapshot",
            {"ticker": "NVDA", "market": "US"},
            MarketDataError(MarketDataErrorCode.RATE_LIMITED, "secret constructor detail"),
            "MARKET_DATA_RATE_LIMITED",
        ),
        (
            "get_market_bars",
            {"ticker": "NVDA", "interval": "1Day", "limit": 5},
            MarketDataError(MarketDataErrorCode.SCOPE_MISMATCH, "secret constructor detail"),
            "MARKET_DATA_SCOPE_MISMATCH",
        ),
        (
            "get_market_snapshot",
            {"ticker": "NVDA", "market": "US"},
            RuntimeError("provider constructor token=leak"),
            "MARKET_DATA_UNAVAILABLE",
        ),
        (
            "get_market_bars",
            {"ticker": "NVDA", "interval": "1Day", "limit": 5},
            ValueError("provider configuration token=leak"),
            "MARKET_DATA_UNAVAILABLE",
        ),
    ],
)
async def test_market_tools_safely_map_gateway_factory_failures(
    name: str,
    arguments: dict[str, object],
    factory_error: Exception,
    public_code: str,
) -> None:
    """Gateway construction failures must use the same safe envelope as fetch failures."""

    def failing_factory():
        raise factory_error

    server, _, _ = _market_server(failing_factory)
    async with Client(server) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == public_code
    assert "secret" not in str(result.structured_content)
    assert "token=leak" not in str(result.structured_content)


async def test_market_tools_publish_pydantic_output_schemas() -> None:
    """Protocol discovery must advertise strict response envelopes for both tools."""
    server, _, _ = _market_server(lambda: RecordingGateway())

    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for name, payload_field in (
        ("get_market_snapshot", "snapshot"),
        ("get_market_bars", "bars"),
    ):
        tool = tools[name]
        schema = tool.outputSchema
        assert schema is not None
        assert payload_field in schema["properties"]
        assert "error" in schema["properties"]
        assert schema["additionalProperties"] is False
        assert "(?!^[-+.]*$)" not in str(schema)
        if tool.annotations is not None:
            assert tool.annotations.readOnlyHint is not True


def test_run_scoped_market_fetch_writer_persists_only_safe_metadata() -> None:
    """Provider attempts bind to the application run without payloads or credentials."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    repository = ResearchRunRepository(engine)
    repository.start(RunStart(run_id="run-market", ticker="NVDA", request="snapshot"))
    writer = RunScopedMarketFetchWriter(repository, run_id="run-market")

    writer.record_fetch(
        MarketFetchWrite(
            provider="alpaca",
            feed="iex",
            symbol="NVDA",
            operation="snapshot",
            requested_at=NOW - timedelta(seconds=1),
            fetched_at=NOW,
            status="completed",
        )
    )

    stored = repository.get("run-market").source_fetches
    assert len(stored) == 1
    assert stored[0].run_id == "run-market"
    assert stored[0].source_kind == "market_snapshot"
    assert stored[0].source_ref == "alpaca:iex:NVDA:snapshot"
    assert "secret" not in stored[0].model_dump_json()
