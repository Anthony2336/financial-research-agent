"""Strict MCP envelopes for normalized market snapshots and bars."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol

from fastmcp import FastMCP
from pydantic import ConfigDict, Field, StrictInt, model_validator

from fra.domain import StrictModel
from fra.market_data.gateway import MarketFetchWrite
from fra.market_data.models import (
    MarketBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
)
from fra.market_data.validation import (
    canonical_market_bars,
    canonical_market_snapshot,
)
from fra.storage.market_repositories import MarketDataRepository
from fra.storage.run_repositories import SourceFetchWrite

_TICKER_PATTERN = r"^[A-Z][A-Z0-9.-]{0,9}$"
_PYDANTIC_DECIMAL_PATTERN = r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$"
_MCP_DECIMAL_PATTERN = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$"
_PUBLIC_ERROR_CODE = Literal[
    "MARKET_DATA_CONFIGURATION_MISSING",
    "MARKET_DATA_UNAVAILABLE",
    "MARKET_DATA_RATE_LIMITED",
    "STALE_MARKET_DATA",
    "MARKET_DATA_SCOPE_MISMATCH",
]


class MarketGateway(Protocol):
    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot: ...

    async def fetch_bars(self, symbol: str) -> list[MarketBar]: ...


class SourceFetchWriter(Protocol):
    def record_fetch(self, value: SourceFetchWrite) -> None: ...


MarketGatewayFactory = Callable[[], MarketGateway | None]


class GetMarketSnapshotInput(StrictModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True
    )

    ticker: str = Field(min_length=1, max_length=10, pattern=_TICKER_PATTERN)
    market: Literal["US"] = "US"


class GetMarketBarsInput(StrictModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True
    )

    ticker: str = Field(min_length=1, max_length=10, pattern=_TICKER_PATTERN)
    interval: Literal["1Day"] = "1Day"
    limit: int = Field(ge=1, le=20)


class MarketToolError(StrictModel):
    """Stable caller-safe recovery information with no provider detail."""

    code: _PUBLIC_ERROR_CODE
    message: str = Field(min_length=1, max_length=100)


class GetMarketSnapshotResponse(StrictModel):
    snapshot: MarketSnapshot | None = None
    error: MarketToolError | None = None

    @model_validator(mode="after")
    def require_one_outcome(self) -> "GetMarketSnapshotResponse":
        if (self.snapshot is None) == (self.error is None):
            raise ValueError("exactly one snapshot outcome is required")
        return self


class GetMarketBarsResponse(StrictModel):
    bars: list[MarketBar] | None = None
    error: MarketToolError | None = None

    @model_validator(mode="after")
    def require_one_outcome(self) -> "GetMarketBarsResponse":
        if (self.bars is None) == (self.error is None):
            raise ValueError("exactly one bars outcome is required")
        return self


class RunScopedMarketFetchWriter:
    """Adapt safe provider-attempt metadata to application run provenance."""

    def __init__(self, writer: SourceFetchWriter, *, run_id: str) -> None:
        self._writer = writer
        self._run_id = run_id

    def record_fetch(self, value: MarketFetchWrite) -> None:
        self._writer.record_fetch(
            SourceFetchWrite(
                run_id=self._run_id,
                source_kind=f"market_{value.operation}",
                source_ref=(
                    f"{value.provider}:{value.feed}:{value.symbol}:{value.operation}"
                ),
                requested_at=value.requested_at,
                fetched_at=value.fetched_at,
                status=value.status,
                error_code=value.error_code,
            )
        )


def register_market_tools(
    mcp: FastMCP,
    gateway_factory: MarketGatewayFactory | None,
    repository: MarketDataRepository,
    *,
    max_bars: int,
) -> None:
    """Register bounded tools while constructing a fresh run-memo gateway per call."""

    if not 1 <= max_bars <= 20:
        raise ValueError("max_bars must be between 1 and 20")

    @mcp.tool(
        name="get_market_snapshot",
        output_schema=_mcp_output_schema(GetMarketSnapshotResponse),
        annotations={"readOnlyHint": False},
    )
    async def get_market_snapshot(
        ticker: Annotated[
            str,
            Field(min_length=1, max_length=10, pattern=_TICKER_PATTERN),
        ],
        market: Literal["US"] = "US",
    ) -> GetMarketSnapshotResponse:
        """Fetch and persist one normalized Alpaca IEX snapshot for a US ticker."""
        request = GetMarketSnapshotInput(ticker=ticker, market=market)
        try:
            gateway = gateway_factory() if gateway_factory is not None else None
            if gateway is None:
                return _snapshot_error(MarketDataErrorCode.UNAVAILABLE)
            snapshot = await gateway.fetch_snapshot(request.ticker)
            observed_at = datetime.now(UTC)
            snapshot = canonical_market_snapshot(
                snapshot,
                ticker=request.ticker,
                now=observed_at,
            )
            if snapshot is None:
                return _snapshot_error(MarketDataErrorCode.SCOPE_MISMATCH)
            repository.save_snapshot(snapshot)
        except MarketDataError as error:
            return _snapshot_error(error.code)
        except Exception:
            return _snapshot_error(MarketDataErrorCode.UNAVAILABLE)
        return GetMarketSnapshotResponse(snapshot=snapshot)

    @mcp.tool(
        name="get_market_bars",
        output_schema=_mcp_output_schema(GetMarketBarsResponse),
        annotations={"readOnlyHint": False},
    )
    async def get_market_bars(
        ticker: Annotated[
            str,
            Field(min_length=1, max_length=10, pattern=_TICKER_PATTERN),
        ],
        interval: Literal["1Day"] = "1Day",
        limit: Annotated[StrictInt, Field(ge=1, le=max_bars)] = max_bars,
    ) -> GetMarketBarsResponse:
        """Fetch and persist bounded chronological Alpaca IEX daily bars."""
        request = GetMarketBarsInput(ticker=ticker, interval=interval, limit=limit)
        if request.limit > max_bars:
            raise ValueError("limit exceeds configured market-data maximum")
        try:
            gateway = gateway_factory() if gateway_factory is not None else None
            if gateway is None:
                return _bars_error(MarketDataErrorCode.UNAVAILABLE)
            bars = await gateway.fetch_bars(request.ticker)
            bars = bars[-request.limit :]
            if not bars:
                raise MarketDataError(
                    MarketDataErrorCode.UNAVAILABLE,
                    "market data unavailable",
                )
            bars = canonical_market_bars(
                bars,
                ticker=request.ticker,
                max_bars=request.limit,
                now=datetime.now(UTC),
            )
            repository.save_bars(bars)
        except MarketDataError as error:
            return _bars_error(error.code)
        except Exception:
            return _bars_error(MarketDataErrorCode.UNAVAILABLE)
        return GetMarketBarsResponse(bars=bars)


def _snapshot_error(code: MarketDataErrorCode) -> GetMarketSnapshotResponse:
    return GetMarketSnapshotResponse(error=_public_error(code))


def _bars_error(code: MarketDataErrorCode) -> GetMarketBarsResponse:
    return GetMarketBarsResponse(error=_public_error(code))


def _public_error(code: MarketDataErrorCode) -> MarketToolError:
    public_code: _PUBLIC_ERROR_CODE
    if code is MarketDataErrorCode.CONFIGURATION_MISSING:
        public_code = "MARKET_DATA_CONFIGURATION_MISSING"
    elif code is MarketDataErrorCode.RATE_LIMITED:
        public_code = "MARKET_DATA_RATE_LIMITED"
    elif code is MarketDataErrorCode.STALE:
        public_code = "STALE_MARKET_DATA"
    elif code is MarketDataErrorCode.SCOPE_MISMATCH:
        public_code = "MARKET_DATA_SCOPE_MISMATCH"
    else:
        public_code = "MARKET_DATA_UNAVAILABLE"
    return MarketToolError(code=public_code, message=_ERROR_MESSAGES[public_code])


_ERROR_MESSAGES: dict[str, str] = {
    "MARKET_DATA_CONFIGURATION_MISSING": "market data configuration is missing or unauthorized",
    "MARKET_DATA_UNAVAILABLE": "market data is unavailable",
    "MARKET_DATA_RATE_LIMITED": "market data is temporarily rate limited",
    "STALE_MARKET_DATA": "market data did not meet freshness requirements",
    "MARKET_DATA_SCOPE_MISMATCH": "market data did not match the requested scope",
}


def _mcp_output_schema(model: type[StrictModel]) -> dict[str, object]:
    """Publish Pydantic schemas with a FastMCP-compatible Decimal regex."""
    schema = model.model_json_schema()
    _replace_decimal_pattern(schema)
    return schema


def _replace_decimal_pattern(value: object) -> None:
    if isinstance(value, dict):
        if value.get("pattern") == _PYDANTIC_DECIMAL_PATTERN:
            value["pattern"] = _MCP_DECIMAL_PATTERN
        for child in value.values():
            _replace_decimal_pattern(child)
    elif isinstance(value, list):
        for child in value:
            _replace_decimal_pattern(child)
