"""Bounded HTTP adapter for Alpaca Basic's fixed IEX market-data scope."""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Literal, cast

import httpx
from pydantic import ValidationError

from financial_evidence_agent.market_data.gateway import (
    MarketFetchWrite,
    MarketFetchWriter,
    NoopMarketFetchWriter,
)
from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
    MarketStatus,
    market_bar_observation_id,
    market_snapshot_observation_id,
)

_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_MAX_BARS = 20


@dataclass(frozen=True)
class _RequestSuccess:
    payload: dict[str, object]
    requested_at: datetime
    fetched_at: datetime


class AlpacaMarketDataProvider:
    """Fetch and normalize one IEX snapshot or bounded daily-bars response."""

    DATA_BASE_URL = "https://data.alpaca.markets"
    PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_TRADING_BASE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        client: httpx.AsyncClient,
        trading_environment: Literal["paper", "live"] = "paper",
        timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
        fetch_writer: MarketFetchWriter | None = None,
    ) -> None:
        if trading_environment not in {"paper", "live"}:
            raise ValueError("trading_environment must be paper or live")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not key_id or not secret_key:
            raise MarketDataError(
                MarketDataErrorCode.CONFIGURATION_MISSING,
                "Alpaca market-data credentials are required",
            )
        self._key_id = key_id
        self._secret_key = secret_key
        self._client = client
        self._trading_environment = trading_environment
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._fetch_writer = fetch_writer or NoopMarketFetchWriter()

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Fetch the required snapshot and a concurrent, degradable market clock."""
        normalized_symbol = _validate_symbol(symbol)
        snapshot_result, clock_result = await asyncio.gather(
            self._request_json(
                f"{self.DATA_BASE_URL}/v2/stocks/{normalized_symbol}/snapshot",
                symbol=normalized_symbol,
                operation="snapshot",
                params={"feed": "iex"},
            ),
            self._request_json(
                f"{_trading_base_url(self._trading_environment)}/v2/clock",
                symbol=normalized_symbol,
                operation="clock",
            ),
            return_exceptions=True,
        )
        market_status = MarketStatus.UNKNOWN
        clock_error: MarketDataError | None = None
        if isinstance(clock_result, BaseException):
            clock_error = _safe_error(clock_result)
        else:
            try:
                market_status = _market_status(clock_result.payload)
            except ValueError:
                clock_error = MarketDataError(
                    MarketDataErrorCode.UNAVAILABLE,
                    "market provider clock was unavailable",
                )
                self._record_fetch(
                    symbol=normalized_symbol,
                    operation="clock",
                    requested_at=clock_result.requested_at,
                    fetched_at=None,
                    status="failed",
                    error_code=clock_error.code.value,
                )
            else:
                self._record_fetch(
                    symbol=normalized_symbol,
                    operation="clock",
                    requested_at=clock_result.requested_at,
                    fetched_at=clock_result.fetched_at,
                    status="completed",
                    error_code=None,
                )

        if isinstance(snapshot_result, BaseException):
            raise _safe_error(snapshot_result)

        try:
            snapshot = _normalize_snapshot(
                normalized_symbol,
                snapshot_result.payload,
                market_status=market_status,
                fetched_at=snapshot_result.fetched_at,
            )
        except MarketDataError as error:
            self._record_fetch(
                symbol=normalized_symbol,
                operation="snapshot",
                requested_at=snapshot_result.requested_at,
                fetched_at=None,
                status="failed",
                error_code=error.code.value,
            )
            raise
        self._record_fetch(
            symbol=normalized_symbol,
            operation="snapshot",
            requested_at=snapshot_result.requested_at,
            fetched_at=snapshot_result.fetched_at,
            status="completed",
            error_code=None,
        )
        if clock_error is not None and clock_error.code in {
            MarketDataErrorCode.CONFIGURATION_MISSING,
            MarketDataErrorCode.RATE_LIMITED,
        }:
            raise clock_error
        return snapshot

    async def get_bars(
        self,
        symbol: str,
        *,
        interval: Literal["1Day"],
        limit: int,
    ) -> list[MarketBar]:
        """Fetch at most twenty chronological IEX daily bars."""
        normalized_symbol = _validate_symbol(symbol)
        if interval != "1Day":
            raise ValueError("interval must be 1Day")
        if not 1 <= limit <= _MAX_BARS:
            raise ValueError(f"limit must be between 1 and {_MAX_BARS}")
        result = await self._request_json(
            f"{self.DATA_BASE_URL}/v2/stocks/{normalized_symbol}/bars",
            symbol=normalized_symbol,
            operation="bars",
            params={"feed": "iex", "timeframe": "1Day", "limit": str(limit)},
        )
        try:
            bars = _normalize_bars(
                normalized_symbol,
                result.payload,
                limit=limit,
                fetched_at=result.fetched_at,
            )
        except MarketDataError as error:
            self._record_fetch(
                symbol=normalized_symbol,
                operation="bars",
                requested_at=result.requested_at,
                fetched_at=None,
                status="failed",
                error_code=error.code.value,
            )
            raise
        self._record_fetch(
            symbol=normalized_symbol,
            operation="bars",
            requested_at=result.requested_at,
            fetched_at=result.fetched_at,
            status="completed",
            error_code=None,
        )
        return bars

    async def _request_json(
        self,
        url: str,
        *,
        symbol: str,
        operation: Literal["snapshot", "clock", "bars"],
        params: dict[str, str] | None = None,
    ) -> _RequestSuccess:
        requested_at = _utc_now(self._clock)
        try:
            response = await self._client.get(
                url,
                params=params,
                headers={
                    "APCA-API-KEY-ID": self._key_id,
                    "APCA-API-SECRET-KEY": self._secret_key,
                },
                timeout=self._timeout_seconds,
            )
            _raise_for_status(response.status_code)
            payload = response.json()
            if not isinstance(payload, dict) or not payload:
                raise MarketDataError(
                    MarketDataErrorCode.UNAVAILABLE,
                    "market provider returned an empty or malformed payload",
                )
        except MarketDataError as error:
            self._record_fetch(
                symbol=symbol,
                operation=operation,
                requested_at=requested_at,
                fetched_at=None,
                status="failed",
                error_code=error.code.value,
            )
            raise
        except (httpx.TimeoutException, httpx.HTTPError, ValueError) as error:
            mapped = MarketDataError(
                MarketDataErrorCode.UNAVAILABLE,
                "market provider request was unavailable",
            )
            self._record_fetch(
                symbol=symbol,
                operation=operation,
                requested_at=requested_at,
                fetched_at=None,
                status="failed",
                error_code=mapped.code.value,
            )
            raise mapped from error
        return _RequestSuccess(
            payload=cast(dict[str, object], payload),
            requested_at=requested_at,
            fetched_at=_utc_now(self._clock),
        )

    def _record_fetch(
        self,
        *,
        symbol: str,
        operation: Literal["snapshot", "clock", "bars"],
        requested_at: datetime,
        fetched_at: datetime | None,
        status: Literal["completed", "failed"],
        error_code: str | None,
    ) -> None:
        self._fetch_writer.record_fetch(
            MarketFetchWrite(
                provider="alpaca",
                feed="iex",
                symbol=symbol,
                operation=operation,
                requested_at=requested_at,
                fetched_at=fetched_at,
                status=status,
                error_code=error_code,
            )
        )


def _trading_base_url(environment: Literal["paper", "live"]) -> str:
    if environment == "paper":
        return AlpacaMarketDataProvider.PAPER_TRADING_BASE_URL
    if environment == "live":
        return AlpacaMarketDataProvider.LIVE_TRADING_BASE_URL
    raise ValueError("trading_environment must be paper or live")


def _raise_for_status(status_code: int) -> None:
    if status_code in {401, 403}:
        raise MarketDataError(
            MarketDataErrorCode.CONFIGURATION_MISSING,
            "market provider authentication or entitlement failed",
        )
    if status_code == 429:
        raise MarketDataError(
            MarketDataErrorCode.RATE_LIMITED,
            "market provider request was rate limited",
        )
    if status_code == 404 or status_code >= 500:
        raise MarketDataError(
            MarketDataErrorCode.UNAVAILABLE,
            "market provider data was unavailable",
        )
    if not 200 <= status_code < 300:
        raise MarketDataError(
            MarketDataErrorCode.UNAVAILABLE,
            "market provider returned an unsupported response",
        )


def _normalize_snapshot(
    symbol: str,
    payload: dict[str, object],
    *,
    market_status: MarketStatus,
    fetched_at: datetime,
) -> MarketSnapshot:
    try:
        if payload.get("symbol") != symbol:
            raise ValueError("snapshot symbol mismatch")
        latest_trade = _mapping(payload, "latestTrade")
        daily_bar = _mapping(payload, "dailyBar")
        previous_bar = _mapping(payload, "prevDailyBar")
        as_of = _utc_timestamp(latest_trade["t"])
        raw_payload_hash = _payload_hash(payload)
        return MarketSnapshot(
            id=market_snapshot_observation_id(
                provider="alpaca",
                feed="iex",
                symbol=symbol,
                as_of=as_of,
                fetched_at=fetched_at,
                raw_payload_hash=raw_payload_hash,
            ),
            provider="alpaca",
            feed="iex",
            coverage="IEX-only",
            symbol=symbol,
            exchange=_exchange(latest_trade["x"]),
            currency="USD",
            price=_decimal(latest_trade["p"]),
            open=_decimal(daily_bar["o"]),
            day_high=_decimal(daily_bar["h"]),
            day_low=_decimal(daily_bar["l"]),
            previous_close=_decimal(previous_bar["c"]),
            as_of=as_of,
            fetched_at=fetched_at,
            market_status=market_status,
            delayed_by_seconds=max(0, int((fetched_at - as_of).total_seconds())),
            raw_payload_hash=raw_payload_hash,
        )
    except (KeyError, TypeError, ValueError, InvalidOperation, ValidationError) as error:
        raise MarketDataError(
            MarketDataErrorCode.UNAVAILABLE,
            "market provider returned malformed snapshot data",
        ) from error


def _normalize_bars(
    symbol: str,
    payload: dict[str, object],
    *,
    limit: int,
    fetched_at: datetime,
) -> list[MarketBar]:
    try:
        if payload.get("symbol") != symbol:
            raise ValueError("bars symbol mismatch")
        raw_bars = payload["bars"]
        if not isinstance(raw_bars, list) or not raw_bars:
            raise ValueError("bars are empty")
        if len(raw_bars) > limit:
            raise ValueError("bars exceed requested limit")
        bars: list[MarketBar] = []
        for value in raw_bars:
            if not isinstance(value, dict):
                raise TypeError("bar is not an object")
            timestamp = _utc_timestamp(value["t"])
            raw_payload_hash = _payload_hash(value)
            bars.append(
                MarketBar(
                    id=market_bar_observation_id(
                        provider="alpaca",
                        feed="iex",
                        symbol=symbol,
                        timestamp=timestamp,
                        fetched_at=fetched_at,
                        raw_payload_hash=raw_payload_hash,
                    ),
                    provider="alpaca",
                    feed="iex",
                    coverage="IEX-only",
                    symbol=symbol,
                    exchange="IEX",
                    currency="USD",
                    interval="1Day",
                    timestamp=timestamp,
                    open=_decimal(value["o"]),
                    high=_decimal(value["h"]),
                    low=_decimal(value["l"]),
                    close=_decimal(value["c"]),
                    volume=value["v"],
                    fetched_at=fetched_at,
                    raw_payload_hash=raw_payload_hash,
                )
            )
        normalized = sorted(bars, key=lambda bar: bar.timestamp)
        timestamps = [bar.timestamp for bar in normalized]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("bars contain duplicate timestamps")
        return normalized
    except (KeyError, TypeError, ValueError, InvalidOperation, ValidationError) as error:
        raise MarketDataError(
            MarketDataErrorCode.UNAVAILABLE,
            "market provider returned malformed bars data",
        ) from error


def _market_status(payload: dict[str, object]) -> MarketStatus:
    is_open = payload.get("is_open")
    if is_open is True:
        return MarketStatus.OPEN
    if is_open is False:
        return MarketStatus.CLOSED
    raise ValueError("clock is_open must be boolean")


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{key} is not an object")
    return value


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise TypeError("market value is not numeric")
    return Decimal(str(value))


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("market timestamp is not text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("market timestamp is missing a timezone")
    return parsed.astimezone(UTC)


def _utc_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _exchange(value: object) -> str:
    if value != "V":
        raise ValueError("snapshot trade is outside the IEX venue")
    return "IEX"


def _payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if _SYMBOL_PATTERN.fullmatch(normalized) is None:
        raise ValueError("symbol must be a valid ticker")
    return normalized


def _safe_error(error: BaseException) -> MarketDataError:
    if isinstance(error, MarketDataError):
        return error
    return MarketDataError(
        MarketDataErrorCode.UNAVAILABLE,
        "market provider request was unavailable",
    )
