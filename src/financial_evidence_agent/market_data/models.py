"""Strict normalized models for the Alpaca Basic IEX market-data boundary."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Literal

from pydantic import Field, field_validator, model_validator

from financial_evidence_agent.domain import SourceKind, SourceTier, StrictModel

_MAX_SIGNED_BIGINT = 9_223_372_036_854_775_807


class MarketStatus(StrEnum):
    """Approved market-status values returned by the provider clock."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class MarketDataErrorCode(StrEnum):
    """Stable market-data failures safe to expose to callers."""

    CONFIGURATION_MISSING = "MARKET_DATA_CONFIGURATION_MISSING"
    UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
    RATE_LIMITED = "MARKET_DATA_RATE_LIMITED"
    STALE = "STALE_MARKET_DATA"
    SCOPE_MISMATCH = "MARKET_DATA_SCOPE_MISMATCH"


class MarketDataError(RuntimeError):
    """Typed market-data failure with a caller-safe detail message."""

    def __init__(self, code: MarketDataErrorCode, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code.value}: {message}")


class _MarketScope(StrictModel):
    """Provider identity and instrument metadata shared by normalized values."""

    provider: Literal["alpaca"]
    feed: Literal["iex"]
    coverage: Literal["IEX-only"]
    symbol: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    exchange: str = Field(min_length=1, max_length=64)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class MarketSnapshot(_MarketScope):
    """One normalized IEX-only latest-trade snapshot."""

    id: str = Field(min_length=1, max_length=512)
    price: Decimal
    open: Decimal
    day_high: Decimal
    day_low: Decimal
    previous_close: Decimal
    as_of: datetime
    fetched_at: datetime
    market_status: MarketStatus
    delayed_by_seconds: int | None = Field(default=None, ge=0)
    raw_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$", frozen=True)

    @field_validator("price", "open", "day_high", "day_low", "previous_close")
    @classmethod
    def require_finite_decimal(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("market Decimal values must be finite")
        return value

    @field_validator("as_of", "fetched_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("market timestamps must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_snapshot_consistency(self) -> "MarketSnapshot":
        if self.as_of > self.fetched_at:
            raise ValueError("as_of must not be later than fetched_at")
        if self.day_low > min(self.open, self.price, self.day_high):
            raise ValueError("day_low must not exceed open, price, or day_high")
        if self.day_high < max(self.open, self.price, self.day_low):
            raise ValueError("day_high must not be below open, price, or day_low")
        return self


class MarketBar(_MarketScope):
    """One normalized daily IEX-only OHLCV bar."""

    id: str = Field(min_length=1, max_length=512)
    interval: Literal["1Day"]
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    fetched_at: datetime
    raw_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$", frozen=True)

    @field_validator("open", "high", "low", "close")
    @classmethod
    def require_finite_decimal(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("market Decimal values must be finite")
        return value

    @field_validator("timestamp", "fetched_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("market timestamps must be timezone-aware UTC")
        return value

    @field_validator("volume", mode="before")
    @classmethod
    def require_nonnegative_integral_volume(cls, value: object) -> int:
        if isinstance(value, bool) or isinstance(value, str):
            raise ValueError("volume must be a nonnegative integral numeric value")
        if isinstance(value, int):
            normalized = value
        elif isinstance(value, float):
            if not isfinite(value) or not value.is_integer():
                raise ValueError("volume must be a nonnegative integral numeric value")
            normalized = int(value)
        elif isinstance(value, Decimal):
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError("volume must be a nonnegative integral numeric value")
            normalized = int(value)
        else:
            raise ValueError("volume must be a nonnegative integral numeric value")
        if normalized < 0 or normalized > _MAX_SIGNED_BIGINT:
            raise ValueError("volume must fit within the supported integer range")
        return normalized

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> "MarketBar":
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must not exceed open, close, or high")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must not be below open, close, or low")
        return self


class MarketDataBundle(StrictModel):
    """One complete or partial normalized market-data retrieval."""

    snapshot: MarketSnapshot
    bars: list[MarketBar]
    status: Literal["completed", "partial"]
    freshness_label: Literal[
        "open-iex", "latest-available-iex", "market-status-unknown"
    ] = "market-status-unknown"
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bars(self) -> "MarketDataBundle":
        previous_timestamp: datetime | None = None
        seen_timestamps: set[datetime] = set()
        for bar in self.bars:
            if (
                bar.provider != self.snapshot.provider
                or bar.feed != self.snapshot.feed
                or bar.symbol != self.snapshot.symbol
                or bar.exchange != self.snapshot.exchange
                or bar.currency != self.snapshot.currency
                or bar.coverage != self.snapshot.coverage
            ):
                raise ValueError("bar scope must match snapshot scope")
            if bar.timestamp in seen_timestamps:
                raise ValueError("bars must have unique timestamps")
            if previous_timestamp is not None and bar.timestamp <= previous_timestamp:
                raise ValueError("bars must be chronological")
            seen_timestamps.add(bar.timestamp)
            previous_timestamp = bar.timestamp
        return self


class MarketEvent(StrictModel):
    """One time-anchored authoritative web event retained for market context."""

    source_ref: str = Field(min_length=1, max_length=512)
    source_url: str = Field(min_length=1, max_length=2_000, pattern=r"^https://")
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=1_200)
    published_at: datetime
    fetched_at: datetime
    relationship: Literal["before_window", "inside_window", "after_window"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: SourceKind
    source_tier: SourceTier
    policy_version: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("published_at", "fetched_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("market context timestamps must be timezone-aware UTC")
        return value


class MarketContext(StrictModel):
    """Strict non-causal market-context material tied to one snapshot timestamp."""

    anchor_as_of: datetime
    window_start: datetime
    window_end: datetime
    events: list[MarketEvent] = Field(default_factory=list, max_length=3)
    counterevidence: list[str] = Field(default_factory=list, max_length=3)
    open_questions: list[str] = Field(default_factory=list, max_length=3)
    cause_assessment: Literal["possibly_related", "cause_unknown"]

    @field_validator("anchor_as_of", "window_start", "window_end")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("market context timestamps must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> "MarketContext":
        if not self.window_start <= self.anchor_as_of <= self.window_end:
            raise ValueError("market context anchor must be inside its window")
        return self


def market_snapshot_observation_id(
    *,
    provider: str,
    feed: str,
    symbol: str,
    as_of: datetime,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> str:
    """Return the opaque immutable ID for one normalized snapshot observation."""
    return _market_observation_id(
        "snapshot",
        provider=provider,
        feed=feed,
        symbol=symbol,
        source_timestamp=as_of,
        fetched_at=fetched_at,
        raw_payload_hash=raw_payload_hash,
    )


def market_bar_observation_id(
    *,
    provider: str,
    feed: str,
    symbol: str,
    timestamp: datetime,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> str:
    """Return the opaque immutable ID for one normalized daily-bar observation."""
    return _market_observation_id(
        "bar",
        provider=provider,
        feed=feed,
        symbol=symbol,
        source_timestamp=timestamp,
        fetched_at=fetched_at,
        raw_payload_hash=raw_payload_hash,
    )


def _market_observation_id(
    kind: Literal["snapshot", "bar"],
    *,
    provider: str,
    feed: str,
    symbol: str,
    source_timestamp: datetime,
    fetched_at: datetime,
    raw_payload_hash: str,
) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "provider": provider,
            "feed": feed,
            "symbol": symbol,
            "source_timestamp": _iso_z(source_timestamp),
            "fetched_at": _iso_z(fetched_at),
            "raw_payload_hash": raw_payload_hash,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"market-{kind}:{sha256(payload).hexdigest()}"


def _iso_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
