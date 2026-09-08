"""Canonical validation shared by market tool, workflow, and reporting boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import ValidationError

from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
    market_bar_observation_id,
    market_snapshot_observation_id,
)

_SUPPORTED_EXCHANGES = frozenset({"AMEX", "ARCA", "IEX", "NASDAQ", "NYSE"})


def canonical_market_snapshot(
    value: object,
    *,
    ticker: str,
    now: datetime | None = None,
) -> MarketSnapshot | None:
    """Return one fully canonical snapshot, or reject untrusted model-bypassed values."""
    if not isinstance(value, MarketSnapshot):
        return None
    fields = tuple(MarketSnapshot.model_fields)
    values = vars(value)
    if any(field not in values for field in fields):
        return None
    try:
        snapshot = MarketSnapshot.model_validate(
            {field: values[field] for field in fields}, strict=True
        )
    except (TypeError, ValidationError, ValueError):
        return None
    observed_at = now.astimezone(UTC) if now is not None else None
    if (
        snapshot.provider != "alpaca"
        or snapshot.feed != "iex"
        or snapshot.coverage != "IEX-only"
        or snapshot.symbol != ticker
        or snapshot.exchange not in _SUPPORTED_EXCHANGES
        or snapshot.currency != "USD"
        or (observed_at is not None and snapshot.as_of > observed_at)
        or (observed_at is not None and snapshot.fetched_at > observed_at)
        or snapshot.id
        != market_snapshot_observation_id(
            provider=snapshot.provider,
            feed=snapshot.feed,
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            fetched_at=snapshot.fetched_at,
            raw_payload_hash=snapshot.raw_payload_hash,
        )
        or any(
            amount <= Decimal(0)
            for amount in (
                snapshot.price,
                snapshot.open,
                snapshot.day_high,
                snapshot.day_low,
                snapshot.previous_close,
            )
        )
    ):
        return None
    expected_delay = max(0, int((snapshot.fetched_at - snapshot.as_of).total_seconds()))
    if snapshot.delayed_by_seconds != expected_delay:
        return None
    return snapshot


def canonical_market_bars(
    value: object,
    *,
    ticker: str,
    max_bars: int,
    now: datetime | None = None,
    snapshot: MarketSnapshot | None = None,
) -> list[MarketBar]:
    """Return canonical bars or raise a typed error safe for tool and guard boundaries."""
    if not isinstance(value, list):
        raise MarketDataError(
            MarketDataErrorCode.UNAVAILABLE,
            "market bars payload was unavailable",
        )
    if len(value) > max_bars:
        raise MarketDataError(
            MarketDataErrorCode.SCOPE_MISMATCH,
            "market bars exceeded the configured bar budget",
        )
    observed_at = now.astimezone(UTC) if now is not None else None
    expected_scope = (
        None
        if snapshot is None
        else (
            snapshot.provider,
            snapshot.feed,
            snapshot.coverage,
            snapshot.symbol,
            snapshot.exchange,
            snapshot.currency,
        )
    )
    bars: list[MarketBar] = []
    fields = tuple(MarketBar.model_fields)
    for raw_bar in value:
        if not isinstance(raw_bar, MarketBar):
            raise MarketDataError(
                MarketDataErrorCode.UNAVAILABLE,
                "market bars payload was malformed",
            )
        values = vars(raw_bar)
        if any(field not in values for field in fields):
            raise MarketDataError(
                MarketDataErrorCode.UNAVAILABLE,
                "market bars payload was malformed",
            )
        try:
            bar = MarketBar.model_validate(
                {field: values[field] for field in fields}, strict=True
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise MarketDataError(
                MarketDataErrorCode.UNAVAILABLE,
                "market bars payload was malformed",
            ) from error
        if (
            bar.provider != "alpaca"
            or bar.feed != "iex"
            or bar.coverage != "IEX-only"
            or bar.symbol != ticker
            or bar.exchange not in _SUPPORTED_EXCHANGES
            or bar.currency != "USD"
            or bar.interval != "1Day"
            or bar.id
            != market_bar_observation_id(
                provider=bar.provider,
                feed=bar.feed,
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                fetched_at=bar.fetched_at,
                raw_payload_hash=bar.raw_payload_hash,
            )
            or any(value <= Decimal(0) for value in (bar.open, bar.high, bar.low, bar.close))
        ):
            raise MarketDataError(
                MarketDataErrorCode.SCOPE_MISMATCH,
                "market bars did not match the supported scope",
            )
        if expected_scope is not None and (
            (
                bar.provider,
                bar.feed,
                bar.coverage,
                bar.symbol,
                bar.exchange,
                bar.currency,
            )
            != expected_scope
        ):
            raise MarketDataError(
                MarketDataErrorCode.SCOPE_MISMATCH,
                "market bars did not match the snapshot scope",
            )
        if bar.timestamp > bar.fetched_at or (
            observed_at is not None
            and (bar.timestamp > observed_at or bar.fetched_at > observed_at)
        ):
            raise MarketDataError(
                MarketDataErrorCode.STALE,
                "market bars did not satisfy the trusted observation window",
            )
        bars.append(bar)
    timestamps = [bar.timestamp for bar in bars]
    if len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps):
        raise MarketDataError(
            MarketDataErrorCode.SCOPE_MISMATCH,
            "market bars must be unique and chronological",
        )
    return bars


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
