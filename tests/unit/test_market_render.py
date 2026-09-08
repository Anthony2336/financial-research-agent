"""Exact fixed-Markdown contracts for guarded IEX-only market reports."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketSnapshot,
)
from financial_evidence_agent.reporting.market_guard import (
    GuardedMarketReport,
    guard_market_bundle,
)
from financial_evidence_agent.reporting.market_render import render_market_markdown

NOW = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)
DISCLAIMER = "Research assistance only; not investment advice."


def _snapshot() -> MarketSnapshot:
    as_of = NOW - timedelta(minutes=1)
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


def _guarded_bundle() -> GuardedMarketReport:
    return guard_market_bundle(
        MarketDataBundle(
            snapshot=_snapshot(),
            bars=[_bar()],
            status="completed",
            freshness_label="open-iex",
        ),
        requested_ticker="NVDA",
        max_bars=5,
        now=NOW,
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


def test_market_report_displays_iex_scope_and_time_metadata() -> None:
    """A valid report must state the exact provider scope and observation times."""
    markdown = render_market_markdown(_guarded_bundle())

    assert "Alpaca" in markdown
    assert "IEX-only" in markdown
    assert "Exchange: IEX" in markdown
    assert "Currency: USD" in markdown
    assert "As of: 2026-08-31T14:00:00Z" in markdown
    assert "Fetched at: 2026-08-31T14:01:00Z" in markdown
    assert "Delay: 60 seconds" in markdown
    assert "Freshness: Open-market IEX observation" in markdown
    assert DISCLAIMER in markdown
    assert markdown.count(DISCLAIMER) == 1
    assert "real-time consolidated" not in markdown.casefold()
    assert "realtime" not in markdown.casefold()


def test_market_report_preserves_decimal_values_and_bounded_daily_bars() -> None:
    """Rendering must not round via float or omit the retained bar provenance."""
    markdown = render_market_markdown(_guarded_bundle())

    assert "Price: 123.450000000000000001 USD" in markdown
    assert "Previous close: 121.00 USD" in markdown
    assert "2026-08-28T04:00:00Z" in markdown
    assert "120.00" in markdown
    assert "124.00" in markdown
    assert "119.00" in markdown
    assert "123.45" in markdown
    assert "1,000" in markdown
    assert _snapshot().id in markdown
    assert _bar().id in markdown


def test_partial_market_report_retains_snapshot_and_names_fixed_gap() -> None:
    """A bars failure must leave valid snapshot facts visible with an explicit limitation."""
    guarded = guard_market_bundle(
        MarketDataBundle(
            snapshot=_snapshot(),
            bars=[],
            status="partial",
            freshness_label="open-iex",
            errors=["MARKET_DATA_RATE_LIMITED"],
        ),
        requested_ticker="NVDA",
        max_bars=5,
        now=NOW,
    )

    markdown = render_market_markdown(guarded)

    assert "Status: partial" in markdown
    assert "Price: 123.450000000000000001 USD" in markdown
    assert "MARKET_DATA_RATE_LIMITED" in markdown
    assert "Daily IEX bars were not retained." in markdown
    assert markdown.count(DISCLAIMER) == 1


def test_failed_market_report_never_renders_rejected_facts() -> None:
    """Unavailable output is fixed and cannot leak values or unsafe provider text."""
    guarded = GuardedMarketReport(
        ticker="NVDA",
        status="failed",
        errors=["MARKET_DATA_UNAVAILABLE"],
        information_gaps=["No valid IEX-only snapshot was retained."],
    )

    markdown = render_market_markdown(guarded)

    assert "Status: failed" in markdown
    assert "Provider: Alpaca" in markdown
    assert "Feed: IEX" in markdown
    assert "Coverage: IEX-only" in markdown
    assert "Market data is unavailable." in markdown
    assert "123.45" not in markdown
    assert "market-snapshot:" not in markdown
    assert markdown.count(DISCLAIMER) == 1


def test_failed_market_report_preserves_first_stable_error_code() -> None:
    for code in (
        "MARKET_DATA_CONFIGURATION_MISSING",
        "MARKET_DATA_UNAVAILABLE",
        "MARKET_DATA_RATE_LIMITED",
        "STALE_MARKET_DATA",
        "MARKET_DATA_SCOPE_MISMATCH",
    ):
        markdown = render_market_markdown(
            GuardedMarketReport(
                ticker="NVDA",
                status="failed",
                errors=[code, "MARKET_DATA_UNAVAILABLE"],
                information_gaps=["No valid IEX-only snapshot was retained."],
            )
        )

        assert f"- {code}" in markdown
        assert "token=secret" not in markdown


def test_renderer_fails_closed_when_guarded_provenance_is_construction_bypassed() -> None:
    """A forged guarded model cannot render facts without exact market source refs."""
    forged = _guarded_bundle().model_copy(update={"source_refs": []})

    markdown = render_market_markdown(forged)

    assert "Status: failed" in markdown
    assert "Market data is unavailable." in markdown
    assert "123.450000000000000001" not in markdown
    assert _snapshot().id not in markdown
