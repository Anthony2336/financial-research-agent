"""Fixed Markdown rendering for guarded Alpaca Basic IEX-only evidence."""

from datetime import UTC, datetime
from html import escape

from pydantic import ValidationError

from fra.reporting.market_guard import GuardedMarketReport

DISCLAIMER = "Research assistance only; not investment advice."
_PUBLIC_ERRORS = (
    "MARKET_DATA_CONFIGURATION_MISSING",
    "MARKET_DATA_UNAVAILABLE",
    "MARKET_DATA_RATE_LIMITED",
    "STALE_MARKET_DATA",
    "MARKET_DATA_SCOPE_MISMATCH",
)
_FAILED_MESSAGES = {
    "MARKET_DATA_CONFIGURATION_MISSING": "Market data configuration is missing or unauthorized.",
    "MARKET_DATA_UNAVAILABLE": "Market data is unavailable.",
    "MARKET_DATA_RATE_LIMITED": "Market data is temporarily rate limited.",
    "STALE_MARKET_DATA": "Market data did not meet freshness requirements.",
    "MARKET_DATA_SCOPE_MISMATCH": "Market data did not match the requested scope.",
}


def render_market_markdown(report: GuardedMarketReport) -> str:
    """Render only canonical guarded values with one fixed disclaimer."""
    guarded = _canonical_report(report)
    if guarded is None or guarded.snapshot is None:
        ticker = _safe_ticker(getattr(report, "ticker", "UNKNOWN"))
        return _failed_markdown(ticker, _first_error_code(getattr(report, "errors", None)))

    snapshot = guarded.snapshot
    freshness = {
        "open-iex": "Open-market IEX observation",
        "latest-available-iex": "Latest available IEX trade while market is closed",
        "market-status-unknown": "Latest available IEX trade; market status unknown",
    }[guarded.freshness_label]
    delay = (
        "unknown"
        if snapshot.delayed_by_seconds is None
        else f"{snapshot.delayed_by_seconds} seconds"
    )
    lines = [
        f"# IEX-only market snapshot — {guarded.ticker}",
        "",
        f"- Status: {guarded.status}",
        "- Provider: Alpaca",
        "- Feed: IEX",
        "- Coverage: IEX-only",
        f"- Exchange: {snapshot.exchange}",
        f"- Currency: {snapshot.currency}",
        f"- Price: {snapshot.price} {snapshot.currency}",
        f"- Open: {snapshot.open} {snapshot.currency}",
        f"- Day high: {snapshot.day_high} {snapshot.currency}",
        f"- Day low: {snapshot.day_low} {snapshot.currency}",
        f"- Previous close: {snapshot.previous_close} {snapshot.currency}",
        f"- As of: {_iso_z(snapshot.as_of)}",
        f"- Fetched at: {_iso_z(snapshot.fetched_at)}",
        f"- Market status: {snapshot.market_status.value}",
        f"- Delay: {delay}",
        f"- Freshness: {freshness}",
        f"- Snapshot source ID: {snapshot.id}",
        "",
        "## Daily bars (IEX-only)",
        "",
    ]
    if guarded.bars:
        lines.extend(
            [
                "| Timestamp | Open | High | Low | Close | Volume | Source ID |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                *(
                    "| "
                    f"{_iso_z(bar.timestamp)} | {bar.open} | {bar.high} | {bar.low} | "
                    f"{bar.close} | {bar.volume:,} | {bar.id} |"
                    for bar in guarded.bars
                ),
            ]
        )
    else:
        lines.append("No daily IEX bars retained.")
    lines.extend(
        [
            "",
            "## Errors",
            "",
            *([f"- {error}" for error in guarded.errors] or ["- None."]),
            "",
            "## Information gaps",
            "",
            *(
                [f"- {gap}" for gap in guarded.information_gaps]
                or ["- None identified by the market-data guard."]
            ),
        ]
    )
    if guarded.market_context is not None:
        context = guarded.market_context
        lines.extend(["", "## Context assessment", ""])
        if context.cause_assessment == "possibly_related":
            lines.append(
                "- Assessment: possibly_related. One or more authoritative events fell "
                "inside the snapshot window; this is time adjacency only; not a causal claim."
            )
        else:
            lines.append(
                "- Assessment: cause_unknown. No authoritative event with adequate "
                "publication-time evidence was retained."
            )
        lines.extend(["", "## Time-adjacent events (not causal)", ""])
        if context.events:
            lines.extend(
                f"- {_controlled_text(event.title)} — {_controlled_text(event.summary)} "
                f"(source ID: {event.source_ref}; URL: {event.source_url}; "
                f"published: {_iso_z(event.published_at)}; fetched: {_iso_z(event.fetched_at)}; "
                f"relationship: {event.relationship})"
                for event in context.events
            )
        else:
            lines.append("- No retained event.")
        lines.extend(["", "## Counterevidence", ""])
        lines.extend([f"- {item}" for item in context.counterevidence] or ["- None retained."])
        lines.extend(["", "## Open questions", ""])
        lines.extend([f"- {item}" for item in context.open_questions] or ["- None."])
    lines.extend(["", DISCLAIMER])
    return "\n".join(lines) + "\n"


def _canonical_report(value: object) -> GuardedMarketReport | None:
    if not isinstance(value, GuardedMarketReport):
        return None
    try:
        return GuardedMarketReport.model_validate(value.model_dump(), strict=True)
    except (TypeError, ValidationError, ValueError):
        return None


def _failed_markdown(ticker: str, error_code: str) -> str:
    return "\n".join(
        [
            f"# IEX-only market snapshot — {ticker}",
            "",
            "- Status: failed",
            "- Provider: Alpaca",
            "- Feed: IEX",
            "- Coverage: IEX-only",
            "",
            _FAILED_MESSAGES[error_code],
            "",
            "## Errors",
            "",
            f"- {error_code}",
            "",
            "## Information gaps",
            "",
            "- No valid IEX-only snapshot was retained.",
            "",
            DISCLAIMER,
            "",
        ]
    )


def _first_error_code(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            if item in _PUBLIC_ERRORS:
                return item
    return "MARKET_DATA_UNAVAILABLE"


def _safe_ticker(value: object) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    normalized = value.strip().upper()
    is_safe = (
        normalized.isascii()
        and normalized.replace(".", "").replace("-", "").isalnum()
    )
    return normalized if is_safe else "UNKNOWN"


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _controlled_text(value: str) -> str:
    """Escape retained text so it cannot introduce renderer-controlled Markdown."""
    return escape(value, quote=False).replace("\n", " ")
