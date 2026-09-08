"""Fail-closed validation between normalized market tools and fixed rendering."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from financial_evidence_agent.domain import SourceRef, SourceRefKind, StrictModel, WebEvidence
from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketContext,
    MarketDataBundle,
    MarketDataError,
    MarketEvent,
    MarketSnapshot,
    MarketStatus,
)
from financial_evidence_agent.market_data.validation import (
    canonical_market_bars,
    canonical_market_snapshot,
)
from financial_evidence_agent.reporting.privacy import retain_public_text
from financial_evidence_agent.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    WebEvidenceValidationError,
)

MarketGuardError = Literal[
    "MARKET_DATA_CONFIGURATION_MISSING",
    "MARKET_DATA_UNAVAILABLE",
    "MARKET_DATA_RATE_LIMITED",
    "STALE_MARKET_DATA",
    "MARKET_DATA_SCOPE_MISMATCH",
]
MarketInformationGap = Literal[
    "No valid IEX-only snapshot was retained.",
    "Daily IEX bars were not retained.",
]
FreshnessLabel = Literal[
    "open-iex",
    "latest-available-iex",
    "market-status-unknown",
]

_ALLOWED_ERRORS = frozenset(
    {
        "MARKET_DATA_CONFIGURATION_MISSING",
        "MARKET_DATA_UNAVAILABLE",
        "MARKET_DATA_RATE_LIMITED",
        "STALE_MARKET_DATA",
        "MARKET_DATA_SCOPE_MISMATCH",
    }
)
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


class GuardedMarketReport(StrictModel):
    """Market facts and provenance that passed deterministic output checks."""

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    snapshot: MarketSnapshot | None = None
    bars: list[MarketBar] = Field(default_factory=list, max_length=20)
    status: Literal["completed", "partial", "failed"]
    freshness_label: FreshnessLabel | None = None
    errors: list[MarketGuardError] = Field(default_factory=list)
    information_gaps: list[MarketInformationGap] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    market_context: MarketContext | None = None

    @model_validator(mode="after")
    def require_guarded_shape(self) -> GuardedMarketReport:
        if self.snapshot is None:
            if self.status != "failed":
                raise ValueError("a report without a snapshot must be failed")
            if self.bars or self.source_refs or self.freshness_label is not None:
                raise ValueError("a failed report cannot retain market facts")
        elif self.status == "failed":
            raise ValueError("a failed report cannot retain a snapshot")
        else:
            expected_refs = [
                SourceRef(
                    ticker=self.ticker,
                    kind=SourceRefKind.MARKET_SNAPSHOT,
                    source_id=self.snapshot.id,
                ),
                *(
                    SourceRef(
                        ticker=self.ticker,
                        kind=SourceRefKind.MARKET_BAR,
                        source_id=bar.id,
                    )
                    for bar in self.bars
                ),
            ]
            if self.market_context is not None:
                expected_refs.extend(
                    SourceRef(
                        ticker=self.ticker,
                        kind=SourceRefKind.WEB,
                        source_id=event.source_ref,
                    )
                    for event in self.market_context.events
                )
            if self.source_refs != expected_refs:
                raise ValueError("market source references must match retained facts")
            if self.snapshot.symbol != self.ticker:
                raise ValueError("market snapshot ticker must match report ticker")
            if self.freshness_label != _canonical_freshness(self.snapshot.market_status):
                raise ValueError("market freshness label must match market status")
            if not self.bars and "Daily IEX bars were not retained." not in self.information_gaps:
                raise ValueError("missing market bars require an information gap")
        return self


def guard_market_bundle(
    bundle: MarketDataBundle | None,
    *,
    requested_ticker: str,
    max_bars: int = 5,
    max_staleness_seconds: int = 90,
    now: datetime | None = None,
    upstream_errors: list[str] | None = None,
) -> GuardedMarketReport:
    """Revalidate market scope, values, time, and provenance before rendering."""
    ticker = requested_ticker.strip().upper()
    observed_at = now or datetime.now(UTC)
    if (
        _TICKER.fullmatch(ticker) is None
        or not 1 <= max_bars <= 20
        or max_staleness_seconds <= 0
        or observed_at.tzinfo is None
        or observed_at.utcoffset() is None
    ):
        return _failed_report(ticker if _TICKER.fullmatch(ticker) else "UNKNOWN")
    observed_at = observed_at.astimezone(UTC)

    snapshot = canonical_market_snapshot(
        getattr(bundle, "snapshot", None), ticker=ticker, now=observed_at
    )
    if snapshot is None:
        return _failed_report(ticker, errors=upstream_errors)
    if (
        snapshot.market_status is MarketStatus.OPEN
        and (observed_at - snapshot.as_of).total_seconds() > max_staleness_seconds
    ):
        return _failed_report(ticker, errors=["STALE_MARKET_DATA"])

    freshness_label = _canonical_freshness(snapshot.market_status)
    if (
        getattr(bundle, "freshness_label", None) != freshness_label
        or getattr(bundle, "status", None) not in {"completed", "partial"}
    ):
        return _failed_report(ticker)

    errors = _canonical_errors(getattr(bundle, "errors", None))
    if "MARKET_DATA_CONFIGURATION_MISSING" in errors:
        return _failed_report(ticker, errors=["MARKET_DATA_CONFIGURATION_MISSING"])

    try:
        bars = canonical_market_bars(
            getattr(bundle, "bars", None),
            snapshot=snapshot,
            ticker=ticker,
            max_bars=max_bars,
            now=observed_at,
        )
    except MarketDataError as error:
        bars = []
        errors = _append_once(errors, error.code.value)
    else:
        if not bars and not errors:
            errors = ["MARKET_DATA_UNAVAILABLE"]
    if not bars and not errors:
        errors = ["MARKET_DATA_UNAVAILABLE"]
    if (
        bars
        and snapshot.market_status is MarketStatus.CLOSED
        and snapshot.as_of < bars[-1].timestamp
    ):
        return _failed_report(ticker, errors=["STALE_MARKET_DATA"])

    status: Literal["completed", "partial"] = (
        "completed"
        if getattr(bundle, "status", None) == "completed" and bars and not errors
        else "partial"
    )
    gaps: list[MarketInformationGap] = []
    if not bars:
        gaps.append("Daily IEX bars were not retained.")
    source_refs = [
        SourceRef(
            ticker=ticker,
            kind=SourceRefKind.MARKET_SNAPSHOT,
            source_id=snapshot.id,
        ),
        *(
            SourceRef(
                ticker=ticker,
                kind=SourceRefKind.MARKET_BAR,
                source_id=bar.id,
            )
            for bar in bars
        ),
    ]
    return GuardedMarketReport(
        ticker=ticker,
        snapshot=snapshot,
        bars=bars,
        status=status,
        freshness_label=freshness_label,
        errors=errors,
        information_gaps=gaps,
        source_refs=source_refs,
    )


_UNSAFE_CONTEXT = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|system\s+prompt|"
    r"(?:disregard|override)\s+(?:all\s+)?(?:instructions|safeguards)|"
    r"caus(?:e|ed|es|ing)\b|because\s+of|due\s+to|driven\s+by|"
    r"(?:led|leads)\s+to|will\s+(?:push|raise|lower|rise|fall)|"
    r"(?:shares?|stock)\s+(?:will|are\s+going\s+to)|"
    r"\b(?:buy|sell|hold|target\s+price|portfolio)\b|"
    r"(?:建议|买入|卖出|持有|目标价|投资组合|导致|造成|一定|必然))",
    re.IGNORECASE,
)
_UNSAFE_CONTEXT_MARKUP = re.compile(
    r"(?:^|\n)\s*#|\[[^\]]*\]\([^)]*\)|<\s*/?\s*[a-z!][^>]*>", re.IGNORECASE
)


def guard_market_event(event: MarketEvent, *, ticker: str = "") -> MarketEvent | None:
    """Retain only a neutral, cited event with complete UTC provenance."""
    try:
        guarded = MarketEvent.model_validate(event.model_dump(), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError):
        return None
    text = f"{guarded.title}\n{guarded.summary}"
    if (
        not retain_public_text(text, ticker=ticker)
        or not retain_public_text(guarded.source_ref, ticker=ticker)
        or not retain_public_text(guarded.source_url, ticker=ticker)
        or _UNSAFE_CONTEXT.search(text)
        or _UNSAFE_CONTEXT_MARKUP.search(text)
    ):
        return None
    return guarded


def guard_market_context(
    context: MarketContext | None,
    *,
    snapshot_as_of: datetime,
    window: timedelta,
    ticker: str | None = None,
    validator: PersistedWebEvidenceValidator | None = None,
) -> MarketContext:
    """Fail closed to cause-unknown context outside the snapshot-derived window."""
    anchor = snapshot_as_of.astimezone(UTC)
    start = anchor - window
    retained: list[MarketEvent] = []
    if isinstance(context, MarketContext):
        for event in context.events:
            guarded = guard_market_event(event, ticker=ticker or "")
            if (
                guarded is not None
                and guarded.relationship == "inside_window"
                and start <= guarded.published_at <= anchor
                and _canonical_event(
                    guarded, ticker=ticker, validator=validator
                ) is not None
            ):
                retained.append(guarded)
    retained = retained[:3]
    return MarketContext(
        anchor_as_of=anchor,
        window_start=start,
        window_end=anchor,
        events=retained,
        counterevidence=[],
        open_questions=(
            []
            if retained
            else ["No authoritative event with adequate publication-time evidence was retained."]
        ),
        cause_assessment="possibly_related" if retained else "cause_unknown",
    )


def attach_market_context(
    report: GuardedMarketReport,
    context: MarketContext | None,
    *,
    window: timedelta,
    validator: PersistedWebEvidenceValidator | None = None,
) -> GuardedMarketReport:
    """Attach only snapshot-anchored context to an already guarded market report."""
    if report.snapshot is None:
        return report
    guarded_context = guard_market_context(
        context,
        snapshot_as_of=report.snapshot.as_of,
        window=window,
        ticker=report.ticker,
        validator=validator,
    )
    source_refs = [
        *report.source_refs,
        *(
            SourceRef(
                ticker=report.ticker,
                kind=SourceRefKind.WEB,
                source_id=event.source_ref,
            )
            for event in guarded_context.events
        ),
    ]
    values = report.model_dump()
    values.update(market_context=guarded_context, source_refs=source_refs)
    return GuardedMarketReport.model_validate(values)


def _canonical_event(
    event: MarketEvent,
    *,
    ticker: str | None,
    validator: PersistedWebEvidenceValidator | None,
) -> MarketEvent | None:
    """Require the exact persisted source row under the same frozen policy version."""
    if ticker is None or validator is None or event.policy_version != validator.policy_version:
        return None
    try:
        validator.validate(
            ticker=ticker,
            evidence=WebEvidence(
                id=event.source_ref,
                ticker=ticker,
                title=event.title,
                content=event.summary,
                source_url=event.source_url,
                source_kind=event.source_kind,
                source_tier=event.source_tier,
                published_at=event.published_at,
                fetched_at=event.fetched_at,
                content_hash=event.content_hash,
            ),
        )
    except (TypeError, ValueError, WebEvidenceValidationError):
        return None
    return event


def _failed_report(
    ticker: str,
    *,
    errors: list[str] | None = None,
) -> GuardedMarketReport:
    guarded_errors = (
        _canonical_errors(errors)
        if errors is not None
        else ["MARKET_DATA_SCOPE_MISMATCH"]
    )
    return GuardedMarketReport(
        ticker=ticker,
        status="failed",
        errors=guarded_errors,
        information_gaps=["No valid IEX-only snapshot was retained."],
    )


def _canonical_errors(value: object) -> list[MarketGuardError]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return ["MARKET_DATA_UNAVAILABLE"]
    errors: list[MarketGuardError] = []
    for item in value:
        if item not in _ALLOWED_ERRORS:
            return ["MARKET_DATA_UNAVAILABLE"]
        if item not in errors:
            errors.append(item)  # type: ignore[arg-type]
    return errors


def _append_once(
    errors: list[MarketGuardError], value: MarketGuardError
) -> list[MarketGuardError]:
    return errors if value in errors else [*errors, value]


def _canonical_freshness(status: MarketStatus) -> FreshnessLabel:
    if status is MarketStatus.OPEN:
        return "open-iex"
    if status is MarketStatus.CLOSED:
        return "latest-available-iex"
    return "market-status-unknown"


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
