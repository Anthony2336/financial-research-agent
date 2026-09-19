"""One-call, time-windowed collection of neutral market event context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from fra.domain import WebEvidence
from fra.market_data.models import MarketContext, MarketEvent, MarketSnapshot

_EVENT_WINDOW = timedelta(days=3)


@dataclass(frozen=True, slots=True)
class EventWindow:
    """The bounded window anchored solely to the guarded snapshot time."""

    start: datetime
    end: datetime


AuthoritativeEventSearch = Callable[..., Awaitable[list[WebEvidence]]]


def event_window(snapshot_as_of: datetime, *, width: timedelta = _EVENT_WINDOW) -> EventWindow:
    """Build the fixed retrospective event window for one approved snapshot."""
    anchor = snapshot_as_of.astimezone(UTC)
    return EventWindow(start=anchor - width, end=anchor)


async def collect_market_context(
    snapshot: MarketSnapshot,
    search: AuthoritativeEventSearch,
) -> MarketContext:
    """Make one bounded event search; unavailable evidence remains cause unknown."""
    window = event_window(snapshot.as_of)
    try:
        evidence = await search(
            ticker=snapshot.symbol,
            start=window.start,
            end=window.end,
            limit=3,
        )
    except Exception:
        evidence = []
    return build_neutral_context(snapshot, evidence, window)


def build_neutral_context(
    snapshot: MarketSnapshot,
    evidence: list[WebEvidence],
    window: EventWindow | None = None,
    policy_version: str | None = None,
) -> MarketContext:
    """Translate already-authoritative evidence without making a causal assertion."""
    resolved_window = window or event_window(snapshot.as_of)
    events: list[MarketEvent] = []
    for item in evidence[:3]:
        if policy_version is None:
            continue
        if item.published_at is None:
            continue
        published_at = item.published_at.astimezone(UTC)
        if not resolved_window.start <= published_at <= resolved_window.end:
            continue
        try:
            event = MarketEvent(
                source_ref=item.id,
                source_url=str(item.source_url),
                title=item.title,
                summary=item.content,
                published_at=published_at,
                fetched_at=item.fetched_at.astimezone(UTC),
                relationship="inside_window",
                content_hash=item.content_hash,
                source_kind=item.source_kind,
                source_tier=item.source_tier,
                policy_version=policy_version,
            )
        except (TypeError, ValidationError, ValueError):
            continue
        events.append(event)
    return MarketContext(
        anchor_as_of=snapshot.as_of,
        window_start=resolved_window.start,
        window_end=resolved_window.end,
        events=events,
        counterevidence=[],
        open_questions=(
            []
            if events
            else ["No authoritative event with adequate publication-time evidence was retained."]
        ),
        cause_assessment="possibly_related" if events else "cause_unknown",
    )
