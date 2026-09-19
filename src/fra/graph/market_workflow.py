"""Short, non-agentic workflow for one guarded IEX-only market report."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import perf_counter

from fastmcp.client.client import CallToolResult
from pydantic import ValidationError

from fra.context import BudgetLimits
from fra.contracts import ResearchCommand
from fra.execution import current_research_context
from fra.graph.models import (
    MarketDependencies,
    MarketResearchResult,
    MCPToolClient,
)
from fra.market_data.context import (
    build_neutral_context,
    event_window,
)
from fra.market_data.models import (
    MarketDataBundle,
    MarketSnapshot,
    MarketStatus,
)
from fra.market_data.validation import canonical_market_snapshot
from fra.mcp_server.market_tools import (
    GetMarketBarsResponse,
    GetMarketSnapshotResponse,
)
from fra.mcp_server.web_tools import (
    SearchAuthoritativeEventsResponse,
)
from fra.observability import ObservationHandle, observe
from fra.reporting.market_guard import (
    GuardedMarketReport,
    attach_market_context,
    guard_market_bundle,
)
from fra.reporting.market_render import render_market_markdown

_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
logger = logging.getLogger(__name__)


def run_market_workflow(
    command: ResearchCommand,
    dependencies: MarketDependencies,
) -> MarketResearchResult:
    """Call each bounded market tool once, then guard and render fixed Markdown."""
    ticker = command.ticker.strip().upper()
    workflow_clock = dependencies.clock or (lambda: datetime.now(UTC))
    context = current_research_context()
    if context is not None and not context.budget.configured:
        context.budget.configure(
            BudgetLimits(
                max_planner_calls=0,
                max_analysis_calls=0,
                max_repair_calls=0,
                max_tool_calls=3,
                max_retrieval_rounds=0,
                max_web_calls=1,
            )
        )
    with dependencies.session() as mcp_client:
        snapshot_response = _call_snapshot(mcp_client, dependencies, ticker, market=command.market)
        bars_response = (
            _call_bars(mcp_client, dependencies, ticker)
            if snapshot_response is not None and snapshot_response.snapshot is not None
            else None
        )
        snapshot = snapshot_response.snapshot if snapshot_response is not None else None
        snapshot_error = _response_error(snapshot_response)
        bars = bars_response.bars if bars_response is not None else None
        bars_error = _response_error(bars_response)

        with observe(
            name="market.guard",
            kind="guardrail",
            metadata={"ticker": ticker, "max_bars": dependencies.max_bars},
        ) as observation:
            guarded = _guard_market_responses(
                ticker=ticker,
                snapshot=snapshot,
                snapshot_error=snapshot_error,
                bars=bars,
                bars_error=bars_error,
                dependencies=dependencies,
                now=_utc_now(workflow_clock),
            )
            observation.update(
                output={
                    "status": guarded.status,
                    "snapshot_retained": guarded.snapshot is not None,
                    "bar_count": len(guarded.bars),
                    "error_codes": list(guarded.errors),
                }
            )
        context_requested = command.with_context or _has_abnormal_move(
            guarded,
            threshold=dependencies.abnormal_move_threshold,
        )
        attempted_context_call = context_requested and guarded.snapshot is not None
        context_response = (
            _call_context(mcp_client, dependencies, guarded.snapshot)
            if attempted_context_call
            else None
        )
        if attempted_context_call:
            guarded = _guard_market_responses(
                ticker=ticker,
                snapshot=snapshot,
                snapshot_error=snapshot_error,
                bars=bars,
                bars_error=bars_error,
                dependencies=dependencies,
                now=_utc_now(workflow_clock),
            )

    if context_requested and guarded.snapshot is not None:
        context = build_neutral_context(
            guarded.snapshot,
            context_response.evidence if context_response is not None else [],
            event_window(
                guarded.snapshot.as_of,
                width=timedelta(days=dependencies.context_window_days),
            ),
            policy_version=(
                dependencies.web_evidence_validator.policy_version
                if dependencies.web_evidence_validator is not None
                else None
            ),
        )
        guarded = attach_market_context(
            guarded,
            context,
            window=timedelta(days=dependencies.context_window_days),
            validator=dependencies.web_evidence_validator,
        )
    _persist_market_bundle(guarded, dependencies)

    with observe(
        name="market.render",
        kind="chain",
        metadata={"format": "markdown", "coverage": "IEX-only"},
    ) as observation:
        rendered = render_market_markdown(guarded)
        observation.update(output={"status": guarded.status})
    return MarketResearchResult.from_guarded(guarded, rendered)


def _persist_market_bundle(
    guarded,
    dependencies: MarketDependencies,
) -> None:
    if guarded.snapshot is None or dependencies.bundle_writer is None:
        return
    bundle = MarketDataBundle(
        snapshot=guarded.snapshot,
        bars=list(guarded.bars),
        status=guarded.status,
        freshness_label=guarded.freshness_label,
        errors=list(guarded.errors),
    )
    with observe(
        name="market.persist_bundle",
        kind="tool",
        metadata={"ticker": guarded.ticker, "status": guarded.status},
    ) as observation:
        try:
            dependencies.bundle_writer.save_bundle(bundle)
        except Exception:
            logger.warning("market bundle persistence failed")
            observation.update(output={"status": "failed"})
        else:
            observation.update(
                output={
                    "status": "completed",
                    "bundle_status": bundle.status,
                    "bar_count": len(bundle.bars),
                }
            )


def _call_snapshot(
    mcp_client: MCPToolClient,
    dependencies: MarketDependencies,
    ticker: str,
    *,
    market: str,
) -> GetMarketSnapshotResponse | None:
    response = _call_and_validate(
        mcp_client,
        name="get_market_snapshot",
        arguments={"ticker": ticker, "market": market},
        response_type=GetMarketSnapshotResponse,
    )
    if response is None or response.snapshot is None:
        return response
    snapshot = canonical_market_snapshot(
        response.snapshot,
        ticker=ticker,
    )
    if snapshot is None:
        return None
    return response.model_copy(update={"snapshot": snapshot})


def _has_abnormal_move(
    guarded: GuardedMarketReport,
    *,
    threshold: Decimal,
) -> bool:
    snapshot = guarded.snapshot
    if guarded.status != "completed" or snapshot is None:
        return False
    previous_close = snapshot.previous_close
    if previous_close <= 0:
        return False
    return abs(snapshot.price - previous_close) / previous_close >= threshold


def _guard_market_responses(
    *,
    ticker: str,
    snapshot: MarketSnapshot | None,
    snapshot_error: str | None,
    bars: list[object] | None,
    bars_error: str | None,
    dependencies: MarketDependencies,
    now: datetime,
):
    if snapshot is None:
        return guard_market_bundle(
            None,
            requested_ticker=ticker,
            max_bars=dependencies.max_bars,
            max_staleness_seconds=dependencies.max_staleness_seconds,
            now=now,
            upstream_errors=[snapshot_error or _UNAVAILABLE],
        )
    if bars_error == "MARKET_DATA_CONFIGURATION_MISSING":
        return guard_market_bundle(
            None,
            requested_ticker=ticker,
            max_bars=dependencies.max_bars,
            max_staleness_seconds=dependencies.max_staleness_seconds,
            now=now,
            upstream_errors=[bars_error],
        )
    errors = [bars_error] if bars_error is not None else []
    bundle = MarketDataBundle.model_construct(
        snapshot=snapshot,
        bars=bars or [],
        status="partial" if bars is None or errors else "completed",
        freshness_label=_freshness_label(snapshot.market_status),
        errors=errors,
    )
    return guard_market_bundle(
        bundle,
        requested_ticker=ticker,
        max_bars=dependencies.max_bars,
        max_staleness_seconds=dependencies.max_staleness_seconds,
        now=now,
    )


def _call_bars(
    mcp_client: MCPToolClient,
    dependencies: MarketDependencies,
    ticker: str,
) -> GetMarketBarsResponse | None:
    return _call_and_validate(
        mcp_client,
        name="get_market_bars",
        arguments={
            "ticker": ticker,
            "interval": "1Day",
            "limit": dependencies.max_bars,
        },
        response_type=GetMarketBarsResponse,
    )


def _call_context(
    mcp_client: MCPToolClient,
    dependencies: MarketDependencies,
    snapshot: object,
) -> SearchAuthoritativeEventsResponse | None:
    if not hasattr(snapshot, "as_of") or not hasattr(snapshot, "symbol"):
        return None
    window = event_window(snapshot.as_of, width=timedelta(days=dependencies.context_window_days))
    response = _call_and_validate(
        mcp_client,
        name="search_authoritative_events",
        arguments={
            "ticker": snapshot.symbol,
            "query": f"{snapshot.symbol} authoritative event",
            "window_start": window.start.isoformat().replace("+00:00", "Z"),
            "window_end": window.end.isoformat().replace("+00:00", "Z"),
            "max_results": 3,
        },
        response_type=SearchAuthoritativeEventsResponse,
        fallback_error_code="WEB_PROVIDER_UNAVAILABLE",
    )
    return response if isinstance(response, SearchAuthoritativeEventsResponse) else None


def _call_and_validate(
    mcp_client: MCPToolClient,
    *,
    name: str,
    arguments: dict[str, object],
    response_type: type[GetMarketSnapshotResponse]
    | type[GetMarketBarsResponse]
    | type[SearchAuthoritativeEventsResponse],
    fallback_error_code: str = _UNAVAILABLE,
) -> GetMarketSnapshotResponse | GetMarketBarsResponse | SearchAuthoritativeEventsResponse | None:
    started_at = perf_counter()
    safe_metadata = {
        "tool": name,
        "ticker": arguments["ticker"],
        "market": arguments.get("market"),
        "interval": arguments.get("interval"),
        "limit": arguments.get("limit"),
    }
    with observe(
        name=f"mcp.{name}",
        kind="tool",
        metadata={key: value for key, value in safe_metadata.items() if value is not None},
    ) as observation:
        try:
            context = current_research_context()
            if context is not None:
                context.budget.consume(
                    tool_calls=1,
                    web_calls=int(name == "search_authoritative_events"),
                )
            raw_response = mcp_client.call_tool(name, arguments)
            payload = _structured_payload(raw_response)
            response = response_type.model_validate(payload)
        except (Exception, ValidationError):
            _finish_tool_observation(observation, started_at, error_code=fallback_error_code)
            return None
        error = _response_error(response)
        _finish_tool_observation(
            observation,
            started_at,
            error_code=error,
            result_count=_result_count(response),
        )
        return response


def _structured_payload(raw_response: object) -> Mapping[str, object]:
    if (
        not isinstance(raw_response, CallToolResult)
        or raw_response.is_error
        or not isinstance(raw_response.structured_content, Mapping)
    ):
        raise ValueError("invalid MCP result envelope")
    return raw_response.structured_content


def _response_error(
    response: (
        GetMarketSnapshotResponse | GetMarketBarsResponse | SearchAuthoritativeEventsResponse | None
    ),
) -> str | None:
    return response.error.code if response is not None and response.error is not None else None


def _result_count(
    response: GetMarketSnapshotResponse | GetMarketBarsResponse | SearchAuthoritativeEventsResponse,
) -> int:
    if isinstance(response, GetMarketSnapshotResponse):
        return int(response.snapshot is not None)
    if isinstance(response, GetMarketBarsResponse):
        return len(response.bars or [])
    return len(response.evidence)


def _finish_tool_observation(
    observation: ObservationHandle,
    started_at: float,
    *,
    error_code: str | None,
    result_count: int = 0,
) -> None:
    output: dict[str, object] = {
        "status": "failed" if error_code is not None else "completed",
        "result_count": result_count,
    }
    if error_code is not None:
        output["error_code"] = error_code
    observation.update(
        output=output,
        metadata={"latency_ms": max(0, int((perf_counter() - started_at) * 1000))},
    )


def _freshness_label(status: MarketStatus) -> str:
    if status is MarketStatus.OPEN:
        return "open-iex"
    if status is MarketStatus.CLOSED:
        return "latest-available-iex"
    return "market-status-unknown"


def _utc_now(clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("market workflow clock must return a timezone-aware datetime")
    return value.astimezone(UTC)
