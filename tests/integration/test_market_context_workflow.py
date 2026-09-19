"""Market-context workflow contracts at the in-process MCP boundary."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

import pytest
from fastmcp.client.client import CallToolResult

from fra.contracts import ResearchCommand, ResearchMode
from fra.domain import (
    SourceKind,
    SourceTier,
    WebEvidence,
    content_addressed_web_evidence_id,
)
from fra.graph.market_workflow import run_market_workflow
from fra.graph.models import MarketDependencies
from fra.market_data.models import MarketStatus
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
)

from .test_market_workflow import (
    RecordingMCP,
    SequenceClock,
    _bar,
    _observation_id,
    _snapshot,
    _success_responses,
    _tool_result,
)


def _result(payload: object) -> CallToolResult:
    return CallToolResult(content=[], structured_content=payload, meta=None, is_error=False)  # type: ignore[arg-type]


def _command(*, with_context: bool) -> ResearchCommand:
    return ResearchCommand(
        ticker="NVDA",
        request="What is the current price?",
        mode=ResearchMode.MARKET_SNAPSHOT,
        with_context=with_context,
    )


class RecordingBundleWriter:
    def __init__(self) -> None:
        self.saved: list[object] = []

    def save_bundle(self, bundle: object) -> None:
        self.saved.append(bundle)


class _EvidenceRepository:
    def __init__(self, evidence: WebEvidence) -> None:
        self.evidence = evidence

    def get_many(self, evidence_ids: list[str]) -> list[WebEvidence]:
        return [self.evidence] if evidence_ids == [self.evidence.id] else []


def _event_response() -> tuple[CallToolResult, PersistedWebEvidenceValidator]:
    content = "The issuer filed a report."
    source_url = "https://www.sec.gov/Archives/example"
    content_hash = sha256(content.encode()).hexdigest()
    evidence = WebEvidence(
        id=content_addressed_web_evidence_id("NVDA", source_url, content_hash),
        ticker="NVDA",
        title="Issuer filing",
        content=content,
        source_url=source_url,
        source_kind=SourceKind.FILING,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime.now(UTC) - timedelta(minutes=2),
        fetched_at=datetime.now(UTC),
        content_hash=content_hash,
    )
    validator = PersistedWebEvidenceValidator(
        SourcePolicy(issuer_domains={"NVDA": frozenset()}),
        _EvidenceRepository(evidence),
    )
    return _result({"evidence": [evidence.model_dump(mode="json")], "error": None}), validator


def test_market_context_is_optional_and_uses_one_event_call_after_snapshot() -> None:
    event_response, validator = _event_response()
    writer = RecordingBundleWriter()
    responses = [
        *_success_responses(),
        event_response,
    ]
    client = RecordingMCP(responses)

    result = run_market_workflow(
        _command(with_context=True),
        MarketDependencies(
            client,
            max_bars=5,
            web_evidence_validator=validator,
            bundle_writer=writer,
        ),
    )

    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
        "search_authoritative_events",
    ]
    assert result.guarded_report.market_context.cause_assessment == "possibly_related"
    assert "possibly_related" in result.rendered_output
    assert "time adjacency only; not a causal claim" in result.rendered_output
    assert result.rendered_output.count("Research assistance only; not investment advice.") == 1
    assert len(writer.saved) == 1


def test_market_context_disabled_makes_no_web_call() -> None:
    client = RecordingMCP(_success_responses())

    run_market_workflow(_command(with_context=False), MarketDependencies(client, max_bars=5))

    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
    ]


def test_abnormal_move_at_exact_decimal_threshold_triggers_one_context_call() -> None:
    event_response, validator = _event_response()
    snapshot = _snapshot().model_copy(
        update={"price": Decimal("127.05"), "day_high": Decimal("128.00")}
    )
    original_snapshot = snapshot.model_dump(mode="python")
    client = RecordingMCP(
        [
            _tool_result({"snapshot": snapshot.model_dump(mode="json"), "error": None}),
            _success_responses()[1],
            event_response,
        ]
    )

    result = run_market_workflow(
        _command(with_context=False),
        MarketDependencies(
            client,
            max_bars=5,
            abnormal_move_threshold=Decimal("0.05"),
            web_evidence_validator=validator,
        ),
    )

    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
        "search_authoritative_events",
    ]
    assert result.guarded_report is not None
    assert result.guarded_report.market_context is not None
    assert result.guarded_report.snapshot is not None
    assert result.guarded_report.snapshot.model_dump(mode="python") == original_snapshot


def test_move_below_exact_decimal_threshold_does_not_trigger_context() -> None:
    snapshot = _snapshot().model_copy(
        update={
            "price": Decimal("127.049999999999999999"),
            "day_high": Decimal("128.00"),
        }
    )
    client = RecordingMCP(
        [
            _tool_result({"snapshot": snapshot.model_dump(mode="json"), "error": None}),
            _success_responses()[1],
        ]
    )

    result = run_market_workflow(
        _command(with_context=False),
        MarketDependencies(
            client,
            max_bars=5,
            abnormal_move_threshold=Decimal("0.05"),
        ),
    )

    assert result.status == "completed"
    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
    ]
    assert result.guarded_report is not None
    assert result.guarded_report.market_context is None


def test_partial_bundle_never_auto_triggers_but_explicit_context_still_calls_once() -> None:
    snapshot = _snapshot().model_copy(
        update={"price": Decimal("127.05"), "day_high": Decimal("128.00")}
    )
    bars_failure = _result(
        {
            "bars": None,
            "error": {"code": "MARKET_DATA_UNAVAILABLE", "message": "bars unavailable"},
        }
    )
    automatic_client = RecordingMCP(
        [
            _tool_result({"snapshot": snapshot.model_dump(mode="json"), "error": None}),
            bars_failure,
        ]
    )
    explicit_client = RecordingMCP(
        [
            _tool_result({"snapshot": snapshot.model_dump(mode="json"), "error": None}),
            bars_failure,
            _result({"evidence": [], "error": None}),
        ]
    )

    automatic = run_market_workflow(
        _command(with_context=False),
        MarketDependencies(
            automatic_client,
            max_bars=5,
            abnormal_move_threshold=Decimal("0.05"),
        ),
    )
    explicit = run_market_workflow(
        _command(with_context=True),
        MarketDependencies(
            explicit_client,
            max_bars=5,
            abnormal_move_threshold=Decimal("0.05"),
        ),
    )

    assert automatic.status == explicit.status == "partial"
    assert [call[0] for call in automatic_client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
    ]
    assert [call[0] for call in explicit_client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
        "search_authoritative_events",
    ]


@pytest.mark.parametrize("previous_close", [None, Decimal("0"), Decimal("-1")])
def test_missing_or_nonpositive_previous_close_never_auto_triggers(
    previous_close: Decimal | None,
) -> None:
    snapshot = _snapshot().model_dump(mode="json")
    if previous_close is None:
        snapshot.pop("previous_close")
    else:
        snapshot["previous_close"] = str(previous_close)
    client = RecordingMCP([_tool_result({"snapshot": snapshot, "error": None})])

    result = run_market_workflow(
        _command(with_context=False),
        MarketDependencies(
            client,
            max_bars=5,
            abnormal_move_threshold=Decimal("0.05"),
        ),
    )

    assert result.status == "failed"
    assert [call[0] for call in client.calls] == ["get_market_snapshot"]


def test_invalid_snapshot_skips_event_search_even_when_context_requested() -> None:
    client = RecordingMCP(
        [
            _result(
                {
                    "snapshot": None,
                    "error": {
                        "code": "MARKET_DATA_UNAVAILABLE",
                        "message": "market data is unavailable",
                    },
                }
            )
        ]
    )

    result = run_market_workflow(
        _command(with_context=True), MarketDependencies(client, max_bars=5)
    )

    assert result.status == "failed"
    assert [call[0] for call in client.calls] == ["get_market_snapshot"]


def test_final_snapshot_guard_failure_skips_event_search() -> None:
    observed_at = datetime.now(UTC) - timedelta(minutes=10)
    stale_snapshot = _snapshot(now=observed_at).model_copy(
        update={"market_status": MarketStatus.OPEN}
    )
    client = RecordingMCP(
        [
            _tool_result({"snapshot": stale_snapshot.model_dump(mode="json"), "error": None}),
            _tool_result({"bars": [_bar(now=observed_at).model_dump(mode="json")], "error": None}),
        ]
    )

    result = run_market_workflow(
        _command(with_context=True), MarketDependencies(client, max_bars=5)
    )

    assert result.status == "failed"
    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
    ]


def test_event_search_failure_preserves_snapshot_with_cause_unknown_context() -> None:
    client = RecordingMCP(
        [
            *_success_responses(),
            _result(
                {
                    "evidence": [],
                    "error": {
                        "code": "WEB_PROVIDER_UNAVAILABLE",
                        "message": "not configured",
                    },
                }
            ),
        ]
    )

    result = run_market_workflow(
        _command(with_context=True), MarketDependencies(client, max_bars=5)
    )

    assert result.status == "completed"
    assert result.guarded_report is not None
    assert result.guarded_report.market_context is not None
    assert result.guarded_report.market_context.cause_assessment == "cause_unknown"


def test_context_flow_revalidates_market_times_after_event_call() -> None:
    """An open snapshot that becomes stale during context fetch must fail final rendering."""
    entry = datetime(2026, 8, 31, 14, 0, tzinfo=UTC)
    snapshot_time = entry
    open_snapshot = _snapshot(now=snapshot_time).model_copy(
        update={
            "id": _observation_id(
                "snapshot",
                symbol="NVDA",
                source_timestamp=snapshot_time - timedelta(seconds=89),
                fetched_at=snapshot_time,
                raw_payload_hash="a" * 64,
            ),
            "market_status": MarketStatus.OPEN,
            "as_of": snapshot_time - timedelta(seconds=89),
            "fetched_at": snapshot_time,
            "delayed_by_seconds": 89,
        }
    )
    event_response, validator = _event_response()
    writer = RecordingBundleWriter()
    client = RecordingMCP(
        [
            _tool_result({"snapshot": open_snapshot.model_dump(mode="json"), "error": None}),
            _tool_result(
                {
                    "bars": [_bar(now=snapshot_time).model_dump(mode="json")],
                    "error": None,
                }
            ),
            event_response,
        ]
    )
    clock = SequenceClock(entry, entry + timedelta(seconds=5))

    result = run_market_workflow(
        _command(with_context=True),
        MarketDependencies(
            client,
            max_bars=5,
            max_staleness_seconds=90,
            web_evidence_validator=validator,
            bundle_writer=writer,
            clock=clock,
        ),
    )

    assert result.status == "failed"
    assert result.errors == ["STALE_MARKET_DATA"]
    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
        "search_authoritative_events",
    ]
    assert clock.calls == 2
    assert writer.saved == []


@pytest.mark.parametrize(
    ("context_response", "label"),
    [
        (
            CallToolResult(
                content=[],
                structured_content={"unexpected": []},
                meta=None,
                is_error=False,
            ),
            "malformed",
        ),
        (
            CallToolResult(
                content=[],
                structured_content={"evidence": [], "error": None},
                meta=None,
                is_error=True,
            ),
            "is_error",
        ),
    ],
)
def test_context_flow_revalidates_after_invalid_event_envelopes(
    context_response: CallToolResult,
    label: str,
) -> None:
    """Any attempted context call must trigger the final guard clock, even on invalid envelopes."""
    del label
    entry = datetime(2026, 8, 31, 14, 0, tzinfo=UTC)
    snapshot_time = entry
    open_snapshot = _snapshot(now=snapshot_time).model_copy(
        update={
            "id": _observation_id(
                "snapshot",
                symbol="NVDA",
                source_timestamp=snapshot_time - timedelta(seconds=89),
                fetched_at=snapshot_time,
                raw_payload_hash="a" * 64,
            ),
            "market_status": MarketStatus.OPEN,
            "as_of": snapshot_time - timedelta(seconds=89),
            "fetched_at": snapshot_time,
            "delayed_by_seconds": 89,
        }
    )
    client = RecordingMCP(
        [
            _tool_result({"snapshot": open_snapshot.model_dump(mode="json"), "error": None}),
            _tool_result(
                {
                    "bars": [_bar(now=snapshot_time).model_dump(mode="json")],
                    "error": None,
                }
            ),
            context_response,
        ]
    )
    clock = SequenceClock(entry, entry + timedelta(seconds=5))

    result = run_market_workflow(
        _command(with_context=True),
        MarketDependencies(
            client,
            max_bars=5,
            max_staleness_seconds=90,
            clock=clock,
        ),
    )

    assert result.status == "failed"
    assert result.errors == ["STALE_MARKET_DATA"]
    assert [call[0] for call in client.calls] == [
        "get_market_snapshot",
        "get_market_bars",
        "search_authoritative_events",
    ]
    assert clock.calls == 2
