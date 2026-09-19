"""Contract tests for the MCP-only P1 collector adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest
from fastmcp.client.client import CallToolResult
from pydantic import ValidationError

from fra.context import (
    BudgetAuthority,
    BudgetExhaustedError,
    BudgetLimits,
)
from fra.domain import (
    EvidenceChunk,
    SourceKind,
    SourceTier,
    WebEvidence,
)
from fra.mcp_server.client_adapters import (
    MCPFilingSearch,
    MCPWebSearch,
)
from fra.observability import ObservationHandle, bind_trace_run
from fra.retrieval.collector import (
    EvidenceQuery,
    RetrievalErrorCode,
    WebSearchRequest,
    rewrite_query,
)
from fra.retrieval.coverage import EvidenceSide
from fra.skills.models import ResearchFacet, WebUsagePolicy
from fra.storage.run_repositories import SourceFetchWrite


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


class RecordingMCP:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = list(
            responses
            or [_tool_result({"chunks": [_chunk().model_dump(mode="json")], "error": None})]
        )
        self.calls: list[ToolCall] = []

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append(ToolCall(name=name, arguments=dict(arguments)))
        return self.responses.pop(0)


class RaisingMCP:
    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        del name, arguments
        raise RuntimeError("private provider failure")


def _tool_result(
    payload: object,
    *,
    is_error: bool = False,
) -> CallToolResult:
    return CallToolResult(
        content=[],
        structured_content=payload,  # type: ignore[arg-type]
        meta=None,
        is_error=is_error,
    )


class RecordingFetchWriter:
    def __init__(self) -> None:
        self.values: list[SourceFetchWrite] = []

    def record_fetch(self, value: SourceFetchWrite) -> None:
        self.values.append(value)


def _chunk(
    source_id: str = "filing-1",
    *,
    ticker: str = "NVDA",
    corpus_version: str = "NVDA-v1",
) -> EvidenceChunk:
    return EvidenceChunk(
        id=source_id,
        ticker=ticker,
        corpus_version=corpus_version,
        content="Revenue increased due to data center demand.",
        source_url=f"https://www.sec.gov/Archives/{source_id}.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=45,
    )


def _web(source_id: str = "web-1", *, ticker: str = "NVDA") -> WebEvidence:
    return WebEvidence(
        id=source_id,
        ticker=ticker,
        title="NVIDIA update",
        content="Issuer evidence for the missing target.",
        source_url="https://investor.nvidia.com/news",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, tzinfo=UTC),
        content_hash=f"sha256:{source_id}",
    )


def _evidence_query(
    ticker: str = "NVDA",
    *,
    question_index: int = 0,
    side: EvidenceSide = EvidenceSide.SUPPORT,
    facet: ResearchFacet = ResearchFacet.EARNINGS_CHANGE,
    query: str = "revenue change",
    limit: int = 2,
) -> EvidenceQuery:
    return EvidenceQuery(
        ticker=ticker,
        question_index=question_index,
        side=side,
        facet=facet,
        query=query,
        limit=limit,
    )


def _web_request() -> WebSearchRequest:
    return WebSearchRequest(
        ticker="NVDA",
        query="NVDA allowlisted evidence target question=1 side=challenge facet=risks",
        missing_facets=(ResearchFacet.RISKS,),
        missing_pairs=((1, EvidenceSide.CHALLENGE),),
        max_results=2,
        usage_policy=WebUsagePolicy.EVIDENCE,
    )


def test_web_request_rejects_user_supplied_source_controls() -> None:
    with pytest.raises(ValidationError):
        WebSearchRequest(
            ticker="NVDA",
            query="company update",
            missing_facets=(ResearchFacet.RISKS,),
            missing_pairs=((0, EvidenceSide.SUPPORT),),
            max_results=1,
            usage_policy=WebUsagePolicy.EVIDENCE,
            domains=("untrusted.example",),  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_filing_adapter_calls_only_hybrid_search_and_preserves_binding() -> None:
    client = RecordingMCP()
    query = _evidence_query(
        question_index=3,
        side=EvidenceSide.CHALLENGE,
        facet=ResearchFacet.GUIDANCE_AND_RISKS,
    )

    result = await MCPFilingSearch(client)((query,))

    assert result.error is None
    assert [call.name for call in client.calls] == ["hybrid_search_filings"]
    assert client.calls[0].arguments == {
        "ticker": "NVDA",
        "query": (
            "revenue change | ticker=NVDA | question=3 | facet=guidance_and_risks | side=challenge"
        ),
        "filing_ids": [],
        "evidence_side": "challenge",
        "k": 2,
    }
    assert [
        (hit.evidence.id, hit.question_index, hit.side, hit.facet) for hit in result.evidence
    ] == [
        (
            "filing-1",
            3,
            EvidenceSide.CHALLENGE,
            ResearchFacet.GUIDANCE_AND_RISKS,
        )
    ]


@pytest.mark.asyncio
async def test_filing_adapter_passes_only_the_application_resolved_scope() -> None:
    client = RecordingMCP()
    query = _evidence_query().model_copy(
        update={
            "corpus_version": "NVDA-v7",
            "filing_ids": ("filing-quarterly",),
        }
    )

    await MCPFilingSearch(client)((query,))

    assert client.calls[0].arguments["corpus_version"] == "NVDA-v7"
    assert client.calls[0].arguments["filing_ids"] == ["filing-quarterly"]


@pytest.mark.asyncio
async def test_filing_adapter_preserves_retry_marker_with_max_length_query() -> None:
    client = RecordingMCP()
    query = rewrite_query(
        _evidence_query(
            side=EvidenceSide.CHALLENGE,
            query="x" * 500,
        )
    )
    suffix = (
        " | ticker=NVDA | question=0 | facet=earnings_change | side=challenge | retry=missing_only"
    )

    result = await MCPFilingSearch(client)((query,))

    emitted_query = str(client.calls[0].arguments["query"])
    assert result.error is None
    assert emitted_query == f"{'x' * (500 - len(suffix))}{suffix}"
    assert len(emitted_query) == 500


@pytest.mark.asyncio
async def test_filing_adapter_validates_all_target_metadata_before_first_call() -> None:
    client = RecordingMCP()
    oversized = _evidence_query(
        question_index=int("9" * 450),
        side=EvidenceSide.CHALLENGE,
        facet=ResearchFacet.GUIDANCE_AND_RISKS,
        query="risk",
    )

    result = await MCPFilingSearch(client)((_evidence_query(), oversized))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert client.calls == []


@pytest.mark.asyncio
async def test_filing_adapter_rejects_tool_error_envelope() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "EMPTY_RETRIEVAL", "message": "none"},
                }
            )
        ]
    )

    result = await MCPFilingSearch(client)((_evidence_query(),))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.EMPTY_RETRIEVAL


@pytest.mark.asyncio
async def test_filing_adapter_keeps_other_targets_when_one_query_is_empty() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "EMPTY_RETRIEVAL", "message": "none"},
                }
            ),
            _tool_result(
                {
                    "chunks": [_chunk("challenge").model_dump(mode="json")],
                    "error": None,
                }
            ),
        ]
    )

    result = await MCPFilingSearch(client)(
        (
            _evidence_query(side=EvidenceSide.SUPPORT),
            _evidence_query(side=EvidenceSide.CHALLENGE),
        )
    )

    assert result.error is None
    assert [call.name for call in client.calls] == [
        "hybrid_search_filings",
        "hybrid_search_filings",
    ]
    assert [(hit.evidence.id, hit.side) for hit in result.evidence] == [
        ("challenge", EvidenceSide.CHALLENGE)
    ]


@pytest.mark.asyncio
async def test_filing_adapter_rejects_bare_mapping_as_protocol_error() -> None:
    response = {"chunks": [_chunk().model_dump(mode="json")], "error": None}

    result = await MCPFilingSearch(RecordingMCP([response]))((_evidence_query(),))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert result.evidence == ()


@pytest.mark.asyncio
async def test_filing_adapter_fails_immediately_for_unsupported_ticker() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {
                        "code": "UNSUPPORTED_TICKER",
                        "message": "upstream-secret-message",
                    },
                }
            ),
            _tool_result({"chunks": [_chunk().model_dump(mode="json")], "error": None}),
        ]
    )

    result = await MCPFilingSearch(client)(
        (
            _evidence_query(side=EvidenceSide.SUPPORT),
            _evidence_query(side=EvidenceSide.CHALLENGE),
        )
    )

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.OPERATION_ERROR
    assert "upstream-secret-message" not in result.error.message
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_filing_adapter_fails_immediately_when_no_filing_corpus_exists() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {
                        "code": "NO_FILINGS",
                        "message": "upstream-secret-message",
                    },
                }
            ),
            _tool_result({"chunks": [_chunk().model_dump(mode="json")], "error": None}),
        ]
    )

    result = await MCPFilingSearch(client)(
        (
            _evidence_query(side=EvidenceSide.SUPPORT),
            _evidence_query(side=EvidenceSide.CHALLENGE),
        )
    )

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.OPERATION_ERROR
    assert "upstream-secret-message" not in result.error.message
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_filing_adapter_preserves_reranker_dependency_error_code() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {
                        "code": "RERANKER_MODEL_UNAVAILABLE",
                        "message": "private dependency detail",
                    },
                }
            )
        ]
    )

    result = await MCPFilingSearch(client)((_evidence_query(),))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.RERANKER_UNAVAILABLE
    assert result.error.message == (
        "RERANKER_MODEL_UNAVAILABLE: configured FlashRank assets are unavailable"
    )
    assert "private dependency detail" not in result.error.message


@pytest.mark.asyncio
async def test_filing_adapter_rejects_unknown_tool_error_code() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "UNKNOWN_ERROR", "message": "unexpected"},
                }
            )
        ]
    )

    result = await MCPFilingSearch(client)((_evidence_query(),))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert result.evidence == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _tool_result(
            {"chunks": [_chunk().model_dump(mode="json")], "error": None},
            is_error=True,
        ),
        _tool_result(None),
        _tool_result([]),
        _tool_result({"chunks": [{"id": "malformed"}], "error": None}),
        _tool_result(
            {
                "chunks": [_chunk().model_dump(mode="json")],
                "error": None,
                "unexpected": True,
            }
        ),
        _tool_result(
            {
                "chunks": [_chunk().model_dump(mode="json")],
                "error": {"code": "EMPTY_RETRIEVAL", "message": "mixed"},
            }
        ),
    ],
)
async def test_filing_adapter_rejects_malformed_or_protocol_error_envelopes(
    response: object,
) -> None:
    result = await MCPFilingSearch(RecordingMCP([response]))((_evidence_query(),))

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert result.evidence == ()


@pytest.mark.asyncio
async def test_filing_adapter_rejects_over_budget_and_scope_mismatches() -> None:
    over_budget = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [
                        _chunk("one").model_dump(mode="json"),
                        _chunk("two").model_dump(mode="json"),
                    ],
                    "error": None,
                }
            )
        ]
    )
    wrong_ticker = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [_chunk(ticker="AMD").model_dump(mode="json")],
                    "error": None,
                }
            )
        ]
    )
    mixed_corpus = RecordingMCP(
        [
            _tool_result(
                {
                    "chunks": [_chunk("one", corpus_version="NVDA-v1").model_dump(mode="json")],
                    "error": None,
                }
            ),
            _tool_result(
                {
                    "chunks": [_chunk("two", corpus_version="NVDA-v2").model_dump(mode="json")],
                    "error": None,
                }
            ),
        ]
    )

    over_budget_result = await MCPFilingSearch(over_budget)((_evidence_query(limit=1),))
    wrong_ticker_result = await MCPFilingSearch(wrong_ticker)((_evidence_query(),))
    mixed_corpus_result = await MCPFilingSearch(mixed_corpus)(
        (
            _evidence_query(question_index=0),
            _evidence_query(question_index=1),
        )
    )

    assert over_budget_result.error is not None
    assert over_budget_result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert wrong_ticker_result.error is not None
    assert wrong_ticker_result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert mixed_corpus_result.error is not None
    assert mixed_corpus_result.error.code is RetrievalErrorCode.PROTOCOL_ERROR


@pytest.mark.asyncio
async def test_web_adapter_preserves_typed_failure_code() -> None:
    client = RecordingMCP(
        [
            _tool_result(
                {
                    "evidence": [],
                    "error": {"code": "WEB_PROVIDER_TIMEOUT", "message": "timeout"},
                }
            )
        ]
    )

    result = await MCPWebSearch(client)(_web_request())

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.OPERATION_ERROR


@pytest.mark.asyncio
async def test_web_adapter_calls_only_allowlisted_tool_without_user_source_controls() -> None:
    client = RecordingMCP(
        [_tool_result({"evidence": [_web().model_dump(mode="json")], "error": None})]
    )

    result = await MCPWebSearch(client)(_web_request())

    assert result.error is None
    assert [call.name for call in client.calls] == ["search_allowlisted_web"]
    assert client.calls[0].arguments == {
        "ticker": "NVDA",
        "query": "NVDA allowlisted evidence target question=1 side=challenge facet=risks",
        "max_results": 2,
    }
    assert "domains" not in client.calls[0].arguments
    assert "urls" not in client.calls[0].arguments
    assert [
        (hit.evidence.id, hit.question_index, hit.side, hit.facet) for hit in result.evidence
    ] == [("web-1", 1, EvidenceSide.CHALLENGE, ResearchFacet.RISKS)]


@pytest.mark.asyncio
async def test_web_adapter_rejects_over_budget_or_wrong_ticker() -> None:
    over_budget = RecordingMCP(
        [
            _tool_result(
                {
                    "evidence": [
                        _web("one").model_dump(mode="json"),
                        _web("two").model_dump(mode="json"),
                        _web("three").model_dump(mode="json"),
                    ],
                    "error": None,
                }
            )
        ]
    )
    wrong_ticker = RecordingMCP(
        [
            _tool_result(
                {
                    "evidence": [_web(ticker="AMD").model_dump(mode="json")],
                    "error": None,
                }
            )
        ]
    )

    over_budget_result = await MCPWebSearch(over_budget)(_web_request())
    wrong_ticker_result = await MCPWebSearch(wrong_ticker)(_web_request())

    assert over_budget_result.error is not None
    assert over_budget_result.error.code is RetrievalErrorCode.PROTOCOL_ERROR
    assert wrong_ticker_result.error is not None
    assert wrong_ticker_result.error.code is RetrievalErrorCode.PROTOCOL_ERROR


@pytest.mark.asyncio
async def test_adapter_records_only_safe_fetch_metadata_for_every_response() -> None:
    writer = RecordingFetchWriter()
    secret_query = "revenue secret-token-123"
    client = RecordingMCP(
        [
            _tool_result({"chunks": [_chunk().model_dump(mode="json")], "error": None}),
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "EMPTY_RETRIEVAL", "message": "raw"},
                }
            ),
        ]
    )
    adapter = MCPFilingSearch(client, source_fetch_writer=writer, run_id="run-1")

    success = await adapter((_evidence_query(query=secret_query),))
    failure = await adapter((_evidence_query(query=secret_query),))

    assert success.error is None
    assert failure.error is not None
    assert len(writer.values) == 2
    assert [value.run_id for value in writer.values] == ["run-1", "run-1"]
    assert [value.source_kind for value in writer.values] == ["filing", "filing"]
    assert [value.status for value in writer.values] == ["completed", "failed"]
    assert [value.error_code for value in writer.values] == [None, "EMPTY_RETRIEVAL"]
    assert all(value.fetched_at is not None for value in writer.values)
    assert all(value.requested_at.tzinfo is UTC for value in writer.values)
    assert all("NVDA" in value.source_ref for value in writer.values)
    assert all("sha256:" in value.source_ref for value in writer.values)
    assert all(secret_query not in value.source_ref for value in writer.values)
    assert all("Revenue increased" not in value.source_ref for value in writer.values)


def test_adapter_requires_writer_and_run_id_together() -> None:
    writer = RecordingFetchWriter()

    with pytest.raises(ValueError, match="together"):
        MCPFilingSearch(RecordingMCP(), source_fetch_writer=writer)
    with pytest.raises(ValueError, match="together"):
        MCPWebSearch(RecordingMCP(), run_id="run-1")


@dataclass
class LiveObservation:
    name: str
    kind: str
    active: bool = False
    output: object | None = None
    metadata: dict[str, object] | None = None

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        assert self.active, f"observation updated after close: {self.name}"
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata = {**(self.metadata or {}), **metadata}


class LiveTraceRun:
    trace_id = "trace-live"

    def __init__(self) -> None:
        self.observations: list[LiveObservation] = []

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: str,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[ObservationHandle]:
        del input
        observation = LiveObservation(name, kind, metadata=dict(metadata or {}))
        self.observations.append(observation)
        observation.active = True
        try:
            yield observation
        finally:
            observation.active = False

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        del output, metadata


@pytest.mark.asyncio
async def test_filing_tool_result_metadata_is_written_before_span_closes() -> None:
    trace = LiveTraceRun()
    adapter = MCPFilingSearch(RecordingMCP())

    with bind_trace_run(trace):
        result = await adapter((_evidence_query(),))

    assert result.error is None
    observation = trace.observations[0]
    assert observation.output == {"status": "completed", "result_count": 1}
    assert observation.metadata is not None
    assert observation.metadata["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_filing_tool_observation_uses_cache_state_from_response_envelope() -> None:
    """A tool span must use this call's metrics, never mutable retriever last-run state."""
    response = _tool_result(
        {
            "chunks": [_chunk().model_dump(mode="json")],
            "metrics": {
                "sparse_candidates": 2,
                "dense_candidates": 2,
                "fused_candidates": 2,
                "fused_ids": ["chunk-1", "chunk-2"],
                "reranker_version": "test-reranker-v1",
                "cache_hit": True,
                "retained_ids": ["chunk-1"],
                "retained_scores": [["chunk-1", 0.9]],
            },
            "error": None,
        }
    )
    trace = LiveTraceRun()

    with bind_trace_run(trace):
        result = await MCPFilingSearch(RecordingMCP([response]))((_evidence_query(),))

    assert result.error is None
    assert trace.observations[0].metadata["cache_hit"] is True


@pytest.mark.asyncio
async def test_empty_web_error_metadata_is_written_before_span_closes() -> None:
    trace = LiveTraceRun()
    adapter = MCPWebSearch(RecordingMCP([_tool_result({"evidence": [], "error": None})]))

    with bind_trace_run(trace):
        result = await adapter(_web_request())

    assert result.error is not None
    assert result.error.code is RetrievalErrorCode.EMPTY_RETRIEVAL
    observation = trace.observations[0]
    assert observation.output == {"status": "failed", "result_count": 0}
    assert observation.metadata is not None
    assert observation.metadata["error_code"] == "EMPTY_RETRIEVAL"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (_tool_result(None), "MCP_PROTOCOL_ERROR"),
        (
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "NO_FILINGS", "message": "none"},
                }
            ),
            "NO_FILINGS",
        ),
        (
            _tool_result(
                {
                    "chunks": [_chunk(ticker="AMD").model_dump(mode="json")],
                    "error": None,
                }
            ),
            "MCP_PROTOCOL_ERROR",
        ),
        (
            _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "EMPTY_RETRIEVAL", "message": "none"},
                }
            ),
            "EMPTY_RETRIEVAL",
        ),
    ],
)
async def test_every_filing_error_updates_its_live_span(
    response: object,
    expected_code: str,
) -> None:
    trace = LiveTraceRun()

    with bind_trace_run(trace):
        result = await MCPFilingSearch(RecordingMCP([response]))((_evidence_query(),))

    assert result.error is not None
    observation = trace.observations[0]
    assert observation.output == {"status": "failed", "result_count": 0}
    assert observation.metadata is not None
    assert observation.metadata["error_code"] == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (_tool_result(None), "MCP_PROTOCOL_ERROR"),
        (
            _tool_result(
                {
                    "evidence": [],
                    "error": {"code": "WEB_PROVIDER_TIMEOUT", "message": "timeout"},
                }
            ),
            "WEB_PROVIDER_TIMEOUT",
        ),
        (
            _tool_result(
                {
                    "evidence": [_web(ticker="AMD").model_dump(mode="json")],
                    "error": None,
                }
            ),
            "MCP_PROTOCOL_ERROR",
        ),
        (_tool_result({"evidence": [], "error": None}), "EMPTY_RETRIEVAL"),
    ],
)
async def test_every_web_error_updates_its_live_span(
    response: object,
    expected_code: str,
) -> None:
    trace = LiveTraceRun()

    with bind_trace_run(trace):
        result = await MCPWebSearch(RecordingMCP([response]))(_web_request())

    assert result.error is not None
    observation = trace.observations[0]
    assert observation.output == {"status": "failed", "result_count": 0}
    assert observation.metadata is not None
    assert observation.metadata["error_code"] == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_kind", ["filing", "web"])
async def test_mcp_call_failure_updates_its_live_span(adapter_kind: str) -> None:
    trace = LiveTraceRun()

    with bind_trace_run(trace):
        if adapter_kind == "filing":
            result = await MCPFilingSearch(RaisingMCP())((_evidence_query(),))
        else:
            result = await MCPWebSearch(RaisingMCP())(_web_request())

    assert result.error is not None
    observation = trace.observations[0]
    assert observation.output == {"status": "failed", "result_count": 0}
    assert observation.metadata is not None
    assert observation.metadata["error_code"] == "MCP_CALL_ERROR"
    assert "private provider failure" not in repr(observation.metadata)


@pytest.mark.asyncio
async def test_failed_mcp_attempt_consumes_shared_budget_before_side_effect() -> None:
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=0,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=1,
            max_retrieval_rounds=0,
            max_web_calls=0,
        )
    )

    trace = LiveTraceRun()
    with bind_trace_run(trace):
        failed = await MCPFilingSearch(
            RaisingMCP(),
            budget_gate=authority,
        )((_evidence_query(),))

    assert failed.error is not None
    assert authority.state.tool_calls == 1
    assert trace.observations[0].metadata["error_code"] == "MCP_CALL_ERROR"

    denied_client = RecordingMCP()
    with pytest.raises(BudgetExhaustedError) as caught:
        await MCPFilingSearch(
            denied_client,
            budget_gate=authority,
        )((_evidence_query(),))

    assert caught.value.dimension == "tool_calls"
    assert denied_client.calls == []
    assert authority.state.tool_calls == 1
