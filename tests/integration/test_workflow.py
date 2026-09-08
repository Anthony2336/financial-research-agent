"""End-to-end tests for the controlled, offline LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastmcp.client.client import CallToolResult
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import financial_evidence_agent.graph.workflow as workflow_module
from financial_evidence_agent.context import BudgetAuthority, BudgetLimits
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    SourceKind,
    SourceTier,
    WebEvidence,
)
from financial_evidence_agent.graph.models import Dependencies, FastMCPToolClient
from financial_evidence_agent.graph.workflow import build_research_graph, run_research
from financial_evidence_agent.mcp_server.server import create_server
from financial_evidence_agent.reporting import guard_memo
from financial_evidence_agent.retrieval.collector import (
    THESIS_COLLECTION_POLICY,
    CollectionErrorCode,
    EvidenceBundle,
    EvidenceCollectionError,
    EvidenceCollector,
    LocalEvidenceHit,
    LocalSearchResponse,
    RetrievalError,
    RetrievalErrorCode,
    WebEvidenceHit,
    WebSearchResponse,
)
from financial_evidence_agent.retrieval.coverage import (
    CoverageReport,
    EvidenceAssignment,
    EvidenceSide,
    FacetAssignment,
)
from financial_evidence_agent.retrieval.hybrid import HashEmbeddingProvider, HybridRetriever
from financial_evidence_agent.retrieval.ingest import ingest_fixture
from financial_evidence_agent.skills.models import ResearchFacet
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.repositories import FilingRepository
from financial_evidence_agent.web_evidence.source_policy import PolicyValidatedWebEvidence


def _chunk(
    chunk_id: str,
    *,
    ticker: str = "NVDA",
    corpus_version: str = "NVDA-v1",
) -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker=ticker,
        corpus_version=corpus_version,
        content=f"Evidence for {chunk_id}.",
        source_url=f"https://www.sec.gov/Archives/{chunk_id}.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=20,
    )


@dataclass
class WorkflowRecorder:
    mcp_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    analyst_inputs: list[tuple[list[ResearchQuestion], list[EvidenceChunk]]] = field(
        default_factory=list
    )
    trace_events: list[str] = field(default_factory=list)


class PlanningModel:
    def route(self, thesis: str) -> RouterDecision:
        return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=f"research: {thesis}")

    def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
        return [
            ResearchQuestion(
                question="Does demand support growth?",
                support_query="data center demand revenue growth",
                challenge_query="data center revenue growth risks",
                forms=["10-Q"],
            ),
            ResearchQuestion(
                question="What capacity constraints challenge growth?",
                support_query="capacity supply growth",
                challenge_query="capacity constraints headwinds",
                forms=["10-Q", "8-K"],
            ),
        ]


class RecordingAnalyst:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        self._recorder.analyst_inputs.append((questions, evidence))
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=evidence[0].content.split(".", maxsplit=1)[0] + ".",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[evidence[0].id],
                )
            ],
            counter_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=evidence[-1].content.split(".", maxsplit=1)[0] + ".",
                    confidence=Confidence.MEDIUM,
                    evidence_chunk_ids=[evidence[-1].id],
                )
            ],
            information_sufficiency="B",
            confidence=Confidence.MEDIUM,
        )


class UnsupportedCounterAnalyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=evidence[0].content,
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[evidence[0].id],
                )
            ],
            counter_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="Unsupported counter-evidence.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=["missing-citation"],
                )
            ],
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


@dataclass
class RecordingThesisRepairer:
    calls: list[tuple[ResearchMemo, tuple[str, ...], object]] = field(
        default_factory=list
    )

    def repair(
        self,
        *,
        draft: ResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: object,
    ) -> ResearchMemo:
        self.calls.append((draft, guard_errors, evidence))
        return draft.model_copy(
            update={
                "counter_claims": [],
                "open_questions": [
                    Claim(
                        kind=ClaimKind.OPEN_QUESTION,
                        text="Unsupported counter-evidence.",
                        confidence=Confidence.LOW,
                    )
                ],
                "information_sufficiency": "C",
                "confidence": Confidence.LOW,
            }
        )


@dataclass
class NonMonotonicThesisRepairer:
    mutation: str

    def repair(
        self,
        *,
        draft: ResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: object,
    ) -> ResearchMemo:
        del guard_errors, evidence
        valid = draft.supporting_claims[0]
        updates: dict[str, object] = {
            "supporting_claims": [],
            "counter_claims": [],
            "inferences": [],
            "open_questions": [],
            "information_sufficiency": "C",
            "confidence": Confidence.LOW,
        }
        if self.mutation == "move":
            updates["counter_claims"] = [valid]
        elif self.mutation == "downgrade":
            updates["open_questions"] = [
                valid.model_copy(
                    update={
                        "kind": ClaimKind.OPEN_QUESTION,
                        "confidence": Confidence.LOW,
                        "evidence_chunk_ids": [],
                    }
                )
            ]
        elif self.mutation == "remove_source":
            updates["supporting_claims"] = [
                valid.model_copy(update={"evidence_chunk_ids": []})
            ]
        return draft.model_copy(update=updates)


class ScopedMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        if arguments["query"] == "data center demand revenue growth":
            return {
                "chunks": [
                    _chunk("shared"),
                    _chunk("wrong-ticker", ticker="AMD", corpus_version="AMD-v1"),
                ],
                "error": None,
            }
        if arguments["query"] == "data center revenue growth risks":
            return {
                "chunks": [
                    _chunk("shared"),
                    _chunk("wrong-corpus", corpus_version="NVDA-v2"),
                ],
                "error": None,
            }
        return {"chunks": [_chunk("second")], "error": None}


class RecordingTraceSink:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def record(self, node: str, attributes: dict[str, object]) -> None:
        del attributes
        self._recorder.trace_events.append(node)


def _dependencies(recorder: WorkflowRecorder, mcp_client: object | None = None) -> Dependencies:
    return Dependencies(
        mcp_client=mcp_client or ScopedMCPClient(recorder),
        fast_model=PlanningModel(),
        analyst_model=RecordingAnalyst(recorder),
        trace_sink=RecordingTraceSink(recorder),
    )


def test_graph_retrieves_support_and_challenge_per_question_then_scopes_and_deduplicates() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        " nvda ",
        "  Does data center demand support revenue growth?  ",
        _dependencies(recorder),
    )

    assert [name for name, _ in recorder.mcp_calls] == ["hybrid_search_filings"] * 4
    assert [arguments["query"] for _, arguments in recorder.mcp_calls] == [
        "data center demand revenue growth",
        "data center revenue growth risks",
        "capacity supply growth",
        "capacity constraints headwinds",
    ]
    assert all(arguments["ticker"] == "NVDA" for _, arguments in recorder.mcp_calls)
    assert set(result.evidence) == {"shared", "second"}
    assert result.corpus_version == "NVDA-v1"
    assert len(recorder.analyst_inputs) == 1
    questions, evidence = recorder.analyst_inputs[0]
    assert questions == result.questions
    assert evidence == list(result.evidence.values())


def test_graph_exposes_compiled_nodes_and_executes_required_order() -> None:
    recorder = WorkflowRecorder()
    dependencies = _dependencies(recorder)
    graph = build_research_graph(dependencies)

    result = run_research("NVDA", "Does data center demand support revenue growth?", dependencies)

    assert set(graph.get_graph().nodes) >= {
        "safety_router",
        "normalize",
        "load_session_memory",
        "load_research_memory",
        "plan_questions",
        "retrieve_evidence",
        "analyze",
        "citation_guard",
        "repair_draft",
        "render",
    }
    assert result.node_trace == [
        "safety_router",
        "normalize",
        "load_session_memory",
        "load_research_memory",
        "plan_questions",
        "retrieve_evidence",
        "analyze",
        "citation_guard",
        "render",
    ]
    assert recorder.trace_events == result.node_trace
    assert result.memo is not None
    assert result.guarded_memo is not None
    assert result.guarded_memo.ticker == "NVDA"
    assert result.guarded_memo.corpus_version == "NVDA-v1"
    assert "## Verified evidence supporting thesis" in result.rendered_output
    assert "## Sources" in result.rendered_output
    assert "Research assistance only; not investment advice." in result.rendered_output


def test_guard_failure_triggers_one_restrictive_repair_and_deterministic_reguard() -> None:
    recorder = WorkflowRecorder()
    repairer = RecordingThesisRepairer()
    dependencies = replace(
        _dependencies(recorder),
        analyst_model=UnsupportedCounterAnalyst(),
        thesis_repair_model=repairer,
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert len(repairer.calls) == 1
    assert "missing-citation" in repr(repairer.calls[0][1])
    assert result.node_trace[-3:] == ["citation_guard", "repair_draft", "render"]
    assert result.guarded_memo is not None
    assert [claim.text for claim in result.guarded_memo.open_questions] == [
        "Unsupported counter-evidence."
    ]
    assert "missing-citation" not in result.rendered_output


def test_clean_guard_does_not_call_repair_model() -> None:
    class UnexpectedRepairer:
        def repair(self, **kwargs: object) -> ResearchMemo:
            raise AssertionError(f"clean guard called repair: {kwargs}")

    recorder = WorkflowRecorder()
    dependencies = replace(
        _dependencies(recorder),
        thesis_repair_model=UnexpectedRepairer(),
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert "repair_draft" not in result.node_trace


def test_repair_cannot_introduce_a_new_citation() -> None:
    class ExpandingRepairer:
        def repair(
            self,
            *,
            draft: ResearchMemo,
            guard_errors: tuple[str, ...],
            evidence: object,
        ) -> ResearchMemo:
            del guard_errors, evidence
            return draft.model_copy(
                update={
                    "counter_claims": [],
                    "open_questions": [
                        Claim(
                            kind=ClaimKind.OPEN_QUESTION,
                            text="Unsupported counter-evidence.",
                            confidence=Confidence.LOW,
                            evidence_chunk_ids=["new-citation"],
                        )
                    ],
                    "information_sufficiency": "C",
                    "confidence": Confidence.LOW,
                }
            )

    recorder = WorkflowRecorder()
    dependencies = replace(
        _dependencies(recorder),
        analyst_model=UnsupportedCounterAnalyst(),
        thesis_repair_model=ExpandingRepairer(),
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert result.guarded_memo is not None
    assert result.guarded_memo.open_questions == []
    assert "new-citation" not in result.rendered_output
    assert "REPAIR_OUTPUT_REJECTED" in result.errors


@pytest.mark.parametrize("mutation", ["delete", "move", "downgrade", "remove_source"])
def test_repair_cannot_reduce_or_move_original_guarded_thesis_content(
    mutation: str,
) -> None:
    recorder = WorkflowRecorder()
    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        replace(
            _dependencies(recorder),
            analyst_model=UnsupportedCounterAnalyst(),
            thesis_repair_model=NonMonotonicThesisRepairer(mutation),
        ),
    )

    assert result.memo is not None
    assert result.corpus_version is not None
    original = guard_memo(
        result.memo,
        result.evidence,
        ticker="NVDA",
        corpus_version=result.corpus_version,
    )
    assert [claim.text for claim in original.supporting_claims] == [
        "Evidence for shared."
    ]
    assert [source.id for source in original.sources] == ["shared"]
    assert result.guarded_memo == original
    assert "REPAIR_OUTPUT_REJECTED" in result.errors


def test_repair_rejects_new_fact_text_and_malformed_structured_output() -> None:
    class NewFactRepairer:
        def repair(self, *, draft, guard_errors, evidence) -> ResearchMemo:
            del guard_errors, evidence
            return draft.model_copy(
                update={
                    "counter_claims": [],
                    "open_questions": [
                        Claim(
                            kind=ClaimKind.OPEN_QUESTION,
                            text="A new unsupported fact was introduced.",
                            confidence=Confidence.LOW,
                        )
                    ],
                    "information_sufficiency": "C",
                    "confidence": Confidence.LOW,
                }
            )

    class MalformedRepairer:
        def repair(self, **kwargs: object) -> object:
            del kwargs
            return {"research_question": "incomplete"}

    results = [
        run_research(
            "NVDA",
            "Does data center demand support revenue growth?",
            replace(
                _dependencies(WorkflowRecorder()),
                analyst_model=UnsupportedCounterAnalyst(),
                thesis_repair_model=repairer,
            ),
        )
        for repairer in (NewFactRepairer(), MalformedRepairer())
    ]

    assert all(result.guarded_memo is not None for result in results)
    assert all(result.guarded_memo.open_questions == [] for result in results)  # type: ignore[union-attr]
    assert all("REPAIR_OUTPUT_REJECTED" in result.errors for result in results)
    assert all("new unsupported fact" not in result.rendered_output for result in results)


def test_repair_budget_exhaustion_and_provider_failure_keep_original_guarded_result() -> None:
    class UnexpectedRepairer:
        def repair(self, **kwargs: object) -> ResearchMemo:
            raise AssertionError(f"exhausted repair budget reached provider: {kwargs}")

    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=1,
            max_analysis_calls=1,
            max_repair_calls=0,
            max_tool_calls=20,
            max_retrieval_rounds=2,
            max_web_calls=1,
        )
    )
    recorder = WorkflowRecorder()
    exhausted = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        replace(
            _dependencies(recorder),
            analyst_model=UnsupportedCounterAnalyst(),
            thesis_repair_model=UnexpectedRepairer(),
            budget=authority,
        ),
    )

    class FailingRepairer:
        def repair(self, **kwargs: object) -> ResearchMemo:
            del kwargs
            raise RuntimeError("private provider failure")

    failed = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        replace(
            _dependencies(WorkflowRecorder()),
            analyst_model=UnsupportedCounterAnalyst(),
            thesis_repair_model=FailingRepairer(),
        ),
    )

    assert exhausted.guarded_memo is not None
    assert failed.guarded_memo is not None
    assert exhausted.guarded_memo.supporting_claims == failed.guarded_memo.supporting_claims
    assert exhausted.guarded_memo.open_questions == failed.guarded_memo.open_questions == []
    assert "REPAIR_SKIPPED: budget_exhausted" in exhausted.errors
    assert "REPAIR_FAILED: RuntimeError" in failed.errors
    assert "private provider failure" not in repr(failed.errors)


def test_repaired_output_is_guarded_again_before_rendering() -> None:
    unsafe_text = "You should buy NVDA now."

    class UnsafeAnalyst(UnsupportedCounterAnalyst):
        def analyze(self, questions, evidence) -> ResearchMemo:
            draft = super().analyze(questions, evidence)
            return draft.model_copy(
                update={
                    "counter_claims": [
                        Claim(
                            kind=ClaimKind.VERIFIED_FACT,
                            text=unsafe_text,
                            confidence=Confidence.HIGH,
                            evidence_chunk_ids=["missing-citation"],
                        )
                    ]
                }
            )

    class UnsafeDowngradeRepairer:
        def repair(self, *, draft, guard_errors, evidence) -> ResearchMemo:
            del guard_errors, evidence
            return draft.model_copy(
                update={
                    "counter_claims": [],
                    "open_questions": [
                        Claim(
                            kind=ClaimKind.OPEN_QUESTION,
                            text=unsafe_text,
                            confidence=Confidence.LOW,
                        )
                    ],
                    "information_sufficiency": "C",
                    "confidence": Confidence.LOW,
                }
            )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        replace(
            _dependencies(WorkflowRecorder()),
            analyst_model=UnsafeAnalyst(),
            thesis_repair_model=UnsafeDowngradeRepairer(),
        ),
    )

    assert result.guarded_memo is not None
    assert result.guarded_memo.open_questions == []
    assert unsafe_text not in result.rendered_output


def test_production_thesis_retries_only_missing_coverage_then_uses_one_scoped_web_call() -> None:
    """Broadening retry scope, bypassing the collector, or flattening web provenance fails."""
    local_calls: list[tuple[object, ...]] = []
    web_calls: list[object] = []
    analyst_bundles: list[EvidenceBundle] = []

    class OneQuestionPlanner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support disclosure",
                    challenge_query="challenge disclosure",
                )
            ]

    async def local_search(queries):
        local_calls.append(queries)
        if len(local_calls) == 1:
            support = next(query for query in queries if query.side is EvidenceSide.SUPPORT)
            return LocalSearchResponse(
                evidence=(
                    LocalEvidenceHit(
                        evidence=_chunk("filing-support"),
                        question_index=0,
                        side=EvidenceSide.SUPPORT,
                        facet=support.facet,
                    ),
                )
            )
        return LocalSearchResponse()

    web_source = WebEvidence(
        id="web-challenge",
        ticker="NVDA",
        title="Issuer risk update",
        content="The issuer disclosed a material challenge.",
        source_url="https://investor.nvidia.com/risk-update",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, tzinfo=UTC),
        content_hash="hash-web-challenge",
    )

    async def web_search(request):
        web_calls.append(request)
        question_index, side = request.missing_pairs[0]
        return WebSearchResponse(
            evidence=(
                WebEvidenceHit(
                    evidence=web_source,
                    question_index=question_index,
                    side=side,
                    facet=request.missing_facets[0],
                ),
            )
        )

    class BundleAnalyst:
        def analyze(
            self,
            questions: list[ResearchQuestion],
            evidence: EvidenceBundle,
        ) -> ResearchMemo:
            analyst_bundles.append(evidence)
            return ResearchMemo(
                research_question=questions[0].question,
                supporting_claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text="The filing supplies supporting evidence.",
                        confidence=Confidence.HIGH,
                        evidence_chunk_ids=["filing-support"],
                    )
                ],
                counter_claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text="The issuer update supplies counter-evidence.",
                        confidence=Confidence.MEDIUM,
                        web_evidence_ids=["web-challenge"],
                    )
                ],
                information_sufficiency="B",
                confidence=Confidence.MEDIUM,
            )

    class NoDirectMCP:
        def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise AssertionError(f"graph bypassed collector: {name} {arguments!r}")

    class Validator:
        policy_version = "policy-v1"

        def validate(self, *, ticker: str, evidence: WebEvidence):
            assert ticker == "NVDA"
            return PolicyValidatedWebEvidence(
                **evidence.model_dump(),
                policy_version=self.policy_version,
                canonical_url=evidence.source_url,
            )

    result = run_research(
        "NVDA",
        "Does the evidence support the thesis despite disclosed challenges?",
        Dependencies(
            mcp_client=NoDirectMCP(),
            fast_model=OneQuestionPlanner(),
            analyst_model=BundleAnalyst(),
            thesis_collector=EvidenceCollector(
                local_search=local_search,
                web_search=web_search,
                corpus_version="NVDA-v1",
                filing_ids=("filing-1",),
            ),
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
            web_evidence_validator=Validator(),  # type: ignore[arg-type]
        ),
        intent=Intent.RESEARCH_REQUEST,
        corpus_version="NVDA-v1",
        filing_ids=("filing-1",),
    )

    assert len(local_calls) == 2
    assert all(query.corpus_version == "NVDA-v1" for call in local_calls for query in call)
    assert all(query.filing_ids == ("filing-1",) for call in local_calls for query in call)
    assert {query.side for query in local_calls[1]} == {EvidenceSide.CHALLENGE}
    assert len(web_calls) == 1
    assert web_calls[0].ticker == "NVDA"
    assert web_calls[0].missing_pairs == ((0, EvidenceSide.CHALLENGE),)
    assert len(analyst_bundles) == 1
    assert [source.id for source in analyst_bundles[0].filing_evidence] == ["filing-support"]
    assert [source.id for source in analyst_bundles[0].web_evidence] == ["web-challenge"]
    assert result.guarded_memo is not None
    assert [source.id for source in result.guarded_memo.web_sources] == ["web-challenge"]
    assert result.node_trace == [
        "safety_router",
        "normalize",
        "load_session_memory",
        "load_research_memory",
        "plan_questions",
        "retrieve_evidence",
        "evidence_coverage",
        "retry_missing_evidence",
        "retry_coverage",
        "web_fallback",
        "final_coverage",
        "analyze",
        "citation_guard",
        "render",
    ]


def test_production_thesis_incomplete_final_coverage_skips_analyst() -> None:
    local_calls: list[tuple[object, ...]] = []
    analyst_calls: list[object] = []

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support disclosure",
                    challenge_query="challenge disclosure",
                )
            ]

    async def local_search(queries):
        local_calls.append(queries)
        if len(local_calls) == 1:
            support = next(query for query in queries if query.side is EvidenceSide.SUPPORT)
            return LocalSearchResponse(
                evidence=(
                    LocalEvidenceHit(
                        evidence=_chunk("support-only"),
                        question_index=0,
                        side=EvidenceSide.SUPPORT,
                        facet=support.facet,
                    ),
                )
            )
        return LocalSearchResponse()

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("incomplete final coverage reached analyst")

    class NoDirectMCP:
        def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise AssertionError(f"graph bypassed collector: {name} {arguments!r}")

    dependencies = Dependencies(
        mcp_client=NoDirectMCP(),
        fast_model=Planner(),
        analyst_model=Analyst(),
        thesis_collector=EvidenceCollector(
            local_search=local_search,
            corpus_version="NVDA-v1",
            filing_ids=("filing-1",),
        ),
        thesis_collection_policy=THESIS_COLLECTION_POLICY,
    )
    state = build_research_graph(dependencies).invoke(
        {
            "ticker": "NVDA",
            "thesis": "Does disclosed demand support growth despite challenges?",
            "node_trace": [],
            "requested_intent": Intent.RESEARCH_REQUEST,
            "corpus_version": "NVDA-v1",
            "filing_ids": ("filing-1",),
            "scope_error": None,
        }
    )
    result = workflow_module._build_result(state, run_id="run-incomplete")

    assert len(local_calls) == 2
    assert analyst_calls == []
    assert state["evidence_bundle"].coverage.complete is False
    assert "insufficient_information" in state["evidence_bundle"].coverage.reason_codes
    assert result.status == "insufficient_evidence"
    assert result.memo is None
    assert result.guarded_memo is None
    assert result.rendered_output == "Insufficient evidence to produce a research memo."
    assert result.node_trace[-3:] == ["web_fallback", "final_coverage", "render"]


def test_production_thesis_budget_exhaustion_fails_closed_before_analysis() -> None:
    """A collector reporting work beyond the run limit must not reach the analyst."""
    analyst_calls: list[object] = []

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support disclosure",
                    challenge_query="challenge disclosure",
                )
            ]

    class OverBudgetCollector:
        gate = None

        def with_budget_gate(self, gate):
            self.gate = gate
            return self

        async def retrieve(self, **values: object) -> EvidenceBundle:
            del values
            assert self.gate is not None
            self.gate.consume(retrieval_rounds=2)
            source = _chunk("support")
            challenge = _chunk("challenge")
            return EvidenceBundle(
                filing_evidence=[source, challenge],
                web_evidence=[],
                assignments=[],
                coverage=CoverageReport(
                    complete=True,
                    missing_facets=(),
                    missing_pairs=(),
                    invalid_source_ids=(),
                    ticker_mismatches=(),
                    date_mismatches=(),
                    new_valid_source_count=2,
                    reason_codes=(),
                ),
                retrieval_rounds=2,
                web_calls=0,
            )

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("budget exhaustion reached analyst")

    one_round_budget = THESIS_COLLECTION_POLICY.budget.model_copy(
        update={"max_retrieval_rounds": 1}
    )
    policy = THESIS_COLLECTION_POLICY.model_copy(update={"budget": one_round_budget})
    result = run_research(
        "NVDA",
        "Does disclosed demand support growth despite challenges?",
        Dependencies(
            mcp_client=object(),  # type: ignore[arg-type]
            fast_model=Planner(),
            analyst_model=Analyst(),
            thesis_collector=OverBudgetCollector(),  # type: ignore[arg-type]
            thesis_collection_policy=policy,
        ),
        intent=Intent.RESEARCH_REQUEST,
        corpus_version="NVDA-v1",
        filing_ids=("filing-1",),
    )

    assert result.status == "insufficient_evidence"
    assert result.memo is None
    assert result.guarded_memo is None
    assert analyst_calls == []
    assert "BUDGET_EXHAUSTED: retrieval_rounds" in result.errors
    assert result.rendered_output == (
        "Insufficient evidence: the frozen research budget was exhausted."
    )


def test_result_builder_rejects_completed_claims_when_final_bundle_is_incomplete() -> None:
    support = _chunk("support-only")
    guarded = guard_memo(
        ResearchMemo(
            research_question="Does evidence support the thesis?",
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="A filing fact was retained.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[support.id],
                )
            ],
            counter_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="A counter fact was retained.",
                    confidence=Confidence.MEDIUM,
                    evidence_chunk_ids=[support.id],
                )
            ],
            information_sufficiency="B",
            confidence=Confidence.MEDIUM,
        ),
        {support.id: support},
        ticker="NVDA",
        corpus_version="NVDA-v1",
    )
    incomplete = EvidenceBundle(
        filing_evidence=[support],
        web_evidence=[],
        assignments=[
            EvidenceAssignment(
                question_index=0,
                side=EvidenceSide.SUPPORT,
                source_id=support.id,
                source_kind=SourceKind.FILING,
            )
        ],
        facet_assignments=[
            FacetAssignment(
                question_index=0,
                side=EvidenceSide.SUPPORT,
                facet=ResearchFacet.INFORMATION_GAPS,
                source_id=support.id,
            )
        ],
        coverage=CoverageReport(
            complete=False,
            missing_facets=(),
            missing_pairs=((0, EvidenceSide.CHALLENGE),),
            invalid_source_ids=(),
            ticker_mismatches=(),
            date_mismatches=(),
            new_valid_source_count=0,
            reason_codes=("challenge_missing", "insufficient_information"),
        ),
        retrieval_rounds=2,
        web_calls=0,
    )
    state = {
        "ticker": "NVDA",
        "thesis": "Does evidence support the thesis?",
        "node_trace": [],
        "decision": RouterDecision(
            intent=Intent.RESEARCH_REQUEST,
            reason="test",
        ),
        "questions": [],
        "evidence": {support.id: support},
        "evidence_bundle": incomplete,
        "memo": guarded,
        "guarded_memo": guarded,
        "errors": [],
        "rendered_output": "must not count as completed",
    }

    result = workflow_module._build_result(state, run_id="run-defense")

    assert result.status == "insufficient_evidence"
    assert result.memo is None
    assert result.guarded_memo is None
    assert result.rendered_output == "Insufficient evidence to produce a research memo."
    assert result.to_run_finish(trace_id=None).claims == []
    assert result.to_run_finish(trace_id=None).prompt_version is None


def test_production_thesis_retry_protocol_error_fails_closed_before_web_or_analyst() -> None:
    local_calls: list[tuple[object, ...]] = []
    web_calls: list[object] = []
    analyst_calls: list[object] = []

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support",
                    challenge_query="challenge",
                )
            ]

    async def local_search(queries):
        local_calls.append(queries)
        if len(local_calls) == 1:
            support = next(query for query in queries if query.side is EvidenceSide.SUPPORT)
            return LocalSearchResponse(
                evidence=(
                    LocalEvidenceHit(
                        evidence=_chunk("support-before-error"),
                        question_index=0,
                        side=EvidenceSide.SUPPORT,
                        facet=support.facet,
                    ),
                )
            )
        return LocalSearchResponse(
            error=RetrievalError(
                code=RetrievalErrorCode.PROTOCOL_ERROR,
                message="malformed retry envelope",
            )
        )

    async def web_search(request):
        web_calls.append(request)
        return WebSearchResponse()

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("protocol failure reached analyst")

    class NoDirectMCP:
        def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise AssertionError(f"graph bypassed collector: {name} {arguments!r}")

    result = run_research(
        "NVDA",
        "Does the evidence support the thesis despite disclosed challenges?",
        Dependencies(
            mcp_client=NoDirectMCP(),
            fast_model=Planner(),
            analyst_model=Analyst(),
            thesis_collector=EvidenceCollector(
                local_search=local_search,
                web_search=web_search,
                corpus_version="NVDA-v1",
                filing_ids=("filing-1",),
            ),
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
        ),
        intent=Intent.RESEARCH_REQUEST,
        corpus_version="NVDA-v1",
        filing_ids=("filing-1",),
    )

    assert len(local_calls) == 2
    assert web_calls == []
    assert analyst_calls == []
    assert result.status == "failed"
    assert result.evidence == {}
    assert result.memo is None
    assert "Unable to retrieve evidence safely" in result.rendered_output
    assert result.errors == ["THESIS_COLLECTION_PROTOCOL_ERROR"]


def test_upstream_collection_detail_never_reaches_result_markdown_or_finish() -> None:
    """Provider exception detail is untrusted and must collapse to a stable code."""
    private = "password=private-password-value"

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support",
                    challenge_query="challenge",
                )
            ]

    class Collector:
        async def retrieve(self, **kwargs):
            del kwargs
            raise EvidenceCollectionError(CollectionErrorCode.RETRIEVAL_ERROR, private)

    class Analyst:
        def analyze(self, questions, evidence):
            raise AssertionError(f"upstream failure reached analyst: {questions!r} {evidence!r}")

    result = run_research(
        "NVDA",
        "Does public evidence support sustained revenue growth?",
        Dependencies(
            mcp_client=object(),  # type: ignore[arg-type]
            fast_model=Planner(),
            analyst_model=Analyst(),
            thesis_collector=Collector(),  # type: ignore[arg-type]
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
        ),
        intent=Intent.RESEARCH_REQUEST,
        corpus_version="NVDA-v1",
        filing_ids=("filing-1",),
    )

    assert result.status == "failed"
    assert result.errors == ["THESIS_COLLECTION_RETRIEVAL_ERROR"]
    assert private not in repr(result)
    assert private not in result.rendered_output
    assert private not in repr(result.to_run_finish("trace-upstream-error"))


def test_production_thesis_planner_provider_error_stops_before_collection() -> None:
    collector_calls: list[str] = []
    analyst_calls: list[object] = []

    class FailingPlanner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker, thesis
            raise RuntimeError("provider unavailable")

    class Collector:
        async def retrieve(self, **kwargs):
            collector_calls.append(repr(kwargs))
            raise AssertionError("planner failure reached collector")

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("planner failure reached analyst")

    class NoDirectMCP:
        def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise AssertionError(f"planner failure reached MCP: {name} {arguments!r}")

    result = run_research(
        "NVDA",
        "Does disclosed demand support sustained revenue growth?",
        Dependencies(
            mcp_client=NoDirectMCP(),
            fast_model=FailingPlanner(),
            analyst_model=Analyst(),
            thesis_collector=Collector(),  # type: ignore[arg-type]
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
        ),
        intent=Intent.RESEARCH_REQUEST,
        corpus_version="NVDA-v1",
        filing_ids=("filing-1",),
    )

    assert result.status == "failed"
    assert collector_calls == []
    assert analyst_calls == []
    assert result.errors == ["FAST_PLAN_ERROR: RuntimeError"]
    assert "Unable to complete thesis research safely" in result.rendered_output


def test_no_filings_scope_uses_web_fallback_without_local_retrieval_or_analysis() -> None:
    local_calls: list[object] = []
    web_calls: list[object] = []
    analyst_calls: list[object] = []

    async def local_search(queries):
        local_calls.append(queries)
        raise AssertionError("NO_FILINGS reached local retrieval")

    async def web_search(request):
        web_calls.append(request)
        return WebSearchResponse()

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support",
                    challenge_query="challenge",
                )
            ]

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("NO_FILINGS reached analyst")

    class NoDirectMCP:
        def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise AssertionError(f"NO_FILINGS reached MCP: {name} {arguments!r}")

    result = run_research(
        "NVDA",
        "Does the selected filing scope support the thesis?",
        Dependencies(
            mcp_client=NoDirectMCP(),
            fast_model=Planner(),
            analyst_model=Analyst(),
            thesis_collector=EvidenceCollector(
                local_search=local_search,
                web_search=web_search,
                corpus_version=None,
                filing_ids=(),
                scope_error="NO_FILINGS: no filings matched forms/date",
            ),
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
        ),
        intent=Intent.RESEARCH_REQUEST,
        scope_error="NO_FILINGS: no filings matched forms/date",
    )

    assert result.status == "insufficient_evidence"
    assert local_calls == []
    assert len(web_calls) == 1
    assert analyst_calls == []
    assert result.errors == ["NO_FILINGS: no filings matched forms/date"]
    assert result.rendered_output == "Insufficient evidence to produce a research memo."
    assert result.node_trace[-3:] == ["web_fallback", "final_coverage", "render"]


@pytest.mark.parametrize(
    "scope_error",
    [
        "MCP_CALL_ERROR: filing scope resolution failed",
        "MCP_PROTOCOL_ERROR: filing scope tool returned is_error",
        "MCP_RESPONSE_ERROR: invalid filing scope response",
    ],
)
def test_scope_transport_or_protocol_failure_is_fatal_without_downstream_calls(
    scope_error: str,
) -> None:
    local_calls: list[object] = []
    analyst_calls: list[object] = []

    async def local_search(queries):
        local_calls.append(queries)
        raise AssertionError("fatal scope error reached local retrieval")

    class Planner:
        def route(self, thesis: str) -> RouterDecision:
            return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=thesis)

        def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
            del ticker
            return [
                ResearchQuestion(
                    question=thesis,
                    support_query="support",
                    challenge_query="challenge",
                )
            ]

    class Analyst:
        def analyze(self, questions, evidence):
            analyst_calls.append((questions, evidence))
            raise AssertionError("fatal scope error reached analyst")

    result = run_research(
        "NVDA",
        "Does the selected filing scope support the thesis?",
        Dependencies(
            mcp_client=object(),  # type: ignore[arg-type]
            fast_model=Planner(),
            analyst_model=Analyst(),
            thesis_collector=EvidenceCollector(
                local_search=local_search,
                corpus_version=None,
                filing_ids=(),
                scope_error=scope_error,
            ),
            thesis_collection_policy=THESIS_COLLECTION_POLICY,
        ),
        intent=Intent.RESEARCH_REQUEST,
        scope_error=scope_error,
    )

    assert result.status == "failed"
    assert local_calls == []
    assert analyst_calls == []
    assert result.errors == ["THESIS_COLLECTION_RETRIEVAL_ERROR"]


def test_caller_selected_thesis_intent_cannot_be_rerouted_to_p1() -> None:
    """An explicit thesis mode must remain on P0 after deterministic safety checks."""
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "介绍一下 NVDA",
        _dependencies(recorder),
        intent=Intent.RESEARCH_REQUEST,
    )

    assert result.decision.intent is Intent.RESEARCH_REQUEST
    assert result.skill_runs == []
    assert recorder.mcp_calls
    assert result.node_trace[-1] == "render"


class EmptyMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        return {
            "chunks": [],
            "error": {
                "code": "EMPTY_RETRIEVAL",
                "message": "No evidence matched the selected filing corpus",
            },
        }


class ErrorWithChunksMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        return {
            "chunks": [_chunk("must-not-reach-analyst")],
            "error": {
                "code": "EMPTY_RETRIEVAL",
                "message": "Search failed closed",
            },
        }


class ProtocolErrorMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        return CallToolResult(
            content=[],
            structured_content={
                "chunks": [_chunk("must-not-reach-analyst").model_dump(mode="json")],
                "error": None,
            },
            meta=None,
            is_error=True,
        )


class MalformedChunkMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        malformed = _chunk("malformed").model_dump(mode="json")
        malformed["raw_end"] = 0
        return CallToolResult(
            content=[],
            structured_content={"chunks": [malformed], "error": None},
            meta=None,
            is_error=False,
        )


class MixedResultMCPClient:
    def __init__(self, recorder: WorkflowRecorder) -> None:
        self._recorder = recorder
        self._call_count = 0

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self._recorder.mcp_calls.append((name, arguments))
        self._call_count += 1
        if self._call_count == 1:
            return {"chunks": [_chunk("partial-success")], "error": None}
        if self._call_count == 2:
            return {
                "chunks": [],
                "error": {"code": "EMPTY_RETRIEVAL", "message": "challenge failed"},
            }
        return {"chunks": [_chunk("later-success")], "error": None}


def test_mcp_empty_error_envelope_returns_insufficient_without_analyst_fabrication() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        _dependencies(recorder, EmptyMCPClient(recorder)),
    )

    assert result.status == "insufficient_evidence"
    assert result.memo is None
    assert result.evidence == {}
    assert recorder.analyst_inputs == []
    assert len(recorder.mcp_calls) == 4
    assert result.errors == ["EMPTY_RETRIEVAL: No evidence matched the selected filing corpus"]
    assert "insufficient" in result.rendered_output.lower()
    assert result.node_trace[-3:] == ["analyze", "citation_guard", "render"]


def test_mcp_error_envelope_never_forwards_returned_chunks_to_analyst() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        _dependencies(recorder, ErrorWithChunksMCPClient(recorder)),
    )

    assert result.status == "insufficient_evidence"
    assert result.evidence == {}
    assert recorder.analyst_inputs == []


def test_fastmcp_protocol_error_is_checked_before_structured_content() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        _dependencies(recorder, ProtocolErrorMCPClient(recorder)),
    )

    assert result.status == "insufficient_evidence"
    assert result.evidence == {}
    assert recorder.analyst_inputs == []
    assert result.errors == ["MCP_PROTOCOL_ERROR: hybrid_search_filings returned is_error"]


def test_malformed_fastmcp_structured_chunk_fails_closed_explicitly() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        _dependencies(recorder, MalformedChunkMCPClient(recorder)),
    )

    assert result.status == "insufficient_evidence"
    assert result.evidence == {}
    assert recorder.analyst_inputs == []
    assert result.errors == ["MCP_RESPONSE_ERROR: Invalid MCP response envelope"]


def test_one_failed_required_retrieval_discards_partial_and_later_evidence() -> None:
    recorder = WorkflowRecorder()

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        _dependencies(recorder, MixedResultMCPClient(recorder)),
    )

    assert result.status == "insufficient_evidence"
    assert result.evidence == {}
    assert recorder.analyst_inputs == []
    assert result.errors == ["EMPTY_RETRIEVAL: challenge failed"]


def test_sync_adapter_runs_workflow_against_real_fastmcp_server_and_fixture() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = FilingRepository(engine)
    assert ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    server = create_server(repository, HybridRetriever(repository, HashEmbeddingProvider()))
    recorder = WorkflowRecorder()
    dependencies = Dependencies(
        mcp_client=FastMCPToolClient(server),
        fast_model=PlanningModel(),
        analyst_model=RecordingAnalyst(recorder),
        trace_sink=RecordingTraceSink(recorder),
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert result.status == "completed"
    assert result.memo is not None
    assert result.guarded_memo is not None
    assert len(result.guarded_memo.verified_claims) == 2
    assert result.evidence
    assert all(chunk.ticker == "NVDA" for chunk in result.evidence.values())
    assert len(recorder.analyst_inputs) == 1
    assert result.node_trace == [
        "safety_router",
        "normalize",
        "load_session_memory",
        "load_research_memory",
        "plan_questions",
        "retrieve_evidence",
        "analyze",
        "citation_guard",
        "render",
    ]


class FabricatingAnalyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        del evidence
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="Fabricated growth fact.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=["not-returned"],
                )
            ],
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


class OneSidedAfterGuardAnalyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=evidence[0].content,
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[evidence[0].id],
                )
            ],
            counter_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="Fabricated counter fact.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=["not-returned"],
                )
            ],
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


def test_graph_status_and_report_are_insufficient_after_one_evidence_side_is_lost() -> None:
    """A C/low one-sided guarded report must not carry a completed run status."""
    recorder = WorkflowRecorder()
    dependencies = Dependencies(
        mcp_client=ScopedMCPClient(recorder),
        fast_model=PlanningModel(),
        analyst_model=OneSidedAfterGuardAnalyst(),
        trace_sink=RecordingTraceSink(recorder),
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert result.status == "insufficient_evidence"
    assert result.guarded_memo is not None
    assert len(result.guarded_memo.supporting_claims) == 1
    assert result.guarded_memo.counter_claims == []
    assert result.guarded_memo.information_sufficiency == "C"
    assert result.guarded_memo.confidence is Confidence.LOW
    assert "Information sufficiency: C" in result.rendered_output
    assert "Confidence: low" in result.rendered_output


def test_graph_guard_drops_fabrication_and_prevents_completed_status() -> None:
    """Retrieved evidence alone must not mark an uncited analyst draft completed."""
    recorder = WorkflowRecorder()
    dependencies = Dependencies(
        mcp_client=ScopedMCPClient(recorder),
        fast_model=PlanningModel(),
        analyst_model=FabricatingAnalyst(),
        trace_sink=RecordingTraceSink(recorder),
    )

    result = run_research(
        "NVDA",
        "Does data center demand support revenue growth?",
        dependencies,
    )

    assert result.status == "insufficient_evidence"
    assert result.memo is not None
    assert result.memo.supporting_claims[0].text == "Fabricated growth fact."
    assert result.guarded_memo is not None
    assert result.guarded_memo.verified_claims == []
    assert "Fabricated growth fact" not in result.rendered_output
    assert "citation 'not-returned' was not provided" in result.rendered_output
