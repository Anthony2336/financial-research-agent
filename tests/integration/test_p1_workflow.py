"""Offline integration coverage for deterministic P1 graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Literal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

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
    content_addressed_web_evidence_id,
)
from financial_evidence_agent.graph import Dependencies, run_research
from financial_evidence_agent.graph.models import SkillAnalysisInput, SkillPlanningInput
from financial_evidence_agent.reporting import guard_skill_memo
from financial_evidence_agent.retrieval.collector import (
    CollectionErrorCode,
    EvidenceBundle,
    EvidenceCollectionError,
)
from financial_evidence_agent.retrieval.coverage import (
    CoverageReport,
    EvidenceAssignment,
    EvidenceSide,
    FacetAssignment,
)
from financial_evidence_agent.skills.models import ResearchRecipe, SkillName
from financial_evidence_agent.skills.recipes import INDUSTRY_RESEARCH, P1_RECIPES
from financial_evidence_agent.skills.schemas import (
    FinancialDataPoint,
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
    VerificationStatus,
)
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
)


class UnusedP0MCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        raise AssertionError("P1 execution must not enter the P0 MCP path")


class P0Model:
    def route(self, thesis: str) -> RouterDecision:
        return RouterDecision(intent=Intent.AMBIGUOUS, reason=f"unused: {thesis}")

    def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
        raise AssertionError(f"P1 must not use P0 planner for {ticker}: {thesis}")


class P0Analyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        raise AssertionError(f"P1 must not use P0 analyst: {questions!r} {evidence!r}")


@dataclass
class P1Recorder:
    planner_recipes: list[SkillName] = field(default_factory=list)
    planner_requests: list[SkillPlanningInput] = field(default_factory=list)
    collector_recipes: list[SkillName] = field(default_factory=list)
    analyst_recipes: list[SkillName] = field(default_factory=list)
    analyst_requests: list[SkillAnalysisInput] = field(default_factory=list)
    repository_starts: list[tuple[str, str, str, dict[str, object], str | None]] = field(
        default_factory=list
    )
    repository_finishes: list[tuple[str, str, list[str], list[str]]] = field(
        default_factory=list
    )
    trace_events: list[tuple[str, dict[str, object]]] = field(default_factory=list)


class FakePlanner:
    def __init__(self, recorder: P1Recorder, *, fail_recipe: SkillName | None = None) -> None:
        self._recorder = recorder
        self._fail_recipe = fail_recipe

    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        self._recorder.planner_recipes.append(request.recipe.name)
        self._recorder.planner_requests.append(request)
        if request.recipe.name is self._fail_recipe:
            raise TimeoutError("provider secret must not be returned")
        return [
            ResearchQuestion(
                question=f"What evidence covers {request.recipe.name.value}?",
                support_query="supporting disclosure",
                challenge_query="challenging disclosure",
            )
        ]


class FakeCollector:
    def __init__(
        self,
        recorder: P1Recorder,
        *,
        incomplete_recipe: SkillName | Literal["all"] | None = None,
        protocol_error_recipe: SkillName | None = None,
        raw_incomplete_recipe: SkillName | None = None,
        secret_reason_recipe: SkillName | None = None,
    ) -> None:
        self._recorder = recorder
        self._incomplete_recipe = incomplete_recipe
        self._protocol_error_recipe = protocol_error_recipe
        self._raw_incomplete_recipe = raw_incomplete_recipe
        self._secret_reason_recipe = secret_reason_recipe

    async def collect(
        self,
        *,
        ticker: str,
        recipe: ResearchRecipe,
        questions: list[ResearchQuestion],
    ) -> EvidenceBundle:
        del questions
        self._recorder.collector_recipes.append(recipe.name)
        if recipe.name is self._protocol_error_recipe:
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "untrusted raw response with secret-token",
            )
        if recipe.name is self._raw_incomplete_recipe:
            return _bundle(
                recipe,
                ticker=ticker,
                complete=False,
                include_raw_evidence=True,
                reason_codes=("ticker_mismatch", "insufficient_information"),
            )
        if recipe.name is self._secret_reason_recipe:
            return _bundle(
                recipe,
                ticker=ticker,
                complete=False,
                reason_codes=("missing_facet", "secret-token-must-not-trace"),
            )
        if self._incomplete_recipe == "all" or recipe.name is self._incomplete_recipe:
            return _bundle(recipe, ticker=ticker, complete=False)
        return _bundle(recipe, ticker=ticker)


class FakeAnalyst:
    def __init__(
        self,
        recorder: P1Recorder,
        *,
        reject_recipe: SkillName | None = None,
        financial_conflict: bool = False,
    ) -> None:
        self._recorder = recorder
        self._reject_recipe = reject_recipe
        self._financial_conflict = financial_conflict

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        self._recorder.analyst_recipes.append(request.recipe.name)
        self._recorder.analyst_requests.append(request)
        source_id = evidence.filing_evidence[0].id
        web_source_id = evidence.web_evidence[0].id
        if request.recipe.name is self._reject_recipe:
            source_id = "hallucinated-source"
        sections = [
            SkillResearchSection(
                facet=facet,
                claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text=f"Source-backed {facet.value} finding.",
                        confidence=Confidence.HIGH,
                        evidence_chunk_ids=[source_id],
                        web_evidence_ids=[web_source_id],
                    )
                ],
            )
            for facet in request.recipe.required_facets
        ]
        data_points = []
        if request.recipe.name is SkillName.FINANCIAL_DATA_VERIFICATION:
            data_points = [_financial_point(source_id)]
            if self._financial_conflict:
                data_points = [
                    _financial_point(
                        source_id,
                        value=Decimal("44.1"),
                    ),
                    _financial_point(
                        source_id,
                        value=Decimal("44.2"),
                    ),
                ]
        return SkillResearchMemo(
            recipe_name=request.recipe.name,
            recipe_version=request.recipe.version,
            research_question=f"Review {request.recipe.name.value}.",
            sections=sections,
            data_points=data_points,
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            information_gaps=[],
            confidence=Decimal("0.9"),
        )


@dataclass
class RestrictiveSkillRepairer:
    calls: list[tuple[SkillAnalysisInput, SkillResearchMemo, tuple[str, ...]]] = field(
        default_factory=list
    )

    async def repair(
        self,
        *,
        request: SkillAnalysisInput,
        draft: SkillResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        del evidence
        self.calls.append((request, draft, guard_errors))
        return draft.model_copy(
            update={
                "sections": [
                    section.model_copy(
                        update={
                            "claims": [
                                claim.model_copy(update={"evidence_chunk_ids": []})
                                for claim in section.claims
                            ]
                        }
                    )
                    for section in draft.sections
                ],
                "information_sufficiency": InformationSufficiency.PARTIAL,
                "confidence": Decimal("0.5"),
            }
        )


class RichConflictAnalyst(FakeAnalyst):
    def __init__(self, recorder: P1Recorder) -> None:
        super().__init__(recorder, financial_conflict=True)

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        memo = await super().analyze(request=request, evidence=evidence)
        if request.recipe.name is not SkillName.FINANCIAL_DATA_VERIFICATION:
            return memo
        sections = list(memo.sections)
        first = sections[0]
        sections[0] = first.model_copy(
            update={
                "claims": [
                    *first.claims,
                    first.claims[0].model_copy(
                        update={"text": "Second retained data-verification finding."}
                    ),
                ]
            }
        )
        return memo.model_copy(
            update={
                "sections": sections,
                "information_gaps": ["Retained information gap."],
            }
        )


@dataclass
class NonMonotonicSkillRepairer:
    mutation: str
    calls: list[SkillResearchMemo] = field(default_factory=list)

    async def repair(
        self,
        *,
        request: SkillAnalysisInput,
        draft: SkillResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        del request, guard_errors, evidence
        self.calls.append(draft)
        sections = [
            section.model_copy(update={"claims": list(section.claims)})
            for section in draft.sections
        ]
        data_points = list(draft.data_points)
        information_gaps = list(draft.information_gaps)
        if self.mutation == "delete_claim":
            sections[0] = sections[0].model_copy(
                update={"claims": sections[0].claims[:1]}
            )
        elif self.mutation == "reorder_claims":
            sections[0] = sections[0].model_copy(
                update={"claims": list(reversed(sections[0].claims))}
            )
        elif self.mutation == "downgrade_claim":
            downgraded = sections[0].claims[0].model_copy(
                update={
                    "kind": ClaimKind.INFERENCE,
                    "confidence": Confidence.LOW,
                    "evidence_chunk_ids": [],
                    "web_evidence_ids": [],
                }
            )
            sections[0] = sections[0].model_copy(
                update={"claims": [downgraded, *sections[0].claims[1:]]}
            )
        elif self.mutation == "drop_data_point":
            data_points = data_points[:1]
        elif self.mutation == "drop_gap":
            information_gaps = []
        elif self.mutation == "remove_source":
            sections = [
                section.model_copy(
                    update={
                        "claims": [
                            claim.model_copy(update={"web_evidence_ids": []})
                            for claim in section.claims
                        ]
                    }
                )
                for section in sections
            ]
        elif self.mutation == "reorder_sections":
            sections.reverse()
        if self.mutation == "lower_guarded_confidence":
            return draft.model_copy(update={"confidence": Decimal("0.5")})
        return draft.model_copy(
            update={
                "sections": sections,
                "data_points": data_points,
                "information_gaps": information_gaps,
                "information_sufficiency": InformationSufficiency.PARTIAL,
                "confidence": Decimal("0.5"),
            }
        )


class TwoRepairAnalyst(FakeAnalyst):
    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        memo = await super().analyze(request=request, evidence=evidence)
        if request.recipe.name not in {
            SkillName.COMPANY_DEEP_RESEARCH,
            SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        }:
            return memo
        sections = list(memo.sections)
        sections[0] = sections[0].model_copy(update={"claims": []})
        return memo.model_copy(update={"sections": sections})


@dataclass
class PassthroughSkillRepairer:
    calls: list[SkillName] = field(default_factory=list)

    async def repair(
        self,
        *,
        request: SkillAnalysisInput,
        draft: SkillResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        del guard_errors, evidence
        self.calls.append(request.recipe.name)
        return draft


class FakeSkillRunRepository:
    def __init__(self, recorder: P1Recorder) -> None:
        self._recorder = recorder

    def start(
        self,
        *,
        ticker: str,
        recipe_name: str,
        recipe_version: str,
        recipe_snapshot: dict[str, object],
        application_run_id: str,
    ) -> str:
        run_id = f"run-{len(self._recorder.repository_starts)}"
        self._recorder.repository_starts.append(
            (ticker, recipe_name, recipe_version, recipe_snapshot, application_run_id)
        )
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: list[str],
        errors: list[str],
    ) -> None:
        self._recorder.repository_finishes.append(
            (run_id, status, list(source_ids), list(errors))
        )


def _financial_point(
    source_id: str,
    *,
    value: Decimal = Decimal("44.1"),
) -> FinancialDataPoint:
    return FinancialDataPoint(
        name="Revenue",
        value=value,
        currency="USD",
        unit="billions",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        definition="GAAP revenue",
        source_ids=[source_id],
    )


class RecordingTraceSink:
    def __init__(self, recorder: P1Recorder) -> None:
        self._recorder = recorder

    def record(self, node: str, attributes: dict[str, object]) -> None:
        self._recorder.trace_events.append((node, dict(attributes)))


def _chunk(recipe: ResearchRecipe, *, ticker: str = "NVDA") -> EvidenceChunk:
    source_id = f"sec-{recipe.name.value}"
    return EvidenceChunk(
        id=source_id,
        ticker=ticker,
        corpus_version="NVDA-p1-v1",
        content=f"CONFIDENTIAL_SOURCE_TEXT for {recipe.name.value}.",
        source_url=f"https://www.sec.gov/Archives/{source_id}.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=50,
    )


def _web(recipe: ResearchRecipe, *, ticker: str = "NVDA") -> WebEvidence:
    source_url = f"https://investor.nvidia.com/{recipe.name.value}"
    content = f"CONFIDENTIAL_WEB_SOURCE_TEXT for {recipe.name.value}."
    content_hash = sha256(content.encode()).hexdigest()
    source_id = content_addressed_web_evidence_id(ticker, source_url, content_hash)
    return WebEvidence(
        id=source_id,
        ticker=ticker,
        title=f"Issuer disclosure for {recipe.name.value}",
        content=content,
        source_url=source_url,
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, tzinfo=UTC),
        content_hash=content_hash,
    )


def _bundle(
    recipe: ResearchRecipe,
    *,
    ticker: str = "NVDA",
    complete: bool = True,
    include_raw_evidence: bool = False,
    reason_codes: tuple[str, ...] | None = None,
) -> EvidenceBundle:
    chunk = _chunk(recipe, ticker=ticker)
    web = _web(recipe, ticker=ticker)
    filing_evidence = [chunk] if complete else []
    if include_raw_evidence:
        filing_evidence = [_chunk(recipe, ticker="AMD")]
    sources = [*filing_evidence, *([web] if complete else [])]
    assignments = [
        EvidenceAssignment(
            question_index=0,
            side=side,
            source_id=source.id,
            source_kind=(
                SourceKind.FILING if isinstance(source, EvidenceChunk) else source.source_kind
            ),
        )
        for source in sources
        for side in EvidenceSide
    ]
    facet_assignments = [
        FacetAssignment(
            question_index=0,
            side=side,
            facet=facet,
            source_id=source.id,
        )
        for source in sources
        for side in EvidenceSide
        for facet in recipe.required_facets
    ]
    return EvidenceBundle(
        filing_evidence=filing_evidence,
        web_evidence=[web] if complete else [],
        assignments=assignments,
        facet_assignments=facet_assignments,
        coverage=CoverageReport(
            complete=complete,
            missing_facets=() if complete else recipe.required_facets,
            missing_pairs=() if complete else ((0, "support"), (0, "challenge")),
            invalid_source_ids=(),
            ticker_mismatches=(),
            date_mismatches=(),
            new_valid_source_count=1 if complete else 0,
            reason_codes=(
                reason_codes
                if reason_codes is not None
                else (() if complete else ("missing_facet", "insufficient_information"))
            ),
        ),
        retrieval_rounds=1 if complete else 2,
        web_calls=0,
    )


def _dependencies(
    recorder: P1Recorder,
    *,
    planner: FakePlanner | None = None,
    collector: FakeCollector | None = None,
    analyst: FakeAnalyst | None = None,
    include_p1_models: bool = True,
) -> tuple[Dependencies, UnusedP0MCPClient]:
    mcp_client = UnusedP0MCPClient()
    values: dict[str, object] = {}
    if include_p1_models:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        create_schema(engine)
        web_repository = WebEvidenceRepository(engine)
        for recipe in (*P1_RECIPES, INDUSTRY_RESEARCH):
            web_repository.upsert(_web(recipe))
        values.update(
            skill_planner=planner or FakePlanner(recorder),
            skill_collector=collector or FakeCollector(recorder),
            skill_analyst=analyst or FakeAnalyst(recorder),
            skill_run_repository=FakeSkillRunRepository(recorder),
            web_evidence_validator=PersistedWebEvidenceValidator(
                SourcePolicy(issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}),
                web_repository,
            ),
        )
    return (
        Dependencies(
            mcp_client=mcp_client,
            fast_model=P0Model(),
            analyst_model=P0Analyst(),
            trace_sink=RecordingTraceSink(recorder),
            **values,
        ),
        mcp_client,
    )


def test_company_profile_executes_three_frozen_recipes_and_combines_guarded_sections() -> None:
    """A company request must persist and execute only its three code-owned recipes."""
    recorder = P1Recorder()
    dependencies, mcp_client = _dependencies(recorder)

    result = run_research(
        " nvda ",
        "介绍 NVDA",
        dependencies,
        run_id="application-run-1",
    )

    expected = [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    recipe_map = {recipe.name: recipe for recipe in P1_RECIPES}
    expected_web_ids = {name: _web(recipe_map[name]).id for name in expected}
    assert result.status == "completed"
    assert [run.recipe_name for run in result.skill_runs] == expected
    assert recorder.planner_recipes == expected
    assert recorder.collector_recipes == expected
    assert recorder.analyst_recipes == expected
    assert mcp_client.calls == []
    assert [start[1] for start in recorder.repository_starts] == [name.value for name in expected]
    assert all(start[0] == "NVDA" for start in recorder.repository_starts)
    assert all(start[3]["budget"] for start in recorder.repository_starts)
    assert all(start[4] == "application-run-1" for start in recorder.repository_starts)
    assert [finish[1] for finish in recorder.repository_finishes] == ["completed"] * 3
    assert [
        [reference.encode() for reference in finish[2]]
        for finish in recorder.repository_finishes
    ] == [
        [
            f"NVDA:filing:sec-{name.value}",
            f"NVDA:web:{expected_web_ids[name]}",
        ]
        for name in expected
    ]
    assert "## Bull case" in result.rendered_output
    assert "## Bear case" in result.rendered_output
    assert [
        [source.id for source in run.guarded_memo.filing_sources]
        for run in result.skill_runs
        if run.guarded_memo is not None
    ] == [[f"sec-{name.value}"] for name in expected]
    assert [
        [source.id for source in run.guarded_memo.web_sources]
        for run in result.skill_runs
        if run.guarded_memo is not None
    ] == [[expected_web_ids[name]] for name in expected]
    assert all("market" not in tool for run in result.skill_runs for tool in run.allowed_tools)

    recipe_events = [
        attributes for node, attributes in recorder.trace_events if node == "p1_recipe"
    ]
    assert [event["recipe_name"] for event in recipe_events] == [name.value for name in expected]
    assert [event["source_ids"] for event in recipe_events] == [
        [
            f"NVDA:filing:sec-{name.value}",
            f"NVDA:web:{expected_web_ids[name]}",
        ]
        for name in expected
    ]
    assert all(
        set(event)
        == {
            "recipe_name",
            "recipe_version",
            "retrieval_rounds",
            "web_calls",
            "coverage_reasons",
            "source_ids",
            "status",
        }
        for event in recipe_events
    )
    p1_node_events = [
        attributes
        for node, attributes in recorder.trace_events
        if node in {"skill_dispatch", "execute_skill_recipes", "combine_skill_sections"}
    ]
    assert p1_node_events == [{}, {}, {}]
    dispatch_events = [
        attributes for node, attributes in recorder.trace_events if node == "p1_dispatch"
    ]
    assert len(dispatch_events) == 1
    assert set(dispatch_events[0]) == {"recipe_names", "recipe_versions"}
    assert "CONFIDENTIAL_SOURCE_TEXT" not in repr(recorder.trace_events)
    assert "CONFIDENTIAL_WEB_SOURCE_TEXT" not in repr(recorder.trace_events)
    assert "secret" not in repr(recorder.trace_events).casefold()


def test_company_recipes_share_one_run_budget_that_cannot_reset() -> None:
    """A fresh recipe-local counter must not bypass the application-owned run cap."""
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=2,
            max_analysis_calls=2,
            max_repair_calls=0,
            max_tool_calls=100,
            max_retrieval_rounds=10,
            max_web_calls=3,
        )
    )

    result = run_research(
        "NVDA",
        "介绍 NVDA",
        replace(dependencies, budget=authority),
        intent=Intent.COMPANY_PROFILE_REQUEST,
    )

    assert result.status == "partial"
    assert recorder.planner_recipes == [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
    ]
    assert recorder.collector_recipes == recorder.planner_recipes
    assert recorder.analyst_recipes == recorder.planner_recipes
    assert [run.status for run in result.skill_runs] == [
        "completed",
        "completed",
        "partial",
    ]
    assert result.skill_runs[2].errors == ["BUDGET_EXHAUSTED: planner_calls"]
    assert [finish[1] for finish in recorder.repository_finishes] == [
        "completed",
        "completed",
        "partial",
    ]
    assert authority.state.planner_calls == 2
    assert authority.state.analysis_calls == 2


def test_caller_selected_company_intent_cannot_be_rerouted_to_earnings() -> None:
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)

    result = run_research(
        "NVDA",
        "分析 NVDA 最近一期财报",
        dependencies,
        intent=Intent.COMPANY_PROFILE_REQUEST,
    )

    assert result.decision.intent is Intent.COMPANY_PROFILE_REQUEST
    assert [run.recipe_name for run in result.skill_runs] == [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]


def test_earnings_executes_exact_two_recipes_and_derives_single_source_status() -> None:
    """One retained filing cannot inherit a model-authored verified/discrepancy label."""
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)

    result = run_research("NVDA", "分析 NVDA 最近一期财报", dependencies)

    assert [run.recipe_name for run in result.skill_runs] == [
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    verification = result.skill_runs[1].guarded_memo
    assert verification is not None
    assert (
        verification.memo.data_points[0].verification_status
        is VerificationStatus.SINGLE_SOURCE
    )
    assert verification.memo.data_points[0].discrepancy_note is None
    assert "Verification: single\\_source" in result.rendered_output


def test_nonfatal_financial_discrepancy_persists_public_partial_status() -> None:
    """A guarded discrepancy remains renderable but must persist as partial."""
    recorder = P1Recorder()
    analyst = FakeAnalyst(recorder, financial_conflict=True)
    dependencies, _ = _dependencies(recorder, analyst=analyst)

    result = run_research("NVDA", "分析 NVDA 最近一期财报", dependencies)

    verification = result.skill_runs[1]
    assert result.status == "partial"
    assert verification.status == "partial"
    assert verification.guarded_memo is not None
    assert verification.guarded_memo.guard_errors == ["FINANCIAL_DATA_DISCREPANCY: Revenue"]
    assert recorder.repository_finishes[1][1] == "partial"


def test_missing_p1_model_dependency_fails_closed_without_entering_p0() -> None:
    """Legacy P0 dependency construction stays valid but cannot silently execute P1."""
    recorder = P1Recorder()
    dependencies, mcp_client = _dependencies(recorder, include_p1_models=False)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "failed"
    assert result.skill_runs == []
    assert result.errors == ["P1_DEPENDENCY_MISSING"]
    assert "unable to execute" in result.rendered_output.lower()
    assert mcp_client.calls == []


def test_incomplete_evidence_is_explicit_and_never_reaches_analyst() -> None:
    """Coverage failure must not become an evidence-free analyst answer."""
    recorder = P1Recorder()
    collector = FakeCollector(recorder, incomplete_recipe=SkillName.COMPANY_DEEP_RESEARCH)
    dependencies, _ = _dependencies(recorder, collector=collector)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "partial"
    failed = result.skill_runs[0]
    assert failed.status == "failed"
    assert failed.errors == [
        "INSUFFICIENT_EVIDENCE",
        "COVERAGE: missing_facet",
        "COVERAGE: insufficient_information",
    ]
    assert SkillName.COMPANY_DEEP_RESEARCH not in recorder.analyst_recipes
    assert "Partial — company_deep_research" in result.rendered_output
    assert [finish[1] for finish in recorder.repository_finishes] == [
        "failed",
        "completed",
        "completed",
    ]


def test_incomplete_raw_bundle_never_persists_or_traces_unguarded_source_ids() -> None:
    """Raw invalid bundle IDs are not auditable cited provenance."""
    recorder = P1Recorder()
    collector = FakeCollector(
        recorder,
        raw_incomplete_recipe=SkillName.COMPANY_DEEP_RESEARCH,
    )
    dependencies, _ = _dependencies(recorder, collector=collector)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "partial"
    assert result.skill_runs[0].guarded_memo is None
    assert recorder.repository_finishes[0][2] == []
    recipe_event = next(
        attributes
        for node, attributes in recorder.trace_events
        if node == "p1_recipe" and attributes["recipe_name"] == "company_deep_research"
    )
    assert recipe_event["source_ids"] == []


def test_trace_filters_unknown_coverage_reason_and_has_exact_p1_metadata() -> None:
    """An injected coverage reason cannot leak through P1 trace metadata."""
    recorder = P1Recorder()
    collector = FakeCollector(
        recorder,
        secret_reason_recipe=SkillName.COMPANY_DEEP_RESEARCH,
    )
    dependencies, _ = _dependencies(recorder, collector=collector)

    run_research("NVDA", "介绍 NVDA", dependencies)

    event = next(
        attributes
        for node, attributes in recorder.trace_events
        if node == "p1_recipe" and attributes["recipe_name"] == "company_deep_research"
    )
    assert set(event) == {
        "recipe_name",
        "recipe_version",
        "retrieval_rounds",
        "web_calls",
        "coverage_reasons",
        "source_ids",
        "status",
    }
    assert event["coverage_reasons"] == ["missing_facet"]
    assert "secret-token-must-not-trace" not in repr(recorder.trace_events)


def test_all_incomplete_recipes_return_insufficient_evidence_without_analysis() -> None:
    """No valid recipe result must produce an overall insufficient-evidence status."""
    recorder = P1Recorder()
    collector = FakeCollector(recorder, incomplete_recipe="all")
    dependencies, _ = _dependencies(recorder, collector=collector)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "insufficient_evidence"
    assert [run.status for run in result.skill_runs] == ["failed", "failed", "failed"]
    assert recorder.analyst_recipes == []
    assert result.rendered_output.count("## Partial —") == 3


def test_one_recipe_research_failure_retains_other_guarded_results() -> None:
    """A provider outage in one recipe must not erase independent valid research."""
    recorder = P1Recorder()
    planner = FakePlanner(
        recorder,
        fail_recipe=SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
    )
    dependencies, _ = _dependencies(recorder, planner=planner)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "partial"
    assert [run.status for run in result.skill_runs] == ["completed", "failed", "completed"]
    assert result.skill_runs[1].errors == ["RECIPE_EXECUTION_ERROR: TimeoutError"]
    assert "Partial — management_and_governance_review" in result.rendered_output
    assert "## Company overview" in result.rendered_output
    assert "## Data verification" in result.rendered_output
    assert "provider secret" not in result.rendered_output
    assert "provider secret" not in repr(recorder.trace_events)


def test_citation_guard_rejection_fails_entire_p1_render_closed() -> None:
    """A citation or source-policy guard error invalidates every P1 rendered section."""
    recorder = P1Recorder()
    analyst = FakeAnalyst(recorder, reject_recipe=SkillName.COMPANY_DEEP_RESEARCH)
    dependencies, _ = _dependencies(recorder, analyst=analyst)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "failed"
    assert result.errors == ["P1_PROTOCOL_FAILURE"]
    assert [run.status for run in result.skill_runs] == ["refused", "refused", "refused"]
    assert "hallucinated-source" not in result.rendered_output
    assert "## Company overview" not in result.rendered_output
    assert "No research sections were rendered." in result.rendered_output
    assert [finish[1] for finish in recorder.repository_finishes] == [
        "refused",
        "refused",
        "refused",
    ]


def test_p1_guard_failure_gets_one_bounded_repair_before_final_guard_result() -> None:
    recorder = P1Recorder()
    repairer = RestrictiveSkillRepairer()
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=2,
            max_analysis_calls=2,
            max_repair_calls=1,
            max_tool_calls=100,
            max_retrieval_rounds=4,
            max_web_calls=2,
        )
    )
    dependencies, _ = _dependencies(
        recorder,
        analyst=FakeAnalyst(recorder, reject_recipe=SkillName.EARNINGS_REVIEW),
    )

    result = run_research(
        "NVDA",
        "分析 NVDA 最近一期财报",
        replace(
            dependencies,
            skill_repair_model=repairer,
            budget=authority,
        ),
        report_as_of=date(2026, 5, 21),
    )

    assert len(repairer.calls) == 1
    assert repairer.calls[0][0].recipe.name is SkillName.EARNINGS_REVIEW
    assert "hallucinated-source" in repr(repairer.calls[0][2])
    assert authority.state.repair_calls == 1
    assert result.skill_runs[0].guarded_memo is not None
    assert result.skill_runs[0].guarded_memo.filing_sources == []
    assert result.skill_runs[0].guarded_memo.web_sources
    assert result.skill_runs[0].guarded_memo.provenance.requested_as_of_dates == (
        date(2026, 5, 21),
    )
    assert result.skill_runs[0].errors[-1] == "REPAIR_APPLIED"
    assert "hallucinated-source" not in result.rendered_output


def _financial_verification_recipe() -> ResearchRecipe:
    return next(
        recipe
        for recipe in P1_RECIPES
        if recipe.name is SkillName.FINANCIAL_DATA_VERIFICATION
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "delete_claim",
        "reorder_claims",
        "downgrade_claim",
        "drop_data_point",
        "drop_gap",
        "remove_source",
        "reorder_sections",
        "lower_guarded_confidence",
    ],
)
def test_p1_repair_preserves_original_guarded_facets_content_and_sources(
    mutation: str,
) -> None:
    recorder = P1Recorder()
    repairer = NonMonotonicSkillRepairer(mutation)
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=2,
            max_analysis_calls=2,
            max_repair_calls=1,
            max_tool_calls=100,
            max_retrieval_rounds=4,
            max_web_calls=2,
        )
    )
    dependencies, _ = _dependencies(recorder, analyst=RichConflictAnalyst(recorder))
    report_as_of = date(2026, 5, 21)

    result = run_research(
        "NVDA",
        "分析 NVDA 最近一期财报",
        replace(
            dependencies,
            skill_repair_model=repairer,
            budget=authority,
        ),
        intent=Intent.EARNINGS_REVIEW_REQUEST,
        report_as_of=report_as_of,
    )

    verification = result.skill_runs[1]
    assert verification.memo is not None
    assert verification.evidence is not None
    original = guard_skill_memo(
        verification.memo,
        verification.evidence,
        ticker="NVDA",
        recipe=_financial_verification_recipe(),
        web_validator=dependencies.web_evidence_validator,
        report_as_of=report_as_of,
    )
    assert [len(section.claims) for section in original.memo.sections] == [2, 1]
    assert len(original.memo.data_points) == 2
    assert original.memo.information_gaps == ["Retained information gap."]
    assert len(original.filing_sources) == len(original.web_sources) == 1
    assert verification.guarded_memo == original
    assert "REPAIR_OUTPUT_REJECTED" in verification.errors
    assert len(repairer.calls) == 1


def test_all_recipes_share_one_repair_call_before_provider_side_effects() -> None:
    recorder = P1Recorder()
    repairer = PassthroughSkillRepairer()
    authority = BudgetAuthority()
    analyst = TwoRepairAnalyst(recorder)
    dependencies, _ = _dependencies(recorder, analyst=analyst)

    result = run_research(
        "NVDA",
        "介绍 NVDA",
        replace(
            dependencies,
            skill_repair_model=repairer,
            budget=authority,
        ),
        intent=Intent.COMPANY_PROFILE_REQUEST,
    )

    assert repairer.calls == [SkillName.COMPANY_DEEP_RESEARCH]
    assert authority.state.limits.max_repair_calls == 1
    assert authority.state.repair_calls == 1
    second = result.skill_runs[1]
    assert second.memo is not None
    assert second.evidence is not None
    original = guard_skill_memo(
        second.memo,
        second.evidence,
        ticker="NVDA",
        recipe=next(
            recipe
            for recipe in P1_RECIPES
            if recipe.name is SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW
        ),
        web_validator=dependencies.web_evidence_validator,
    )
    assert second.guarded_memo == original
    assert "REPAIR_SKIPPED: budget_exhausted" in second.errors


def test_collector_protocol_failure_aborts_all_rendering_and_finishes_remaining_runs() -> None:
    """A malformed provider boundary must fail the whole P1 workflow closed."""
    recorder = P1Recorder()
    collector = FakeCollector(
        recorder,
        protocol_error_recipe=SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
    )
    dependencies, _ = _dependencies(recorder, collector=collector)

    result = run_research("NVDA", "介绍 NVDA", dependencies)

    assert result.status == "failed"
    assert result.errors == ["P1_PROTOCOL_FAILURE"]
    assert "Source-backed company_overview finding." not in result.rendered_output
    assert "secret-token" not in result.rendered_output
    assert [run.status for run in result.skill_runs] == ["completed", "refused", "refused"]
    assert [finish[1] for finish in recorder.repository_finishes] == [
        "completed",
        "refused",
        "refused",
    ]
