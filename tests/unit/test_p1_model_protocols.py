"""Provider-neutral contracts for P1 planning and evidence analysis."""

from dataclasses import dataclass, field
from decimal import Decimal

from financial_evidence_agent.domain import ResearchQuestion
from financial_evidence_agent.graph.models import (
    SkillAnalysisInput,
    SkillAnalystModel,
    SkillPlannerModel,
    SkillPlanningInput,
)
from financial_evidence_agent.retrieval.collector import EvidenceBundle
from financial_evidence_agent.retrieval.coverage import CoverageReport
from financial_evidence_agent.skills.models import SkillName
from financial_evidence_agent.skills.recipes import P1_RECIPES
from financial_evidence_agent.skills.schemas import (
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)


def _empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        filing_evidence=[],
        web_evidence=[],
        assignments=[],
        coverage=CoverageReport(
            complete=False,
            missing_facets=(),
            missing_pairs=(),
            invalid_source_ids=(),
            ticker_mismatches=(),
            date_mismatches=(),
            new_valid_source_count=0,
            reason_codes=("fixture",),
        ),
        retrieval_rounds=1,
        web_calls=0,
    )


@dataclass
class FakeSkillPlanner:
    requests: list[SkillPlanningInput] = field(default_factory=list)

    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        self.requests.append(request)
        return [
            ResearchQuestion(
                question="What evidence addresses the request?",
                support_query="supporting evidence",
                challenge_query="challenging evidence",
            )
        ]


@dataclass
class FakeSkillAnalyst:
    requests: list[SkillAnalysisInput] = field(default_factory=list)

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        self.requests.append(request)
        return SkillResearchMemo(
            recipe_name=request.recipe.name,
            recipe_version=request.recipe.version,
            research_question=request.user_request,
            sections=[
                SkillResearchSection(facet=request.recipe.required_facets[0], claims=[])
            ],
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
            information_gaps=["No evidence was supplied."],
            confidence=Decimal("0"),
        )


async def _run_recipe(
    planner: SkillPlannerModel,
    analyst: SkillAnalystModel,
    request: SkillPlanningInput,
) -> tuple[list[ResearchQuestion], SkillResearchMemo]:
    questions = await planner.plan(request)
    analysis_request = SkillAnalysisInput(
        ticker=request.ticker,
        user_request=request.user_request,
        recipe=request.recipe,
    )
    memo = await analyst.analyze(request=analysis_request, evidence=_empty_bundle())
    return questions, memo


async def test_all_registered_recipes_cross_the_same_model_protocols() -> None:
    """Adding recipe-specific fake branches would hide a provider-neutral contract break."""
    planner = FakeSkillPlanner()
    analyst = FakeSkillAnalyst()

    results = [
        await _run_recipe(
            planner,
            analyst,
            SkillPlanningInput(ticker="NVDA", user_request="Research NVIDIA.", recipe=recipe),
        )
        for recipe in P1_RECIPES
    ]

    assert [request.recipe.name for request in planner.requests] == [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    assert [request.recipe for request in analyst.requests] == [
        request.recipe for request in planner.requests
    ]
    assert all(len(questions) == 1 for questions, _ in results)
    assert [memo.recipe_name for _, memo in results] == [recipe.name for recipe in P1_RECIPES]


def test_planning_input_keeps_the_selected_recipe_snapshot_frozen() -> None:
    """Replacing a selected recipe after dispatch would let mutable state change model limits."""
    request = SkillPlanningInput(
        ticker="NVDA",
        user_request="Research NVIDIA.",
        recipe=P1_RECIPES[0],
    )

    try:
        request.recipe = P1_RECIPES[1]
    except Exception:
        pass

    assert request.recipe is P1_RECIPES[0]


async def test_analyst_request_type_cannot_expose_planner_memory_hints() -> None:
    """An alternate analyst must receive no field capable of carrying session hints."""
    from financial_evidence_agent.context import MemoryHint

    planner = FakeSkillPlanner()
    analyst = FakeSkillAnalyst()
    planning_request = SkillPlanningInput(
        ticker="NVDA",
        user_request="What changed since then?",
        recipe=P1_RECIPES[0],
        memory_hints=(MemoryHint(text="private planner continuity hint"),),
    )

    await _run_recipe(planner, analyst, planning_request)

    assert "memory_hints" not in SkillAnalysisInput.model_fields
    assert not hasattr(analyst.requests[0], "memory_hints")
    assert "private planner continuity hint" not in repr(analyst.requests[0])
