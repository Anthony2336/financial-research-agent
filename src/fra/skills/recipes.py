"""Explicit registered research-recipe declarations."""

from fra.domain import Intent, SourceKind
from fra.skills.models import (
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    SkillName,
    WebUsagePolicy,
)
from fra.skills.registry import ExactIntentRecipeContract

_STANDARD_SOURCE_POLICY = (
    SourceKind.FILING,
    SourceKind.ISSUER_IR,
    SourceKind.AUTHORITATIVE_WEB,
)
_FILING_TOOLS = frozenset({"hybrid_search_filings", "get_source_spans"})
_WEB_RESEARCH_TOOLS = _FILING_TOOLS | {"search_allowlisted_web"}
_STANDARD_BUDGET = RecipeBudget(
    max_questions=3,
    max_local_results_per_query=5,
    max_retrieval_rounds=2,
    max_web_calls=1,
    max_web_results=3,
    max_planner_output_tokens=350,
    max_analysis_output_tokens=1_200,
    max_repair_output_tokens=600,
    max_evidence_tokens=3_000,
)
_NO_WEB_EXPANSION_BUDGET = RecipeBudget(
    max_questions=1,
    max_local_results_per_query=1,
    max_retrieval_rounds=1,
    max_web_calls=0,
    max_web_results=0,
    max_planner_output_tokens=350,
    max_analysis_output_tokens=1_200,
    max_repair_output_tokens=600,
    max_evidence_tokens=3_000,
)

COMPANY_DEEP_RESEARCH = ResearchRecipe(
    name=SkillName.COMPANY_DEEP_RESEARCH,
    version="1.0.0",
    accepted_intents=frozenset({Intent.COMPANY_PROFILE_REQUEST}),
    allowed_tools=_WEB_RESEARCH_TOOLS,
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(
        ResearchFacet.COMPANY_OVERVIEW,
        ResearchFacet.BUSINESS_MODEL,
        ResearchFacet.COMPETITIVE_POSITION,
        ResearchFacet.BULL_CASE,
        ResearchFacet.BEAR_CASE,
        ResearchFacet.INFORMATION_GAPS,
    ),
    budget=_STANDARD_BUDGET,
    web_usage_policy=WebUsagePolicy.EVIDENCE,
    input_schema="ResearchInput",
    output_schema="ResearchMemo",
    guard_profile="strict_citation",
)

MANAGEMENT_AND_GOVERNANCE_REVIEW = ResearchRecipe(
    name=SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
    version="1.0.0",
    accepted_intents=frozenset({Intent.COMPANY_PROFILE_REQUEST}),
    allowed_tools=_WEB_RESEARCH_TOOLS,
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(
        ResearchFacet.MANAGEMENT_AND_GOVERNANCE,
        ResearchFacet.CAPITAL_ALLOCATION,
        ResearchFacet.RISKS,
        ResearchFacet.INFORMATION_GAPS,
    ),
    budget=_STANDARD_BUDGET,
    web_usage_policy=WebUsagePolicy.EVIDENCE,
    input_schema="ResearchInput",
    output_schema="ResearchMemo",
    guard_profile="strict_citation",
)

EARNINGS_REVIEW = ResearchRecipe(
    name=SkillName.EARNINGS_REVIEW,
    version="1.0.0",
    accepted_intents=frozenset({Intent.EARNINGS_REVIEW_REQUEST}),
    allowed_tools=_WEB_RESEARCH_TOOLS,
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(
        ResearchFacet.EARNINGS_CHANGE,
        ResearchFacet.GUIDANCE_AND_RISKS,
        ResearchFacet.BULL_CASE,
        ResearchFacet.BEAR_CASE,
        ResearchFacet.INFORMATION_GAPS,
    ),
    budget=_STANDARD_BUDGET,
    web_usage_policy=WebUsagePolicy.EVIDENCE,
    input_schema="ResearchInput",
    output_schema="ResearchMemo",
    guard_profile="strict_citation",
)

FINANCIAL_DATA_VERIFICATION = ResearchRecipe(
    name=SkillName.FINANCIAL_DATA_VERIFICATION,
    version="1.0.0",
    accepted_intents=frozenset(
        {Intent.COMPANY_PROFILE_REQUEST, Intent.EARNINGS_REVIEW_REQUEST}
    ),
    allowed_tools=_WEB_RESEARCH_TOOLS,
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(ResearchFacet.DATA_VERIFICATION, ResearchFacet.INFORMATION_GAPS),
    budget=_STANDARD_BUDGET,
    web_usage_policy=WebUsagePolicy.PRIMARY_SOURCE_LOCATOR,
    input_schema="ResearchInput",
    output_schema="ResearchMemo",
    guard_profile="strict_citation",
)

INDUSTRY_RESEARCH = ResearchRecipe(
    name=SkillName.INDUSTRY_RESEARCH,
    version="1.0.0",
    accepted_intents=frozenset({Intent.INDUSTRY_RESEARCH_REQUEST}),
    allowed_tools=_WEB_RESEARCH_TOOLS,
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(
        ResearchFacet.INDUSTRY_SCOPE,
        ResearchFacet.VALUE_CHAIN,
        ResearchFacet.SUPPLY_AND_DEMAND,
        ResearchFacet.COMPETITION,
        ResearchFacet.REGULATION,
        ResearchFacet.DRIVERS,
        ResearchFacet.RISKS,
        ResearchFacet.INFORMATION_GAPS,
    ),
    budget=_STANDARD_BUDGET,
    web_usage_policy=WebUsagePolicy.EVIDENCE,
    input_schema="ResearchInput",
    output_schema="ResearchMemo",
    guard_profile="strict_citation",
)

RESEARCH_QUALITY_SCREEN = ResearchRecipe(
    name=SkillName.RESEARCH_QUALITY_SCREEN,
    version="1.0.0",
    accepted_intents=frozenset({Intent.RESEARCH_QUALITY_SCREEN_REQUEST}),
    allowed_tools=frozenset({"get_source_spans", "get_web_evidence"}),
    source_policy=_STANDARD_SOURCE_POLICY,
    required_facets=(ResearchFacet.INFORMATION_GAPS,),
    budget=_NO_WEB_EXPANSION_BUDGET,
    web_usage_policy=WebUsagePolicy.NONE,
    input_schema="GuardedResearchPackage",
    output_schema="ResearchQualityResult",
    guard_profile="research_quality_only",
)

P1_RECIPES = (
    COMPANY_DEEP_RESEARCH,
    MANAGEMENT_AND_GOVERNANCE_REVIEW,
    EARNINGS_REVIEW,
    FINANCIAL_DATA_VERIFICATION,
)
RESEARCH_INPUT_RECIPES = (
    *P1_RECIPES,
    INDUSTRY_RESEARCH,
)
RESEARCH_INPUT_DISPATCH_CONTRACT = ExactIntentRecipeContract(
    version="1.0.0",
    ordered_recipe_names={
        Intent.COMPANY_PROFILE_REQUEST: (
            SkillName.COMPANY_DEEP_RESEARCH,
            SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
            SkillName.FINANCIAL_DATA_VERIFICATION,
        ),
        Intent.EARNINGS_REVIEW_REQUEST: (
            SkillName.EARNINGS_REVIEW,
            SkillName.FINANCIAL_DATA_VERIFICATION,
        ),
        Intent.INDUSTRY_RESEARCH_REQUEST: (SkillName.INDUSTRY_RESEARCH,),
    },
)

RECIPES = (
    *RESEARCH_INPUT_RECIPES,
    RESEARCH_QUALITY_SCREEN,
)
