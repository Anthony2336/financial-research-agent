"""Immutable contracts for registered research recipes."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fra.domain import Intent, SourceKind


class SkillName(StrEnum):
    """Immutable identifiers for approved research recipes."""

    COMPANY_DEEP_RESEARCH = "company_deep_research"
    MANAGEMENT_AND_GOVERNANCE_REVIEW = "management_and_governance_review"
    EARNINGS_REVIEW = "earnings_review"
    FINANCIAL_DATA_VERIFICATION = "financial_data_verification"
    INDUSTRY_RESEARCH = "industry_research"
    RESEARCH_QUALITY_SCREEN = "research_quality_screen"


class ResearchFacet(StrEnum):
    """Required subject areas for a research recipe."""

    COMPANY_OVERVIEW = "company_overview"
    BUSINESS_MODEL = "business_model"
    COMPETITIVE_POSITION = "competitive_position"
    MANAGEMENT_AND_GOVERNANCE = "management_and_governance"
    CAPITAL_ALLOCATION = "capital_allocation"
    RISKS = "risks"
    EARNINGS_CHANGE = "earnings_change"
    GUIDANCE_AND_RISKS = "guidance_and_risks"
    BULL_CASE = "bull_case"
    BEAR_CASE = "bear_case"
    DATA_VERIFICATION = "data_verification"
    INFORMATION_GAPS = "information_gaps"
    INDUSTRY_SCOPE = "industry_scope"
    VALUE_CHAIN = "value_chain"
    SUPPLY_AND_DEMAND = "supply_and_demand"
    COMPETITION = "competition"
    REGULATION = "regulation"
    DRIVERS = "drivers"


class WebUsagePolicy(StrEnum):
    """Permitted purpose for a recipe's allowlisted web fallback."""

    NONE = "none"
    EVIDENCE = "evidence"
    PRIMARY_SOURCE_LOCATOR = "primary_source_locator"


class RecipeBudget(BaseModel):
    """Hard limits for one recipe execution."""

    model_config = ConfigDict(frozen=True)

    max_questions: int = Field(ge=1, le=8)
    max_local_results_per_query: int = Field(ge=1, le=10)
    max_retrieval_rounds: Literal[1, 2]
    max_web_calls: Literal[0, 1]
    max_web_results: int = Field(ge=0, le=3)
    max_planner_output_tokens: int = Field(ge=1, le=350)
    max_analysis_output_tokens: int = Field(ge=1, le=1_200)
    max_repair_output_tokens: int = Field(ge=1, le=600)
    max_evidence_tokens: int = Field(ge=1, le=3_000)


class EvidenceCollectionPolicy(BaseModel):
    """Fixed retrieval authority for a workflow that is not a registered recipe."""

    model_config = ConfigDict(frozen=True)

    allowed_tools: frozenset[str] = Field(min_length=1)
    source_policy: tuple[SourceKind, ...] = Field(min_length=1)
    required_facets: tuple[ResearchFacet, ...] = Field(min_length=1)
    budget: RecipeBudget
    web_usage_policy: WebUsagePolicy


class ResearchRecipe(BaseModel):
    """A static recipe that constrains one approved research workflow."""

    model_config = ConfigDict(frozen=True)

    name: SkillName
    version: str = Field(min_length=1)
    accepted_intents: frozenset[Intent] = Field(min_length=1)
    allowed_tools: frozenset[str] = Field(min_length=1)
    source_policy: tuple[SourceKind, ...] = Field(min_length=1)
    required_facets: tuple[ResearchFacet, ...] = Field(min_length=1)
    budget: RecipeBudget
    web_usage_policy: WebUsagePolicy
    input_schema: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    guard_profile: str = Field(min_length=1)
