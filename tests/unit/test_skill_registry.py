"""Contracts for the static, versioned P1 research-recipe registry."""

import pytest
from pydantic import ValidationError

from fra.domain import Intent, SourceKind
from fra.skills.models import (
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    SkillName,
    WebUsagePolicy,
)
from fra.skills.recipes import (
    COMPANY_DEEP_RESEARCH,
    EARNINGS_REVIEW,
    FINANCIAL_DATA_VERIFICATION,
    INDUSTRY_RESEARCH,
    MANAGEMENT_AND_GOVERNANCE_REVIEW,
    RECIPES,
    RESEARCH_INPUT_DISPATCH_CONTRACT,
    RESEARCH_INPUT_RECIPES,
)
from fra.skills.registry import ResearchRecipeRegistry


def test_registry_declares_all_six_recipes_in_stable_order() -> None:
    """Changing declaration order would change deterministic dispatch for approved recipes."""
    registry = ResearchRecipeRegistry(RECIPES)
    dispatchable = ResearchRecipeRegistry(
        RESEARCH_INPUT_RECIPES,
        dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
    )

    assert tuple(recipe.name for recipe in RECIPES) == (
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
        SkillName.INDUSTRY_RESEARCH,
        SkillName.RESEARCH_QUALITY_SCREEN,
    )
    assert tuple(recipe.version for recipe in RECIPES) == ("1.0.0",) * 6
    assert tuple(
        recipe.name for recipe in dispatchable.resolve(Intent.COMPANY_PROFILE_REQUEST)
    ) == (
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )
    assert tuple(
        recipe.name for recipe in dispatchable.resolve(Intent.EARNINGS_REVIEW_REQUEST)
    ) == (
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )
    assert tuple(
        recipe.name for recipe in dispatchable.resolve(Intent.INDUSTRY_RESEARCH_REQUEST)
    ) == (
        SkillName.INDUSTRY_RESEARCH,
    )
    assert dispatchable.resolve(Intent.RESEARCH_QUALITY_SCREEN_REQUEST) == ()
    assert tuple(
        recipe.name for recipe in registry.resolve(Intent.RESEARCH_QUALITY_SCREEN_REQUEST)
    ) == (SkillName.RESEARCH_QUALITY_SCREEN,)
    assert registry.get(SkillName.EARNINGS_REVIEW) is RECIPES[2]
    assert registry.get(SkillName.RESEARCH_QUALITY_SCREEN) is RECIPES[5]


def test_recipe_mapping_matches_p1_facets_and_source_order() -> None:
    """Removing a required facet or reordering sources weakens the fixed research policy."""
    recipes = {recipe.name: recipe for recipe in RECIPES}

    assert recipes[SkillName.COMPANY_DEEP_RESEARCH].required_facets == (
        ResearchFacet.COMPANY_OVERVIEW,
        ResearchFacet.BUSINESS_MODEL,
        ResearchFacet.COMPETITIVE_POSITION,
        ResearchFacet.BULL_CASE,
        ResearchFacet.BEAR_CASE,
        ResearchFacet.INFORMATION_GAPS,
    )
    assert recipes[SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW].required_facets == (
        ResearchFacet.MANAGEMENT_AND_GOVERNANCE,
        ResearchFacet.CAPITAL_ALLOCATION,
        ResearchFacet.RISKS,
        ResearchFacet.INFORMATION_GAPS,
    )
    assert recipes[SkillName.EARNINGS_REVIEW].required_facets == (
        ResearchFacet.EARNINGS_CHANGE,
        ResearchFacet.GUIDANCE_AND_RISKS,
        ResearchFacet.BULL_CASE,
        ResearchFacet.BEAR_CASE,
        ResearchFacet.INFORMATION_GAPS,
    )
    assert recipes[SkillName.FINANCIAL_DATA_VERIFICATION].required_facets == (
        ResearchFacet.DATA_VERIFICATION,
        ResearchFacet.INFORMATION_GAPS,
    )
    assert recipes[SkillName.INDUSTRY_RESEARCH].required_facets == (
        ResearchFacet.INDUSTRY_SCOPE,
        ResearchFacet.VALUE_CHAIN,
        ResearchFacet.SUPPLY_AND_DEMAND,
        ResearchFacet.COMPETITION,
        ResearchFacet.REGULATION,
        ResearchFacet.DRIVERS,
        ResearchFacet.RISKS,
        ResearchFacet.INFORMATION_GAPS,
    )
    assert recipes[SkillName.RESEARCH_QUALITY_SCREEN].required_facets == (
        ResearchFacet.INFORMATION_GAPS,
    )
    assert all(
        recipe.source_policy
        == (SourceKind.FILING, SourceKind.ISSUER_IR, SourceKind.AUTHORITATIVE_WEB)
        for recipe in RECIPES[:5]
    )
    assert recipes[SkillName.RESEARCH_QUALITY_SCREEN].source_policy == (
        SourceKind.FILING,
        SourceKind.ISSUER_IR,
        SourceKind.AUTHORITATIVE_WEB,
    )


def test_recipes_declare_immutable_web_usage_policies() -> None:
    """Changing a web policy could expand a recipe's authorized evidence use."""
    financial_data = RECIPES[3]
    industry = RECIPES[4]
    quality = RECIPES[5]

    assert all(recipe.web_usage_policy is WebUsagePolicy.EVIDENCE for recipe in RECIPES[:3])
    assert industry.web_usage_policy is WebUsagePolicy.EVIDENCE
    assert quality.web_usage_policy is WebUsagePolicy.NONE
    assert financial_data.web_usage_policy is WebUsagePolicy.PRIMARY_SOURCE_LOCATOR
    assert "search_allowlisted_web" in financial_data.allowed_tools
    assert financial_data.budget.max_web_calls == 1
    assert financial_data.budget.max_web_results == 3
    assert industry.budget.max_web_calls == 1
    assert industry.budget.max_web_results == 3
    assert quality.allowed_tools == frozenset({"get_source_spans", "get_web_evidence"})
    assert quality.budget.max_web_calls == 0
    assert quality.budget.max_web_results == 0

    with pytest.raises(ValidationError):
        financial_data.web_usage_policy = WebUsagePolicy.NONE


def test_all_recipes_declare_the_frozen_model_and_evidence_token_limits() -> None:
    """A missing per-recipe limit would leave one production workflow unbounded."""
    for recipe in RECIPES:
        assert recipe.budget.max_planner_output_tokens == 350
        assert recipe.budget.max_analysis_output_tokens == 1_200
        assert recipe.budget.max_repair_output_tokens == 600
        assert recipe.budget.max_evidence_tokens == 3_000


def test_registry_rejects_duplicate_recipe_name_and_version() -> None:
    """Allowing a duplicate identity makes persisted recipe provenance ambiguous."""
    recipe = RECIPES[0]

    with pytest.raises(ValueError, match="duplicate recipe"):
        ResearchRecipeRegistry((recipe, recipe))


def test_research_input_dispatch_contract_is_versioned_and_exact() -> None:
    assert RESEARCH_INPUT_DISPATCH_CONTRACT.version == "1.0.0"
    assert RESEARCH_INPUT_DISPATCH_CONTRACT.ordered_recipe_names == {
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
    }


def test_registry_rejects_reordered_research_input_dispatch_contract() -> None:
    with pytest.raises(ValueError, match="company_profile_request"):
        ResearchRecipeRegistry(
            (
                COMPANY_DEEP_RESEARCH,
                FINANCIAL_DATA_VERIFICATION,
                MANAGEMENT_AND_GOVERNANCE_REVIEW,
                EARNINGS_REVIEW,
                INDUSTRY_RESEARCH,
            ),
            dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
        )


def test_registry_rejects_missing_research_input_dispatch_recipe() -> None:
    with pytest.raises(ValueError, match="company_profile_request"):
        ResearchRecipeRegistry(
            (
                COMPANY_DEEP_RESEARCH,
                MANAGEMENT_AND_GOVERNANCE_REVIEW,
                EARNINGS_REVIEW,
                INDUSTRY_RESEARCH,
            ),
            dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
        )


def test_registry_rejects_extra_same_intent_research_input_recipe() -> None:
    extra_recipe = ResearchRecipe(
        name=SkillName.RESEARCH_QUALITY_SCREEN,
        version="2.0.0",
        accepted_intents=frozenset({Intent.COMPANY_PROFILE_REQUEST}),
        allowed_tools=frozenset({"get_source_spans"}),
        source_policy=(SourceKind.FILING,),
        required_facets=(ResearchFacet.INFORMATION_GAPS,),
        budget=RecipeBudget(
            max_questions=1,
            max_local_results_per_query=1,
            max_retrieval_rounds=1,
            max_web_calls=0,
            max_web_results=0,
            max_planner_output_tokens=350,
            max_analysis_output_tokens=1_200,
            max_repair_output_tokens=600,
            max_evidence_tokens=3_000,
        ),
        web_usage_policy=WebUsagePolicy.NONE,
        input_schema="ResearchInput",
        output_schema="ResearchMemo",
        guard_profile="strict_citation",
    )

    with pytest.raises(ValueError, match="company_profile_request"):
        ResearchRecipeRegistry(
            (*RESEARCH_INPUT_RECIPES, extra_recipe),
            dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
        )


def test_recipe_rejects_empty_allowed_tools() -> None:
    """An empty tool set would make the recipe unusable and bypass its declared boundary."""
    with pytest.raises(ValidationError, match="allowed_tools"):
        ResearchRecipe(
            name=SkillName.COMPANY_DEEP_RESEARCH,
            version="1.0.0",
            accepted_intents=frozenset({Intent.COMPANY_PROFILE_REQUEST}),
            allowed_tools=frozenset(),
            source_policy=(SourceKind.FILING,),
            required_facets=(ResearchFacet.COMPANY_OVERVIEW,),
            budget=RecipeBudget(
                max_questions=1,
                max_local_results_per_query=1,
                max_retrieval_rounds=1,
                max_web_calls=0,
                max_web_results=0,
                max_planner_output_tokens=350,
                max_analysis_output_tokens=1_200,
                max_repair_output_tokens=600,
                max_evidence_tokens=3_000,
            ),
            web_usage_policy=WebUsagePolicy.NONE,
            input_schema="ResearchInput",
            output_schema="ResearchMemo",
            guard_profile="strict_citation",
        )


def test_registered_recipe_configuration_is_immutable() -> None:
    """Mutable recipe policy could let a caller expand a registered recipe's authority."""
    recipe = ResearchRecipeRegistry(RECIPES).get(SkillName.COMPANY_DEEP_RESEARCH)

    with pytest.raises(ValidationError):
        recipe.budget.max_questions = 4
    with pytest.raises(ValidationError):
        recipe.source_policy += (SourceKind.FILING,)
    with pytest.raises(AttributeError):
        recipe.allowed_tools.add("search_anywhere")


def test_registry_preserves_existing_p1_resolution_order_after_p2_registration() -> None:
    """Adding P2 recipes must not reorder the previously approved company or earnings flows."""
    registry = ResearchRecipeRegistry(
        RESEARCH_INPUT_RECIPES,
        dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
    )

    assert tuple(recipe.name for recipe in registry.resolve(Intent.COMPANY_PROFILE_REQUEST)) == (
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )
    assert tuple(recipe.name for recipe in registry.resolve(Intent.EARNINGS_REVIEW_REQUEST)) == (
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )
