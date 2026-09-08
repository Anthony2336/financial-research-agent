"""Deterministic P1 recipe dispatch contracts."""

import pytest

from financial_evidence_agent.domain import Intent
from financial_evidence_agent.graph import SkillDispatchError, dispatch_skill_recipes
from financial_evidence_agent.skills.models import SkillName


def test_company_profile_dispatch_is_exact_and_stable() -> None:
    """Reordering or widening company dispatch would execute an unapproved workflow."""
    recipes = dispatch_skill_recipes(Intent.COMPANY_PROFILE_REQUEST)

    assert tuple(recipe.name for recipe in recipes) == (
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )


def test_earnings_dispatch_is_exact_and_stable() -> None:
    """An earnings request must not invoke company, market, or industry recipes."""
    recipes = dispatch_skill_recipes(Intent.EARNINGS_REVIEW_REQUEST)

    assert tuple(recipe.name for recipe in recipes) == (
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    )


def test_industry_dispatch_is_exact_and_stable() -> None:
    """An industry request must execute only the single approved industry recipe."""
    recipes = dispatch_skill_recipes(Intent.INDUSTRY_RESEARCH_REQUEST)

    assert tuple(recipe.name for recipe in recipes) == (SkillName.INDUSTRY_RESEARCH,)


@pytest.mark.parametrize(
    "intent",
    [
        Intent.RESEARCH_REQUEST,
        Intent.MARKET_SNAPSHOT_REQUEST,
        Intent.RESEARCH_QUALITY_SCREEN_REQUEST,
        Intent.AMBIGUOUS,
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
    ],
)
def test_dispatch_refuses_every_non_p1_intent(intent: Intent) -> None:
    """Unknown dispatch intents must not silently resolve to a default recipe."""
    with pytest.raises(SkillDispatchError, match="unsupported P1 intent"):
        dispatch_skill_recipes(intent)


def test_dispatch_has_no_model_selected_recipe_input() -> None:
    """A model-supplied recipe name must not alter code-owned dispatch."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        dispatch_skill_recipes(  # type: ignore[call-arg]
            Intent.COMPANY_PROFILE_REQUEST,
            requested_recipe=SkillName.EARNINGS_REVIEW,
        )
