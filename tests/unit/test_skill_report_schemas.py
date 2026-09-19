"""Schema boundaries for structured P1 research reports."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fra.skills.models import ResearchFacet, SkillName
from fra.skills.schemas import (
    FinancialDataPoint,
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)


def _data_point(**updates: object) -> FinancialDataPoint:
    values: dict[str, object] = {
        "name": "Revenue",
        "value": Decimal("44.062"),
        "currency": "USD",
        "unit": "billions",
        "period_start": date(2025, 1, 27),
        "period_end": date(2025, 4, 27),
        "definition": "GAAP revenue",
        "source_ids": ["sec-1"],
    }
    values.update(updates)
    return FinancialDataPoint(**values)


def _memo() -> SkillResearchMemo:
    return SkillResearchMemo(
        recipe_name=SkillName.FINANCIAL_DATA_VERIFICATION,
        recipe_version="1.0.0",
        research_question="Does reported revenue reconcile across sources?",
        sections=[
            SkillResearchSection(
                facet=ResearchFacet.DATA_VERIFICATION,
                claims=[],
            )
        ],
        data_points=[_data_point()],
        information_sufficiency=InformationSufficiency.SUFFICIENT,
        information_gaps=[],
        confidence=Decimal("0.90"),
    )


def test_financial_values_and_confidence_remain_decimal() -> None:
    """Replacing Decimal fields with floats would lose exact financial values."""
    memo = _memo()

    assert memo.data_points[0].value == Decimal("44.062")
    assert type(memo.data_points[0].value) is Decimal
    assert memo.confidence == Decimal("0.90")
    assert type(memo.confidence) is Decimal


def test_financial_data_point_rejects_reversed_period() -> None:
    """Removing period validation would permit a start date after the period end."""
    with pytest.raises(ValidationError, match="period_start must not be after period_end"):
        _data_point(period_start=date(2025, 5, 1), period_end=date(2025, 4, 27))


def test_financial_data_point_requires_a_source_id() -> None:
    """An empty source list must not produce an apparently verified number."""
    with pytest.raises(ValidationError):
        _data_point(source_ids=[])


def test_financial_data_point_rejects_a_blank_source_id() -> None:
    """A present-but-empty ID must not satisfy the financial provenance contract."""
    with pytest.raises(ValidationError):
        _data_point(source_ids=["  "])


@pytest.mark.parametrize("field", ["verification_status", "source_tier", "observation_id"])
def test_analyst_financial_draft_rejects_guard_owned_fields(field: str) -> None:
    """Allowing guard-owned fields in the draft would let the analyst forge final provenance."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _data_point(**{field: "forged"})


@pytest.mark.parametrize("field", ["recipe_name", "recipe_version"])
def test_recipe_identity_cannot_be_mutated_after_validation(field: str) -> None:
    """Changing a validated memo's recipe identity would detach it from its frozen recipe."""
    memo = _memo()

    with pytest.raises(ValidationError):
        setattr(memo, field, "changed")
