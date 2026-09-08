"""Deterministic coverage-gate tests for P1 evidence collection."""

from datetime import date

import pytest

from financial_evidence_agent.domain import EvidenceChunk, Intent, ResearchQuestion, SourceKind
from financial_evidence_agent.retrieval.coverage import (
    EvidenceAssignment,
    EvidenceSide,
    FacetAssignment,
    assess_coverage,
)
from financial_evidence_agent.skills.models import (
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    SkillName,
    WebUsagePolicy,
)


def _recipe(*facets: ResearchFacet) -> ResearchRecipe:
    return ResearchRecipe(
        name=SkillName.EARNINGS_REVIEW,
        version="test",
        accepted_intents=frozenset({Intent.EARNINGS_REVIEW_REQUEST}),
        allowed_tools=frozenset({"hybrid_search_filings"}),
        source_policy=(SourceKind.FILING,),
        required_facets=facets or (ResearchFacet.EARNINGS_CHANGE,),
        budget=RecipeBudget(
            max_questions=3,
            max_local_results_per_query=5,
            max_retrieval_rounds=2,
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


def _question(*, period: str | None = "2025") -> ResearchQuestion:
    return ResearchQuestion(
        question="What changed?",
        support_query="revenue growth",
        challenge_query="revenue headwinds",
        period=period,
        forms=["10-Q"],
    )


def _chunk(chunk_id: str, *, ticker: str = "NVDA", filed_at: date | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker=ticker,
        corpus_version="fixture-v1",
        content=f"Evidence for {chunk_id}",
        source_url=f"https://www.sec.gov/Archives/{chunk_id}",
        form="10-Q",
        filed_at=filed_at or date(2025, 5, 29),
        accession_no=f"000-{chunk_id}",
        section="MD&A",
        raw_start=0,
        raw_end=20,
    )


def _assignment(source_id: str, side: EvidenceSide) -> EvidenceAssignment:
    return EvidenceAssignment(
        question_index=0,
        side=side,
        source_id=source_id,
        source_kind=SourceKind.FILING,
    )


def _facet(
    source_id: str,
    facet: ResearchFacet = ResearchFacet.EARNINGS_CHANGE,
) -> FacetAssignment:
    side = EvidenceSide.CHALLENGE if source_id == "challenge" else EvidenceSide.SUPPORT
    return FacetAssignment(
        question_index=0,
        side=side,
        facet=facet,
        source_id=source_id,
    )


@pytest.mark.parametrize(
    ("case", "evidence", "assignments", "facets", "expected_code", "expected_field"),
    [
        (
            "missing facet",
            [_chunk("support"), _chunk("challenge")],
            [
                _assignment("support", EvidenceSide.SUPPORT),
                _assignment("challenge", EvidenceSide.CHALLENGE),
            ],
            [],
            "missing_facet",
            "missing_facets",
        ),
        (
            "support missing",
            [_chunk("challenge")],
            [_assignment("challenge", EvidenceSide.CHALLENGE)],
            [_facet("challenge")],
            "support_missing",
            "missing_pairs",
        ),
        (
            "challenge missing",
            [_chunk("support")],
            [_assignment("support", EvidenceSide.SUPPORT)],
            [_facet("support")],
            "challenge_missing",
            "missing_pairs",
        ),
        (
            "invalid source id",
            [_chunk("support"), _chunk("challenge")],
            [
                _assignment("support", EvidenceSide.SUPPORT),
                _assignment("missing", EvidenceSide.CHALLENGE),
            ],
            [_facet("support")],
            "invalid_source_id",
            "invalid_source_ids",
        ),
        (
            "ticker mismatch",
            [_chunk("support", ticker="AMD"), _chunk("challenge")],
            [
                _assignment("support", EvidenceSide.SUPPORT),
                _assignment("challenge", EvidenceSide.CHALLENGE),
            ],
            [
                _facet("support"),
                _facet("challenge"),
            ],
            "ticker_mismatch",
            "ticker_mismatches",
        ),
        (
            "date mismatch",
            [_chunk("support", filed_at=date(2024, 5, 29)), _chunk("challenge")],
            [
                _assignment("support", EvidenceSide.SUPPORT),
                _assignment("challenge", EvidenceSide.CHALLENGE),
            ],
            [
                _facet("support"),
                _facet("challenge"),
            ],
            "date_mismatch",
            "date_mismatches",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_coverage_gate_reports_each_invalid_or_missing_condition(
    case: str,
    evidence: list[EvidenceChunk],
    assignments: list[EvidenceAssignment],
    facets: list[FacetAssignment],
    expected_code: str,
    expected_field: str,
) -> None:
    """Removing the matching validation branch would incorrectly mark the case complete."""
    del case

    report = assess_coverage(
        ticker="NVDA",
        recipe=_recipe(ResearchFacet.EARNINGS_CHANGE),
        questions=[_question()],
        filing_evidence=evidence,
        web_evidence=[],
        assignments=assignments,
        facet_assignments=facets,
    )

    assert report.complete is False
    assert expected_code in report.reason_codes
    assert getattr(report, expected_field)


def test_coverage_gate_counts_only_new_valid_evidence_after_round_two() -> None:
    """Repeated IDs must not masquerade as new evidence and suppress the web fallback."""
    evidence = [_chunk("support"), _chunk("challenge")]

    report = assess_coverage(
        ticker="NVDA",
        recipe=_recipe(ResearchFacet.EARNINGS_CHANGE, ResearchFacet.GUIDANCE_AND_RISKS),
        questions=[_question()],
        filing_evidence=evidence,
        web_evidence=[],
        assignments=[
            _assignment("support", EvidenceSide.SUPPORT),
            _assignment("challenge", EvidenceSide.CHALLENGE),
        ],
        facet_assignments=[
            _facet("support"),
            _facet("challenge"),
        ],
        previous_valid_source_ids=frozenset({"support", "challenge"}),
        after_second_round=True,
    )

    assert report.complete is False
    assert report.new_valid_source_count == 0
    assert report.missing_facets == (ResearchFacet.GUIDANCE_AND_RISKS,)
    assert "zero_new_evidence" in report.reason_codes


def test_coverage_gate_accepts_complete_valid_bindings() -> None:
    """A complete support/challenge pair with a required facet must pass the gate."""
    evidence = [_chunk("support"), _chunk("challenge")]

    report = assess_coverage(
        ticker="nvda",
        recipe=_recipe(ResearchFacet.EARNINGS_CHANGE),
        questions=[_question()],
        filing_evidence=evidence,
        web_evidence=[],
        assignments=[
            _assignment("support", EvidenceSide.SUPPORT),
            _assignment("challenge", EvidenceSide.CHALLENGE),
        ],
        facet_assignments=[
            _facet("support"),
            _facet("challenge"),
        ],
    )

    assert report.complete is True
    assert report.reason_codes == ()
    assert report.new_valid_source_count == 2


def test_question_side_pair_requires_an_exact_facet_backed_binding() -> None:
    """A facet on one side of a shared source ID must not cover the other side."""
    evidence = [_chunk("shared")]

    report = assess_coverage(
        ticker="NVDA",
        recipe=_recipe(ResearchFacet.EARNINGS_CHANGE),
        questions=[_question()],
        filing_evidence=evidence,
        web_evidence=[],
        assignments=[
            _assignment("shared", EvidenceSide.SUPPORT),
            _assignment("shared", EvidenceSide.CHALLENGE),
        ],
        facet_assignments=[
            FacetAssignment(
                question_index=0,
                side=EvidenceSide.SUPPORT,
                facet=ResearchFacet.EARNINGS_CHANGE,
                source_id="shared",
            )
        ],
    )

    assert report.complete is False
    assert report.missing_pairs == ((0, EvidenceSide.CHALLENGE),)
    assert report.reason_codes == ("challenge_missing",)
