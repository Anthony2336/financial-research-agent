"""Pure, deterministic coverage checks for P1 evidence bundles."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from financial_evidence_agent.domain import (
    EvidenceChunk,
    ResearchQuestion,
    SourceKind,
    SourceTier,
    WebEvidence,
)
from financial_evidence_agent.skills.models import (
    EvidenceCollectionPolicy,
    ResearchFacet,
    ResearchRecipe,
    WebUsagePolicy,
)

CollectionPolicy = ResearchRecipe | EvidenceCollectionPolicy


class EvidenceSide(StrEnum):
    """The claim direction for one source assignment."""

    SUPPORT = "support"
    CHALLENGE = "challenge"


class EvidenceAssignment(BaseModel):
    """Bind a stable source ID to one planned question and side."""

    model_config = ConfigDict(frozen=True)

    question_index: int = Field(ge=0)
    side: EvidenceSide
    source_id: str = Field(min_length=1)
    source_kind: SourceKind


class FacetAssignment(BaseModel):
    """Bind a stable source ID to one explicitly planned recipe facet."""

    model_config = ConfigDict(frozen=True)

    question_index: int = Field(ge=0)
    side: EvidenceSide
    facet: ResearchFacet
    source_id: str = Field(min_length=1)


class CoverageReport(BaseModel):
    """Machine-readable result of the deterministic coverage gate."""

    model_config = ConfigDict(frozen=True)

    complete: bool
    missing_facets: tuple[ResearchFacet, ...]
    missing_pairs: tuple[tuple[int, EvidenceSide], ...]
    invalid_source_ids: tuple[str, ...]
    ticker_mismatches: tuple[str, ...]
    date_mismatches: tuple[str, ...]
    new_valid_source_count: int = Field(ge=0)
    reason_codes: tuple[str, ...]


def assess_coverage(
    *,
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
    previous_valid_source_ids: frozenset[str] = frozenset(),
    after_second_round: bool = False,
) -> CoverageReport:
    """Validate typed bindings and report deterministic coverage gaps.

    Source text is intentionally never inspected. A source counts as newly valid only
    when at least one question/side assignment and one required-facet assignment are
    both valid for its stable ID.
    """

    report, _, _ = _assess_coverage(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
        previous_valid_source_ids=previous_valid_source_ids,
        after_second_round=after_second_round,
    )
    return report


def _assess_coverage(
    *,
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
    previous_valid_source_ids: frozenset[str] = frozenset(),
    after_second_round: bool = False,
) -> tuple[
    CoverageReport,
    frozenset[str],
    frozenset[tuple[int, EvidenceSide, str]],
]:

    normalized_ticker = ticker.strip().upper()
    sources, source_kinds, ambiguous_ids = _source_index(filing_evidence, web_evidence)
    invalid_ids = set(ambiguous_ids)
    ticker_mismatches: set[str] = set()
    date_mismatches: set[str] = set()
    valid_pairs: set[tuple[int, EvidenceSide]] = set()
    source_valid_bindings: set[tuple[int, EvidenceSide, str]] = set()
    facet_backed_bindings: set[tuple[int, EvidenceSide, str]] = set()
    valid_assignment_ids: set[str] = set()

    for assignment in assignments:
        source = sources.get(assignment.source_id)
        if source is None or assignment.source_id in ambiguous_ids:
            invalid_ids.add(assignment.source_id)
            continue
        actual_kind = source_kinds[assignment.source_id]
        if assignment.source_kind is not actual_kind or not source_allowed_for_recipe(
            source, recipe
        ):
            invalid_ids.add(assignment.source_id)
            continue
        if source.ticker.strip().upper() != normalized_ticker:
            ticker_mismatches.add(assignment.source_id)
            continue
        if assignment.question_index >= len(questions):
            invalid_ids.add(assignment.source_id)
            continue
        question = questions[assignment.question_index]
        if not _matches_question_date(source, question):
            date_mismatches.add(assignment.source_id)
            continue
        source_valid_bindings.add(
            (assignment.question_index, assignment.side, assignment.source_id)
        )
        valid_assignment_ids.add(assignment.source_id)

    valid_facets: set[ResearchFacet] = set()
    valid_facet_source_ids: set[str] = set()
    required_facets = frozenset(recipe.required_facets)
    for facet_assignment in facet_assignments:
        if facet_assignment.facet not in required_facets:
            continue
        binding = (
            facet_assignment.question_index,
            facet_assignment.side,
            facet_assignment.source_id,
        )
        if binding not in source_valid_bindings:
            if facet_assignment.source_id not in sources:
                invalid_ids.add(facet_assignment.source_id)
            continue
        valid_facets.add(facet_assignment.facet)
        valid_facet_source_ids.add(facet_assignment.source_id)
        facet_backed_bindings.add(binding)
        valid_pairs.add((facet_assignment.question_index, facet_assignment.side))

    missing_facets = tuple(facet for facet in recipe.required_facets if facet not in valid_facets)
    expected_pairs = tuple(
        (question_index, side)
        for question_index in range(len(questions))
        for side in (EvidenceSide.SUPPORT, EvidenceSide.CHALLENGE)
    )
    missing_pairs = tuple(pair for pair in expected_pairs if pair not in valid_pairs)
    valid_source_ids = valid_assignment_ids & valid_facet_source_ids
    new_valid_source_count = len(valid_source_ids - previous_valid_source_ids)

    reason_codes: list[str] = []
    if missing_facets:
        reason_codes.append("missing_facet")
    if any(side is EvidenceSide.SUPPORT for _, side in missing_pairs):
        reason_codes.append("support_missing")
    if any(side is EvidenceSide.CHALLENGE for _, side in missing_pairs):
        reason_codes.append("challenge_missing")
    if invalid_ids:
        reason_codes.append("invalid_source_id")
    if ticker_mismatches:
        reason_codes.append("ticker_mismatch")
    if date_mismatches:
        reason_codes.append("date_mismatch")
    if after_second_round and new_valid_source_count == 0:
        reason_codes.append("zero_new_evidence")

    return (
        CoverageReport(
            complete=not missing_facets and not missing_pairs,
            missing_facets=missing_facets,
            missing_pairs=missing_pairs,
            invalid_source_ids=tuple(sorted(invalid_ids)),
            ticker_mismatches=tuple(sorted(ticker_mismatches)),
            date_mismatches=tuple(sorted(date_mismatches)),
            new_valid_source_count=new_valid_source_count,
            reason_codes=tuple(reason_codes),
        ),
        frozenset(valid_source_ids),
        frozenset(facet_backed_bindings),
    )


def _source_index(
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
) -> tuple[
    dict[str, EvidenceChunk | WebEvidence],
    dict[str, SourceKind],
    frozenset[str],
]:
    sources: dict[str, EvidenceChunk | WebEvidence] = {}
    source_kinds: dict[str, SourceKind] = {}
    ambiguous_ids: set[str] = set()
    for source, kind in (
        *((source, SourceKind.FILING) for source in filing_evidence),
        *((source, source.source_kind) for source in web_evidence),
    ):
        if source.id in sources:
            ambiguous_ids.add(source.id)
            continue
        sources[source.id] = source
        source_kinds[source.id] = kind
    return sources, source_kinds, frozenset(ambiguous_ids)


def source_allowed_for_recipe(
    source: EvidenceChunk | WebEvidence,
    recipe: CollectionPolicy,
) -> bool:
    source_kind = SourceKind.FILING if isinstance(source, EvidenceChunk) else source.source_kind
    if source_kind not in recipe.source_policy:
        return False
    if not isinstance(source, WebEvidence):
        return True
    if recipe.web_usage_policy is not WebUsagePolicy.PRIMARY_SOURCE_LOCATOR:
        return True
    return (
        source.source_tier is SourceTier.PRIMARY
        and source.source_kind in {SourceKind.FILING, SourceKind.ISSUER_IR}
    )


def _matches_question_date(
    source: EvidenceChunk | WebEvidence,
    question: ResearchQuestion,
) -> bool:
    if isinstance(source, EvidenceChunk):
        if question.forms and source.form not in question.forms:
            return False
        source_date: date | None = source.filed_at
    else:
        if question.forms:
            return False
        source_date = source.published_at.date() if source.published_at is not None else None
    if question.period is None:
        return True
    if source_date is None:
        return False
    return _date_matches_period(source_date, question.period)


def _date_matches_period(source_date: date, period: str) -> bool:
    normalized = period.strip()
    if len(normalized) == 4 and normalized.isdigit():
        return source_date.year == int(normalized)
    if len(normalized) == 7 and normalized[4] == "-":
        return source_date.isoformat().startswith(f"{normalized}-")
    try:
        return source_date == date.fromisoformat(normalized)
    except ValueError:
        return False


def valid_source_ids(
    *,
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
) -> frozenset[str]:
    """Return source IDs that can contribute to both pair and facet coverage."""

    _, valid, _ = _assess_coverage(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
    )
    return valid


def valid_assignment_keys(
    *,
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
) -> frozenset[tuple[int, EvidenceSide, str]]:
    """Return only question/side/source bindings that pass every source check."""

    _, _, valid = _assess_coverage(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
    )
    return valid
