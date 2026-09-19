"""Fail-closed validation between analyst output and report rendering."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError

from fra.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    ResearchMemo,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    WebEvidence,
)
from fra.prompts import current_prompt_version
from fra.reporting.privacy import (
    PRIVATE_DETAIL_CODE,
    collapse_private_detail_errors,
    redact_private_text,
    retain_public_text,
)
from fra.retrieval.collector import EvidenceBundle
from fra.retrieval.coverage import EvidenceSide, source_allowed_for_recipe
from fra.safety.router import contains_price_prediction
from fra.skills.models import ResearchFacet, ResearchRecipe
from fra.skills.schemas import (
    FinancialDataPoint,
    FinancialSourceProvenance,
    GuardedFinancialDataPoint,
    GuardedSkillMemo,
    GuardedSkillResearchMemo,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    SkillResearchMemo,
    SkillResearchSection,
    VerificationStatus,
    canonical_financial_source_identity,
    canonical_financial_text,
    financial_observation_id,
)
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    PolicyValidatedWebEvidence,
    WebEvidenceValidationError,
)

_RECOMMENDED_TRADE_ACTION = (
    r"(?:buy(?:ing)?|purchas(?:e|ing)|acquir(?:e|ing)|sell(?:ing)?|"
    r"dispos(?:e|ing)\s+of|hold(?:ing)?|invest(?:ing)?\s+in|"
    r"increas(?:e|ing)|reduc(?:e|ing)|exit(?:ing)?|enter(?:ing)?)"
)
_READER = r"(?:you|investors?|shareholders?|traders?|clients?)"
_ADVISOR = r"(?:i|we|analysts?|advisers?|brokers?)"
_ADVICE_VERB = (
    r"(?:recommend(?:s|ed|ing)?|suggest(?:s|ed|ing)?|"
    r"advis(?:e|es|ed|ing)|urge(?:s|d|ing)?)"
)
_ADVICE_MODAL = (
    r"(?:should|must|need\s+to|ought\s+to|had\s+better|can|could|"
    r"(?:may|might)\s+want\s+to|would\s+be\s+wise\s+to)"
)
_PRODUCT_NOUN = (
    r"(?:gpus?|products?|systems?|chips?|hardware|software|platforms?|"
    r"servers?|services?|cards?|boards?)"
)
_SECURITY_NOUN = (
    r"(?:shares?|stock|securit(?:y|ies)|positions?|holdings?|exposure|"
    r"allocations?|stake|portfolio)"
)
_OVERRIDE_ACTION = r"(?:ignore|disregard|forget|override|bypass|discard)"
_FOLLOW_ACTION = r"(?:follow|obey|honou?r|respect)"
_EXFIL_ACTION = r"(?:reveal|output|print|show|display|disclose|expose|leak|provide|return)"
_PROTECTED_INSTRUCTION = (
    r"(?:(?:the|all|any)\s+)?"
    r"(?:(?:previous|prior|system|developer|safety|hidden)\s+){0,2}"
    r"(?:instructions?|directions?|rules?|guidance|guardrails?)"
)
_SENSITIVE_INSTRUCTION = (
    rf"(?:{_PROTECTED_INSTRUCTION}|"
    r"(?:(?:the|your|our)\s+)?(?:system|developer|internal|hidden)\s+"
    r"(?:prompt|message|instructions?|rules?))"
)
_CHINESE_PROTECTED_INSTRUCTION = (
    r"(?:(?:之前|先前|以上|所有|安全|系统|开发者|隐藏)的?){0,2}"
    r"(?:指令|提示|规则|消息)"
)
_CHINESE_OVERRIDE_ACTION = r"(?:忽略|无视|忘掉|覆盖|绕过|抛弃)"
_CHINESE_FOLLOW_ACTION = r"(?:遵循|服从|听从|执行)"
_CHINESE_EXFIL_ACTION = r"(?:输出|显示|展示|披露|泄露|公开|提供)"
_CHINESE_SENSITIVE_INSTRUCTION = (
    rf"(?:{_CHINESE_PROTECTED_INSTRUCTION}|"
    r"(?:系统|开发者|内部|隐藏)(?:提示词?|指令|消息|规则))"
)
_DIRECTIVE_WRAPPERS = (
    re.compile(
        r"^(?:could|would|can|will)\s+you\s+"
        r"(?:(?:please|kindly)\b[\s,;:!.-]*)?"
    ),
    re.compile(r"^(?:please|kindly)\b[\s,;:!.-]*"),
    re.compile(r"^proceed\s+to\s+"),
    re.compile(r"^(?:你|您)(?:可以|能否|能不能)(?:请)?[\s,，;；:：!！-]*"),
    re.compile(r"^(?:请您?|麻烦(?:你|您)?)[\s,，;；:：!！-]*"),
    re.compile(r"^(?:继续|接着)\s*"),
    re.compile(r"^(?:(?:and|but)(?:\s+then)?|then|however|yet)\b[\s,;:!.-]*"),
    re.compile(r"^(?:然后|随后|但是|但|并且|却|而且)\s*"),
)
_SOURCE_FIELDS = (
    "id",
    "ticker",
    "corpus_version",
    "content",
    "source_url",
    "form",
    "filed_at",
    "accession_no",
    "section",
    "raw_start",
    "raw_end",
)
_SOURCE_TEXT_FIELDS = (
    "id",
    "ticker",
    "corpus_version",
    "source_url",
    "form",
    "accession_no",
    "section",
)
_SEC_ARCHIVE_PATH = re.compile(r"^/Archives/[A-Za-z0-9._~/%+\-]+$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_URL_DELIMITERS = frozenset("<>[](){}\\`\"'")
_WEB_SOURCE_FIELDS = (
    "id",
    "ticker",
    "title",
    "content",
    "source_url",
    "source_kind",
    "source_tier",
    "published_at",
    "fetched_at",
    "content_hash",
)


class GuardedMemo(ResearchMemo):
    """A memo whose factual claims and retained sources passed deterministic checks."""

    ticker: str = Field(min_length=1, max_length=10)
    corpus_version: str = Field(min_length=1)
    sources: list[EvidenceChunk] = Field(default_factory=list)
    web_sources: list[WebEvidence] = Field(default_factory=list)
    source_policy_version: str | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def verified_claims(self) -> list[Claim]:
        """Return guarded facts without creating a second mutable storage field."""
        return [*self.supporting_claims, *self.counter_claims]


def guard_memo(
    memo: ResearchMemo,
    evidence: dict[str, EvidenceChunk] | EvidenceBundle,
    ticker: str,
    corpus_version: str,
    *,
    web_validator: PersistedWebEvidenceValidator | None = None,
) -> GuardedMemo:
    """Validate factual citations and safely retain explicitly typed non-facts."""
    supports_web_evidence = isinstance(evidence, EvidenceBundle)
    if supports_web_evidence:
        evidence, web_errors, rejected_web_ids = _revalidated_web_evidence(
            evidence,
            ticker=ticker,
            validator=web_validator,
        )
        filing_evidence = {source.id: source for source in evidence.filing_evidence}
        web_evidence = {source.id: source for source in evidence.web_evidence}
        side_bindings: frozenset[tuple[EvidenceSide, str]] | None = frozenset(
            (assignment.side, assignment.source_id) for assignment in evidence.assignments
        )
    else:
        filing_evidence = evidence
        web_evidence = {}
        web_errors = []
        rejected_web_ids = frozenset()
        side_bindings = None
    errors: list[str] = list(web_errors)
    unsafe_content_was_dropped = False
    supporting: list[Claim] = []
    counter: list[Claim] = []
    inferences: list[Claim] = []
    open_questions: list[Claim] = []
    canonical_sources: dict[str, EvidenceChunk] = {}
    canonical_web_sources: dict[str, WebEvidence] = {}
    research_question, private_question = redact_private_text(
        memo.research_question,
        ticker=ticker,
    )
    if private_question:
        errors.append(PRIVATE_DETAIL_CODE)
        unsafe_content_was_dropped = True
    elif _is_unsafe_output_text(research_question, ticker):
        research_question = "Removed by output safety guard."
        errors.append("RESEARCH_QUESTION_REDACTED: unsafe output text")
        unsafe_content_was_dropped = True

    def handle(claim: Claim, section: str, index: int) -> None:
        nonlocal unsafe_content_was_dropped
        label = section.upper()
        try:
            claim = Claim.model_validate(claim.model_dump(warnings=False))
        except (AttributeError, TypeError, ValidationError, ValueError):
            errors.append(f"{label}_CLAIM_DROPPED[{index}]: invalid claim schema")
            return
        if _is_unsafe_output_text(claim.text, ticker):
            errors.append(f"{label}_CLAIM_DROPPED[{index}]: unsafe claim text")
            unsafe_content_was_dropped = True
            return

        if claim.kind is ClaimKind.VERIFIED_FACT:
            if section not in {"supporting", "counter"}:
                display_section = "inference" if section == "inferences" else "open question"
                errors.append(
                    f"{label}_CLAIM_DROPPED[{index}]: verified fact in {display_section} section"
                )
                return
            guarded_fact, error = _guard_fact(
                claim,
                filing_evidence,
                web_evidence,
                ticker,
                corpus_version,
                canonical_sources,
                canonical_web_sources,
                rejected_web_ids,
                supports_web_evidence,
                side_bindings,
                (
                    EvidenceSide.SUPPORT
                    if section == "supporting"
                    else EvidenceSide.CHALLENGE
                ),
            )
            if error is not None:
                errors.append(f"{label}_CLAIM_DROPPED[{index}]: {error}")
                return
            destination = supporting if section == "supporting" else counter
            _append_claim(destination, guarded_fact)
            return

        destination_name = "inferences" if claim.kind is ClaimKind.INFERENCE else "open questions"
        expected_section = "inferences" if claim.kind is ClaimKind.INFERENCE else "open_questions"
        if section != expected_section:
            errors.append(
                f"{label}_CLAIM_MOVED[{index}]: {claim.kind.value} moved to {destination_name}"
            )
        sanitized, citations_were_removed = _sanitize_optional_citations(
            claim,
            filing_evidence,
            web_evidence,
            ticker,
            corpus_version,
            errors,
            label,
            index,
            canonical_sources,
            canonical_web_sources,
            rejected_web_ids,
        )
        unsafe_content_was_dropped = (
            unsafe_content_was_dropped or citations_were_removed
        )
        destination = inferences if claim.kind is ClaimKind.INFERENCE else open_questions
        _append_claim(destination, sanitized)

    for section, claims in (
        ("supporting", memo.supporting_claims),
        ("counter", memo.counter_claims),
        ("inferences", memo.inferences),
        ("open_questions", memo.open_questions),
    ):
        for index, claim in enumerate(claims):
            handle(claim, section, index)

    original_verified_count = sum(
        _is_declared_verified(claim)
        for claims in (
            memo.supporting_claims,
            memo.counter_claims,
            memo.inferences,
            memo.open_questions,
        )
        for claim in claims
    )
    cited_ids = _cited_ids((*supporting, *counter, *inferences, *open_questions))
    sources = [canonical_sources[chunk_id] for chunk_id in cited_ids]
    cited_web_ids = _cited_web_ids((*supporting, *counter, *inferences, *open_questions))
    web_sources = [canonical_web_sources[source_id] for source_id in cited_web_ids]
    sufficiency = memo.information_sufficiency
    confidence = memo.confidence
    if unsafe_content_was_dropped:
        sufficiency = _cap_sufficiency(sufficiency, "B")
        confidence = _cap_confidence(confidence, Confidence.MEDIUM)
    if len(supporting) + len(counter) < original_verified_count:
        sufficiency = _cap_sufficiency(sufficiency, "B")
        confidence = _cap_confidence(confidence, Confidence.MEDIUM)
    if not supporting or not counter:
        sufficiency = _cap_sufficiency(sufficiency, "C")
        confidence = _cap_confidence(confidence, Confidence.LOW)

    return GuardedMemo(
        ticker=ticker,
        corpus_version=corpus_version,
        research_question=research_question,
        supporting_claims=supporting,
        counter_claims=counter,
        inferences=inferences,
        open_questions=open_questions,
        information_sufficiency=sufficiency,
        confidence=confidence,
        sources=sources,
        web_sources=web_sources,
        source_policy_version=(
            web_validator.policy_version if web_validator is not None else None
        ),
        errors=collapse_private_detail_errors(errors),
    )


def guard_skill_memo(
    memo: SkillResearchMemo,
    evidence: EvidenceBundle,
    ticker: str,
    recipe: ResearchRecipe,
    *,
    web_validator: PersistedWebEvidenceValidator | None = None,
    report_as_of: date | None = None,
) -> GuardedSkillMemo:
    """Guard one structured P1 memo against its exact recipe and evidence bundle."""
    normalized_ticker = ticker.strip().upper()
    validated_evidence, web_validation_errors, rejected_web_ids = _revalidated_web_evidence(
        evidence,
        ticker=normalized_ticker,
        validator=web_validator,
    )
    errors: list[str] = list(web_validation_errors)
    filing_index, web_index, invalid_ids, ambiguous_ids = _p1_source_indexes(
        validated_evidence
    )
    citation_bindings = _p1_citation_bindings(validated_evidence)
    retained_filing_ids: list[str] = []
    retained_web_ids: list[str] = []
    sections: list[SkillResearchSection] = []
    evidence_was_dropped = bool(rejected_web_ids)

    identity_mismatch = memo.recipe_name is not recipe.name or memo.recipe_version != recipe.version
    if identity_mismatch:
        errors.append("RECIPE_IDENTITY_MISMATCH")

    research_question, private_question = redact_private_text(
        memo.research_question,
        ticker=normalized_ticker,
    )
    if private_question:
        errors.append(PRIVATE_DETAIL_CODE)
        evidence_was_dropped = True
    elif _is_unsafe_output_text(research_question, normalized_ticker):
        research_question = "Removed by output safety guard."
        errors.append("RESEARCH_QUESTION_REDACTED: unsafe output text")
        evidence_was_dropped = True

    information_gaps: list[str] = []
    for index, gap in enumerate(memo.information_gaps):
        if _is_unsafe_output_text(gap, normalized_ticker):
            errors.append(f"INFORMATION_GAP_DROPPED[{index}]: unsafe output text")
            evidence_was_dropped = True
        else:
            information_gaps.append(gap)

    for section in memo.sections:
        claims: list[Claim] = []
        for index, claim in enumerate(section.claims):
            guarded_claim, claim_errors = _guard_p1_claim(
                claim=claim,
                facet=section.facet,
                index=index,
                ticker=normalized_ticker,
                recipe=recipe,
                filing_index=filing_index,
                web_index=web_index,
                invalid_ids=invalid_ids,
                ambiguous_ids=ambiguous_ids,
                citation_bindings=citation_bindings,
                rejected_web_ids=rejected_web_ids,
            )
            errors.extend(claim_errors)
            if claim_errors:
                evidence_was_dropped = True
            if guarded_claim is None:
                evidence_was_dropped = True
                continue
            claims.append(guarded_claim)
            retained_filing_ids.extend(guarded_claim.evidence_chunk_ids)
            retained_web_ids.extend(guarded_claim.web_evidence_ids)
        sections.append(section.model_copy(update={"claims": claims}))

    data_points: list[GuardedFinancialDataPoint] = []
    for index, point in enumerate(memo.data_points):
        unsafe_field = _unsafe_financial_field(point, normalized_ticker)
        if unsafe_field is not None:
            errors.append(f"FINANCIAL_DATA_DROPPED[{index}]: unsafe {unsafe_field}")
            evidence_was_dropped = True
            continue
        if rejected_web_ids.intersection(point.source_ids):
            evidence_was_dropped = True
            continue
        guarded_point, point_error, filing_ids, web_ids = _guard_financial_point(
            point=point,
            index=index,
            ticker=normalized_ticker,
            recipe=recipe,
            filing_index=filing_index,
            web_index=web_index,
            invalid_ids=invalid_ids,
            ambiguous_ids=ambiguous_ids,
            citation_bindings=citation_bindings,
        )
        if point_error is not None:
            errors.append(point_error)
            evidence_was_dropped = True
            continue
        assert guarded_point is not None
        data_points.append(guarded_point)
        retained_filing_ids.extend(filing_ids)
        retained_web_ids.extend(web_ids)

    data_points, conflict_errors = _mark_financial_conflicts(data_points)
    errors.extend(conflict_errors)

    present_facets = {section.facet for section in sections if section.claims}
    if data_points:
        present_facets.add(ResearchFacet.DATA_VERIFICATION)
    if information_gaps:
        present_facets.add(ResearchFacet.INFORMATION_GAPS)
    missing_facets = [facet for facet in recipe.required_facets if facet not in present_facets]
    errors.extend(f"MISSING_REQUIRED_FACET: {facet.value}" for facet in missing_facets)

    bull_bear_incomplete = _bull_bear_incomplete(sections, recipe)
    if bull_bear_incomplete:
        errors.append("BULL_BEAR_EVIDENCE_INCOMPLETE: both sides require cited evidence")

    sufficiency = memo.information_sufficiency
    confidence = memo.confidence
    if identity_mismatch:
        sufficiency = InformationSufficiency.INSUFFICIENT
    if missing_facets:
        cap = (
            InformationSufficiency.INSUFFICIENT
            if len(missing_facets) == len(recipe.required_facets)
            else InformationSufficiency.PARTIAL
        )
        sufficiency = _cap_p1_sufficiency(sufficiency, cap)
    if evidence_was_dropped or bull_bear_incomplete or conflict_errors:
        sufficiency = _cap_p1_sufficiency(sufficiency, InformationSufficiency.PARTIAL)
    if evidence_was_dropped:
        confidence = min(confidence, Decimal("0.5"))

    retained_filing_ids = list(dict.fromkeys(retained_filing_ids))
    retained_web_ids = list(dict.fromkeys(retained_web_ids))
    if not retained_filing_ids and not retained_web_ids:
        sufficiency = InformationSufficiency.INSUFFICIENT

    guarded_memo = GuardedSkillResearchMemo.model_validate(
        {
            **memo.model_dump(exclude={"data_points"}),
            "recipe_name": recipe.name,
            "recipe_version": recipe.version,
            "research_question": research_question,
            "sections": sections,
            "data_points": data_points,
            "information_gaps": information_gaps,
            "information_sufficiency": sufficiency,
            "confidence": confidence,
        }
    )
    filing_sources = [filing_index[source_id] for source_id in retained_filing_ids]
    web_sources = [web_index[source_id] for source_id in retained_web_ids]
    provenance = _p1_report_provenance(
        ticker=normalized_ticker,
        recipe=recipe,
        filing_sources=filing_sources,
        web_sources=web_sources,
        source_policy_version=(
            web_validator.policy_version if web_validator is not None else None
        ),
        prompt_version=current_prompt_version(),
        report_as_of=report_as_of,
        information_sufficiency=sufficiency,
    )
    return GuardedSkillMemo(
        ticker=normalized_ticker,
        memo=guarded_memo,
        filing_sources=filing_sources,
        web_sources=web_sources,
        provenance=provenance,
        guard_errors=collapse_private_detail_errors(errors),
    )


def _revalidated_web_evidence(
    evidence: EvidenceBundle,
    *,
    ticker: str,
    validator: PersistedWebEvidenceValidator | None,
) -> tuple[EvidenceBundle, list[str], frozenset[str]]:
    if validator is None:
        rejected_ids = frozenset(source.id for source in evidence.web_evidence)
        errors = [
            (
                f"WEB_SOURCE_VALIDATOR_REQUIRED: {source.id}"
                if retain_public_text(source.id, ticker=ticker)
                and retain_public_text(source.title, ticker=ticker)
                else PRIVATE_DETAIL_CODE
            )
            for source in evidence.web_evidence
        ]
        return evidence.model_copy(update={"web_evidence": []}), errors, rejected_ids
    retained: list[WebEvidence] = []
    errors: list[str] = []
    rejected_ids: set[str] = set()
    for source in evidence.web_evidence:
        if any(
            not retain_public_text(value, ticker=ticker)
            for value in (source.id, source.title, str(source.source_url))
        ):
            errors.append(PRIVATE_DETAIL_CODE)
            rejected_ids.add(source.id)
            continue
        try:
            validated = validator.validate(ticker=ticker, evidence=source)
        except WebEvidenceValidationError as error:
            errors.append(f"{error.code}: {source.id}")
            rejected_ids.add(source.id)
            continue
        if any(
            not retain_public_text(value, ticker=ticker)
            for value in (str(validated.source_url), str(validated.canonical_url))
        ):
            errors.append(PRIVATE_DETAIL_CODE)
            rejected_ids.add(source.id)
            continue
        retained.append(validated)
    return (
        evidence.model_copy(update={"web_evidence": retained}),
        errors,
        frozenset(rejected_ids),
    )


def _unsafe_financial_field(point: FinancialDataPoint, ticker: str) -> str | None:
    fields = (
        ("name", point.name),
        ("currency", point.currency),
        ("unit", point.unit),
        ("definition", point.definition),
    )
    combined = " ".join(value for _, value in fields if value is not None)
    normalized = " ".join(unicodedata.normalize("NFKC", combined).casefold().split())
    if (
        re.search(r"\b(?:stock\s+|share\s+)?price\b|(?:股价|股票价格|目标价)", normalized)
        and re.search(
            r"\b(?:predict|forecast|project|estimate|target)\b|(?:预测|预估|预计|估算)",
            normalized,
        )
    ):
        return "price prediction"
    return next(
        (
            field
            for field, value in fields
            if value is not None and _is_unsafe_output_text(value, ticker)
        ),
        None,
    )


def _guard_p1_claim(
    *,
    claim: Claim,
    facet: ResearchFacet,
    index: int,
    ticker: str,
    recipe: ResearchRecipe,
    filing_index: dict[str, EvidenceChunk],
    web_index: dict[str, WebEvidence],
    invalid_ids: frozenset[str],
    ambiguous_ids: frozenset[str],
    citation_bindings: frozenset[tuple[ResearchFacet, EvidenceSide, str]],
    rejected_web_ids: frozenset[str],
) -> tuple[Claim | None, list[str]]:
    label = f"{facet.value.upper()}_CLAIM"
    try:
        canonical_claim = Claim.model_validate(claim.model_dump(warnings=False), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError):
        return None, [f"{label}_DROPPED[{index}]: invalid claim schema"]
    if _is_unsafe_output_text(canonical_claim.text, ticker):
        return None, [f"{label}_DROPPED[{index}]: unsafe claim text"]

    claim_errors: list[str] = []
    valid_filing_ids: list[str] = []
    valid_web_ids: list[str] = []
    for source_id in canonical_claim.evidence_chunk_ids:
        citation_error = _p1_citation_error(
            source_id=source_id,
            source_label="SEC",
            ticker=ticker,
            recipe=recipe,
            source=filing_index.get(source_id),
            invalid_ids=invalid_ids,
            ambiguous_ids=ambiguous_ids,
        )
        if citation_error is None:
            binding_error = _p1_binding_error(
                source_id=source_id,
                facet=facet,
                citation_bindings=citation_bindings,
            )
            if binding_error is None:
                valid_filing_ids.append(source_id)
            else:
                claim_errors.append(binding_error)
        else:
            claim_errors.append(citation_error)
    for source_id in canonical_claim.web_evidence_ids:
        if source_id in rejected_web_ids:
            continue
        source = web_index.get(source_id)
        citation_error = _p1_citation_error(
            source_id=source_id,
            source_label="web",
            ticker=ticker,
            recipe=recipe,
            source=source,
            invalid_ids=invalid_ids,
            ambiguous_ids=ambiguous_ids,
        )
        if citation_error is None:
            binding_error = _p1_binding_error(
                source_id=source_id,
                facet=facet,
                citation_bindings=citation_bindings,
            )
            if binding_error is None:
                valid_web_ids.append(source_id)
            else:
                claim_errors.append(binding_error)
        else:
            claim_errors.append(citation_error)

    if canonical_claim.kind is ClaimKind.VERIFIED_FACT and claim_errors:
        return None, [f"{label}_DROPPED[{index}]: {error}" for error in claim_errors]
    if canonical_claim.kind is ClaimKind.VERIFIED_FACT and not (
        valid_filing_ids or valid_web_ids
    ):
        return None, []

    guarded_claim = canonical_claim.model_copy(
        update={
            "evidence_chunk_ids": valid_filing_ids,
            "web_evidence_ids": valid_web_ids,
        }
    )
    errors = [f"{label}_CITATION_REMOVED[{index}]: {error}" for error in claim_errors]
    if facet in {ResearchFacet.BULL_CASE, ResearchFacet.BEAR_CASE} and not (
        valid_filing_ids or valid_web_ids
    ):
        return None, [*errors, f"{label}_DROPPED[{index}]: claim has no valid evidence"]
    return guarded_claim, errors


def _guard_financial_point(
    *,
    point: FinancialDataPoint,
    index: int,
    ticker: str,
    recipe: ResearchRecipe,
    filing_index: dict[str, EvidenceChunk],
    web_index: dict[str, WebEvidence],
    invalid_ids: frozenset[str],
    ambiguous_ids: frozenset[str],
    citation_bindings: frozenset[tuple[ResearchFacet, EvidenceSide, str]],
) -> tuple[GuardedFinancialDataPoint | None, str | None, list[str], list[str]]:
    canonical_values = {
        field: getattr(point, field)
        for field in (
            "name",
            "value",
            "currency",
            "unit",
            "period_start",
            "period_end",
            "definition",
            "source_ids",
        )
    }
    canonical_values["source_ids"] = list(dict.fromkeys(point.source_ids))
    canonical_point = FinancialDataPoint.model_validate(canonical_values, strict=True)
    filing_ids: list[str] = []
    web_ids: list[str] = []
    source_provenance: list[FinancialSourceProvenance] = []
    for source_id in canonical_point.source_ids:
        filing = filing_index.get(source_id)
        web = web_index.get(source_id)
        source = filing if filing is not None else web
        error = _p1_citation_error(
            source_id=source_id,
            source_label="financial",
            ticker=ticker,
            recipe=recipe,
            source=source,
            invalid_ids=invalid_ids,
            ambiguous_ids=ambiguous_ids,
        )
        if error is not None:
            return None, f"FINANCIAL_DATA_DROPPED[{index}]: {error}", [], []
        binding_error = _p1_binding_error(
            source_id=source_id,
            facet=ResearchFacet.DATA_VERIFICATION,
            citation_bindings=citation_bindings,
        )
        if binding_error is not None:
            return None, f"FINANCIAL_DATA_DROPPED[{index}]: {binding_error}", [], []
        if filing is not None:
            filing_ids.append(source_id)
            source_provenance.append(
                FinancialSourceProvenance(
                    source_ref=SourceRef(
                        ticker=ticker,
                        kind=SourceRefKind.FILING,
                        source_id=source_id,
                    ),
                    source_kind=SourceKind.FILING,
                    source_tier=SourceTier.PRIMARY,
                    canonical_source_identity=canonical_financial_source_identity(
                        storage_kind=SourceRefKind.FILING,
                        source_kind=SourceKind.FILING,
                        source_url=filing.source_url,
                        content_hash=None,
                        accession_no=filing.accession_no,
                    ),
                )
            )
        else:
            assert web is not None
            web_ids.append(source_id)
            canonical_url = str(
                web.canonical_url
                if isinstance(web, PolicyValidatedWebEvidence)
                else web.source_url
            )
            source_provenance.append(
                FinancialSourceProvenance(
                    source_ref=SourceRef(
                        ticker=ticker,
                        kind=SourceRefKind.WEB,
                        source_id=source_id,
                    ),
                    source_kind=web.source_kind,
                    source_tier=web.source_tier,
                    canonical_source_identity=canonical_financial_source_identity(
                        storage_kind=SourceRefKind.WEB,
                        source_kind=web.source_kind,
                        source_url=canonical_url,
                        content_hash=web.content_hash,
                    ),
                )
            )
    independent_sources = {
        source.canonical_source_identity for source in source_provenance
    }
    status = (
        VerificationStatus.MISSING
        if canonical_point.value is None
        else VerificationStatus.VERIFIED
        if len(independent_sources) >= 2
        else VerificationStatus.SINGLE_SOURCE
    )
    guarded = GuardedFinancialDataPoint(
        **canonical_point.model_dump(),
        ticker=ticker,
        observation_id=financial_observation_id(
            ticker=ticker,
            name=canonical_point.name,
            period_start=canonical_point.period_start,
            period_end=canonical_point.period_end,
            currency=canonical_point.currency,
            unit=canonical_point.unit,
            definition=canonical_point.definition,
        ),
        source_provenance=source_provenance,
        verification_status=status,
    )
    return guarded, None, filing_ids, web_ids


def _p1_citation_error(
    *,
    source_id: str,
    source_label: str,
    ticker: str,
    recipe: ResearchRecipe,
    source: EvidenceChunk | WebEvidence | None,
    invalid_ids: frozenset[str],
    ambiguous_ids: frozenset[str],
) -> str | None:
    if not retain_public_text(source_id, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    if source_id in ambiguous_ids:
        return f"{source_label} evidence '{source_id}' is ambiguous in the supplied bundle"
    if source_id in invalid_ids:
        return f"{source_label} evidence '{source_id}' has invalid citation metadata"
    if source is None:
        return f"{source_label} evidence '{source_id}' was not provided"
    if source.ticker.strip().upper() != ticker:
        return f"citation '{source_id}' ticker does not match run ticker"
    if _has_unsafe_provenance_display_text(source, ticker):
        return (
            PRIVATE_DETAIL_CODE
            if not retain_public_text(
                source.section if isinstance(source, EvidenceChunk) else source.title,
                ticker=ticker,
            )
            else f"citation '{source_id}' has unsafe provenance display text"
        )
    if not source_allowed_for_recipe(source, recipe):
        return (
            f"citation '{source_id}' is not eligible under "
            f"{recipe.web_usage_policy.value} policy"
        )
    return None


def _p1_citation_bindings(
    evidence: EvidenceBundle,
) -> frozenset[tuple[ResearchFacet, EvidenceSide, str]]:
    source_kinds: dict[str, SourceKind] = {}
    ambiguous_ids: set[str] = set()
    for source, source_kind in (
        *((source, SourceKind.FILING) for source in evidence.filing_evidence),
        *((source, source.source_kind) for source in evidence.web_evidence),
    ):
        if source.id in source_kinds:
            ambiguous_ids.add(source.id)
        else:
            source_kinds[source.id] = source_kind
    assignment_keys = {
        (assignment.question_index, assignment.side, assignment.source_id)
        for assignment in evidence.assignments
        if assignment.source_id not in ambiguous_ids
        and source_kinds.get(assignment.source_id) is assignment.source_kind
    }
    return frozenset(
        (assignment.facet, assignment.side, assignment.source_id)
        for assignment in evidence.facet_assignments
        if (assignment.question_index, assignment.side, assignment.source_id)
        in assignment_keys
    )


def _p1_binding_error(
    *,
    source_id: str,
    facet: ResearchFacet,
    citation_bindings: frozenset[tuple[ResearchFacet, EvidenceSide, str]],
) -> str | None:
    required_sides = (
        frozenset({EvidenceSide.SUPPORT})
        if facet is ResearchFacet.BULL_CASE
        else frozenset({EvidenceSide.CHALLENGE})
        if facet is ResearchFacet.BEAR_CASE
        else frozenset(EvidenceSide)
    )
    if any((facet, side, source_id) in citation_bindings for side in required_sides):
        return None
    side_label = (
        EvidenceSide.SUPPORT.value
        if facet is ResearchFacet.BULL_CASE
        else EvidenceSide.CHALLENGE.value
        if facet is ResearchFacet.BEAR_CASE
        else "support-or-challenge"
    )
    return f"citation '{source_id}' lacks an exact {facet.value}/{side_label} binding"


def _has_unsafe_provenance_display_text(
    source: EvidenceChunk | WebEvidence,
    ticker: str,
) -> bool:
    value = source.section if isinstance(source, EvidenceChunk) else source.title
    return _is_unsafe_output_text(value, ticker)


def _p1_source_indexes(
    evidence: EvidenceBundle,
) -> tuple[
    dict[str, EvidenceChunk],
    dict[str, WebEvidence],
    frozenset[str],
    frozenset[str],
]:
    filing_index: dict[str, EvidenceChunk] = {}
    web_index: dict[str, WebEvidence] = {}
    invalid_ids: set[str] = set()
    seen_ids: set[str] = set()
    ambiguous_ids: set[str] = set()

    for source in evidence.filing_evidence:
        source_id = getattr(source, "id", None)
        if not isinstance(source_id, str):
            continue
        if source_id in seen_ids:
            ambiguous_ids.add(source_id)
        seen_ids.add(source_id)
        canonical = _canonical_evidence_chunk(source)
        if canonical is None:
            invalid_ids.add(source_id)
        else:
            filing_index.setdefault(source_id, canonical)

    for source in evidence.web_evidence:
        source_id = getattr(source, "id", None)
        if not isinstance(source_id, str):
            continue
        if source_id in seen_ids:
            ambiguous_ids.add(source_id)
        seen_ids.add(source_id)
        canonical = _canonical_web_evidence(source)
        if canonical is None:
            invalid_ids.add(source_id)
        else:
            web_index.setdefault(source_id, canonical)

    return filing_index, web_index, frozenset(invalid_ids), frozenset(ambiguous_ids)


def _canonical_web_evidence(source: object) -> WebEvidence | None:
    if not isinstance(source, WebEvidence):
        return None
    values = vars(source)
    if any(field not in values for field in _WEB_SOURCE_FIELDS):
        return None
    if any(
        type(values[field]) is not str
        for field in ("id", "ticker", "title", "content", "content_hash")
    ):
        return None
    if not isinstance(values["source_kind"], SourceKind) or not isinstance(
        values["source_tier"], SourceTier
    ):
        return None
    if any(
        not values[field].strip()
        for field in ("id", "ticker", "title", "content", "content_hash")
    ):
        return None
    if any(
        _has_control_character(values[field])
        for field in ("id", "ticker", "title", "content_hash")
    ):
        return None
    if any(
        unicodedata.category(character).startswith("C") and character not in "\n\t"
        for character in values["content"]
    ):
        return None
    try:
        parsed = urlsplit(str(values["source_url"]))
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    try:
        canonical = WebEvidence.model_validate(
            {field: values[field] for field in _WEB_SOURCE_FIELDS},
            strict=True,
        )
        if isinstance(source, PolicyValidatedWebEvidence):
            return PolicyValidatedWebEvidence(
                **canonical.model_dump(),
                policy_version=source.policy_version,
                canonical_url=source.canonical_url,
            )
        return canonical
    except (TypeError, ValidationError, ValueError):
        return None


def _mark_financial_conflicts(
    points: list[GuardedFinancialDataPoint],
) -> tuple[list[GuardedFinancialDataPoint], list[str]]:
    grouped: dict[str, list[GuardedFinancialDataPoint]] = {}
    for point in points:
        normalized_name = canonical_financial_text(point.name)
        grouped.setdefault(normalized_name, []).append(point)

    guarded: list[GuardedFinancialDataPoint] = []
    errors: list[str] = []
    for group in grouped.values():
        nonmissing = [point for point in group if point.value is not None]
        identities = {point.observation_id for point in group}
        conflict_status: VerificationStatus | None = None
        if len(identities) > 1 and nonmissing:
            status = VerificationStatus.NOT_COMPARABLE
            note = _financial_conflict_note(nonmissing, not_comparable=True)
            code = "FINANCIAL_DATA_NOT_COMPARABLE"
            conflict_status = status
        elif len({point.value for point in nonmissing}) > 1:
            status = VerificationStatus.DISCREPANCY
            note = _financial_conflict_note(nonmissing, not_comparable=False)
            code = "FINANCIAL_DATA_DISCREPANCY"
            conflict_status = status
        else:
            independent_sources = {
                source.canonical_source_identity
                for point in nonmissing
                for source in point.source_provenance
            }
            status = (
                VerificationStatus.VERIFIED
                if len(independent_sources) >= 2
                else VerificationStatus.SINGLE_SOURCE
            )
            note = None

        reconciled: list[GuardedFinancialDataPoint] = []
        for point in group:
            point_status = (
                VerificationStatus.MISSING
                if point.value is None
                else status
            )
            reconciled.append(
                GuardedFinancialDataPoint.model_validate(
                    {
                        **point.model_dump(),
                        "verification_status": point_status,
                        "discrepancy_note": (
                            note
                            if point_status
                            in {
                                VerificationStatus.DISCREPANCY,
                                VerificationStatus.NOT_COMPARABLE,
                            }
                            else None
                        ),
                    }
                )
            )
        if conflict_status is not None:
            reconciled.sort(key=lambda point: not _is_primary_financial_observation(point))
            errors.append(f"{code}: {group[0].name}")
        guarded.extend(reconciled)
    return guarded, errors


def _is_primary_financial_observation(point: GuardedFinancialDataPoint) -> bool:
    return any(
        source.source_tier is SourceTier.PRIMARY for source in point.source_provenance
    )


def _financial_conflict_note(
    points: list[GuardedFinancialDataPoint],
    *,
    not_comparable: bool,
) -> str:
    has_primary = any(_is_primary_financial_observation(point) for point in points)
    has_secondary = any(
        source.source_tier is SourceTier.AUTHORITATIVE_SECONDARY
        for point in points
        for source in point.source_provenance
    )
    conflict = (
        "Conflicting period, currency, unit, or definition observations were retained."
        if not_comparable
        else "Conflicting exact values were retained."
    )
    if has_primary and has_secondary:
        precedence = (
            "Primary disclosure appears first; secondary values were not silently "
            "substituted or averaged."
        )
    elif has_primary:
        precedence = (
            "Primary-source values were retained; no value was silently substituted or "
            "averaged."
        )
    else:
        precedence = (
            "No primary observation was available; no value was silently substituted or "
            "averaged."
        )
    return f"{conflict} {precedence}"


def _p1_report_provenance(
    *,
    ticker: str,
    recipe: ResearchRecipe,
    filing_sources: list[EvidenceChunk],
    web_sources: list[WebEvidence],
    source_policy_version: str | None,
    prompt_version: str | None,
    report_as_of: date | None,
    information_sufficiency: InformationSufficiency,
) -> ReportProvenance:
    source_dates = [source.filed_at for source in filing_sources]
    source_dates.extend(
        (
            source.published_at.date()
            if source.published_at is not None
            else source.fetched_at.date()
        )
        for source in web_sources
    )
    source_refs = (
        *(
            SourceRef(ticker=ticker, kind=SourceRefKind.FILING, source_id=source.id)
            for source in filing_sources
        ),
        *(
            SourceRef(ticker=ticker, kind=SourceRefKind.WEB, source_id=source.id)
            for source in web_sources
        ),
    )
    return ReportProvenance(
        recipes=(RecipeProvenance(name=recipe.name, version=recipe.version),),
        source_policy_versions=(
            () if source_policy_version is None else (source_policy_version,)
        ),
        corpus_versions=tuple(
            dict.fromkeys(source.corpus_version for source in filing_sources)
        ),
        prompt_versions=(() if prompt_version is None else (prompt_version,)),
        requested_as_of_dates=(
            () if report_as_of is None else (report_as_of,)
        ),
        evidence_cutoff_dates=(() if not source_dates else (max(source_dates),)),
        information_sufficiency=information_sufficiency,
        source_refs=source_refs,
    )


def _bull_bear_incomplete(
    sections: list[SkillResearchSection],
    recipe: ResearchRecipe,
) -> bool:
    relevant_facets = {ResearchFacet.BULL_CASE, ResearchFacet.BEAR_CASE}
    if not relevant_facets.intersection(recipe.required_facets):
        return False
    evidence_by_facet = {
        section.facet: any(
            claim.evidence_chunk_ids or claim.web_evidence_ids for claim in section.claims
        )
        for section in sections
        if section.facet in relevant_facets
    }
    return not all(evidence_by_facet.get(facet, False) for facet in relevant_facets)


def _cap_p1_sufficiency(
    value: InformationSufficiency,
    cap: InformationSufficiency,
) -> InformationSufficiency:
    order = {
        InformationSufficiency.INSUFFICIENT: 0,
        InformationSufficiency.PARTIAL: 1,
        InformationSufficiency.SUFFICIENT: 2,
    }
    return value if order[value] <= order[cap] else cap


def _guard_fact(
    claim: Claim,
    evidence: dict[str, EvidenceChunk],
    web_evidence: dict[str, WebEvidence],
    ticker: str,
    corpus_version: str,
    canonical_sources: dict[str, EvidenceChunk],
    canonical_web_sources: dict[str, WebEvidence],
    rejected_web_ids: frozenset[str],
    supports_web_evidence: bool,
    side_bindings: frozenset[tuple[EvidenceSide, str]] | None,
    expected_side: EvidenceSide,
) -> tuple[Claim, str | None]:
    if claim.web_evidence_ids and not supports_web_evidence:
        return claim, "web evidence citations are not supported"
    for chunk_id in claim.evidence_chunk_ids:
        error = _citation_error(
            chunk_id,
            evidence,
            ticker,
            corpus_version,
            canonical_sources,
        )
        if error is not None:
            return claim, error
    for source_id in claim.web_evidence_ids:
        error = _web_citation_error(
            source_id,
            web_evidence,
            ticker,
            canonical_web_sources,
            rejected_web_ids,
        )
        if error is not None:
            return claim, error
    if side_bindings is not None:
        for source_id in (*claim.evidence_chunk_ids, *claim.web_evidence_ids):
            if (expected_side, source_id) not in side_bindings:
                return (
                    claim,
                    f"citation '{source_id}' lacks a {expected_side.value} assignment",
                )
    return claim.model_copy(
        update={
            "evidence_chunk_ids": list(dict.fromkeys(claim.evidence_chunk_ids)),
            "web_evidence_ids": list(dict.fromkeys(claim.web_evidence_ids)),
        }
    ), None


def _sanitize_optional_citations(
    claim: Claim,
    evidence: dict[str, EvidenceChunk],
    web_evidence: dict[str, WebEvidence],
    ticker: str,
    corpus_version: str,
    errors: list[str],
    label: str,
    index: int,
    canonical_sources: dict[str, EvidenceChunk],
    canonical_web_sources: dict[str, WebEvidence],
    rejected_web_ids: frozenset[str],
) -> tuple[Claim, bool]:
    retained: list[str] = []
    removed = False
    for chunk_id in dict.fromkeys(claim.evidence_chunk_ids):
        error = _citation_error(
            chunk_id,
            evidence,
            ticker,
            corpus_version,
            canonical_sources,
        )
        if error is None:
            retained.append(chunk_id)
        else:
            removed = True
            errors.append(f"{label}_CITATION_REMOVED[{index}]: {error}")
    retained_web: list[str] = []
    for source_id in dict.fromkeys(claim.web_evidence_ids):
        error = _web_citation_error(
            source_id,
            web_evidence,
            ticker,
            canonical_web_sources,
            rejected_web_ids,
        )
        if error is None:
            retained_web.append(source_id)
        else:
            removed = True
            errors.append(f"{label}_CITATION_REMOVED[{index}]: {error}")
    return (
        claim.model_copy(
            update={
                "evidence_chunk_ids": retained,
                "web_evidence_ids": retained_web,
            }
        ),
        removed,
    )


def _web_citation_error(
    source_id: str,
    evidence: dict[str, WebEvidence],
    ticker: str,
    canonical_sources: dict[str, WebEvidence],
    rejected_web_ids: frozenset[str],
) -> str | None:
    if not retain_public_text(source_id, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    if source_id in rejected_web_ids:
        return f"web citation '{source_id}' failed final policy validation"
    source = evidence.get(source_id)
    if source is None:
        return f"web citation '{source_id}' was not provided"
    if source.id != source_id:
        return f"web evidence key '{source_id}' does not match source id '{source.id}'"
    canonical = _canonical_web_evidence(source)
    if canonical is None or not isinstance(canonical, PolicyValidatedWebEvidence):
        return f"web citation '{source_id}' has invalid citation metadata"
    if not retain_public_text(canonical.title, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    if canonical.ticker.strip().upper() != ticker.strip().upper():
        return f"web citation '{source_id}' ticker does not match '{ticker}'"
    canonical_sources[source_id] = canonical
    return None


def _citation_error(
    chunk_id: str,
    evidence: dict[str, EvidenceChunk],
    ticker: str,
    corpus_version: str,
    canonical_sources: dict[str, EvidenceChunk],
) -> str | None:
    if not retain_public_text(chunk_id, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    chunk = evidence.get(chunk_id)
    if chunk is None:
        return f"citation '{chunk_id}' was not provided"
    actual_id = getattr(chunk, "id", None)
    if isinstance(actual_id, str) and not retain_public_text(actual_id, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    if actual_id is not None and actual_id != chunk_id:
        return f"evidence key '{chunk_id}' does not match chunk id '{actual_id}'"
    canonical = _canonical_evidence_chunk(chunk)
    if canonical is None:
        return f"citation '{chunk_id}' has invalid citation metadata"
    if not retain_public_text(canonical.section, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    if canonical.ticker != ticker:
        return f"citation '{chunk_id}' ticker '{canonical.ticker}' does not match '{ticker}'"
    if canonical.corpus_version != corpus_version:
        return (
            f"citation '{chunk_id}' corpus '{canonical.corpus_version}' "
            f"does not match '{corpus_version}'"
        )
    canonical_sources[chunk_id] = canonical
    return None


def _canonical_evidence_chunk(chunk: object) -> EvidenceChunk | None:
    if not isinstance(chunk, EvidenceChunk):
        return None
    values = vars(chunk)
    if any(field not in values for field in _SOURCE_FIELDS):
        return None
    if any(type(values[field]) is not str for field in _SOURCE_TEXT_FIELDS):
        return None
    if type(values["content"]) is not str or not values["content"].strip():
        return None
    if type(values["filed_at"]) is not date:
        return None
    if type(values["raw_start"]) is not int or type(values["raw_end"]) is not int:
        return None
    if any(
        not values[field]
        or values[field] != values[field].strip()
        or _has_control_character(values[field])
        for field in _SOURCE_TEXT_FIELDS
    ):
        return None
    if any(
        unicodedata.category(character).startswith("C") and character not in "\n\t"
        for character in values["content"]
    ):
        return None
    if not _is_admissible_source_url(values["source_url"]):
        return None
    if _ACCESSION.fullmatch(values["accession_no"]) is None:
        return None
    try:
        return EvidenceChunk.model_validate(
            {field: values[field] for field in _SOURCE_FIELDS},
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        return None


def _is_admissible_source_url(value: str) -> bool:
    if any(character.isspace() or character in _URL_DELIMITERS for character in value):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"sec.gov", "www.sec.gov"}
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
        and _SEC_ARCHIVE_PATH.fullmatch(parsed.path) is not None
    )


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _is_unsafe_output_text(text: str, ticker: str) -> bool:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return (
        not retain_public_text(text, ticker=ticker)
        or _contains_advisory_directive(normalized, ticker)
        or contains_price_prediction(normalized, ticker)
        or _contains_instruction_override(normalized)
    )


def _contains_instruction_override(text: str) -> bool:
    if _has_unsafe_action_after_negated_report(text):
        return True
    clauses = re.split(
        r"(?:[.!?。！？;；:：,，—–]\s*|\s+(?:and|but)(?:\s+then)?\s+|"
        r"\s+yet\s+|(?:然后|随后|但是|但|并且|却|而且))",
        text,
    )
    return any(_contains_instruction_override_clause(clause) for clause in clauses if clause)


def _has_unsafe_action_after_negated_report(text: str) -> bool:
    """A safe reported negation cannot exempt a later directive relation."""
    reporting_subject = (
        r"(?:the\s+)?(?:company|management|system|issuer|board|customers?|"
        r"institutions?|analysts?|brokers?)"
    )
    action = rf"(?:{_OVERRIDE_ACTION}|{_FOLLOW_ACTION}|{_EXFIL_ACTION})"
    first = re.match(
        rf"{reporting_subject}\s+did\s+not\s+{action}\b"
        rf".{{0,32}}\b{_SENSITIVE_INSTRUCTION}\b",
        text,
    )
    if first is None:
        return False

    tail = text[first.end() :]
    for match in re.finditer(
        rf"(?P<action>{action})\b.{{0,32}}\b{_SENSITIVE_INSTRUCTION}\b",
        tail,
    ):
        prefix = tail[max(0, match.start() - 16) : match.start()]
        if re.search(r"\bdid\s+not\s+$", prefix):
            continue
        if re.search(r"\b(?:do\s+not|don't|never)\s+$", prefix) and re.fullmatch(
            rf"(?:{_OVERRIDE_ACTION}|{_EXFIL_ACTION})", match.group("action")
        ):
            continue
        return True
    return False


def _contains_instruction_override_clause(text: str) -> bool:
    if _is_negated_reporting_statement(text):
        return False

    directive = _strip_directive_wrappers(text)
    sensitive = rf".{{0,32}}\b{_SENSITIVE_INSTRUCTION}\b"
    chinese_sensitive = rf".{{0,16}}{_CHINESE_SENSITIVE_INSTRUCTION}"

    # Negating override/exfiltration preserves the protected instruction;
    # negating follow/obey explicitly breaks it and remains unsafe.
    if re.search(
        rf"^(?:do\s+not|don't|never)\s+(?:{_OVERRIDE_ACTION}|{_EXFIL_ACTION})"
        rf"\b{sensitive}",
        directive,
    ):
        return False
    if re.search(rf"^(?:do\s+not|don't|never)\s+{_FOLLOW_ACTION}\b{sensitive}", directive):
        return True
    if re.search(rf"^(?:{_OVERRIDE_ACTION}|{_EXFIL_ACTION})\b{sensitive}", directive):
        return True
    if re.search(
        r"^(?:you|we|i|users?|readers?|assistants?)\s+"
        r"(?:should|must|need\s+to|can|could|may|will)\s+"
        rf"(?:{_OVERRIDE_ACTION}|{_EXFIL_ACTION})\b{sensitive}",
        directive,
    ):
        return True

    if re.search(
        rf"^(?:不要|别|不得)(?:{_CHINESE_OVERRIDE_ACTION}|{_CHINESE_EXFIL_ACTION})"
        rf"{chinese_sensitive}",
        directive,
    ):
        return False
    if re.search(
        rf"^(?:不要|别|不得){_CHINESE_FOLLOW_ACTION}{chinese_sensitive}",
        directive,
    ):
        return True
    if re.search(
        rf"^(?:{_CHINESE_OVERRIDE_ACTION}|{_CHINESE_EXFIL_ACTION}){chinese_sensitive}",
        directive,
    ):
        return True
    return bool(
        re.search(
            r"^(?:你|您|我们|我|用户|读者|助手)(?:应该|必须|需要|可以)"
            rf"(?:{_CHINESE_OVERRIDE_ACTION}|{_CHINESE_EXFIL_ACTION})"
            rf"{chinese_sensitive}",
            directive,
        )
    )


def _strip_directive_wrappers(text: str) -> str:
    directive = text
    while True:
        for wrapper in _DIRECTIVE_WRAPPERS:
            stripped = wrapper.sub("", directive, count=1)
            if stripped != directive:
                directive = stripped
                break
        else:
            return directive


def _is_negated_reporting_statement(text: str) -> bool:
    reporting_subject = (
        r"(?:the\s+)?(?:company|management|system|issuer|board|customers?|"
        r"institutions?|analysts?|brokers?)"
    )
    reporting_action = rf"(?:{_OVERRIDE_ACTION}|{_FOLLOW_ACTION}|{_EXFIL_ACTION})"
    return bool(
        re.fullmatch(
            rf"{reporting_subject}\s+did\s+not\s+{reporting_action}\b"
            rf"[^.!?]{{0,64}}\b{_SENSITIVE_INSTRUCTION}\b[^.!?]{{0,32}}[.!?]?",
            text,
        )
    )


def _contains_advisory_directive(text: str, ticker: str) -> bool:
    normalized_ticker = re.escape(unicodedata.normalize("NFKC", ticker).casefold())
    if _has_recommendation_relation(text, normalized_ticker):
        return True

    # Advice and bare imperatives are unsafe only when the action targets a
    # security or portfolio object. This preserves filing noun phrases such as
    # "Purchase of systems" and "Increase in customer purchases."
    trade_action = re.compile(rf"\b{_RECOMMENDED_TRADE_ACTION}\b")
    for match in trade_action.finditer(text):
        prefix = text[: match.start()].rstrip()
        if _is_security_object(text[match.end() :], normalized_ticker) and (
            _has_trade_advice_relation(prefix)
        ):
            return True

    chinese_action = re.compile(
        rf"(?:购入|买入|增持|卖出|减持|清仓|持有|投资)\s*"
        rf"(?:{re.escape(unicodedata.normalize('NFKC', ticker).casefold())}|"
        r"股票|股份|仓位|持仓|敞口|配置|头寸|证券)"
    )
    for match in chinese_action.finditer(text):
        prefix = text[: match.start()].rstrip()
        if _has_chinese_trade_advice_relation(prefix):
            return True
    return False


def _has_recommendation_relation(text: str, ticker: str) -> bool:
    """Recognize only recommendation and rating relations bound to the security."""
    target = rf"{ticker}(?:\s+{_SECURITY_NOUN})?"
    trade = r"(?:buy|sell|hold)"
    trade_action = r"(?:buy|buying|purchase|purchasing|sell|selling|hold|holding)"
    advice = _ADVICE_VERB
    local_tail = (
        r"(?=$|[.!?,;:]|\s+(?:after|before|during|because|following|ahead|on|at|in|with)\b)"
    )
    security_context_tail = (
        rf"(?:{local_tail}|(?=\s+(?:by\s+{_READER}\b|for\s+"
        rf"(?:(?:a|an|the|your|our|their)\s+)?{_SECURITY_NOUN}\b|"
        r"as\s+(?:an?\s+)?investment\b)))"
    )
    request_context_tail = (
        rf"(?:\s+(?:right\s+now|based\s+on\s+[^.!?]{{1,80}}|for\s+"
        rf"(?:(?:my|your|our|the|a)\s+)?{_SECURITY_NOUN}))?[.!?]?"
    )

    request_actor = rf"(?:{_READER}|analysts?|advisers?|brokers?)"
    request_relations = (
        rf"(?:would|do|does|could|can|will)\s+{request_actor}\s+{advice}\s+"
        rf"(?:the\s+)?{target}{request_context_tail}",
        rf"(?:would|do|does|could|can|will)\s+{request_actor}\s+{advice}\s+"
        rf"(?:investing|(?:an?\s+)?investment)\s+in\s+{target}[.!?]?",
        rf"(?:do|would|could|can|will)\s+(?:analysts?|advisers?|brokers?)\s+"
        rf"{advice}\s+{trade_action}(?:\s+now)?[.!?]?",
        rf"(?:would|could|should|can|do|will)\s+(?:you|i|we)\s+invest\s+in\s+"
        rf"{target}[.!?]?",
        rf"is\s+{target}\s+(?:a\s+)?good\s+investment{request_context_tail}",
        rf"{target}\s+is\s+(?:a\s+)?recommended\s+purchase[.!?]?",
    )
    if any(re.fullmatch(pattern, text) for pattern in request_relations):
        return True

    # Adviser clauses must directly name the security or its purchase/rating.
    direct_advisor_relations = (
        rf"{_ADVISOR}\s+(?:strongly\s+)?{advice}\s+{target}{local_tail}",
        rf"{_ADVISOR}\s+(?:strongly\s+)?{advice}\s+(?:the\s+)?"
        rf"(?:purchase|sale)\s+of\s+{target}{local_tail}",
    )
    if any(re.match(pattern, text) for pattern in direct_advisor_relations):
        return True
    rating_relations = (
        rf"{_ADVISOR}\s+(?:rate(?:s|d)?|call(?:s|ed)?|assign(?:s|ed)?|give(?:s)?|gave)"
        rf"\s+{target}\s+(?:a\s+)?(?:strong\s+)?{trade}(?:\s+rating)?\b",
        rf"{_ADVISOR}\s+(?:issue(?:s|d)?|reiterate(?:s|d)?|maintain(?:s|ed)?)\s+"
        rf"(?:a|the|their|its)\s+(?:strong\s+)?{trade}\s+rating\s+on\s+"
        rf"{target}\b",
        rf"(?:the\s+)?consensus\s+rating\s+on\s+{target}\s+"
        rf"(?:is|remains)\s+{trade}\b",
    )
    if any(re.match(pattern, text) for pattern in rating_relations):
        return True

    # Ticker-led clauses must express the rating/evaluation immediately.
    rating_copula = (
        r"(?:(?:is|was)(?:\s+(?:currently\s+)?rated)?|"
        r"has\s+been\s+(?:currently\s+)?rated|remains)"
    )
    direct_rating = rf"{target}\s+{rating_copula}\s+(?:a\s+)?(?:strong\s+)?{trade}\b"
    if re.match(direct_rating, text):
        return True
    ticker_relations = (
        rf"{target}\s+(?:(?:is|looks|seems|appears)(?:\s+clearly)?|"
        rf"(?:may|might|could)\s+be)\s+worth\s+{trade_action}{security_context_tail}",
        rf"{target}\s+(?:(?:is|looks|seems|appears)(?:\s+"
        rf"(?:clearly|widely|strongly|highly))?|comes(?:\s+highly)?)\s+"
        rf"(?:recommended|advised)\s+(?:for|to)\s+"
        rf"(?:purchase|buying|buy|acquisition|selling|sell|sale|holding|hold)"
        rf"{security_context_tail}",
    )
    if any(re.match(pattern, text) for pattern in ticker_relations):
        return True

    chinese_trade = r"(?:购入|买入|增持|卖出|减持|清仓|持有|投资)"
    chinese_relations = (
        rf"{ticker}\s*(?:(?:看起来|似乎|显然)\s*)?值得(?:现在|立即)?{chinese_trade}\b",
        rf"{ticker}\s*被(?:建议|推荐|评为){chinese_trade}\b",
        rf"{ticker}\s*获得{chinese_trade}评级\b",
        rf"(?:推荐|建议)(?:购买|买入|持有|卖出)?\s*{ticker}\s*"
        r"(?:股票|股份|证券)?(?=$|[。.!?，,；;：:])",
        rf"(?:分析师|顾问|经纪人)(?:给予|给出|评为)\s*{ticker}\s*"
        rf"{chinese_trade}评级\b",
        rf"(?:建议|推荐)(?:投资者|客户)(?:考虑)?{chinese_trade}\s*{ticker}\b",
        rf"(?:你|您|分析师|顾问)\s*(?:会|是否|能否|可以)?\s*"
        rf"(?:推荐|建议)(?:买(?:入)?|购买|卖出|持有)?\s*{target}\s*"
        r"(?:吗|呢)?[。.!?]?",
        rf"(?:你|您)\s*(?:会|是否|能否|可以)?\s*投资\s*{target}\s*"
        r"(?:吗|呢)?[。.!?]?",
        rf"{target}\s*是?\s*值得投资的?(?:股票|证券)?(?:吗)?[。.!?]?",
    )
    return any(re.match(pattern, text) for pattern in chinese_relations)


def _ticker_names_product(tail: str) -> bool:
    """Return whether a ticker immediately introduces a known product phrase."""
    if tail.startswith("'s "):
        phrase = tail[3:]
    elif tail.startswith(" "):
        phrase = tail.lstrip()
    else:
        return False
    model = r"(?:h100|dgx|grace|geforce(?:\s+rtx)?(?:\s+\d+)?)"
    descriptor = rf"(?:{model}|data\s+center|networking|graphics|accelerator)"
    return bool(re.match(rf"(?:{model}\b|(?:(?:{descriptor})\s+)?{_PRODUCT_NOUN}\b)", phrase))


def _is_security_object(tail: str, ticker: str) -> bool:
    candidate = tail.lstrip()
    candidate = re.sub(r"^(?:of\s+)?(?:(?:more|additional|all|some)\s+)?", "", candidate)
    security_noun = (
        rf"(?:(?:your|the|a|an|this|that|our|their)\s+)?(?:{ticker}\s+)?"
        rf"{_SECURITY_NOUN}\b"
    )
    if re.match(security_noun, candidate):
        return True
    ticker_match = re.match(rf"{ticker}\b", candidate)
    return ticker_match is not None and not _ticker_names_product(candidate[ticker_match.end() :])


def _has_trade_advice_relation(prefix: str) -> bool:
    if not prefix or re.fullmatch(r"(?:please|now|immediately)", prefix):
        return True
    if re.search(rf"\b{_READER}\s+{_ADVICE_MODAL}(?:\s+immediately)?$", prefix):
        return True
    if re.search(rf"\b{_READER}\s+{_ADVICE_MODAL}\s+(?:strongly\s+)?consider$", prefix):
        return True
    if re.search(
        rf"\b{_READER}\s+(?:is|are)\s+(?:strongly\s+)?"
        rf"(?:recommended|advised|urged)\s+to$",
        prefix,
    ):
        return True
    if re.search(
        rf"\b{_ADVISOR}\s+(?:strongly\s+)?{_ADVICE_VERB}\b"
        rf"(?:\s+that)?(?:\s+{_READER})?(?:\s+(?:to|consider))?$",
        prefix,
    ):
        return True
    return bool(
        re.search(r"\b(?:my|our)\s+(?:advice|recommendation)\s+is\b.{0,24}$", prefix)
        or re.fullmatch(
            r"(?:maybe\s+)?(?:please\s+)?consider(?:\s+(?:now|immediately))?",
            prefix,
        )
    )


def _has_chinese_trade_advice_relation(prefix: str) -> bool:
    if re.fullmatch(r"(?:请|立即|现在|务必)?", prefix):
        return True
    if re.fullmatch(
        r"(?:(?:你|您|投资者|股东|交易者)(?:可以|不妨|或许可以|可能想要)?\s*)?"
        r"(?:请)?考虑(?:现在|立即)?",
        prefix,
    ):
        return True
    if re.fullmatch(
        r"(?:你|您|投资者|股东|交易者)"
        r"(?:可以|可|不妨|应该|应当|必须|需要|最好|务必)(?:现在|立即)?",
        prefix,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:(?:我|我们|分析师|顾问|经纪人)(?:强烈)?)?"
            r"(?:建议|推荐|劝告|主张)(?:你|您|投资者|客户)?(?:现在|立即)?",
            prefix,
        )
    )


def _is_declared_verified(claim: object) -> bool:
    kind = getattr(claim, "kind", None)
    return kind is ClaimKind.VERIFIED_FACT or kind == ClaimKind.VERIFIED_FACT.value


def _cap_sufficiency(value: str, cap: str) -> str:
    order = {"C": 0, "B": 1, "A": 2}
    return value if order[value] <= order[cap] else cap


def _cap_confidence(value: Confidence, cap: Confidence) -> Confidence:
    order = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    return value if order[value] <= order[cap] else cap


def _append_claim(destination: list[Claim], claim: Claim) -> None:
    key = (
        claim.kind,
        claim.text,
        claim.confidence,
        tuple(claim.evidence_chunk_ids),
        tuple(claim.web_evidence_ids),
    )
    if all(
        (
            item.kind,
            item.text,
            item.confidence,
            tuple(item.evidence_chunk_ids),
            tuple(item.web_evidence_ids),
        )
        != key
        for item in destination
    ):
        destination.append(claim)


def _cited_ids(claims: Iterable[Claim]) -> list[str]:
    return list(
        dict.fromkeys(chunk_id for claim in claims for chunk_id in claim.evidence_chunk_ids)
    )


def _cited_web_ids(claims: Iterable[Claim]) -> list[str]:
    return list(
        dict.fromkeys(source_id for claim in claims for source_id in claim.web_evidence_ids)
    )
