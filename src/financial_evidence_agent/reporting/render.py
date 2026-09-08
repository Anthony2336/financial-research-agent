"""Deterministic Markdown rendering for guarded research memos."""

from __future__ import annotations

import re

from financial_evidence_agent.domain import Claim, EvidenceChunk, WebEvidence
from financial_evidence_agent.reporting.guard import (
    GuardedMemo,
    _canonical_evidence_chunk,
    _is_unsafe_output_text,
)
from financial_evidence_agent.reporting.privacy import (
    PRIVATE_DETAIL_CODE,
    retain_public_text,
    safe_reference_label,
)
from financial_evidence_agent.skills.schemas import (
    GuardedFinancialDataPoint,
    GuardedSkillMemo,
    canonical_financial_text,
)
from financial_evidence_agent.web_evidence.source_policy import (
    PolicyValidatedWebEvidence,
    canonicalize_source_url,
)

DISCLAIMER = "Research assistance only; not investment advice."
_MARKDOWN_CONTROL = re.compile(r"([\\`*_[\]<>#|&])")


def render_markdown(result: GuardedMemo) -> str:
    """Render a fixed report structure without inventing source locations."""
    canonical_sources = [
        canonical
        for source in result.sources
        if (canonical := _canonical_evidence_chunk(source)) is not None
        and retain_public_text(canonical.id, ticker=result.ticker)
        and retain_public_text(canonical.section, ticker=result.ticker)
    ]
    canonical_web_sources = [
        source
        for source in result.web_sources
        if _has_current_canonical_policy(source, result.source_policy_version)
        and retain_public_text(source.id, ticker=result.ticker)
        and retain_public_text(source.title, ticker=result.ticker)
        and retain_public_text(str(source.source_url), ticker=result.ticker)
        and retain_public_text(str(source.canonical_url), ticker=result.ticker)
    ]
    cutoff = max((source.filed_at for source in canonical_sources), default=None)
    lines = [
        f"# Financial evidence report — {_safe_text(result.ticker)}",
        "",
        f"- Ticker: {_safe_text(result.ticker)}",
        f"- Corpus version: {_safe_text(result.corpus_version)}",
        f"- Filing cutoff: {cutoff.isoformat() if cutoff else 'No retained cited filing'}",
        "",
        "## Research question",
        "",
        _safe_text(result.research_question),
        "",
        "## Verified evidence supporting thesis",
        "",
        *_claim_lines(
            result.supporting_claims,
            ticker=result.ticker,
            empty="No verified supporting evidence retained.",
        ),
        "",
        "## Counter-evidence and risks",
        "",
        *_claim_lines(
            result.counter_claims,
            ticker=result.ticker,
            empty="No verified counter-evidence retained.",
        ),
        "",
        "## Inferences",
        "",
        *_claim_lines(
            result.inferences,
            ticker=result.ticker,
            label="Inference",
            empty="No inference retained.",
        ),
        "",
        "## Open questions / insufficient evidence",
        "",
        *_claim_lines(
            result.open_questions,
            ticker=result.ticker,
            label="Open question",
            empty="No open question recorded.",
        ),
        "",
        "## Information sufficiency and confidence",
        "",
        f"- Information sufficiency: {_safe_text(result.information_sufficiency)}",
        f"- Confidence: {_safe_text(getattr(result.confidence, 'value', result.confidence))}",
        "",
        "## Sources",
        "",
        *_source_lines(
            canonical_sources,
            had_sources=bool(result.sources),
            ticker=result.ticker,
        ),
        "",
        "## Web sources",
        "",
        *_p1_web_source_lines(canonical_web_sources, ticker=result.ticker),
    ]
    if result.errors:
        lines.extend(
            [
                "",
                "## Guard notes",
                "",
                *[
                    f"- {_safe_detail(error, ticker=result.ticker)}"
                    for error in result.errors
                ],
            ]
        )
    lines.extend(["", f"> {DISCLAIMER}"])
    return "\n".join(lines) + "\n"


def render_skill_markdown(result: GuardedSkillMemo, *, ticker: str | None = None) -> str:
    """Render a guarded P1 memo with distinct SEC and web provenance."""
    if ticker is not None and ticker.strip().upper() != result.ticker:
        raise ValueError(
            f"render ticker '{ticker.strip().upper()}' does not match guarded ticker "
            f"'{result.ticker}'"
        )
    result = _renderable_skill_result(result)
    result = result.model_copy(
        update={
            "guard_errors": [
                _safe_detail(error, ticker=result.ticker)
                for error in result.guard_errors
            ]
        }
    )
    _reject_unsafe_skill_render_text(result)
    memo = result.memo
    lines = [
        f"# Structured financial research — {_safe_text(result.ticker)}",
        "",
        f"- Confidence: {_safe_text(memo.confidence)}",
        "",
        "## Reproducibility",
        "",
        *_provenance_lines(result),
        "",
        "## Research question",
        "",
        _safe_text(memo.research_question),
    ]
    for section in memo.sections:
        lines.extend(
            [
                "",
                f"## {_safe_text(section.facet.value.replace('_', ' ').capitalize())}",
                "",
                *_p1_claim_lines(section.claims),
            ]
        )
    lines.extend(
        [
            "",
            "## Financial data points",
            "",
            *_financial_data_lines(memo.data_points),
            "",
            "## Information gaps",
            "",
            *(
                [f"- {_safe_text(gap)}" for gap in memo.information_gaps]
                or ["No information gaps recorded."]
            ),
            "",
            "## SEC filing sources",
            "",
            *_p1_filing_source_lines(result.filing_sources, ticker=result.ticker),
            "",
            "## Web sources",
            "",
            *_p1_web_source_lines(result.web_sources, ticker=result.ticker),
        ]
    )
    if result.guard_errors:
        lines.extend(
            [
                "",
                "## Guard notes",
                "",
                *[
                    f"- {_safe_detail(error, ticker=result.ticker)}"
                    for error in result.guard_errors
                ],
            ]
        )
    lines.extend(["", f"> {DISCLAIMER}"])
    return "\n".join(lines) + "\n"


def _renderable_skill_result(result: GuardedSkillMemo) -> GuardedSkillMemo:
    policy_version = _single_policy_version(result)
    valid_sources = [
        source
        for source in result.web_sources
        if _has_current_canonical_policy(source, policy_version)
        and retain_public_text(source.id, ticker=result.ticker)
        and retain_public_text(source.title, ticker=result.ticker)
        and retain_public_text(str(source.source_url), ticker=result.ticker)
        and retain_public_text(str(source.canonical_url), ticker=result.ticker)
    ]
    valid_web_ids = {source.id for source in valid_sources}
    valid_filing_sources = [
        source
        for source in result.filing_sources
        if retain_public_text(source.id, ticker=result.ticker)
        and retain_public_text(source.section, ticker=result.ticker)
    ]
    retained_filing_ids = {source.id for source in valid_filing_sources}
    invalid_ids = {source.id for source in result.web_sources if source not in valid_sources}
    invalid_ids.update(
        source.id for source in result.filing_sources if source not in valid_filing_sources
    )
    referenced_web_ids = {
        source_id
        for section in result.memo.sections
        for claim in section.claims
        for source_id in claim.web_evidence_ids
        if source_id not in valid_web_ids
    }
    referenced_filing_ids = {
        source_id
        for section in result.memo.sections
        for claim in section.claims
        for source_id in claim.evidence_chunk_ids
        if source_id not in retained_filing_ids
    }
    missing_source_ids = {
        source_id
        for point in result.memo.data_points
        for source_id in point.source_ids
        if source_id not in retained_filing_ids and source_id not in valid_web_ids
    }
    invalid_ids.update(referenced_web_ids)
    invalid_ids.update(referenced_filing_ids)
    invalid_ids.update(missing_source_ids)
    if not invalid_ids:
        return result

    sections = []
    for section in result.memo.sections:
        claims = []
        for claim in section.claims:
            filing_ids = [
                source_id
                for source_id in claim.evidence_chunk_ids
                if source_id not in invalid_ids
            ]
            web_ids = [
                source_id
                for source_id in claim.web_evidence_ids
                if source_id not in invalid_ids
            ]
            if claim.kind.value == "verified_fact" and not (
                filing_ids or web_ids
            ):
                continue
            claims.append(
                claim.model_copy(
                    update={
                        "evidence_chunk_ids": filing_ids,
                        "web_evidence_ids": web_ids,
                    }
                )
            )
        sections.append(section.model_copy(update={"claims": claims}))
    data_points = [
        point
        for point in result.memo.data_points
        if not invalid_ids.intersection(point.source_ids)
    ]
    sufficiency = result.memo.information_sufficiency
    if sufficiency.value == "sufficient":
        sufficiency = type(sufficiency).PARTIAL
    memo = result.memo.model_copy(
        update={
            "sections": sections,
            "data_points": data_points,
            "information_sufficiency": sufficiency,
        }
    )
    valid_source_refs = tuple(
        source_ref
        for source_ref in result.provenance.source_refs
        if source_ref.source_id not in invalid_ids
    )
    provenance = result.provenance.model_copy(
        update={
            "information_sufficiency": sufficiency,
            "source_refs": valid_source_refs,
        }
    )
    return result.model_copy(
        update={
            "memo": memo,
            "filing_sources": valid_filing_sources,
            "web_sources": valid_sources,
            "provenance": provenance,
            "guard_errors": [
                *result.guard_errors,
                "WEB_SOURCE_OMITTED_AT_RENDER: policy provenance mismatch",
            ],
        }
    )


def _has_current_canonical_policy(
    source: WebEvidence,
    policy_version: str | None,
) -> bool:
    if not isinstance(source, PolicyValidatedWebEvidence) or policy_version is None:
        return False
    try:
        canonical = canonicalize_source_url(source.source_url)
    except ValueError:
        return False
    return (
        source.policy_version == policy_version
        and source.canonical_url == source.source_url
        and canonical == source.canonical_url
    )


def _single_policy_version(result: GuardedSkillMemo) -> str | None:
    versions = result.provenance.source_policy_versions
    return versions[0] if len(versions) == 1 else None


def _reject_unsafe_skill_render_text(result: GuardedSkillMemo) -> None:
    memo = result.memo
    rendered_text = [
        memo.recipe_version,
        memo.research_question,
        *memo.information_gaps,
        *(claim.text for section in memo.sections for claim in section.claims),
        *(point.name for point in memo.data_points),
        *(point.currency for point in memo.data_points if point.currency is not None),
        *(point.unit for point in memo.data_points),
        *(point.definition for point in memo.data_points),
        *(
            point.discrepancy_note
            for point in memo.data_points
            if point.discrepancy_note is not None
        ),
        *(source.section for source in result.filing_sources),
        *(source.title for source in result.web_sources),
        *(str(source.source_url) for source in result.web_sources),
        *(str(source.canonical_url) for source in result.web_sources),
        *result.provenance.source_policy_versions,
        *result.provenance.corpus_versions,
        *result.provenance.prompt_versions,
        *result.guard_errors,
    ]
    if any(_is_unsafe_output_text(value, result.ticker) for value in rendered_text):
        raise ValueError("guarded skill memo contains unsafe rendered text")


def _claim_lines(
    claims: list[Claim],
    *,
    ticker: str,
    label: str | None = None,
    empty: str,
) -> list[str]:
    if not claims:
        return [empty]
    lines: list[str] = []
    for claim in claims:
        prefix = f"**{label}:** " if label else ""
        citation_values: list[str] = []
        if claim.evidence_chunk_ids:
            rendered_ids = ", ".join(
                _safe_reference(value, ticker=ticker)
                for value in claim.evidence_chunk_ids
            )
            citation_values.append(f"Evidence: {rendered_ids}")
        if claim.web_evidence_ids:
            rendered_ids = ", ".join(
                _safe_reference(value, ticker=ticker) for value in claim.web_evidence_ids
            )
            citation_values.append(f"Web: {rendered_ids}")
        citations = f" ({'; '.join(citation_values)})" if citation_values else ""
        lines.append(f"- {prefix}{_safe_text(claim.text)}{citations}")
    return lines


def _p1_claim_lines(claims: list[Claim]) -> list[str]:
    if not claims:
        return ["No supported claim retained."]
    lines: list[str] = []
    for claim in claims:
        citations: list[str] = []
        if claim.evidence_chunk_ids:
            citations.append(
                "SEC: " + ", ".join(_safe_text(value) for value in claim.evidence_chunk_ids)
            )
        if claim.web_evidence_ids:
            citations.append(
                "Web: " + ", ".join(_safe_text(value) for value in claim.web_evidence_ids)
            )
        suffix = f" ({'; '.join(citations)})" if citations else ""
        lines.append(f"- {_safe_text(claim.text)}{suffix}")
    return lines


def _financial_data_lines(points: list[GuardedFinancialDataPoint]) -> list[str]:
    if not points:
        return ["No financial data points retained."]
    names_with_primary = {
        canonical_financial_text(point.name)
        for point in points
        if any(source.source_tier.value == "primary" for source in point.source_provenance)
    }
    lines: list[str] = []
    for point in points:
        value = "Missing" if point.value is None else _safe_text(point.value)
        currency = f" {_safe_text(point.currency)}" if point.currency else ""
        unit = f" {_safe_text(point.unit)}" if point.unit else ""
        period_start = point.period_start.isoformat() if point.period_start else "Not provided"
        lines.extend(
            [
                f"- **{_safe_text(point.name)}:** {value}{currency}{unit}",
                f"  - Period: {period_start} to {point.period_end.isoformat()}",
                f"  - Definition: {_safe_text(point.definition)}",
                f"  - Observation ID: {_safe_text(point.observation_id)}",
                f"  - Verification: {_safe_text(point.verification_status.value)}",
                (
                    "  - Precedence: primary"
                    if any(
                        source.source_tier.value == "primary"
                        for source in point.source_provenance
                    )
                    else (
                        "  - Precedence: secondary"
                        if canonical_financial_text(point.name) in names_with_primary
                        else "  - Precedence: secondary (no primary observation retained)"
                    )
                ),
            ]
        )
        for source in point.source_provenance:
            lines.extend(
                [
                    f"  - Exact source: {_safe_text(source.source_ref.encode())}",
                    f"    - Source kind: {_safe_text(source.source_kind.value)}",
                    f"    - Source tier: {_safe_text(source.source_tier.value)}",
                    (
                        "    - Canonical source identity: "
                        f"{_safe_text(source.canonical_source_identity)}"
                    ),
                ]
            )
        if point.discrepancy_note:
            lines.append(f"  - Discrepancy: {_safe_text(point.discrepancy_note)}")
    return lines


def _provenance_lines(result: GuardedSkillMemo) -> list[str]:
    provenance = result.provenance
    lines = [
        *(
            f"- Recipe: {_safe_text(recipe.name.value)} @ {_safe_text(recipe.version)}"
            for recipe in provenance.recipes
        ),
        (
            "- Source policy version: "
            + ", ".join(_safe_text(value) for value in provenance.source_policy_versions)
            if provenance.source_policy_versions
            else "- Source policy version: none (no active validated policy)"
        ),
        (
            "- Corpus versions: "
            + ", ".join(_safe_text(value) for value in provenance.corpus_versions)
            if provenance.corpus_versions
            else "- Corpus versions: none (no retained filing corpus)"
        ),
        (
            "- Prompt versions: "
            + ", ".join(_safe_text(value) for value in provenance.prompt_versions)
            if provenance.prompt_versions
            else "- Prompt versions: none (no prompt used)"
        ),
        (
            "- Requested as-of: "
            + ", ".join(value.isoformat() for value in provenance.requested_as_of_dates)
            if provenance.requested_as_of_dates
            else "- Requested as-of: none (not requested)"
        ),
        (
            "- Retained evidence cutoff: "
            + ", ".join(value.isoformat() for value in provenance.evidence_cutoff_dates)
            if provenance.evidence_cutoff_dates
            else "- Retained evidence cutoff: none"
        ),
        (
            "- Information sufficiency: "
            f"{_safe_text(provenance.information_sufficiency.value)}"
        ),
    ]
    lines.extend(
        f"- Exact retained source: {_safe_text(reference.encode())}"
        for reference in provenance.source_refs
    )
    if not provenance.source_refs:
        lines.append("- Exact retained source: none")
    return lines


def _source_lines(
    sources: list[EvidenceChunk],
    *,
    had_sources: bool,
    ticker: str,
) -> list[str]:
    if not sources:
        if had_sources:
            return ["Invalid source omitted."]
        return ["No cited source retained."]
    lines: list[str] = []
    for source in sources:
        lines.extend(
            [
                f"- **{_safe_reference(source.id, ticker=ticker)}**",
                f"  - URL: <{source.source_url}>",
                f"  - Form: {_safe_text(source.form)}",
                f"  - Filed: {source.filed_at.isoformat()}",
                f"  - Accession: {_safe_text(source.accession_no)}",
                f"  - Section: {_safe_text(source.section)}",
                f"  - Raw characters: {source.raw_start}–{source.raw_end}",
            ]
        )
    return lines


def _p1_filing_source_lines(sources: list[EvidenceChunk], *, ticker: str) -> list[str]:
    if not sources:
        return ["No cited SEC filing source retained."]
    lines: list[str] = []
    for source in sources:
        lines.extend(
            [
                f"- **{_safe_reference(source.id, ticker=ticker)}**",
                f"  - URL: <{source.source_url}>",
                "  - Source tier: primary",
                f"  - Form: {_safe_text(source.form)}",
                f"  - Filed: {source.filed_at.isoformat()}",
                f"  - Accession: {_safe_text(source.accession_no)}",
                f"  - Section: {_safe_text(source.section)}",
                f"  - Raw characters: {source.raw_start}–{source.raw_end}",
            ]
        )
    return lines


def _p1_web_source_lines(sources: list[WebEvidence], *, ticker: str) -> list[str]:
    if not sources:
        return ["No cited web source retained."]
    lines: list[str] = []
    for source in sources:
        published = source.published_at.isoformat() if source.published_at else "Not provided"
        lines.extend(
            [
                f"- **{_safe_reference(source.id, ticker=ticker)}** — {_safe_text(source.title)}",
                f"  - URL: <{source.source_url}>",
                f"  - Source kind: {_safe_text(source.source_kind.value)}",
                f"  - Source tier: {_safe_text(source.source_tier.value)}",
                f"  - Published: {published}",
                f"  - Fetched: {source.fetched_at.isoformat()}",
            ]
        )
    return lines


def _safe_text(value: object) -> str:
    flattened = " ".join(str(value).split())
    return _MARKDOWN_CONTROL.sub(r"\\\1", flattened)


def _safe_reference(value: str, *, ticker: str) -> str:
    label = safe_reference_label(value, ticker=ticker)
    return label if label != value else _safe_text(label)


def _safe_detail(value: str, *, ticker: str) -> str:
    if value == PRIVATE_DETAIL_CODE or not retain_public_text(value, ticker=ticker):
        return PRIVATE_DETAIL_CODE
    return _safe_text(value)
