"""Fixed Markdown composer for unified P2 reports."""

from __future__ import annotations

import re
from datetime import date, datetime

from fra.domain import canonicalize_source_url
from fra.reporting.privacy import (
    PRIVATE_DETAIL_CODE,
    retain_public_text,
    safe_reference_label,
)
from fra.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedP2Report,
    MetricComparison,
)
from fra.skills.schemas import canonical_financial_text

DISCLAIMER = "Research assistance only; not investment advice."
_MARKDOWN_CONTROL = re.compile(r"([\\`*_[\]<>#|&])")


def render_p2_markdown(report: GuardedP2Report) -> str:
    """Render one fixed report shape from the final guarded P2 model."""

    lines = [
        f"# Unified P2 research - {_safe_text(report.scope.primary_ticker)}",
        "",
        "## Scope",
        "",
        f"- Primary ticker: {_safe_text(report.scope.primary_ticker)}",
        f"- Peer scope: {_safe_text(report.scope.description)}",
        (
            "- Peer tickers: "
            + ", ".join(_safe_text(ticker) for ticker in report.scope.peer_tickers)
            if report.scope.peer_tickers
            else "- Peer tickers: none"
        ),
        "",
        "## Reproducibility",
        "",
        *_provenance_lines(report),
        "",
        "## Per-company evidence",
        "",
        *_package_lines(report),
        "",
        "## Peer comparison",
        "",
        *_comparison_lines(report),
        "",
        "## Industry evidence",
        "",
        *_industry_evidence_lines(report),
        "",
        "## Comparability limits",
        "",
        *_comparability_limit_lines(report),
        "",
        "## Information sufficiency",
        "",
        f"- Overall: {_safe_text(report.information_sufficiency.value)}",
        *_quality_lines(report),
        "",
        "## Sources",
        "",
        *_source_lines(report),
        "",
        "## Guard notes",
        "",
        *_guard_note_lines(report),
        "",
        f"> {DISCLAIMER}",
    ]
    return "\n".join(lines) + "\n"


def _package_lines(report: GuardedP2Report) -> list[str]:
    if not report.packages:
        return ["No retained company evidence."]
    lines: list[str] = []
    for package in report.packages:
        lines.extend(
            [
                f"### {_safe_text(package.ticker)}",
                f"- Coverage: {_safe_text(package.coverage)}",
                (
                    "- Recipes: "
                    + ", ".join(
                        f"{_safe_text(recipe.name.value)} @ {_safe_text(recipe.version)}"
                        for recipe in package.provenance.recipes
                    )
                    if package.provenance.recipes
                    else "- Recipes: none"
                ),
                (
                    "- Corpus versions: "
                    + ", ".join(
                        _safe_text(version)
                        for version in package.provenance.corpus_versions
                    )
                    if package.provenance.corpus_versions
                    else "- Corpus versions: none"
                ),
            ]
        )
        if package.claims:
            lines.extend(_claim_lines(package))
        else:
            lines.append("- No retained claim.")
        lines.extend(_package_metric_lines(package))
        if package.information_gaps:
            lines.extend(
                f"- Information gap: {_safe_text(gap)}" for gap in package.information_gaps
            )
    return lines


def _comparison_lines(report: GuardedP2Report) -> list[str]:
    lines: list[str] = []
    missing = sorted(
        set(report.scope.peer_tickers) - {package.ticker for package in report.packages}
    )
    for ticker in missing:
        lines.append(f"- {_safe_text(ticker)}: insufficient evidence")
    if not report.comparisons:
        lines.append("No retained peer comparisons.")
        return lines
    for comparison in report.comparisons:
        lines.extend(_one_comparison_lines(comparison))
    return lines


def _one_comparison_lines(comparison: MetricComparison) -> list[str]:
    lines = [
        f"### {_safe_text(comparison.name)}",
        f"- Status: {_safe_text(comparison.status.value)}",
    ]
    if comparison.status is ComparabilityStatus.COMPARABLE and comparison.delta is not None:
        lines.append(f"- Delta: {_safe_text(comparison.delta)}")
    if comparison.limitation:
        lines.append(f"- Limitation: {_safe_text(comparison.limitation)}")
    names_with_primary = {
        canonical_financial_text(observation.name)
        for observation in comparison.observations
        if any(source.source_tier.value == "primary" for source in observation.source_provenance)
    }
    for observation in comparison.observations:
        lines.extend(_observation_lines(observation, names_with_primary=names_with_primary))
    return lines


def _observation_lines(
    observation: ComparableMetric,
    *,
    names_with_primary: set[str],
) -> list[str]:
    value = "missing" if observation.value is None else _safe_text(observation.value)
    currency = f" {_safe_text(observation.currency)}" if observation.currency else ""
    period_start = (
        observation.period_start.isoformat()
        if observation.period_start is not None
        else "not provided"
    )
    sources = (
        ", ".join(
            _safe_reference_text(ref.encode(), ticker=observation.ticker)
            for ref in observation.source_refs
        )
        or "none"
    )
    lines = [
        f"- {_safe_text(observation.ticker)}: {value}{currency} {_safe_text(observation.unit)}",
        f"  - Period: {period_start} to {observation.period_end.isoformat()}",
        f"  - Definition: {_safe_text(observation.definition)}",
        f"  - Sources: {sources}",
    ]
    if observation.source_provenance:
        lines.append(
            "  - Precedence: primary"
            if any(
                source.source_tier.value == "primary"
                for source in observation.source_provenance
            )
            else (
                "  - Precedence: secondary"
                if canonical_financial_text(observation.name) in names_with_primary
                else "  - Precedence: secondary (no primary observation retained)"
            )
        )
    if observation.limitation:
        lines.append(f"  - Limitation: {_safe_text(observation.limitation)}")
    return lines


def _industry_evidence_lines(report: GuardedP2Report) -> list[str]:
    claims = [
        (package.ticker, claim.text, [reference.encode() for reference in claim.source_refs])
        for package in report.packages
        for claim in package.claims
    ]
    if not claims:
        return ["No retained company evidence."]
    return [
        f"- {_safe_text(ticker)}: {_safe_text(text)}"
        + (
            " (Sources: "
            + ", ".join(_safe_reference_text(reference, ticker=ticker) for reference in refs)
            + ")"
            if refs
            else ""
        )
        for ticker, text, refs in claims
    ]


def _comparability_limit_lines(report: GuardedP2Report) -> list[str]:
    lines = [
        f"- {_safe_text(comparison.name)}: {_safe_text(comparison.limitation)}"
        for comparison in report.comparisons
        if comparison.status is not ComparabilityStatus.COMPARABLE and comparison.limitation
    ]
    missing = sorted(
        set(report.scope.peer_tickers) - {package.ticker for package in report.packages}
    )
    lines.extend(f"- {_safe_text(ticker)}: insufficient evidence" for ticker in missing)
    return lines or ["No comparability limits recorded."]


def _quality_lines(report: GuardedP2Report) -> list[str]:
    if report.quality is None:
        return []
    lines = [
        "",
        "### Research quality",
        "",
        f"- Decision: {_safe_text(report.quality.decision.value)}",
    ]
    lines.extend(f"- Reason: {_safe_text(reason)}" for reason in report.quality.reasons)
    if report.quality.source_refs:
        lines.append(
            "- Source refs: "
            + ", ".join(
                _safe_reference_text(
                    reference.encode(),
                    ticker=report.scope.primary_ticker,
                )
                for reference in report.quality.source_refs
            )
        )
    return lines


def _source_lines(report: GuardedP2Report) -> list[str]:
    if not report.retained_sources:
        return ["No retained source links."]
    lines: list[str] = []
    for source in report.retained_sources:
        if not retain_public_text(
            source.ref.source_id,
            ticker=report.scope.primary_ticker,
        ) or any(
            not retain_public_text(value, ticker=report.scope.primary_ticker)
            for value in (
                source.title,
                str(source.source_url),
                source.canonical_source_identity,
            )
        ):
            lines.append(f"- {PRIVATE_DETAIL_CODE}")
            continue
        published = _render_date(source.published_or_filed_at)
        reference_label = safe_reference_label(
            source.ref.encode(),
            ticker=report.scope.primary_ticker,
        )
        lines.extend(
            [
                "- **"
                f"{_safe_text(reference_label)}"
                f"** - {_safe_text(source.title)}",
                f"  - URL: <{canonicalize_source_url(source.source_url)}>",
                f"  - Date: {published}",
                f"  - Source kind: {_safe_text(source.source_kind.value)}",
                f"  - Source tier: {_safe_text(source.source_tier.value)}",
                (
                    "  - Canonical source identity: "
                    f"{_safe_text(source.canonical_source_identity)}"
                ),
            ]
        )
    return lines


def _guard_note_lines(report: GuardedP2Report) -> list[str]:
    if not report.guard_errors:
        return ["No guard notes."]
    return [
        (
            f"- {_safe_text(error)}"
            if error != PRIVATE_DETAIL_CODE
            and retain_public_text(error, ticker=report.scope.primary_ticker)
            else f"- {PRIVATE_DETAIL_CODE}"
        )
        for error in report.guard_errors
    ]


def _claim_lines(package) -> list[str]:
    return [
        f"- {_safe_text(claim.text)}"
        + (
            " (Sources: "
            + ", ".join(
                _safe_reference_text(reference.encode(), ticker=package.ticker)
                for reference in claim.source_refs
            )
            + ")"
            if claim.source_refs
            else ""
        )
        for claim in package.claims
    ]


def _package_metric_lines(package) -> list[str]:
    if not package.financial_metrics:
        return []
    lines = ["#### Financial metrics"]
    names_with_primary = {
        canonical_financial_text(metric.name)
        for metric in package.financial_metrics
        if any(source.source_tier.value == "primary" for source in metric.source_provenance)
    }
    for metric in package.financial_metrics:
        lines.extend(
            _package_metric_detail_lines(metric, names_with_primary=names_with_primary)
        )
    return lines


def _package_metric_detail_lines(
    metric: ComparableMetric,
    *,
    names_with_primary: set[str],
) -> list[str]:
    value = "missing" if metric.value is None else _safe_text(metric.value)
    currency = f" {_safe_text(metric.currency)}" if metric.currency else ""
    period_start = (
        metric.period_start.isoformat() if metric.period_start is not None else "not provided"
    )
    sources = (
        ", ".join(
            _safe_reference_text(ref.encode(), ticker=metric.ticker)
            for ref in metric.source_refs
        )
        or "none"
    )
    lines = [
        f"- **{_safe_text(metric.name)}:** {value}{currency} {_safe_text(metric.unit)}",
        f"  - Ticker: {_safe_text(metric.ticker)}",
        f"  - Period: {period_start} to {metric.period_end.isoformat()}",
        f"  - Definition: {_safe_text(metric.definition)}",
        f"  - Observation ID: {_safe_text(metric.observation_id)}",
        f"  - Verification: {_safe_text(metric.verification_status.value)}",
        f"  - Status: {_safe_text(metric.status.value)}",
        f"  - Sources: {sources}",
    ]
    if metric.source_provenance:
        lines.append(
            "  - Precedence: primary"
            if any(
                source.source_tier.value == "primary"
                for source in metric.source_provenance
            )
            else (
                "  - Precedence: secondary"
                if canonical_financial_text(metric.name) in names_with_primary
                else "  - Precedence: secondary (no primary observation retained)"
            )
        )
    for source in metric.source_provenance:
        lines.extend(
            [
                "  - Exact source: "
                f"{_safe_reference_text(source.source_ref.encode(), ticker=metric.ticker)}",
                f"    - Source kind: {_safe_text(source.source_kind.value)}",
                f"    - Source tier: {_safe_text(source.source_tier.value)}",
                (
                    "    - Canonical source identity: "
                    f"{_safe_text(source.canonical_source_identity)}"
                ),
            ]
        )
    if metric.limitation:
        lines.append(f"  - Limitation: {_safe_text(metric.limitation)}")
    return lines


def _provenance_lines(report: GuardedP2Report) -> list[str]:
    provenance = report.provenance
    lines = [
        *(
            f"- Recipe: {_safe_text(recipe.name.value)} @ {_safe_text(recipe.version)}"
            for recipe in provenance.recipes
        ),
        *(
            f"- Source policy version: {_safe_text(version)}"
            for version in provenance.source_policy_versions
        ),
        *(
            f"- Corpus version: {_safe_text(version)}"
            for version in provenance.corpus_versions
        ),
        *(
            f"- Prompt version: {_safe_text(version)}"
            for version in provenance.prompt_versions
        ),
        *(
            f"- Requested as-of: {requested.isoformat()}"
            for requested in provenance.requested_as_of_dates
        ),
        *(
            f"- Retained evidence cutoff: {cutoff.isoformat()}"
            for cutoff in provenance.evidence_cutoff_dates
        ),
        (
            "- Information sufficiency: "
            f"{_safe_text(provenance.information_sufficiency.value)}"
        ),
        *(
            "- Exact retained source: "
            f"{_safe_reference_text(reference.encode(), ticker=report.scope.primary_ticker)}"
            for reference in provenance.source_refs
        ),
    ]
    if not provenance.source_policy_versions:
        lines.append("- Source policy version: none (no active validated policy)")
    if not provenance.corpus_versions:
        lines.append("- Corpus version: none (no retained filing corpus)")
    if not provenance.prompt_versions:
        lines.append("- Prompt version: none (no prompt used)")
    if not provenance.requested_as_of_dates:
        lines.append("- Requested as-of: none (not requested)")
    if not provenance.evidence_cutoff_dates:
        lines.append("- Retained evidence cutoff: none")
    if not provenance.source_refs:
        lines.append("- Exact retained source: none")
    return lines


def _render_date(value: datetime | date) -> str:
    return value.isoformat()


def _safe_text(value: object) -> str:
    return _MARKDOWN_CONTROL.sub(r"\\\1", " ".join(str(value).split()))


def _safe_reference_text(value: str, *, ticker: str) -> str:
    label = safe_reference_label(value, ticker=ticker)
    return label if label != value else _safe_text(label)
