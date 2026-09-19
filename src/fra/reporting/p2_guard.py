"""Final guard for unified P2 industry, peer, and quality reports."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import date

from pydantic import ValidationError

from fra.domain import (
    Confidence,
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    WebEvidence,
    canonicalize_source_url,
)
from fra.reporting.guard import (
    _canonical_evidence_chunk,
    _is_unsafe_output_text,
)
from fra.reporting.privacy import (
    PRIVATE_DETAIL_CODE,
    PRIVATE_TEXT_REDACTION,
    collapse_private_detail_errors,
    redact_private_text,
    retain_public_text,
)
from fra.research_packages.comparability import (
    compare_metric_observations,
)
from fra.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedP2Report,
    GuardedResearchPackage,
    MetricComparison,
    MultiTickerResearchPackage,
    PackageClaim,
    PeerScope,
    ResearchQualityDecision,
    ResearchQualityResult,
    ResolvedSource,
)
from fra.skills.recipes import RESEARCH_QUALITY_SCREEN
from fra.skills.schemas import (
    FinancialSourceProvenance,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
    canonical_financial_source_identity,
    canonical_financial_text,
    financial_metric_identity_components,
    financial_observation_id,
)
from fra.storage.run_repositories import PersistedClaim
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    PolicyValidatedWebEvidence,
    WebEvidenceValidationError,
)

_UNSAFE_MARKUP = re.compile(
    r"(?:^|\n)\s*#|\[[^\]]*\]\([^)]*\)|<\s*/?\s*[a-z!][^>]*>",
    re.IGNORECASE,
)
_UNSAFE_CAUSAL = re.compile(
    r"caus(?:e|ed|es|ing)\b|because\s+of|due\s+to|driven\s+by|"
    r"(?:led|leads)\s+to|will\s+(?:push|raise|lower|rise|fall)|"
    r"(?:shares?|stock)\s+(?:will|are\s+going\s+to)|(?:一定|必然)",
    re.IGNORECASE,
)
_UNSAFE_RANKING = re.compile(
    r"\b(?:best|winner|rank(?:ed|ing)?|preferred|top\s+\d+)\b|"
    r"(?:目标价|投资组合|最佳|赢家)",
    re.IGNORECASE,
)
_CROSS_TICKER_NOTE = re.compile(
    r"^CROSS_TICKER_SOURCE_REJECTED:\s*(?P<ticker>[A-Z][A-Z0-9.-]*)$"
)
_CROSS_TICKER_BRACKET = re.compile(
    r"\[(?P<ticker>[A-Z][A-Z0-9.-]*)(?::[^\]]*)?\].*CROSS_TICKER_SOURCE_REJECTED\b"
)
_LEAKAGE_COUNT_NOTE = re.compile(r"^CROSS_TICKER_LEAKAGE_COUNT:\s*\d+$")


def guard_p2_report(
    *,
    scope: PeerScope,
    package: GuardedResearchPackage | MultiTickerResearchPackage | None,
    comparisons: list[MetricComparison],
    quality: ResearchQualityResult | None,
    web_validator: PersistedWebEvidenceValidator | None,
    requested_as_of: date | None = None,
) -> GuardedP2Report:
    """Retain only source-resolved, non-advisory P2 content."""

    errors: list[str] = []
    normalized_scope = _guard_scope(scope, errors)
    guarded_packages: list[GuardedResearchPackage] = []
    sources_by_ticker: dict[str, dict[str, ResolvedSource]] = {}
    missing_tickers: list[str] = []
    carried_rejection_count = _carried_cross_ticker_rejection_count(package)
    rejection_findings = _carried_cross_ticker_rejection_findings(package)
    package_free_out_of_scope = (
        package is None
        and quality is not None
        and quality.decision is ResearchQualityDecision.OUT_OF_SCOPE
    )

    if package is None:
        source_packages = []
    elif isinstance(package, MultiTickerResearchPackage):
        source_packages = list(package.packages)
        missing_tickers = list(package.missing_tickers)
        errors.extend(
            _displayable_guard_notes(
                package.guard_notes,
                ticker=normalized_scope.primary_ticker,
            )
        )
    else:
        source_packages = [package]

    for source_package in source_packages:
        (
            guarded_package,
            source_index,
            package_errors,
            package_rejection_findings,
        ) = _guard_package(
            source_package,
            web_validator=web_validator,
        )
        guarded_packages.append(guarded_package)
        sources_by_ticker[guarded_package.ticker] = source_index
        errors.extend(
            _displayable_guard_notes(
                guarded_package.guard_notes,
                ticker=guarded_package.ticker,
            )
        )
        errors.extend(package_errors)
        rejection_findings.extend(package_rejection_findings)

    present_tickers = {pkg.ticker for pkg in guarded_packages}
    if (
        normalized_scope.primary_ticker not in present_tickers
        and not package_free_out_of_scope
    ):
        errors.append("PRIMARY_PACKAGE_MISSING")
    for ticker in missing_tickers:
        if not retain_public_text(ticker, ticker=normalized_scope.primary_ticker):
            errors.append(PRIVATE_DETAIL_CODE)
            continue
        errors.append(f"PEER_PACKAGE_MISSING: {ticker}")
    for ticker in normalized_scope.peer_tickers:
        if ticker not in present_tickers:
            errors.append(f"PEER_PACKAGE_MISSING: {ticker}")
    guarded_comparisons: list[MetricComparison] = []
    for index, comparison in enumerate(comparisons):
        guarded, error = _guard_comparison(comparison, index=index, packages=guarded_packages)
        if guarded is None:
            errors.append(error)
            rejection_findings.append(error)
            continue
        guarded_comparisons.append(guarded)

    guarded_quality, quality_errors = _guard_quality(
        quality,
        packages=guarded_packages,
    )
    errors.extend(quality_errors)
    cross_ticker_rejection_count = max(
        carried_rejection_count,
        _cross_ticker_rejection_count(rejection_findings),
    )
    retained_refs = _retained_ref_keys(
        packages=guarded_packages,
        comparisons=guarded_comparisons,
        quality=guarded_quality,
    )
    retained_sources = _retained_sources(sources_by_ticker, retained_refs)
    cross_ticker_leakage_count = _retained_cross_ticker_leakage_count(
        guarded_packages,
        retained_refs,
    )
    if cross_ticker_leakage_count:
        errors.append(f"CROSS_TICKER_LEAKAGE_COUNT: {cross_ticker_leakage_count}")

    sufficiency = _information_sufficiency(
        scope=normalized_scope,
        packages=guarded_packages,
        retained_sources=retained_sources,
        errors=errors,
    )
    provenance = _p2_report_provenance(
        packages=guarded_packages,
        retained_sources=retained_sources,
        information_sufficiency=sufficiency,
        web_validator=web_validator,
        include_quality=guarded_quality is not None,
        requested_as_of=requested_as_of,
    )
    return GuardedP2Report(
        scope=normalized_scope,
        packages=guarded_packages,
        comparisons=guarded_comparisons,
        quality=guarded_quality,
        retained_sources=retained_sources,
        provenance=provenance,
        information_sufficiency=sufficiency,
        cross_ticker_rejection_count=cross_ticker_rejection_count,
        cross_ticker_leakage_count=cross_ticker_leakage_count,
        guard_errors=collapse_private_detail_errors(errors),
    )


def p2_persisted_claims(report: GuardedP2Report) -> list[PersistedClaim]:
    """Map the final guarded report to persisted application claims."""

    claims: list[PersistedClaim] = []
    for package in report.packages:
        for claim in package.claims:
            claims.append(
                PersistedClaim(
                    kind=claim.kind.value,
                    text=claim.text,
                    confidence=claim.confidence.value,
                    source_refs=list(claim.source_refs),
                    guard_status="retained",
                )
            )
        for metric in package.financial_metrics:
            claims.append(
                PersistedClaim(
                    kind="comparable_metric",
                    text=_metric_claim_text(metric),
                    confidence=Confidence.HIGH.value,
                    source_refs=list(metric.source_refs),
                    guard_status="retained",
                )
            )
    if report.quality is not None:
        claims.append(
            PersistedClaim(
                kind="research_quality_decision",
                text=(
                    f"{report.scope.primary_ticker} research quality decision: "
                    f"{report.quality.decision.value}."
                ),
                confidence=Confidence.HIGH.value,
                source_refs=list(report.quality.source_refs),
                guard_status="retained",
            )
        )
        for reason in report.quality.reasons:
            claims.append(
                PersistedClaim(
                    kind="research_quality_reason",
                    text=reason,
                    confidence=Confidence.HIGH.value,
                    source_refs=list(report.quality.source_refs),
                    guard_status="retained",
                )
            )
    for error in report.guard_errors:
        claims.append(
            PersistedClaim(
                kind="p2_guard_note",
                text=error,
                confidence=Confidence.HIGH.value,
                source_refs=[],
                guard_status="rejected",
            )
        )
    return claims


def p2_source_refs(report: GuardedP2Report) -> list[SourceRef]:
    """Return the canonical retained source references for root metadata."""

    return [source.ref for source in report.retained_sources]


def _guard_scope(scope: PeerScope, errors: list[str]) -> PeerScope:
    description, private_description = redact_private_text(
        scope.description,
        ticker=scope.primary_ticker,
    )
    if private_description:
        description = PRIVATE_TEXT_REDACTION
        errors.append(PRIVATE_DETAIL_CODE)
    elif _is_unsafe_report_text(description, scope.primary_ticker):
        description = "Removed by output safety guard."
        errors.append("SCOPE_DESCRIPTION_REDACTED")
    peer_tickers = tuple(
        ticker
        for ticker in scope.peer_tickers
        if retain_public_text(ticker, ticker=scope.primary_ticker)
    )
    if len(peer_tickers) != len(scope.peer_tickers):
        errors.append(PRIVATE_DETAIL_CODE)
    return scope.model_copy(
        update={"description": description, "peer_tickers": peer_tickers}
    )


def _guard_package(
    package: GuardedResearchPackage,
    *,
    web_validator: PersistedWebEvidenceValidator | None,
) -> tuple[GuardedResearchPackage, dict[str, ResolvedSource], list[str], list[str]]:
    errors: list[str] = []
    filing_sources, filing_index = _guard_filing_sources(package, errors)
    web_sources, web_index = _guard_web_sources(package, web_validator, errors)
    source_index = {**filing_index, **web_index}

    claims: list[PackageClaim] = []
    for index, claim in enumerate(package.claims):
        guarded, error = _guard_claim(
            claim,
            ticker=package.ticker,
            index=index,
            source_index=source_index,
        )
        if guarded is None:
            errors.append(error)
            continue
        claims.append(guarded)

    metrics: list[ComparableMetric] = []
    for index, metric in enumerate(package.financial_metrics):
        guarded, error = _guard_metric(
            metric,
            ticker=package.ticker,
            index=index,
            source_index=source_index,
        )
        if guarded is None:
            errors.append(error)
            continue
        metrics.append(guarded)

    metrics, reconciliation_errors = _reconcile_package_metrics(metrics)
    errors.extend(reconciliation_errors)
    information_gaps = _guard_text_list(
        package.information_gaps,
        ticker=package.ticker,
        label="INFORMATION_GAP_DROPPED",
        errors=errors,
    )
    coverage = _guarded_package_coverage(
        incoming=package.coverage,
        claims=claims,
        metrics=metrics,
        errors=errors,
    )

    guarded_package = GuardedResearchPackage(
        ticker=package.ticker,
        claims=claims,
        financial_metrics=metrics,
        filing_sources=filing_sources,
        web_sources=web_sources,
        provenance=_guarded_package_provenance(
            package=package,
            filing_sources=filing_sources,
            web_sources=web_sources,
            web_validator=web_validator,
            coverage=coverage,
        ),
        zero_recipe_outcome=package.zero_recipe_outcome,
        evidence_dates=_source_dates(filing_sources, web_sources),
        coverage=coverage,
        information_gaps=information_gaps,
        guard_notes=_safe_guard_notes(package.guard_notes, ticker=package.ticker),
    )
    return (
        guarded_package,
        source_index,
        errors,
        _cross_ticker_rejection_findings([*package.guard_notes, *errors]),
    )


def _guard_filing_sources(
    package: GuardedResearchPackage,
    errors: list[str],
) -> tuple[list[EvidenceChunk], dict[str, ResolvedSource]]:
    retained: list[EvidenceChunk] = []
    resolved: dict[str, ResolvedSource] = {}
    duplicate_keys: set[str] = set()
    for source in package.filing_sources:
        if not retain_public_text(source.id, ticker=package.ticker):
            errors.append(PRIVATE_DETAIL_CODE)
            continue
        key = SourceRef(
            ticker=package.ticker,
            kind=SourceRefKind.FILING,
            source_id=source.id,
        ).encode()
        if key in resolved or key in duplicate_keys:
            duplicate_keys.add(key)
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: DUPLICATE_RETAINED_SOURCE")
            resolved.pop(key, None)
            retained = [
                kept
                for kept in retained
                if SourceRef(
                    ticker=package.ticker,
                    kind=SourceRefKind.FILING,
                    source_id=kept.id,
                ).encode()
                != key
            ]
            continue
        if source.ticker != package.ticker:
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: CROSS_TICKER_SOURCE_REJECTED")
            continue
        canonical = _canonical_evidence_chunk(source)
        if canonical is None:
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: INVALID_FILING_SOURCE")
            continue
        if _is_unsafe_report_text(canonical.section, package.ticker):
            errors.append("RETAINED_SOURCE_REJECTED: UNSAFE_OUTPUT_TEXT")
            continue
        retained.append(canonical)
        resolved[key] = ResolvedSource(
            ref=SourceRef(ticker=package.ticker, kind=SourceRefKind.FILING, source_id=source.id),
            title=f"{canonical.form} filed {canonical.filed_at.isoformat()} - {canonical.section}",
            source_url=canonicalize_source_url(canonical.source_url),
            published_or_filed_at=canonical.filed_at,
            source_kind=SourceKind.FILING,
            source_tier=SourceTier.PRIMARY,
            canonical_source_identity=canonical_financial_source_identity(
                storage_kind=SourceRefKind.FILING,
                source_kind=SourceKind.FILING,
                source_url=canonical.source_url,
                content_hash=None,
                accession_no=canonical.accession_no,
            ),
        )
    return retained, resolved


def _guard_web_sources(
    package: GuardedResearchPackage,
    web_validator: PersistedWebEvidenceValidator | None,
    errors: list[str],
) -> tuple[list[PolicyValidatedWebEvidence], dict[str, ResolvedSource]]:
    retained: list[PolicyValidatedWebEvidence] = []
    resolved: dict[str, ResolvedSource] = {}
    duplicate_keys: set[str] = set()
    for source in package.web_sources:
        if any(
            not retain_public_text(value, ticker=package.ticker)
            for value in (source.id, source.title, str(source.source_url))
        ):
            errors.append(PRIVATE_DETAIL_CODE)
            continue
        key = SourceRef(
            ticker=package.ticker,
            kind=SourceRefKind.WEB,
            source_id=source.id,
        ).encode()
        if key in resolved or key in duplicate_keys:
            duplicate_keys.add(key)
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: DUPLICATE_RETAINED_SOURCE")
            resolved.pop(key, None)
            retained = [kept for kept in retained if kept.id != source.id]
            continue
        if web_validator is None:
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: WEB_SOURCE_VALIDATOR_REQUIRED")
            continue
        try:
            validated = web_validator.validate(
                ticker=package.ticker,
                evidence=WebEvidence.model_validate(
                    {
                        field: getattr(source, field)
                        for field in (
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
                    },
                    strict=True,
                ),
            )
        except WebEvidenceValidationError as error:
            errors.append(f"RETAINED_SOURCE_REJECTED[{key}]: {error.code}")
            continue
        if any(
            not retain_public_text(value, ticker=package.ticker)
            for value in (str(validated.source_url), str(validated.canonical_url))
        ):
            errors.append(PRIVATE_DETAIL_CODE)
            continue
        if _is_unsafe_report_text(validated.title, package.ticker):
            errors.append("RETAINED_SOURCE_REJECTED: UNSAFE_OUTPUT_TEXT")
            continue
        retained.append(validated)
        published_or_filed_at = (
            validated.published_at
            if validated.published_at is not None
            else validated.fetched_at
        )
        resolved[key] = ResolvedSource(
            ref=SourceRef(ticker=package.ticker, kind=SourceRefKind.WEB, source_id=validated.id),
            title=validated.title,
            source_url=canonicalize_source_url(validated.source_url),
            published_or_filed_at=published_or_filed_at,
            source_kind=validated.source_kind,
            source_tier=validated.source_tier,
            canonical_source_identity=canonical_financial_source_identity(
                storage_kind=SourceRefKind.WEB,
                source_kind=validated.source_kind,
                source_url=str(validated.canonical_url),
                content_hash=validated.content_hash,
            ),
        )
    return retained, resolved


def _guard_claim(
    claim: PackageClaim,
    *,
    ticker: str,
    index: int,
    source_index: Mapping[str, ResolvedSource],
) -> tuple[PackageClaim | None, str]:
    if _is_unsafe_report_text(claim.text, ticker):
        return None, f"CLAIM_DROPPED[{ticker}:{index}]: UNSAFE_OUTPUT_TEXT"
    refs, error = _resolve_source_refs(
        claim.source_refs,
        ticker=ticker,
        source_index=source_index,
        label=f"CLAIM_DROPPED[{ticker}:{index}]",
    )
    if error is not None:
        return None, error
    return claim.model_copy(update={"source_refs": refs}), ""


def _guard_metric(
    metric: ComparableMetric,
    *,
    ticker: str,
    index: int,
    source_index: Mapping[str, ResolvedSource],
) -> tuple[ComparableMetric | None, str]:
    if metric.ticker != ticker:
        return None, f"METRIC_DROPPED[{ticker}:{index}]: CROSS_TICKER_SOURCE_REJECTED"
    if any(
        _is_unsafe_report_text(value, ticker)
        for value in (
            metric.name,
            metric.currency,
            metric.unit,
            metric.definition,
        )
        if value is not None
    ):
        return None, f"METRIC_DROPPED[{ticker}:{index}]: UNSAFE_OUTPUT_TEXT"
    refs, error = _resolve_source_refs(
        metric.source_refs,
        ticker=ticker,
        source_index=source_index,
        label=f"METRIC_DROPPED[{ticker}:{index}]",
        allow_empty=metric.value is None,
    )
    if error is not None:
        return None, error
    source_provenance = [
        FinancialSourceProvenance(
            source_ref=ref,
            source_kind=source_index[ref.encode()].source_kind,
            source_tier=source_index[ref.encode()].source_tier,
            canonical_source_identity=(
                source_index[ref.encode()].canonical_source_identity
            ),
        )
        for ref in refs
    ]
    verification_status = _source_verification_status(
        value_missing=metric.value is None,
        source_provenance=source_provenance,
    )
    status = _metric_status(verification_status)
    try:
        return (
            ComparableMetric(
                ticker=metric.ticker,
                name=metric.name,
                value=metric.value,
                period_start=metric.period_start,
                period_end=metric.period_end,
                currency=metric.currency,
                unit=metric.unit,
                definition=metric.definition,
                observation_id=financial_observation_id(
                    ticker=metric.ticker,
                    name=metric.name,
                    period_start=metric.period_start,
                    period_end=metric.period_end,
                    currency=metric.currency,
                    unit=metric.unit,
                    definition=metric.definition,
                ),
                source_refs=refs,
                source_provenance=source_provenance,
                verification_status=verification_status,
                status=status,
                limitation=_metric_limitation(verification_status, None),
            ),
            "",
        )
    except ValidationError:
        return None, f"METRIC_DROPPED[{ticker}:{index}]: INVALID_METRIC"


def _reconcile_package_metrics(
    metrics: list[ComparableMetric],
) -> tuple[list[ComparableMetric], list[str]]:
    grouped: dict[str, list[ComparableMetric]] = {}
    for metric in metrics:
        grouped.setdefault(canonical_financial_text(metric.name), []).append(metric)

    reconciled: list[ComparableMetric] = []
    errors: list[str] = []
    for group in grouped.values():
        nonmissing = [metric for metric in group if metric.value is not None]
        identities = {_metric_identity(metric) for metric in nonmissing}
        conflict_status: VerificationStatus | None = None
        if len(identities) > 1:
            status = VerificationStatus.NOT_COMPARABLE
            note = _financial_conflict_note(nonmissing, not_comparable=True)
            code = "FINANCIAL_DATA_NOT_COMPARABLE"
            conflict_status = status
        elif len({metric.value for metric in nonmissing}) > 1:
            status = VerificationStatus.DISCREPANCY
            note = _financial_conflict_note(nonmissing, not_comparable=False)
            code = "FINANCIAL_DATA_DISCREPANCY"
            conflict_status = status
        else:
            independent_sources = {
                source.canonical_source_identity
                for metric in nonmissing
                for source in metric.source_provenance
            }
            status = (
                VerificationStatus.VERIFIED
                if len(independent_sources) >= 2
                else VerificationStatus.SINGLE_SOURCE
            )
            note = None

        resolved = [
            metric.model_copy(
                update={
                    "verification_status": point_status,
                    "status": _metric_status(point_status),
                    "limitation": _metric_limitation(point_status, note),
                }
            )
            for metric in group
            for point_status in (
                VerificationStatus.MISSING if metric.value is None else status,
            )
        ]
        if conflict_status is not None:
            resolved.sort(key=lambda metric: not _is_primary_metric(metric))
            errors.append(f"{code}: {group[0].name}")
        reconciled.extend(resolved)
    return reconciled, errors


def _source_verification_status(
    *,
    value_missing: bool,
    source_provenance: list[FinancialSourceProvenance],
) -> VerificationStatus:
    if value_missing:
        return VerificationStatus.MISSING
    independent_sources = {
        source.canonical_source_identity for source in source_provenance
    }
    return (
        VerificationStatus.VERIFIED
        if len(independent_sources) >= 2
        else VerificationStatus.SINGLE_SOURCE
    )


def _metric_status(status: VerificationStatus) -> ComparabilityStatus:
    return {
        VerificationStatus.VERIFIED: ComparabilityStatus.COMPARABLE,
        VerificationStatus.SINGLE_SOURCE: ComparabilityStatus.COMPARABLE,
        VerificationStatus.DISCREPANCY: ComparabilityStatus.DISCREPANCY,
        VerificationStatus.NOT_COMPARABLE: ComparabilityStatus.NOT_COMPARABLE,
        VerificationStatus.MISSING: ComparabilityStatus.MISSING,
    }[status]


def _metric_limitation(
    status: VerificationStatus,
    conflict_note: str | None,
) -> str | None:
    if status is VerificationStatus.VERIFIED:
        return None
    if status is VerificationStatus.SINGLE_SOURCE:
        return "Only one independent canonical source was retained."
    if status is VerificationStatus.MISSING:
        return "Metric value is missing from retained evidence."
    assert conflict_note is not None
    return conflict_note


def _metric_identity(metric: ComparableMetric) -> tuple[str, ...]:
    return financial_metric_identity_components(
        name=metric.name,
        period_start=metric.period_start,
        period_end=metric.period_end,
        currency=metric.currency,
        unit=metric.unit,
        definition=metric.definition,
    )


def _is_primary_metric(metric: ComparableMetric) -> bool:
    return any(
        source.source_tier is SourceTier.PRIMARY
        for source in metric.source_provenance
    )


def _financial_conflict_note(
    metrics: list[ComparableMetric],
    *,
    not_comparable: bool,
) -> str:
    conflict = (
        "Conflicting period, currency, unit, or definition observations were retained."
        if not_comparable
        else "Conflicting exact values were retained."
    )
    has_primary = any(_is_primary_metric(metric) for metric in metrics)
    has_secondary = any(
        source.source_tier is SourceTier.AUTHORITATIVE_SECONDARY
        for metric in metrics
        for source in metric.source_provenance
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


def _guard_comparison(
    comparison: MetricComparison,
    *,
    index: int,
    packages: Iterable[GuardedResearchPackage],
) -> tuple[MetricComparison | None, str]:
    package_map = {package.ticker: package for package in packages}
    observations: list[ComparableMetric] = []
    for observation_index, observation in enumerate(comparison.observations):
        package = package_map.get(observation.ticker)
        if package is None:
            return (
                None,
                f"COMPARISON_DROPPED[{index}]: UNKNOWN_TICKER:{observation.ticker}",
            )
        source_index = _package_source_index(package)
        guarded, error = _guard_metric(
            observation,
            ticker=observation.ticker,
            index=observation_index,
            source_index=source_index,
        )
        if guarded is None:
            if _is_cross_ticker_finding(error):
                return (
                    None,
                    (
                        f"COMPARISON_DROPPED[{index}]: "
                        f"[{observation.ticker}] CROSS_TICKER_SOURCE_REJECTED"
                    ),
                )
            code = error.split(": ", maxsplit=1)[-1]
            return None, f"COMPARISON_DROPPED[{index}]: {code}"
        observations.append(guarded)
    if _is_unsafe_report_text(comparison.name, observations[0].ticker):
        return None, f"COMPARISON_DROPPED[{index}]: UNSAFE_OUTPUT_TEXT"
    try:
        recomputed = compare_metric_observations(observations)
    except (TypeError, ValidationError, ValueError):
        return None, f"COMPARISON_DROPPED[{index}]: INVALID_COMPARISON"
    if recomputed.status is ComparabilityStatus.DISCREPANCY:
        observations.sort(key=lambda observation: not _is_primary_metric(observation))
        limitation = _financial_conflict_note(
            [observation for observation in observations if observation.value is not None],
            not_comparable=False,
        )
    else:
        observations = recomputed.observations
        limitation = recomputed.limitation
    if limitation is not None and _is_unsafe_report_text(limitation, observations[0].ticker):
        return None, f"COMPARISON_DROPPED[{index}]: UNSAFE_OUTPUT_TEXT"
    try:
        return (
            MetricComparison(
                name=recomputed.name,
                observations=observations,
                status=recomputed.status,
                delta=recomputed.delta,
                limitation=limitation,
            ),
            "",
        )
    except ValidationError:
        return None, f"COMPARISON_DROPPED[{index}]: INVALID_COMPARISON"


def _guard_quality(
    quality: ResearchQualityResult | None,
    *,
    packages: list[GuardedResearchPackage],
) -> tuple[ResearchQualityResult | None, list[str]]:
    if quality is None:
        return None, []
    source_index: dict[str, ResolvedSource] = {}
    for package in packages:
        source_index.update(_package_source_index(package))
    refs, ref_error = _resolve_source_refs(
        quality.source_refs,
        ticker=None,
        source_index=source_index,
        label="QUALITY_DROPPED",
        allow_empty=quality.decision is ResearchQualityDecision.OUT_OF_SCOPE,
        allow_cross_ticker=True,
    )
    local_errors: list[str] = []
    if ref_error is not None:
        return None, [ref_error]
    if quality.decision is ResearchQualityDecision.OUT_OF_SCOPE:
        reasons = ["Request is outside research-quality scope."]
    else:
        reasons = _guard_text_list(
            quality.reasons,
            ticker=packages[0].ticker if packages else "NVDA",
            label="QUALITY_REASON_DROPPED",
            errors=local_errors,
        )
        if not reasons:
            return None, [*local_errors, "QUALITY_DROPPED: NO_RETAINED_REASONS"]
    return (
        ResearchQualityResult(
            decision=quality.decision,
            reasons=reasons,
            source_refs=refs,
        ),
        local_errors,
    )


def _resolve_source_refs(
    refs: list[SourceRef],
    *,
    ticker: str | None,
    source_index: Mapping[str, ResolvedSource],
    label: str,
    allow_empty: bool = False,
    allow_cross_ticker: bool = False,
) -> tuple[list[SourceRef], str | None]:
    if not refs and allow_empty:
        return [], None
    seen: set[str] = set()
    resolved: list[SourceRef] = []
    for ref in refs:
        encoded = ref.encode()
        if encoded in seen:
            return [], f"{label}: DUPLICATE_SOURCE_REF"
        seen.add(encoded)
        if not allow_cross_ticker and ticker is not None and ref.ticker != ticker:
            return [], f"{label}: CROSS_TICKER_SOURCE_REJECTED"
        if ref.kind not in {SourceRefKind.FILING, SourceRefKind.WEB}:
            return [], f"{label}: SOURCE_KIND_REJECTED"
        if encoded not in source_index:
            return [], f"{label}: SOURCE_REF_MISSING"
        resolved.append(ref)
    if not resolved and not allow_empty:
        return [], f"{label}: SOURCE_REF_MISSING"
    return resolved, None


def _retained_ref_keys(
    *,
    packages: list[GuardedResearchPackage],
    comparisons: list[MetricComparison],
    quality: ResearchQualityResult | None,
) -> set[str]:
    retained = {
        reference.encode()
        for package in packages
        for claim in package.claims
        for reference in claim.source_refs
    }
    retained.update(
        reference.encode()
        for package in packages
        for metric in package.financial_metrics
        for reference in metric.source_refs
    )
    retained.update(
        reference.encode()
        for comparison in comparisons
        for observation in comparison.observations
        for reference in observation.source_refs
    )
    if quality is not None:
        retained.update(reference.encode() for reference in quality.source_refs)
    return retained


def _retained_sources(
    sources_by_ticker: Mapping[str, Mapping[str, ResolvedSource]],
    retained_refs: set[str],
) -> list[ResolvedSource]:
    retained: list[ResolvedSource] = []
    for ticker in sorted(sources_by_ticker):
        for key in sorted(sources_by_ticker[ticker]):
            if key in retained_refs:
                retained.append(sources_by_ticker[ticker][key])
    return retained


def _retained_cross_ticker_leakage_count(
    packages: list[GuardedResearchPackage],
    retained_refs: set[str],
) -> int:
    leakage_count = 0
    for package in packages:
        for sources, kind in (
            (package.filing_sources, SourceRefKind.FILING),
            (package.web_sources, SourceRefKind.WEB),
        ):
            for source in sources:
                ref = SourceRef(ticker=package.ticker, kind=kind, source_id=source.id)
                if ref.encode() in retained_refs and source.ticker != package.ticker:
                    leakage_count += 1
    return leakage_count


def _package_source_index(package: GuardedResearchPackage) -> dict[str, ResolvedSource]:
    index: dict[str, ResolvedSource] = {}
    for source in package.filing_sources:
        key = SourceRef(
            ticker=package.ticker,
            kind=SourceRefKind.FILING,
            source_id=source.id,
        ).encode()
        index[key] = ResolvedSource(
            ref=SourceRef(ticker=package.ticker, kind=SourceRefKind.FILING, source_id=source.id),
            title=f"{source.form} filed {source.filed_at.isoformat()} - {source.section}",
            source_url=canonicalize_source_url(source.source_url),
            published_or_filed_at=source.filed_at,
            source_kind=SourceKind.FILING,
            source_tier=SourceTier.PRIMARY,
            canonical_source_identity=canonical_financial_source_identity(
                storage_kind=SourceRefKind.FILING,
                source_kind=SourceKind.FILING,
                source_url=source.source_url,
                content_hash=None,
                accession_no=source.accession_no,
            ),
        )
    for source in package.web_sources:
        key = SourceRef(
            ticker=package.ticker,
            kind=SourceRefKind.WEB,
            source_id=source.id,
        ).encode()
        index[key] = ResolvedSource(
            ref=SourceRef(ticker=package.ticker, kind=SourceRefKind.WEB, source_id=source.id),
            title=source.title,
            source_url=canonicalize_source_url(source.source_url),
            published_or_filed_at=(
                source.published_at if source.published_at is not None else source.fetched_at
            ),
            source_kind=source.source_kind,
            source_tier=source.source_tier,
            canonical_source_identity=canonical_financial_source_identity(
                storage_kind=SourceRefKind.WEB,
                source_kind=source.source_kind,
                source_url=str(source.canonical_url),
                content_hash=source.content_hash,
            ),
        )
    return index


def _guarded_package_coverage(
    *,
    incoming: str,
    claims: list[PackageClaim],
    metrics: list[ComparableMetric],
    errors: list[str],
) -> str:
    if not claims and not metrics:
        return "insufficient"
    if incoming == "insufficient":
        return "insufficient"
    if incoming == "partial" or errors:
        return "partial"
    return "complete"


def _source_dates(
    filing_sources: list[EvidenceChunk],
    web_sources: list[PolicyValidatedWebEvidence],
) -> list[date]:
    dates = [source.filed_at for source in filing_sources]
    dates.extend(
        source.published_at.date()
        if source.published_at is not None
        else source.fetched_at.date()
        for source in web_sources
    )
    return sorted(set(dates))


def _guarded_package_provenance(
    *,
    package: GuardedResearchPackage,
    filing_sources: list[EvidenceChunk],
    web_sources: list[PolicyValidatedWebEvidence],
    web_validator: PersistedWebEvidenceValidator | None,
    coverage: str,
) -> ReportProvenance:
    source_refs = (
        *(
            SourceRef(
                ticker=package.ticker,
                kind=SourceRefKind.FILING,
                source_id=source.id,
            )
            for source in filing_sources
        ),
        *(
            SourceRef(
                ticker=package.ticker,
                kind=SourceRefKind.WEB,
                source_id=source.id,
            )
            for source in web_sources
        ),
    )
    source_dates = _source_dates(filing_sources, web_sources)
    return ReportProvenance(
        recipes=package.provenance.recipes,
        source_policy_versions=(
            () if web_validator is None else (web_validator.policy_version,)
        ),
        corpus_versions=tuple(
            dict.fromkeys(source.corpus_version for source in filing_sources)
        ),
        prompt_versions=package.provenance.prompt_versions,
        requested_as_of_dates=package.provenance.requested_as_of_dates,
        evidence_cutoff_dates=(() if not source_dates else (max(source_dates),)),
        information_sufficiency={
            "complete": InformationSufficiency.SUFFICIENT,
            "partial": InformationSufficiency.PARTIAL,
            "insufficient": InformationSufficiency.INSUFFICIENT,
        }[coverage],
        source_refs=source_refs,
    )


def _p2_report_provenance(
    *,
    packages: list[GuardedResearchPackage],
    retained_sources: list[ResolvedSource],
    information_sufficiency: InformationSufficiency,
    web_validator: PersistedWebEvidenceValidator | None,
    include_quality: bool,
    requested_as_of: date | None,
) -> ReportProvenance:
    recipes = list(
        dict.fromkeys(
            recipe
            for package in packages
            for recipe in package.provenance.recipes
        )
    )
    if include_quality:
        quality_recipe = RecipeProvenance(
            name=RESEARCH_QUALITY_SCREEN.name,
            version=RESEARCH_QUALITY_SCREEN.version,
        )
        if quality_recipe not in recipes:
            recipes.append(quality_recipe)
    retained_filing_refs = {
        source.ref.encode()
        for source in retained_sources
        if source.ref.kind is SourceRefKind.FILING
    }
    corpus_versions = tuple(
        dict.fromkeys(
            source.corpus_version
            for package in packages
            for source in package.filing_sources
            if SourceRef(
                ticker=package.ticker,
                kind=SourceRefKind.FILING,
                source_id=source.id,
            ).encode()
            in retained_filing_refs
        )
    )
    prompt_versions = tuple(
        dict.fromkeys(
            version
            for package in packages
            for version in package.provenance.prompt_versions
        )
    )
    requested_as_of_dates = tuple(
        dict.fromkeys(
            [
                *(
                    requested
                    for package in packages
                    for requested in package.provenance.requested_as_of_dates
                ),
                *((requested_as_of,) if requested_as_of is not None else ()),
            ]
        )
    )
    evidence_cutoff_dates = (
        ()
        if not retained_sources
        else (
            max(
                source.published_or_filed_at.date()
                if hasattr(source.published_or_filed_at, "date")
                else source.published_or_filed_at
                for source in retained_sources
            ),
        )
    )
    return ReportProvenance(
        recipes=tuple(recipes),
        source_policy_versions=(
            () if web_validator is None else (web_validator.policy_version,)
        ),
        corpus_versions=corpus_versions,
        prompt_versions=prompt_versions,
        requested_as_of_dates=requested_as_of_dates,
        evidence_cutoff_dates=evidence_cutoff_dates,
        information_sufficiency=information_sufficiency,
        source_refs=tuple(source.ref for source in retained_sources),
    )


def _guard_text_list(
    values: list[str],
    *,
    ticker: str,
    label: str,
    errors: list[str],
) -> list[str]:
    retained: list[str] = []
    for index, value in enumerate(values):
        if _is_unsafe_report_text(value, ticker):
            errors.append(f"{label}[{index}]: UNSAFE_OUTPUT_TEXT")
            continue
        retained.append(value)
    return retained


def _safe_guard_notes(values: list[str], *, ticker: str) -> list[str]:
    retained: list[str] = []
    for value in values:
        if not retain_public_text(value, ticker=ticker) or any(
            unicodedata.category(character).startswith("C") for character in value
        ):
            retained.append(PRIVATE_DETAIL_CODE)
            continue
        retained.append(value)
    return _unique(retained)


def _carried_cross_ticker_rejection_count(
    package: GuardedResearchPackage | MultiTickerResearchPackage | None,
) -> int:
    if not isinstance(package, MultiTickerResearchPackage):
        return 0
    return package.cross_ticker_leakage_count


def _carried_cross_ticker_rejection_findings(
    package: GuardedResearchPackage | MultiTickerResearchPackage | None,
) -> list[str]:
    if isinstance(package, MultiTickerResearchPackage):
        values = list(package.guard_notes)
        values.extend(
            note
            for guarded_package in package.packages
            for note in guarded_package.guard_notes
        )
        return _cross_ticker_rejection_findings(values)
    return (
        []
        if package is None
        else _cross_ticker_rejection_findings(package.guard_notes)
    )


def _is_cross_ticker_finding(value: str) -> bool:
    return "CROSS_TICKER_SOURCE_REJECTED" in value


def _displayable_guard_notes(values: list[str], *, ticker: str) -> list[str]:
    return [
        value
        for value in _safe_guard_notes(values, ticker=ticker)
        if not _LEAKAGE_COUNT_NOTE.fullmatch(value)
    ]


def _cross_ticker_rejection_findings(values: list[str]) -> list[str]:
    return [value for value in values if _is_cross_ticker_finding(value)]


def _cross_ticker_rejection_count(values: list[str]) -> int:
    notes: set[str] = set()
    concrete_findings: set[str] = set()
    concrete_tickers: set[str] = set()
    for value in values:
        if not _is_cross_ticker_finding(value):
            continue
        if note_match := _CROSS_TICKER_NOTE.fullmatch(value):
            notes.add(note_match.group("ticker"))
            continue
        concrete_findings.add(value)
        if bracket_match := _CROSS_TICKER_BRACKET.search(value):
            concrete_tickers.add(bracket_match.group("ticker"))
    return len(concrete_findings) + len(notes.difference(concrete_tickers))


def _information_sufficiency(
    *,
    scope: PeerScope,
    packages: list[GuardedResearchPackage],
    retained_sources: list[ResolvedSource],
    errors: list[str],
) -> InformationSufficiency:
    primary = next(
        (package for package in packages if package.ticker == scope.primary_ticker),
        None,
    )
    if primary is None:
        return InformationSufficiency.INSUFFICIENT
    if primary.coverage == "insufficient":
        return InformationSufficiency.INSUFFICIENT
    if not primary.claims and not primary.financial_metrics:
        return InformationSufficiency.INSUFFICIENT
    if not retained_sources:
        return InformationSufficiency.INSUFFICIENT
    if errors:
        return InformationSufficiency.PARTIAL
    if any(package.coverage != "complete" for package in packages):
        return InformationSufficiency.PARTIAL
    if any(package.information_gaps or package.guard_notes for package in packages):
        return InformationSufficiency.PARTIAL
    if set(scope.peer_tickers) - {package.ticker for package in packages}:
        return InformationSufficiency.PARTIAL
    return InformationSufficiency.SUFFICIENT


def _is_unsafe_report_text(text: str, ticker: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    return (
        _is_unsafe_output_text(normalized, ticker)
        or _UNSAFE_MARKUP.search(normalized) is not None
        or _UNSAFE_CAUSAL.search(normalized) is not None
        or _UNSAFE_RANKING.search(normalized) is not None
        or any(
            unicodedata.category(character).startswith("C") and character not in "\n\t"
            for character in normalized
        )
    )


def _metric_claim_text(metric: ComparableMetric) -> str:
    value = "missing" if metric.value is None else str(metric.value)
    return (
        f"{metric.ticker} {metric.name} {value} {metric.unit} for {metric.period_end.isoformat()}"
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
