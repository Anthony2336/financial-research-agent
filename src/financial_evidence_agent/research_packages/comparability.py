"""Exact comparability checks for guarded peer metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    MetricComparison,
)
from financial_evidence_agent.skills.schemas import (
    VerificationStatus,
    canonical_financial_text,
    financial_metric_identity_components,
    financial_observation_id,
)

_IDENTITY_FIELDS = (
    "name",
    "period_start",
    "period_end",
    "currency",
    "unit",
    "definition",
)


def compare_metric_pair(
    left: ComparableMetric,
    right: ComparableMetric,
) -> MetricComparison:
    """Compare two metrics without converting units, periods, or definitions."""

    return compare_metric_observations([left, right])


def compare_metric_observations(
    observations: list[ComparableMetric],
) -> MetricComparison:
    """Derive comparison status and delta solely from guarded observations."""
    if len(observations) < 2:
        raise ValueError("a comparison requires at least two observations")

    first = observations[0]
    first_identity = _identity(first)
    for observation in observations[1:]:
        identity = _identity(observation)
        mismatch = next(
            (
                field
                for field, first_value, value in zip(
                    _IDENTITY_FIELDS,
                    first_identity,
                    identity,
                    strict=True,
                )
                if first_value != value
            ),
            None,
        )
        if mismatch is not None:
            return MetricComparison(
                name=first.name,
                observations=observations,
                status=ComparabilityStatus.NOT_COMPARABLE,
                delta=None,
                limitation=f"{mismatch} mismatch",
            )

    ticker_order = list(dict.fromkeys(observation.ticker for observation in observations))
    by_ticker = {
        ticker: [observation for observation in observations if observation.ticker == ticker]
        for ticker in ticker_order
    }
    if any(_has_conflicting_duplicates(group) for group in by_ticker.values()):
        return _discrepancy_comparison(observations)
    if any(
        observation.status is ComparabilityStatus.MISSING
        for observation in observations
    ):
        return MetricComparison(
            name=first.name,
            observations=observations,
            status=ComparabilityStatus.MISSING,
            delta=None,
            limitation=next(
                (
                    observation.limitation
                    for observation in observations
                    if observation.limitation
                ),
                "metric missing",
            ),
        )
    if any(
        observation.status is ComparabilityStatus.DISCREPANCY
        for observation in observations
    ):
        discrepancy = _discrepancy_comparison(observations)
        limitation = next(
            (
                observation.limitation
                for observation in observations
                if observation.status is ComparabilityStatus.DISCREPANCY
                and observation.limitation
            ),
            discrepancy.limitation,
        )
        return discrepancy.model_copy(update={"limitation": limitation})
    if any(
        observation.status is ComparabilityStatus.NOT_COMPARABLE
        for observation in observations
    ):
        return MetricComparison(
            name=first.name,
            observations=observations,
            status=ComparabilityStatus.NOT_COMPARABLE,
            delta=None,
            limitation=next(
                (
                    observation.limitation
                    for observation in observations
                    if observation.limitation
                ),
                "metric is not comparable",
            ),
        )
    if len(ticker_order) != 2:
        return MetricComparison(
            name=first.name,
            observations=observations,
            status=ComparabilityStatus.NOT_COMPARABLE,
            delta=None,
            limitation="comparison requires exactly two tickers",
        )
    left = _canonical_metric(by_ticker[ticker_order[0]])
    right = _canonical_metric(by_ticker[ticker_order[1]])
    assert left.value is not None and right.value is not None
    return MetricComparison(
        name=first.name,
        observations=[
            *_sorted_metrics(by_ticker[ticker_order[0]]),
            *_sorted_metrics(by_ticker[ticker_order[1]]),
        ],
        status=ComparabilityStatus.COMPARABLE,
        delta=left.value - right.value,
        limitation=None,
    )


def compare_metrics(packages: list[GuardedResearchPackage]) -> list[MetricComparison]:
    """Compare the primary ticker against each successful peer package."""

    if len(packages) < 2:
        return []
    primary = packages[0]
    comparisons: list[MetricComparison] = []
    for peer in packages[1:]:
        comparisons.extend(_compare_package_pair(primary, peer))
    return comparisons


def _compare_package_pair(
    primary: GuardedResearchPackage,
    peer: GuardedResearchPackage,
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    primary_by_name = _metrics_by_name(primary.financial_metrics)
    peer_by_name = _metrics_by_name(peer.financial_metrics)
    for name in sorted(primary_by_name):
        primary_identities = _metrics_by_identity(primary_by_name[name])
        peer_identities = _metrics_by_identity(peer_by_name.get(name, []))
        for identity in sorted(primary_identities, key=_identity_sort_key):
            primary_metrics = primary_identities[identity]
            comparisons.append(
                _build_metric_comparison(
                    primary_metrics=primary_metrics,
                    peer_exact_metrics=peer_identities.get(identity, []),
                    peer_same_name_metrics=peer_by_name.get(name, []),
                    peer_ticker=peer.ticker,
                )
            )
        comparisons.extend(
            _unmatched_peer_conflict_comparisons(
                primary_ticker=primary.ticker,
                primary_identities=primary_identities,
                peer_identities=peer_identities,
            )
        )
    return comparisons


def _build_metric_comparison(
    *,
    primary_metrics: list[ComparableMetric],
    peer_exact_metrics: list[ComparableMetric],
    peer_same_name_metrics: list[ComparableMetric],
    peer_ticker: str,
) -> MetricComparison:
    primary = _canonical_metric(primary_metrics)
    primary_conflict = _has_conflicting_duplicates(primary_metrics)
    if primary_conflict:
        observations = _sorted_metrics(primary_metrics)
        if peer_exact_metrics:
            if _has_conflicting_duplicates(peer_exact_metrics):
                observations.extend(_sorted_metrics(peer_exact_metrics))
            else:
                observations.append(_canonical_metric(peer_exact_metrics))
        return _discrepancy_comparison(observations)
    if peer_exact_metrics:
        if _has_conflicting_duplicates(peer_exact_metrics):
            return _discrepancy_comparison([primary, *_sorted_metrics(peer_exact_metrics)])
        comparison = compare_metric_pair(primary, _canonical_metric(peer_exact_metrics))
        return comparison.model_copy(
            update={
                "observations": [
                    *_sorted_metrics(primary_metrics),
                    *_sorted_metrics(peer_exact_metrics),
                ]
            }
        )
    if peer_same_name_metrics:
        return MetricComparison(
            name=primary.name,
            observations=[primary, *_sorted_metrics(peer_same_name_metrics)],
            status=ComparabilityStatus.NOT_COMPARABLE,
            delta=None,
            limitation=(
                "same-name peer metric did not match the exact identity "
                "(name, period_start, period_end, currency, unit, definition)"
            ),
        )
    return MetricComparison(
        name=primary.name,
        observations=[primary, _missing_peer_metric(primary, peer_ticker)],
        status=ComparabilityStatus.MISSING,
        delta=None,
        limitation=f"{peer_ticker} missing retained metric for {primary.name}",
    )


def _unmatched_peer_conflict_comparisons(
    *,
    primary_ticker: str,
    primary_identities: dict[tuple[object, ...], list[ComparableMetric]],
    peer_identities: dict[tuple[object, ...], list[ComparableMetric]],
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for identity in sorted(peer_identities, key=_identity_sort_key):
        if identity in primary_identities:
            continue
        peer_metrics = peer_identities[identity]
        if not _has_conflicting_duplicates(peer_metrics):
            continue
        peer_canonical = _canonical_metric(peer_metrics)
        comparisons.append(
            _discrepancy_comparison(
                [
                    _missing_primary_metric(peer_canonical, primary_ticker),
                    *_sorted_metrics(peer_metrics),
                ]
            )
        )
    return comparisons


def _metrics_by_name(metrics: Iterable[ComparableMetric]) -> dict[str, list[ComparableMetric]]:
    grouped: dict[str, list[ComparableMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[canonical_financial_text(metric.name)].append(metric)
    return grouped


def _metrics_by_identity(
    metrics: Iterable[ComparableMetric],
) -> dict[tuple[object, ...], list[ComparableMetric]]:
    grouped: dict[tuple[object, ...], list[ComparableMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[_identity(metric)].append(metric)
    return grouped


def _has_conflicting_duplicates(metrics: list[ComparableMetric]) -> bool:
    for duplicates in _metrics_by_identity(metrics).values():
        if len(duplicates) <= 1:
            continue
        head = duplicates[0]
        if any(
            metric.value != head.value or metric.status is not head.status
            for metric in duplicates[1:]
        ):
            return True
    return False


def _canonical_metric(metrics: list[ComparableMetric]) -> ComparableMetric:
    return _sorted_metrics(metrics)[0]


def _identity(metric: ComparableMetric) -> tuple[object, ...]:
    return financial_metric_identity_components(
        name=metric.name,
        period_start=metric.period_start,
        period_end=metric.period_end,
        currency=metric.currency,
        unit=metric.unit,
        definition=metric.definition,
    )


def _identity_sort_key(identity: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(_sort_component(value) for value in identity)


def _metric_sort_key(metric: ComparableMetric) -> tuple[str, ...]:
    return (
        _sort_component(metric.ticker),
        *_identity_sort_key(_identity(metric)),
        _sort_component(metric.status.value),
        _sort_component(metric.value),
        _sort_component(metric.limitation),
        _sort_component(tuple(reference.encode() for reference in metric.source_refs)),
    )


def _sorted_metrics(metrics: Iterable[ComparableMetric]) -> list[ComparableMetric]:
    return sorted(metrics, key=_metric_sort_key)


def _sort_component(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return "|".join(_sort_component(item) for item in value)
    return str(value)


def _missing_peer_metric(primary: ComparableMetric, peer_ticker: str) -> ComparableMetric:
    return ComparableMetric(
        ticker=peer_ticker,
        name=primary.name,
        value=None,
        period_start=primary.period_start,
        period_end=primary.period_end,
        currency=primary.currency,
        unit=primary.unit,
        definition=primary.definition,
        observation_id=financial_observation_id(
            ticker=peer_ticker,
            name=primary.name,
            period_start=primary.period_start,
            period_end=primary.period_end,
            currency=primary.currency,
            unit=primary.unit,
            definition=primary.definition,
        ),
        source_refs=[],
        source_provenance=[],
        verification_status=VerificationStatus.MISSING,
        status=ComparabilityStatus.MISSING,
        limitation=f"{peer_ticker} missing retained metric for {primary.name}",
    )


def _missing_primary_metric(peer: ComparableMetric, primary_ticker: str) -> ComparableMetric:
    return ComparableMetric(
        ticker=primary_ticker,
        name=peer.name,
        value=None,
        period_start=peer.period_start,
        period_end=peer.period_end,
        currency=peer.currency,
        unit=peer.unit,
        definition=peer.definition,
        observation_id=financial_observation_id(
            ticker=primary_ticker,
            name=peer.name,
            period_start=peer.period_start,
            period_end=peer.period_end,
            currency=peer.currency,
            unit=peer.unit,
            definition=peer.definition,
        ),
        source_refs=[],
        source_provenance=[],
        verification_status=VerificationStatus.MISSING,
        status=ComparabilityStatus.MISSING,
        limitation=f"{primary_ticker} missing retained metric for {peer.name}",
    )


def _discrepancy_comparison(observations: list[ComparableMetric]) -> MetricComparison:
    return MetricComparison(
        name=observations[0].name,
        observations=observations,
        status=ComparabilityStatus.DISCREPANCY,
        delta=None,
        limitation="conflicting observations for the same metric identity",
    )
