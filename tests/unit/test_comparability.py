"""Exact comparability checks for multi-ticker industry metrics."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from financial_evidence_agent.domain import (
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
)
from financial_evidence_agent.research_packages.comparability import (
    compare_metric_pair,
    compare_metrics,
)
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    MetricComparison,
)
from financial_evidence_agent.skills.models import SkillName
from financial_evidence_agent.skills.schemas import (
    FinancialSourceProvenance,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
    financial_observation_id,
)


def _ref(source_id: str, *, ticker: str) -> SourceRef:
    return SourceRef(ticker=ticker, kind=SourceRefKind.FILING, source_id=source_id)


def _metric(
    *,
    ticker: str,
    name: str = "Revenue",
    value: Decimal | None = Decimal("44.1"),
    period_start: date | None = date(2026, 1, 1),
    period_end: date = date(2026, 3, 31),
    currency: str | None = "USD",
    unit: str = "billions",
    definition: str = "GAAP revenue",
    status: ComparabilityStatus = ComparabilityStatus.COMPARABLE,
    limitation: str | None = None,
    source_id: str | None = None,
) -> ComparableMetric:
    refs = [] if source_id is None else [_ref(source_id, ticker=ticker)]
    source_provenance = [
        FinancialSourceProvenance(
            source_ref=reference,
            source_kind=SourceKind.FILING,
            source_tier=SourceTier.PRIMARY,
            canonical_source_identity="filing-accession:0001045810-26-000001",
        )
        for reference in refs
    ]
    verification_status = {
        ComparabilityStatus.COMPARABLE: VerificationStatus.SINGLE_SOURCE,
        ComparabilityStatus.NOT_COMPARABLE: VerificationStatus.NOT_COMPARABLE,
        ComparabilityStatus.MISSING: VerificationStatus.MISSING,
        ComparabilityStatus.DISCREPANCY: VerificationStatus.DISCREPANCY,
    }[status]
    return ComparableMetric(
        ticker=ticker,
        name=name,
        value=value,
        period_start=period_start,
        period_end=period_end,
        currency=currency,
        unit=unit,
        definition=definition,
        observation_id=financial_observation_id(
            ticker=ticker,
            name=name,
            period_start=period_start,
            period_end=period_end,
            currency=currency,
            unit=unit,
            definition=definition,
        ),
        source_refs=refs,
        source_provenance=source_provenance,
        verification_status=verification_status,
        status=status,
        limitation=limitation,
    )


def _package(
    ticker: str,
    *metrics: ComparableMetric,
) -> GuardedResearchPackage:
    filing_sources = [
        EvidenceChunk(
            id=reference.source_id,
            ticker=ticker,
            corpus_version=f"{ticker}-v1",
            content=f"Evidence for {reference.source_id}.",
            source_url=f"https://www.sec.gov/Archives/{reference.source_id}.htm",
            form="10-Q",
            filed_at=date(2026, 5, 20),
            accession_no="0001045810-26-000001",
            section="MD&A",
            raw_start=0,
            raw_end=20,
        )
        for metric in metrics
        for reference in metric.source_refs
        if reference.kind is SourceRefKind.FILING
    ]
    return GuardedResearchPackage(
        ticker=ticker,
        claims=[],
        financial_metrics=list(metrics),
        filing_sources=filing_sources,
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=SkillName.INDUSTRY_RESEARCH,
                    version="1.0.0",
                ),
            ),
            corpus_versions=(f"{ticker}-v1",) if filing_sources else (),
            evidence_cutoff_dates=(date(2026, 5, 20),) if filing_sources else (),
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            source_refs=tuple(
                reference
                for metric in metrics
                for reference in metric.source_refs
            ),
        ),
        evidence_dates=[],
        coverage="complete",
        information_gaps=[],
        guard_notes=[],
    )


@pytest.mark.parametrize(
    ("field", "updated"),
    [
        ("name", {"name": "Gross margin"}),
        ("period_start", {"period_start": date(2025, 1, 1)}),
        ("period_end", {"period_end": date(2026, 6, 30)}),
        ("currency", {"currency": "EUR"}),
        ("unit", {"unit": "percent"}),
        ("definition", {"definition": "Non-GAAP revenue"}),
    ],
)
def test_metric_mismatch_is_not_comparable(
    field: str,
    updated: dict[str, object],
) -> None:
    left = _metric(ticker="NVDA", source_id="sec-left")
    right = _metric(ticker="AMD", source_id="sec-right", **updated)

    comparison = compare_metric_pair(left, right)

    assert comparison.name == "Revenue"
    assert comparison.status is ComparabilityStatus.NOT_COMPARABLE
    assert comparison.delta is None
    assert comparison.limitation == f"{field} mismatch"


def test_metric_pair_marks_missing_without_delta() -> None:
    left = _metric(ticker="NVDA", source_id="sec-left")
    right = _metric(
        ticker="AMD",
        value=None,
        status=ComparabilityStatus.MISSING,
        limitation="Metric missing from retained evidence.",
    )

    comparison = compare_metric_pair(left, right)

    assert comparison.status is ComparabilityStatus.MISSING
    assert comparison.delta is None
    assert comparison.limitation == "Metric missing from retained evidence."


def test_metric_pair_preserves_discrepancy_without_averaging() -> None:
    left = _metric(
        ticker="NVDA",
        value=Decimal("44.1"),
        status=ComparabilityStatus.DISCREPANCY,
        limitation="Issuer and filing values differ.",
        source_id="sec-left",
    )
    right = _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-right")

    comparison = compare_metric_pair(left, right)

    assert comparison.status is ComparabilityStatus.DISCREPANCY
    assert comparison.delta is None
    assert comparison.limitation == "Issuer and filing values differ."


def test_metric_pair_preserves_guarded_observation_and_source_identity() -> None:
    """Peer comparison must not flatten canonical observation or source provenance."""
    left = _metric(ticker="NVDA", source_id="sec-left")
    right = _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-right")

    comparison = compare_metric_pair(left, right)

    assert [item.observation_id for item in comparison.observations] == [
        left.observation_id,
        right.observation_id,
    ]
    assert [
        source.source_tier.value
        for item in comparison.observations
        for source in item.source_provenance
    ] == ["primary", "primary"]
    assert [
        source.canonical_source_identity
        for item in comparison.observations
        for source in item.source_provenance
    ] == [
        "filing-accession:0001045810-26-000001",
        "filing-accession:0001045810-26-000001",
    ]


def test_metric_pair_uses_canonical_nfkc_case_and_whitespace_identity() -> None:
    """Presentation-only identity differences must not turn exact peers non-comparable."""
    left = _metric(ticker="NVDA", source_id="sec-left")
    right = _metric(
        ticker="AMD",
        name="  ＲＥＶＥＮＵＥ  ",
        value=Decimal("32.0"),
        unit=" BILLIONS ",
        definition=" gaap   REVENUE ",
        source_id="sec-right",
    )

    comparison = compare_metric_pair(left, right)

    assert comparison.status is ComparabilityStatus.COMPARABLE
    assert comparison.delta == Decimal("12.1")
    assert comparison.observations == [left, right]


def test_same_value_source_provenance_differences_are_not_discrepancies() -> None:
    """Two sources for one exact value establish evidence rather than a numeric conflict."""
    primary_observations = [
        _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
        _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-b"),
    ]
    peer = _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-c")

    comparisons = compare_metrics(
        [
            _package("NVDA", *primary_observations),
            _package("AMD", peer),
        ]
    )

    assert len(comparisons) == 1
    assert comparisons[0].status is ComparabilityStatus.COMPARABLE
    assert comparisons[0].delta == Decimal("12.1")
    assert comparisons[0].observations == [*primary_observations, peer]


def test_compare_metrics_reports_conflicting_same_identity_as_discrepancy() -> None:
    packages = [
        _package(
            "NVDA",
            _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
            _metric(ticker="NVDA", value=Decimal("44.2"), source_id="sec-b"),
        ),
        _package("AMD", _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-c")),
    ]

    comparison = compare_metrics(packages)[0]

    assert comparison.name == "Revenue"
    assert comparison.status is ComparabilityStatus.DISCREPANCY
    assert comparison.delta is None
    assert len(comparison.observations) == 3
    assert comparison.limitation == "conflicting observations for the same metric identity"


def test_primary_conflict_stays_discrepancy_when_peer_metric_is_missing() -> None:
    comparison = compare_metrics(
        [
            _package(
                "NVDA",
                _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
                _metric(ticker="NVDA", value=Decimal("44.2"), source_id="sec-b"),
            ),
            _package(
                "AMD",
                _metric(
                    ticker="AMD",
                    name="Gross margin",
                    value=Decimal("52.0"),
                    unit="percent",
                    definition="GAAP gross margin",
                    source_id="sec-c",
                ),
            ),
        ]
    )[0]

    assert comparison.status is ComparabilityStatus.DISCREPANCY
    assert comparison.delta is None
    assert comparison.observations == [
        _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
        _metric(ticker="NVDA", value=Decimal("44.2"), source_id="sec-b"),
    ]


def test_primary_conflict_stays_discrepancy_when_peer_has_same_name_different_identity() -> None:
    quarter_conflicts = [
        _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
        _metric(ticker="NVDA", value=Decimal("44.2"), source_id="sec-b"),
    ]
    annual_peer = _metric(
        ticker="AMD",
        value=Decimal("118.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-c",
    )

    comparison = compare_metrics(
        [
            _package("NVDA", *quarter_conflicts),
            _package("AMD", annual_peer),
        ]
    )[0]

    assert comparison.status is ComparabilityStatus.DISCREPANCY
    assert comparison.delta is None
    assert comparison.observations == quarter_conflicts


def test_peer_exact_conflict_is_discrepancy_without_delta() -> None:
    comparison = compare_metrics(
        [
            _package(
                "NVDA",
                _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
            ),
            _package(
                "AMD",
                _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-b"),
                _metric(ticker="AMD", value=Decimal("32.1"), source_id="sec-c"),
            ),
        ]
    )[0]

    assert comparison.status is ComparabilityStatus.DISCREPANCY
    assert comparison.delta is None
    assert comparison.observations == [
        _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
        _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-b"),
        _metric(ticker="AMD", value=Decimal("32.1"), source_id="sec-c"),
    ]


def test_compare_metrics_computes_delta_only_for_exact_match() -> None:
    comparison = compare_metrics(
        [
            _package("NVDA", _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a")),
            _package("AMD", _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-b")),
        ]
    )[0]

    assert comparison == MetricComparison(
        name="Revenue",
        observations=[
            _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-a"),
            _metric(ticker="AMD", value=Decimal("32.0"), source_id="sec-b"),
        ],
        status=ComparabilityStatus.COMPARABLE,
        delta=Decimal("12.1"),
        limitation=None,
    )


def test_compare_metrics_tracks_same_name_metrics_by_full_identity() -> None:
    annual_nvda = _metric(
        ticker="NVDA",
        value=Decimal("130.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-nvda-annual",
    )
    quarter_nvda = _metric(
        ticker="NVDA",
        value=Decimal("44.1"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-nvda-quarter",
    )
    quarter_amd = _metric(
        ticker="AMD",
        value=Decimal("32.0"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-amd-quarter",
    )
    annual_amd = _metric(
        ticker="AMD",
        value=Decimal("118.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-amd-annual",
    )

    comparisons = compare_metrics(
        [
            _package("NVDA", quarter_nvda, annual_nvda),
            _package("AMD", annual_amd, quarter_amd),
        ]
    )

    assert comparisons == [
        MetricComparison(
            name="Revenue",
            observations=[annual_nvda, annual_amd],
            status=ComparabilityStatus.COMPARABLE,
            delta=Decimal("12.0"),
            limitation=None,
        ),
        MetricComparison(
            name="Revenue",
            observations=[quarter_nvda, quarter_amd],
            status=ComparabilityStatus.COMPARABLE,
            delta=Decimal("12.1"),
            limitation=None,
        ),
    ]


def test_compare_metrics_same_name_without_exact_identity_is_not_comparable() -> None:
    quarter_nvda = _metric(
        ticker="NVDA",
        value=Decimal("44.1"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-nvda-quarter",
    )
    annual_amd = _metric(
        ticker="AMD",
        value=Decimal("118.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-amd-annual",
    )

    comparison = compare_metrics(
        [
            _package("NVDA", quarter_nvda),
            _package("AMD", annual_amd),
        ]
    )[0]

    assert comparison.status is ComparabilityStatus.NOT_COMPARABLE
    assert comparison.delta is None
    assert comparison.observations == [quarter_nvda, annual_amd]
    assert "exact identity" in comparison.limitation


def test_compare_metrics_missing_same_name_metric_is_missing() -> None:
    comparison = compare_metrics(
        [
            _package(
                "NVDA",
                _metric(ticker="NVDA", value=Decimal("44.1"), source_id="sec-nvda"),
            ),
            _package(
                "AMD",
                _metric(
                    ticker="AMD",
                    name="Gross margin",
                    value=Decimal("52.0"),
                    unit="percent",
                    definition="GAAP gross margin",
                    source_id="sec-amd",
                ),
            ),
        ]
    )[0]

    assert comparison.status is ComparabilityStatus.MISSING
    assert comparison.delta is None
    assert [observation.ticker for observation in comparison.observations] == ["NVDA", "AMD"]
    assert comparison.observations[1].status is ComparabilityStatus.MISSING


def test_compare_metrics_is_independent_of_metric_order_with_same_name_periods() -> None:
    annual_nvda = _metric(
        ticker="NVDA",
        value=Decimal("130.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-nvda-annual",
    )
    quarter_nvda = _metric(
        ticker="NVDA",
        value=Decimal("44.1"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-nvda-quarter",
    )
    annual_amd = _metric(
        ticker="AMD",
        value=Decimal("118.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-amd-annual",
    )
    quarter_amd = _metric(
        ticker="AMD",
        value=Decimal("32.0"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-amd-quarter",
    )

    first = compare_metrics(
        [
            _package("NVDA", annual_nvda, quarter_nvda),
            _package("AMD", quarter_amd, annual_amd),
        ]
    )
    second = compare_metrics(
        [
            _package("NVDA", quarter_nvda, annual_nvda),
            _package("AMD", annual_amd, quarter_amd),
        ]
    )

    assert first == second


def test_unmatched_peer_same_name_conflicts_are_surfaced_deterministically() -> None:
    quarter_nvda = _metric(
        ticker="NVDA",
        value=Decimal("44.1"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-nvda-quarter",
    )
    quarter_amd = _metric(
        ticker="AMD",
        value=Decimal("32.0"),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        source_id="sec-amd-quarter",
    )
    annual_amd_conflicts = [
        _metric(
            ticker="AMD",
            value=Decimal("118.0"),
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            source_id="sec-amd-annual-a",
        ),
        _metric(
            ticker="AMD",
            value=Decimal("119.0"),
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            source_id="sec-amd-annual-b",
        ),
    ]

    first = compare_metrics(
        [
            _package("NVDA", quarter_nvda),
            _package("AMD", quarter_amd, *annual_amd_conflicts),
        ]
    )
    second = compare_metrics(
        [
            _package("NVDA", quarter_nvda),
            _package("AMD", *reversed(annual_amd_conflicts), quarter_amd),
        ]
    )

    assert first == second
    assert first == [
        MetricComparison(
            name="Revenue",
            observations=[quarter_nvda, quarter_amd],
            status=ComparabilityStatus.COMPARABLE,
            delta=Decimal("12.1"),
            limitation=None,
        ),
        MetricComparison(
            name="Revenue",
            observations=[
                _metric(
                    ticker="NVDA",
                    value=None,
                    period_start=date(2025, 1, 1),
                    period_end=date(2025, 12, 31),
                    status=ComparabilityStatus.MISSING,
                    limitation="NVDA missing retained metric for Revenue",
                ),
                annual_amd_conflicts[0],
                annual_amd_conflicts[1],
            ],
            status=ComparabilityStatus.DISCREPANCY,
            delta=None,
            limitation="conflicting observations for the same metric identity",
        ),
    ]


def test_primary_identities_do_not_consume_later_matching_peer_conflict() -> None:
    fy2024_nvda = _metric(
        ticker="NVDA",
        value=Decimal("100.0"),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        source_id="sec-nvda-fy2024",
    )
    fy2025_nvda = _metric(
        ticker="NVDA",
        value=Decimal("120.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-nvda-fy2025",
    )
    fy2025_amd_conflicts = [
        _metric(
            ticker="AMD",
            value=Decimal("90.0"),
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            source_id="sec-amd-fy2025-a",
        ),
        _metric(
            ticker="AMD",
            value=Decimal("91.0"),
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            source_id="sec-amd-fy2025-b",
        ),
    ]

    first = compare_metrics(
        [
            _package("NVDA", fy2024_nvda, fy2025_nvda),
            _package("AMD", *fy2025_amd_conflicts),
        ]
    )
    second = compare_metrics(
        [
            _package("NVDA", fy2025_nvda, fy2024_nvda),
            _package("AMD", *reversed(fy2025_amd_conflicts)),
        ]
    )

    assert first == second
    assert first == [
        MetricComparison(
            name="Revenue",
            observations=[fy2024_nvda, *fy2025_amd_conflicts],
            status=ComparabilityStatus.NOT_COMPARABLE,
            delta=None,
            limitation=(
                "same-name peer metric did not match the exact identity "
                "(name, period_start, period_end, currency, unit, definition)"
            ),
        ),
        MetricComparison(
            name="Revenue",
            observations=[fy2025_nvda, *fy2025_amd_conflicts],
            status=ComparabilityStatus.DISCREPANCY,
            delta=None,
            limitation="conflicting observations for the same metric identity",
        ),
    ]


def test_unmatched_peer_conflicting_identity_adds_one_extra_discrepancy_only() -> None:
    fy2024_nvda = _metric(
        ticker="NVDA",
        value=Decimal("100.0"),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        source_id="sec-nvda-fy2024",
    )
    fy2025_nvda = _metric(
        ticker="NVDA",
        value=Decimal("120.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-nvda-fy2025",
    )
    fy2024_amd = _metric(
        ticker="AMD",
        value=Decimal("80.0"),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        source_id="sec-amd-fy2024",
    )
    fy2025_amd = _metric(
        ticker="AMD",
        value=Decimal("95.0"),
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        source_id="sec-amd-fy2025",
    )
    fy2026_amd_conflicts = [
        _metric(
            ticker="AMD",
            value=Decimal("105.0"),
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            source_id="sec-amd-fy2026-a",
        ),
        _metric(
            ticker="AMD",
            value=Decimal("106.0"),
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            source_id="sec-amd-fy2026-b",
        ),
    ]

    first = compare_metrics(
        [
            _package("NVDA", fy2024_nvda, fy2025_nvda),
            _package("AMD", fy2025_amd, *fy2026_amd_conflicts, fy2024_amd),
        ]
    )
    second = compare_metrics(
        [
            _package("NVDA", fy2025_nvda, fy2024_nvda),
            _package("AMD", fy2024_amd, *reversed(fy2026_amd_conflicts), fy2025_amd),
        ]
    )

    assert first == second
    assert first == [
        MetricComparison(
            name="Revenue",
            observations=[fy2024_nvda, fy2024_amd],
            status=ComparabilityStatus.COMPARABLE,
            delta=Decimal("20.0"),
            limitation=None,
        ),
        MetricComparison(
            name="Revenue",
            observations=[fy2025_nvda, fy2025_amd],
            status=ComparabilityStatus.COMPARABLE,
            delta=Decimal("25.0"),
            limitation=None,
        ),
        MetricComparison(
            name="Revenue",
            observations=[
                _metric(
                    ticker="NVDA",
                    value=None,
                    period_start=date(2026, 1, 1),
                    period_end=date(2026, 12, 31),
                    status=ComparabilityStatus.MISSING,
                    limitation="NVDA missing retained metric for Revenue",
                ),
                fy2026_amd_conflicts[0],
                fy2026_amd_conflicts[1],
            ],
            status=ComparabilityStatus.DISCREPANCY,
            delta=None,
            limitation="conflicting observations for the same metric identity",
        ),
    ]
