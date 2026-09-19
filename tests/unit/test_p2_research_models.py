"""Strict contracts for P2 guarded research packages and comparisons."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fra.domain import (
    ClaimKind,
    Confidence,
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
)
from fra.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    MetricComparison,
    MultiTickerResearchPackage,
    PackageClaim,
    PeerResearchRequest,
    ResearchQualityDecision,
    ResearchQualityResult,
)
from fra.skills.models import ResearchFacet, SkillName
from fra.skills.schemas import (
    FinancialSourceProvenance,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
    financial_observation_id,
)
from fra.web_evidence.source_policy import PolicyValidatedWebEvidence


def _ref(
    source_id: str = "sec-1",
    *,
    ticker: str = "NVDA",
    kind: SourceRefKind = SourceRefKind.FILING,
) -> SourceRef:
    return SourceRef(ticker=ticker, kind=kind, source_id=source_id)


def _filing(source_id: str = "sec-1", *, ticker: str = "NVDA") -> EvidenceChunk:
    return EvidenceChunk(
        id=source_id,
        ticker=ticker,
        corpus_version="fixture-v1",
        content=f"Filing evidence for {source_id}.",
        source_url=f"https://www.sec.gov/Archives/{source_id}",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        accession_no="0001045810-25-000041",
        section="MD&A",
        raw_start=0,
        raw_end=40,
    )


def _web(source_id: str = "web-1", *, ticker: str = "NVDA") -> PolicyValidatedWebEvidence:
    return PolicyValidatedWebEvidence(
        id=source_id,
        ticker=ticker,
        title="Issuer results",
        content=f"Web evidence for {source_id}.",
        source_url=f"https://investor.nvidia.com/{source_id}",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2025, 5, 28, 20, 0, tzinfo=UTC),
        fetched_at=datetime(2025, 5, 29, 9, 30, tzinfo=UTC),
        content_hash=f"sha256:{source_id}",
        policy_version="policy-v1",
        canonical_url=f"https://investor.nvidia.com/{source_id}",
    )


def _package_claim(
    *,
    kind: ClaimKind = ClaimKind.VERIFIED_FACT,
    source_refs: list[SourceRef] | None = None,
) -> PackageClaim:
    return PackageClaim(
        facet=ResearchFacet.INDUSTRY_SCOPE,
        kind=kind,
        text="AI accelerators remain the key demand driver.",
        confidence=Confidence.HIGH,
        source_refs=[] if source_refs is None else source_refs,
    )


def _metric(
    *,
    ticker: str = "NVDA",
    source_refs: list[SourceRef] | None = None,
    status: ComparabilityStatus = ComparabilityStatus.COMPARABLE,
    limitation: str | None = None,
) -> ComparableMetric:
    refs = [_ref()] if source_refs is None else source_refs
    provenance = [
        FinancialSourceProvenance(
            source_ref=reference,
            source_kind=(
                SourceKind.FILING
                if reference.kind is SourceRefKind.FILING
                else SourceKind.ISSUER_IR
            ),
            source_tier=SourceTier.PRIMARY,
            canonical_source_identity=(
                "filing-accession:0001045810-25-000041"
                if reference.kind is SourceRefKind.FILING
                else f"web-url:https://investor.nvidia.com/{reference.source_id}"
            ),
        )
        for reference in refs
    ]
    return ComparableMetric(
        ticker=ticker,
        name="Revenue",
        value=Decimal("44.062"),
        period_start=date(2025, 1, 27),
        period_end=date(2025, 4, 27),
        currency="USD",
        unit="billions",
        definition="GAAP revenue",
        observation_id=financial_observation_id(
            ticker=ticker,
            name="Revenue",
            period_start=date(2025, 1, 27),
            period_end=date(2025, 4, 27),
            currency="USD",
            unit="billions",
            definition="GAAP revenue",
        ),
        source_refs=refs,
        source_provenance=provenance,
        verification_status={
            ComparabilityStatus.COMPARABLE: VerificationStatus.SINGLE_SOURCE,
            ComparabilityStatus.NOT_COMPARABLE: VerificationStatus.NOT_COMPARABLE,
            ComparabilityStatus.MISSING: VerificationStatus.MISSING,
            ComparabilityStatus.DISCREPANCY: VerificationStatus.DISCREPANCY,
        }[status],
        status=status,
        limitation=limitation,
    )


def _package(**updates: object) -> GuardedResearchPackage:
    values: dict[str, object] = {
        "ticker": "NVDA",
        "claims": [_package_claim(source_refs=[_ref()])],
        "financial_metrics": [_metric()],
        "filing_sources": [_filing()],
        "web_sources": [_web()],
        "evidence_dates": [date(2025, 5, 28)],
        "coverage": "partial",
        "information_gaps": ["Competitor share data remains sparse."],
        "guard_notes": [],
    }
    values.update(updates)
    if "provenance" not in updates:
        ticker = str(values["ticker"])
        filings = list(values["filing_sources"])
        web_sources = list(values["web_sources"])
        source_refs = tuple(
            [
                *(
                    SourceRef(
                        ticker=ticker,
                        kind=SourceRefKind.FILING,
                        source_id=source.id,
                    )
                    for source in filings
                ),
                *(
                    SourceRef(
                        ticker=ticker,
                        kind=SourceRefKind.WEB,
                        source_id=source.id,
                    )
                    for source in web_sources
                ),
            ]
        )
        values["provenance"] = ReportProvenance.model_construct(
            recipes=(
                RecipeProvenance(
                    name=SkillName.INDUSTRY_RESEARCH,
                    version="1.0.0",
                ),
            ),
            source_policy_versions=("policy-v1",),
            corpus_versions=tuple(
                dict.fromkeys(source.corpus_version for source in filings)
            ),
            prompt_versions=("research-v2",),
            evidence_cutoff_dates=(date(2025, 5, 28),),
            information_sufficiency={
                "complete": InformationSufficiency.SUFFICIENT,
                "partial": InformationSufficiency.PARTIAL,
                "insufficient": InformationSufficiency.INSUFFICIENT,
            }[values["coverage"]],
            source_refs=source_refs,
        )
    return GuardedResearchPackage(**values)


def _package_for_ticker(ticker: str) -> GuardedResearchPackage:
    return _package(
        ticker=ticker,
        claims=[_package_claim(source_refs=[_ref(ticker=ticker)])],
        financial_metrics=[_metric(ticker=ticker, source_refs=[_ref(ticker=ticker)])],
        filing_sources=[_filing(ticker=ticker)],
        web_sources=[_web(ticker=ticker)],
    )


def _refused_package(
    refusal_code: str = "prohibited_advice",
    **updates: object,
) -> GuardedResearchPackage:
    values: dict[str, object] = {
        "claims": [],
        "financial_metrics": [],
        "filing_sources": [],
        "web_sources": [],
        "provenance": ReportProvenance(
            recipes=(),
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
        ),
        "zero_recipe_outcome": "refused",
        "evidence_dates": [],
        "coverage": "insufficient",
        "information_gaps": [],
        "guard_notes": [refusal_code],
    }
    values.update(updates)
    return _package(**values)


def test_source_ref_reuses_existing_namespaced_contract() -> None:
    """P2 packages must reuse the shared ticker/kind/id encoding from the domain layer."""
    ref = _ref("chunk-1")

    assert ref.encode() == "NVDA:filing:chunk-1"

    with pytest.raises(ValidationError):
        SourceRef(ticker="", kind=SourceRefKind.FILING, source_id="chunk-1")


def test_package_claim_requires_namespaced_sources_for_verified_facts() -> None:
    """A guarded factual package claim cannot survive without at least one retained source ref."""
    with pytest.raises(ValidationError, match="verified facts require at least one source ref"):
        _package_claim(source_refs=[])


def test_comparable_metric_requires_limitation_for_non_comparable_status() -> None:
    """Missing comparability rationale would hide why a value cannot be compared."""
    with pytest.raises(ValidationError, match="limitation"):
        _metric(status=ComparabilityStatus.NOT_COMPARABLE, limitation=None)


def test_guarded_package_rejects_cross_ticker_and_absent_retained_source_refs() -> None:
    """Every package claim and metric must point only at retained sources for that ticker."""
    with pytest.raises(ValidationError, match="does not match package ticker"):
        _package(claims=[_package_claim(source_refs=[_ref(ticker="AMD")])])

    with pytest.raises(ValidationError, match="retained source set"):
        _package(
            claims=[_package_claim(source_refs=[_ref("web-missing", kind=SourceRefKind.WEB)])]
        )


def test_guarded_package_accepts_single_ticker_retained_sources() -> None:
    """A valid package keeps one ticker across claims, metrics, and retained evidence."""
    package = _package(
        claims=[
            _package_claim(
                source_refs=[_ref("sec-1"), _ref("web-1", kind=SourceRefKind.WEB)]
            )
        ],
        financial_metrics=[
            _metric(source_refs=[_ref("sec-1"), _ref("web-1", kind=SourceRefKind.WEB)])
        ],
    )

    assert package.ticker == "NVDA"
    assert package.claims[0].source_refs[1].kind is SourceRefKind.WEB
    assert [recipe.name for recipe in package.provenance.recipes] == [
        SkillName.INDUSTRY_RESEARCH
    ]


@pytest.mark.parametrize(
    "refusal_code",
    ["prohibited_advice", "prompt_injection", "unsafe_source_request"],
)
def test_guarded_package_allows_only_canonical_zero_recipe_refusal_shapes(
    refusal_code: str,
) -> None:
    package = _refused_package(refusal_code)

    assert package.provenance.recipes == ()
    assert package.coverage == "insufficient"
    assert package.information_gaps == []
    assert package.guard_notes == [refusal_code]


@pytest.mark.parametrize(
    ("updates", "pattern"),
    [
        ({"coverage": "complete"}, "zero-recipe packages must use insufficient coverage"),
        ({"coverage": "partial"}, "zero-recipe packages must use insufficient coverage"),
        (
            {"information_gaps": ["prohibited_advice"]},
            "zero-recipe packages cannot report information gaps",
        ),
        (
            {"information_gaps": ["arbitrary gap"]},
            "zero-recipe packages cannot report information gaps",
        ),
        (
            {"guard_notes": []},
            "zero-recipe refusals must encode exactly one approved refusal code",
        ),
        (
            {"guard_notes": ["prohibited_advice", "prompt_injection"]},
            "zero-recipe refusals must encode exactly one approved refusal code",
        ),
        (
            {"guard_notes": ["arbitrary-note"]},
            "zero-recipe refusals must encode exactly one approved refusal code",
        ),
        (
            {"evidence_dates": [date(2025, 5, 28)]},
            "packages without executed recipes cannot retain evidence",
        ),
        (
            {"claims": [_package_claim(source_refs=[_ref()])]},
            "packages without executed recipes cannot retain evidence",
        ),
        (
            {"financial_metrics": [_metric()]},
            "packages without executed recipes cannot retain evidence",
        ),
        (
            {"filing_sources": [_filing()]},
            "packages without executed recipes cannot retain evidence",
        ),
        (
            {"web_sources": [_web()]},
            "packages without executed recipes cannot retain evidence",
        ),
    ],
)
def test_guarded_package_rejects_invalid_zero_recipe_shapes(
    updates: dict[str, object],
    pattern: str,
) -> None:
    with pytest.raises(ValidationError, match=pattern):
        _refused_package(**updates)


def test_guarded_package_rejects_duplicate_retained_source_identities_per_namespace() -> None:
    """Retained filing/web source lists must not contain duplicate namespaced source identities."""
    with pytest.raises(ValidationError, match="unique and ordered|duplicate filing source"):
        _package(
            claims=[_package_claim(source_refs=[_ref("dup")])],
            financial_metrics=[_metric(source_refs=[_ref("dup")])],
            filing_sources=[_filing("dup"), _filing("dup")],
            web_sources=[],
        )

    with pytest.raises(ValidationError, match="unique and ordered|duplicate web source"):
        _package(
            claims=[_package_claim(source_refs=[_ref("dup", kind=SourceRefKind.WEB)])],
            financial_metrics=[_metric(source_refs=[_ref("dup", kind=SourceRefKind.WEB)])],
            filing_sources=[],
            web_sources=[_web("dup"), _web("dup")],
        )


def test_guarded_package_allows_same_raw_id_in_different_retained_namespaces() -> None:
    """The same raw source ID may appear once in filings and once in web because the refs differ."""
    package = _package(
        claims=[
            _package_claim(
                source_refs=[_ref("shared"), _ref("shared", kind=SourceRefKind.WEB)]
            )
        ],
        financial_metrics=[
            _metric(
                source_refs=[_ref("shared"), _ref("shared", kind=SourceRefKind.WEB)]
            )
        ],
        filing_sources=[_filing("shared")],
        web_sources=[_web("shared")],
    )

    assert [reference.encode() for reference in package.claims[0].source_refs] == [
        "NVDA:filing:shared",
        "NVDA:web:shared",
    ]


def test_metric_comparison_and_quality_result_use_strict_p2_status_enums() -> None:
    """Comparison and quality results must stay within the fixed deterministic outcome set."""
    comparison = MetricComparison(
        name="Revenue",
        observations=[_metric()],
        status=ComparabilityStatus.COMPARABLE,
        delta=Decimal("1.0"),
    )
    result = ResearchQualityResult(
        decision=ResearchQualityDecision.WORTH_FURTHER_RESEARCH,
        reasons=["Coverage spans both supporting and counterevidence."],
        source_refs=[_ref()],
    )

    assert comparison.status is ComparabilityStatus.COMPARABLE
    assert result.decision is ResearchQualityDecision.WORTH_FURTHER_RESEARCH


def test_peer_request_and_multi_ticker_package_enforce_bounded_scope() -> None:
    """Peer inputs stay explicit, bounded, and free of duplicate package tickers."""
    request = PeerResearchRequest(
        primary_ticker="nvda",
        peer_tickers=("amd", "avgo"),
        peer_scope="Datacenter accelerators",
        question="How do margins differ across direct peers?",
    )

    assert request.primary_ticker == "NVDA"
    assert request.peer_tickers == ("AMD", "AVGO")

    with pytest.raises(ValidationError, match="primary ticker cannot appear in peer_tickers"):
        PeerResearchRequest(
            primary_ticker="NVDA",
            peer_tickers=("NVDA",),
            peer_scope="Datacenter accelerators",
            question="Invalid scope",
        )

    with pytest.raises(ValidationError, match="duplicate package ticker"):
        MultiTickerResearchPackage(
            primary_ticker="NVDA",
            packages=[_package(), _package()],
            missing_tickers=[],
            status="partial",
        )


def test_multi_ticker_completed_allows_exactly_one_primary_package() -> None:
    """Completed or partial multi-ticker results must include the primary package exactly once."""
    result = MultiTickerResearchPackage(
        primary_ticker="NVDA",
        packages=[_package_for_ticker("NVDA"), _package_for_ticker("AMD")],
        missing_tickers=[],
        status="completed",
    )

    assert [package.ticker for package in result.packages] == ["NVDA", "AMD"]


def test_multi_ticker_rejects_completed_or_partial_peer_only_packages() -> None:
    """A completed or partial result cannot omit the primary package and keep only peers."""
    with pytest.raises(ValidationError, match="exactly one primary package"):
        MultiTickerResearchPackage(
            primary_ticker="NVDA",
            packages=[_package_for_ticker("AMD")],
            missing_tickers=[],
            status="partial",
        )


def test_multi_ticker_failed_accepts_missing_primary_without_packages() -> None:
    """A failed result is valid only when the missing primary ticker explains the failure."""
    result = MultiTickerResearchPackage(
        primary_ticker="NVDA",
        packages=[],
        missing_tickers=["NVDA"],
        status="failed",
    )

    assert result.status == "failed"
    assert result.missing_tickers == ["NVDA"]


def test_multi_ticker_failed_rejects_peer_only_packages() -> None:
    """A failed result cannot carry only peer packages once the primary is missing."""
    with pytest.raises(
        ValidationError,
        match="failed multi-ticker package cannot carry peer-only packages",
    ):
        MultiTickerResearchPackage(
            primary_ticker="NVDA",
            packages=[_package_for_ticker("AMD")],
            missing_tickers=["NVDA"],
            status="failed",
        )
