"""Unit coverage for the unified P2 renderer."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from financial_evidence_agent.domain import (
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
)
from financial_evidence_agent.reporting.p2_guard import GuardedP2Report
from financial_evidence_agent.reporting.p2_render import render_p2_markdown
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    MetricComparison,
    PackageClaim,
    PeerScope,
    ResearchQualityDecision,
    ResearchQualityResult,
    ResolvedSource,
)
from financial_evidence_agent.skills.models import ResearchFacet, SkillName
from financial_evidence_agent.skills.schemas import (
    FinancialSourceProvenance,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
    financial_observation_id,
)


def _ref(source_id: str, *, ticker: str, kind: SourceRefKind = SourceRefKind.FILING) -> SourceRef:
    return SourceRef(ticker=ticker, kind=kind, source_id=source_id)


def _claim(ticker: str, source_id: str, *, text: str) -> PackageClaim:
    return PackageClaim.model_construct(
        facet=ResearchFacet.INDUSTRY_SCOPE,
        kind="verified_fact",
        text=text,
        confidence="high",
        source_refs=[_ref(source_id, ticker=ticker)],
    )


def _metric(
    ticker: str,
    source_id: str,
    *,
    name: str = "Revenue",
    value: str | None = "44.062",
    status: ComparabilityStatus = ComparabilityStatus.COMPARABLE,
    limitation: str | None = None,
    unit: str = "billions",
    definition: str = "GAAP revenue",
) -> ComparableMetric:
    source_refs = (
        [_ref(source_id, ticker=ticker)]
        if status is not ComparabilityStatus.MISSING
        else []
    )
    source_provenance = [
        FinancialSourceProvenance(
            source_ref=reference,
            source_kind=SourceKind.FILING,
            source_tier=SourceTier.PRIMARY,
            canonical_source_identity="filing-accession:0001045810-26-000001",
        )
        for reference in source_refs
    ]
    return ComparableMetric(
        ticker=ticker,
        name=name,
        value=None if value is None else Decimal(value),
        period_start=date(2026, 2, 1),
        period_end=date(2026, 4, 30),
        currency="USD",
        unit=unit,
        definition=definition,
        observation_id=financial_observation_id(
            ticker=ticker,
            name=name,
            period_start=date(2026, 2, 1),
            period_end=date(2026, 4, 30),
            currency="USD",
            unit=unit,
            definition=definition,
        ),
        source_refs=source_refs,
        source_provenance=source_provenance,
        verification_status={
            ComparabilityStatus.COMPARABLE: VerificationStatus.SINGLE_SOURCE,
            ComparabilityStatus.NOT_COMPARABLE: VerificationStatus.NOT_COMPARABLE,
            ComparabilityStatus.MISSING: VerificationStatus.MISSING,
            ComparabilityStatus.DISCREPANCY: VerificationStatus.DISCREPANCY,
        }[status],
        status=status,
        limitation=limitation,
    )


def _package(
    ticker: str,
    *,
    claims: list[PackageClaim],
    metrics: list[ComparableMetric],
    information_gaps: list[str] | None = None,
) -> GuardedResearchPackage:
    source_ids = {
        reference.source_id
        for claim in claims
        for reference in claim.source_refs
    }
    source_ids.update(
        reference.source_id
        for metric in metrics
        for reference in metric.source_refs
    )
    return GuardedResearchPackage.model_construct(
        ticker=ticker,
        claims=claims,
        financial_metrics=metrics,
        filing_sources=[
            EvidenceChunk(
                id=source_id,
                ticker=ticker,
                corpus_version=f"{ticker}-v1",
                content=f"{ticker} filing evidence for {source_id}.",
                source_url=f"https://www.sec.gov/Archives/{source_id}.htm",
                form="10-Q",
                filed_at=date(2026, 5, 20),
                accession_no="0001045810-26-000001",
                section="MD&A",
                raw_start=0,
                raw_end=20,
            )
            for source_id in sorted(source_ids)
        ],
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=(
                        SkillName.INDUSTRY_RESEARCH
                        if ticker == "NVDA"
                        else SkillName.FINANCIAL_DATA_VERIFICATION
                    ),
                    version="1.0.0",
                ),
            ),
            source_policy_versions=("policy-v3",),
            corpus_versions=(f"{ticker}-v1",) if source_ids else (),
            prompt_versions=("research-v2",),
            requested_as_of_dates=(date(2026, 5, 31),),
            evidence_cutoff_dates=(date(2026, 5, 20),),
            information_sufficiency=InformationSufficiency.PARTIAL,
            source_refs=tuple(
                _ref(source_id, ticker=ticker) for source_id in sorted(source_ids)
            ),
        ),
        evidence_dates=[date(2026, 5, 20)],
        coverage="partial",
        information_gaps=[] if information_gaps is None else information_gaps,
        guard_notes=[],
    )


def _comparison(
    name: str,
    observations: list[ComparableMetric],
    *,
    status: ComparabilityStatus,
    delta: str | None = None,
    limitation: str | None = None,
) -> MetricComparison:
    return MetricComparison(
        name=name,
        observations=observations,
        status=status,
        delta=None if delta is None else Decimal(delta),
        limitation=limitation,
    )


def _source(ref: SourceRef, *, title: str, url: str, published: datetime | date) -> ResolvedSource:
    return ResolvedSource(
        ref=ref,
        title=title,
        source_url=url,
        published_or_filed_at=published,
        source_kind=(
            SourceKind.FILING
            if ref.kind is SourceRefKind.FILING
            else SourceKind.AUTHORITATIVE_WEB
        ),
        source_tier=(
            SourceTier.PRIMARY
            if ref.kind is SourceRefKind.FILING
            else SourceTier.AUTHORITATIVE_SECONDARY
        ),
        canonical_source_identity=(
            "filing-accession:0001045810-26-000001"
            if ref.kind is SourceRefKind.FILING
            else f"web-url:{url.split('#', maxsplit=1)[0]}"
        ),
    )


def _guarded_report(*, include_quality: bool = True, empty: bool = False) -> GuardedP2Report:
    if empty:
        return GuardedP2Report(
            scope=PeerScope(
                primary_ticker="NVDA",
                peer_tickers=("AMD",),
                description="US semiconductors",
            ),
            packages=[],
            comparisons=[],
            quality=None,
            retained_sources=[],
            provenance=ReportProvenance(
                recipes=(),
                information_sufficiency=InformationSufficiency.INSUFFICIENT,
            ),
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
            guard_errors=["PRIMARY_PACKAGE_MISSING"],
        )

    nvda_revenue = _metric("NVDA", "sec-nvda-revenue", value="44.062")
    amd_revenue = _metric("AMD", "sec-amd-revenue", value="5.835")
    nvda_margin = _metric(
        "NVDA",
        "sec-nvda-margin",
        name="Gross margin",
        value=None,
        status=ComparabilityStatus.NOT_COMPARABLE,
        limitation="unit mismatch",
        unit="percent",
    )
    amd_margin = _metric(
        "AMD",
        "sec-amd-margin",
        name="Gross margin",
        value=None,
        status=ComparabilityStatus.NOT_COMPARABLE,
        limitation="unit mismatch",
        unit="ratio",
    )
    nvda_inventory = _metric(
        "NVDA",
        "sec-nvda-inventory",
        name="Inventory days",
        value="82",
    )
    amd_inventory = _metric(
        "AMD",
        "sec-amd-inventory",
        name="Inventory days",
        value=None,
        status=ComparabilityStatus.MISSING,
        limitation="AMD missing retained metric for Inventory days",
    )
    nvda_eps = _metric(
        "NVDA",
        "sec-nvda-eps",
        name="Diluted EPS",
        value=None,
        status=ComparabilityStatus.DISCREPANCY,
        limitation="conflicting observations for the same metric identity",
        unit="USD/share",
        definition="GAAP diluted EPS",
    )
    amd_eps = _metric(
        "AMD",
        "sec-amd-eps",
        name="Diluted EPS",
        value=None,
        status=ComparabilityStatus.DISCREPANCY,
        limitation="conflicting observations for the same metric identity",
        unit="USD/share",
        definition="GAAP diluted EPS",
    )
    return GuardedP2Report(
        scope=PeerScope(
            primary_ticker="NVDA",
            peer_tickers=("AMD", "INTC"),
            description="US semiconductors & accelerators",
        ),
        packages=[
            _package(
                "NVDA",
                claims=[_claim("NVDA", "sec-nvda-claim", text="NVDA retained claim.")],
                metrics=[nvda_revenue],
                information_gaps=["Allowlisted web fallback unavailable."],
            ),
            _package(
                "AMD",
                claims=[_claim("AMD", "sec-amd-claim", text="AMD retained claim.")],
                metrics=[amd_revenue],
            ),
        ],
        comparisons=[
            _comparison(
                "Revenue",
                [nvda_revenue, amd_revenue],
                status=ComparabilityStatus.COMPARABLE,
                delta="38.227",
            ),
            _comparison(
                "Gross margin",
                [nvda_margin, amd_margin],
                status=ComparabilityStatus.NOT_COMPARABLE,
                limitation="unit mismatch",
            ),
            _comparison(
                "Inventory days",
                [nvda_inventory, amd_inventory],
                status=ComparabilityStatus.MISSING,
                limitation="AMD missing retained metric for Inventory days",
            ),
            _comparison(
                "Diluted EPS",
                [nvda_eps, amd_eps],
                status=ComparabilityStatus.DISCREPANCY,
                limitation="conflicting observations for the same metric identity",
            ),
        ],
        quality=(
            ResearchQualityResult(
                decision=ResearchQualityDecision.WORTH_FURTHER_RESEARCH,
                reasons=[
                    "Balanced supporting and counterevidence remain retained.",
                    (
                        "Comparable metrics cite NVDA:filing:sec-nvda-revenue and "
                        "AMD:filing:sec-amd-revenue."
                    ),
                ],
                source_refs=[
                    _ref("sec-nvda-revenue", ticker="NVDA"),
                    _ref("sec-amd-revenue", ticker="AMD"),
                ],
            )
            if include_quality
            else None
        ),
        retained_sources=[
            _source(
                _ref("sec-nvda-claim", ticker="NVDA"),
                title="NVDA MD&A [claim]",
                url="https://www.sec.gov/Archives/sec-nvda-claim.htm",
                published=date(2026, 5, 20),
            ),
            _source(
                _ref("sec-amd-claim", ticker="AMD"),
                title="AMD MD&A & outlook",
                url="https://www.sec.gov/Archives/sec-amd-claim.htm",
                published=date(2026, 5, 20),
            ),
            _source(
                _ref("ftc-web", ticker="NVDA", kind=SourceRefKind.WEB),
                title="FTC market [update]",
                url="https://www.ftc.gov/news-events/update?ref=1#fragment",
                published=datetime(2026, 5, 18, 15, 0, tzinfo=UTC),
            ),
        ],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=SkillName.INDUSTRY_RESEARCH,
                    version="1.0.0",
                ),
                RecipeProvenance(
                    name=SkillName.FINANCIAL_DATA_VERIFICATION,
                    version="1.0.0",
                ),
            ),
            source_policy_versions=("policy-v3",),
            corpus_versions=("NVDA-v1", "AMD-v1"),
            prompt_versions=("research-v2",),
            requested_as_of_dates=(date(2026, 5, 31),),
            evidence_cutoff_dates=(date(2026, 5, 20),),
            information_sufficiency=InformationSufficiency.PARTIAL,
            source_refs=(
                _ref("sec-nvda-claim", ticker="NVDA"),
                _ref("sec-amd-claim", ticker="AMD"),
                _ref("ftc-web", ticker="NVDA", kind=SourceRefKind.WEB),
            ),
        ),
        information_sufficiency=InformationSufficiency.PARTIAL,
        guard_errors=["PEER_PACKAGE_MISSING: INTC"],
    )


def _headings(markdown: str) -> list[str]:
    return [line.removeprefix("## ") for line in markdown.splitlines() if line.startswith("## ")]


def test_render_p2_markdown_has_fixed_sections_one_disclaimer_and_quality_subsection() -> None:
    markdown = render_p2_markdown(_guarded_report())

    assert markdown.count("Research assistance only; not investment advice.") == 1
    assert _headings(markdown) == [
        "Scope",
        "Reproducibility",
        "Per-company evidence",
        "Peer comparison",
        "Industry evidence",
        "Comparability limits",
        "Information sufficiency",
        "Sources",
        "Guard notes",
    ]
    assert "### Research quality" in markdown
    assert "winner" not in markdown.casefold()
    assert "best stock" not in markdown.casefold()
    assert "portfolio" not in markdown.casefold()


def test_render_p2_markdown_aggregates_multi_package_reproducibility_metadata() -> None:
    """P2 must expose every isolated package recipe/corpus without losing ticker namespaces."""
    report = render_p2_markdown(_guarded_report())

    assert "## Reproducibility" in report
    assert "- Recipe: industry\\_research @ 1.0.0" in report
    assert "- Recipe: financial\\_data\\_verification @ 1.0.0" in report
    assert "- Source policy version: policy-v3" in report
    assert "- Corpus version: NVDA-v1" in report
    assert "- Corpus version: AMD-v1" in report
    assert "- Prompt version: research-v2" in report
    assert "- Requested as-of: 2026-05-31" in report
    assert "- Retained evidence cutoff: 2026-05-20" in report
    assert "- Information sufficiency: partial" in report
    assert "- Exact retained source: NVDA:filing:sec-nvda-claim" in report
    assert "- Exact retained source: AMD:filing:sec-amd-claim" in report


def test_render_p2_markdown_shows_all_comparison_statuses_without_rank_or_invalid_delta() -> None:
    markdown = render_p2_markdown(_guarded_report(include_quality=False))

    assert "Revenue" in markdown
    assert "Status: comparable" in markdown
    assert "Delta: 38.227" in markdown
    assert "Gross margin" in markdown
    assert "Status: not\\_comparable" in markdown
    assert "Inventory days" in markdown
    assert "Status: missing" in markdown
    assert "Diluted EPS" in markdown
    assert "Status: discrepancy" in markdown
    assert markdown.count("Delta:") == 1
    assert "average" not in markdown.casefold()
    assert "rank" not in markdown.casefold()


def test_render_p2_markdown_escapes_text_canonicalizes_links_and_handles_empty_report() -> None:
    full_markdown = render_p2_markdown(_guarded_report())
    empty_markdown = render_p2_markdown(_guarded_report(empty=True))

    assert "US semiconductors \\& accelerators" in full_markdown
    assert "NVDA MD\\&A \\[claim\\]" in full_markdown
    assert "FTC market \\[update\\]" in full_markdown
    assert "<https://www.ftc.gov/news-events/update?ref=1>" in full_markdown
    assert "INTC: insufficient evidence" in full_markdown
    assert "No retained company evidence." in empty_markdown
    assert "No retained peer comparisons." in empty_markdown
    assert "No retained source links." in empty_markdown
    assert "PRIMARY\\_PACKAGE\\_MISSING" in empty_markdown


def test_render_p2_markdown_shows_package_metrics_when_comparisons_are_empty() -> None:
    report = GuardedP2Report(
        scope=PeerScope(
            primary_ticker="NVDA",
            peer_tickers=(),
            description="Single-company industry research",
        ),
        packages=[
            _package(
                "NVDA",
                claims=[_claim("NVDA", "sec-nvda-claim", text="NVDA retained claim.")],
                metrics=[
                    _metric(
                        "NVDA",
                        "sec-nvda-revenue",
                        value="44.062",
                    ),
                    _metric(
                        "NVDA",
                        "sec-nvda-margin",
                        name="Gross margin",
                        value=None,
                        status=ComparabilityStatus.NOT_COMPARABLE,
                        limitation="unit mismatch",
                        unit="percent",
                    ),
                ],
            )
        ],
        comparisons=[],
        quality=None,
        retained_sources=[
            _source(
                _ref("sec-nvda-claim", ticker="NVDA"),
                title="NVDA MD&A",
                url="https://www.sec.gov/Archives/sec-nvda-claim.htm",
                published=date(2026, 5, 20),
            )
        ],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=SkillName.INDUSTRY_RESEARCH,
                    version="1.0.0",
                ),
            ),
            source_policy_versions=("policy-v3",),
            corpus_versions=("NVDA-v1",),
            prompt_versions=("research-v2",),
            requested_as_of_dates=(date(2026, 5, 31),),
            evidence_cutoff_dates=(date(2026, 5, 20),),
            information_sufficiency=InformationSufficiency.PARTIAL,
            source_refs=(_ref("sec-nvda-claim", ticker="NVDA"),),
        ),
        information_sufficiency=InformationSufficiency.PARTIAL,
        cross_ticker_leakage_count=0,
        guard_errors=[],
    )

    markdown = render_p2_markdown(report)

    assert "Revenue" in markdown
    assert "Gross margin" in markdown
    assert "Status: comparable" in markdown
    assert "Status: not\\_comparable" in markdown
    assert "44.062 USD billions" in markdown
    assert "missing USD percent" in markdown
    assert "Period: 2026-02-01 to 2026-04-30" in markdown
    assert "Definition: GAAP revenue" in markdown
    assert "Sources: NVDA:filing:sec-nvda-revenue" in markdown
    assert "Limitation: unit mismatch" in markdown
