"""Offline integration coverage for explicit peer orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

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
    MultiTickerResearchPackage,
    PackageClaim,
    PeerResearchRequest,
)
from fra.research_packages.orchestrator import (
    PeerResearchOrchestrator,
    PeerResearchResult,
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


def _ref(source_id: str, *, ticker: str) -> SourceRef:
    return SourceRef(ticker=ticker, kind=SourceRefKind.FILING, source_id=source_id)


def _metric(
    *,
    ticker: str,
    value: Decimal = Decimal("44.1"),
    source_id: str = "sec-1",
) -> ComparableMetric:
    return ComparableMetric(
        ticker=ticker,
        name="Revenue",
        value=value,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        currency="USD",
        unit="billions",
        definition="GAAP revenue",
        observation_id=financial_observation_id(
            ticker=ticker,
            name="Revenue",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            currency="USD",
            unit="billions",
            definition="GAAP revenue",
        ),
        source_refs=[_ref(source_id, ticker=ticker)],
        source_provenance=[
            FinancialSourceProvenance(
                source_ref=_ref(source_id, ticker=ticker),
                source_kind=SourceKind.FILING,
                source_tier=SourceTier.PRIMARY,
                canonical_source_identity="filing-accession:0001045810-26-000002",
            )
        ],
        verification_status=VerificationStatus.SINGLE_SOURCE,
        status=ComparabilityStatus.COMPARABLE,
        limitation=None,
    )


def _package(
    ticker: str,
    *,
    claim_ticker: str | None = None,
    bypass_validation: bool = False,
) -> GuardedResearchPackage:
    scoped_ticker = claim_ticker or ticker
    filing_sources = [
        EvidenceChunk(
            id=f"claim-{ticker}",
            ticker=ticker,
            corpus_version=f"{ticker}-v1",
            content=f"{ticker} claim evidence.",
            source_url=f"https://www.sec.gov/Archives/claim-{ticker}.htm",
            form="10-Q",
            filed_at=date(2026, 5, 20),
            accession_no="0001045810-26-000001",
            section="MD&A",
            raw_start=0,
            raw_end=20,
        ),
        EvidenceChunk(
            id=f"metric-{ticker}",
            ticker=ticker,
            corpus_version=f"{ticker}-v1",
            content=f"{ticker} metric evidence.",
            source_url=f"https://www.sec.gov/Archives/metric-{ticker}.htm",
            form="10-Q",
            filed_at=date(2026, 5, 20),
            accession_no="0001045810-26-000002",
            section="Financial statements",
            raw_start=0,
            raw_end=20,
        ),
    ]
    values = dict(
        ticker=ticker,
        claims=[
            PackageClaim(
                facet=ResearchFacet.INDUSTRY_SCOPE,
                kind=ClaimKind.VERIFIED_FACT,
                text=f"{ticker} claim.",
                confidence=Confidence.HIGH,
                source_refs=[_ref(f"claim-{ticker}", ticker=scoped_ticker)],
            )
        ],
        financial_metrics=[_metric(ticker=ticker, source_id=f"metric-{ticker}")],
        filing_sources=filing_sources,
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=SkillName.INDUSTRY_RESEARCH,
                    version="1.0.0",
                ),
            ),
            corpus_versions=(f"{ticker}-v1",),
            prompt_versions=("research-v2",),
            evidence_cutoff_dates=(date(2026, 5, 20),),
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            source_refs=tuple(
                _ref(source.id, ticker=ticker) for source in filing_sources
            ),
        ),
        evidence_dates=[],
        coverage="complete",
        information_gaps=[],
        guard_notes=[],
    )
    if bypass_validation:
        return GuardedResearchPackage.model_construct(**values)
    return GuardedResearchPackage(**values)


def _request(primary: str, peers: list[str]) -> PeerResearchRequest:
    return PeerResearchRequest(
        primary_ticker=primary,
        peer_tickers=tuple(peers),
        peer_scope="US semiconductors",
        question="Compare direct peers using exact reported metrics.",
    )


@dataclass
class RecordingRunner:
    packages: dict[str, GuardedResearchPackage]
    failures: set[str] = field(default_factory=set)
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def run(self, ticker: str, question: str, *, run_id: str) -> GuardedResearchPackage:
        self.calls.append((ticker, question, run_id))
        await asyncio.sleep(0)
        if ticker in self.failures:
            raise RuntimeError(f"{ticker} unavailable")
        return self.packages[ticker]


def test_peer_orchestrator_runs_primary_then_peers_with_distinct_run_ids() -> None:
    runner = RecordingRunner(
        packages={
            "NVDA": _package("NVDA"),
            "AMD": _package("AMD"),
            "INTC": _package("INTC"),
        }
    )

    result = asyncio.run(
        PeerResearchOrchestrator(runner.run).run(_request("NVDA", ["AMD", "INTC"]))
    )

    assert isinstance(result, PeerResearchResult)
    assert result.status == "completed"
    assert isinstance(result.package, MultiTickerResearchPackage)
    assert [ticker for ticker, _, _ in runner.calls] == ["NVDA", "AMD", "INTC"]
    assert len({run_id for _, _, run_id in runner.calls}) == 3
    assert result.package.missing_tickers == []


def test_peer_failure_is_partial_without_cross_ticker_fill() -> None:
    runner = RecordingRunner(
        packages={
            "NVDA": _package("NVDA"),
            "AMD": _package("AMD"),
            "INTC": _package("INTC"),
        },
        failures={"INTC"},
    )

    result = asyncio.run(
        PeerResearchOrchestrator(runner.run).run(_request("NVDA", ["AMD", "INTC"]))
    )

    assert result.status == "partial"
    assert {package.ticker for package in result.package.packages} == {"NVDA", "AMD"}
    assert result.package.missing_tickers == ["INTC"]
    assert result.package.cross_ticker_leakage_count == 0


def test_primary_failure_fails_closed_without_peer_only_result() -> None:
    runner = RecordingRunner(
        packages={"NVDA": _package("NVDA"), "AMD": _package("AMD")},
        failures={"NVDA"},
    )

    result = asyncio.run(PeerResearchOrchestrator(runner.run).run(_request("NVDA", ["AMD"])))

    assert result.status == "failed"
    assert result.package.packages == []
    assert result.package.missing_tickers == ["NVDA"]
    assert [ticker for ticker, _, _ in runner.calls] == ["NVDA"]


def test_cross_ticker_source_rejection_does_not_fill_another_ticker() -> None:
    runner = RecordingRunner(
        packages={
            "NVDA": _package("NVDA"),
            "AMD": _package("AMD", claim_ticker="NVDA", bypass_validation=True),
        }
    )

    result = asyncio.run(PeerResearchOrchestrator(runner.run).run(_request("NVDA", ["AMD"])))

    assert result.status == "partial"
    assert [package.ticker for package in result.package.packages] == ["NVDA"]
    assert result.package.missing_tickers == ["AMD"]
    assert result.package.cross_ticker_leakage_count == 1
    assert result.package.guard_notes == ["CROSS_TICKER_SOURCE_REJECTED: AMD"]
    assert result.errors == ["CROSS_TICKER_SOURCE_REJECTED: AMD"]
    assert "CROSS_TICKER_SOURCE_REJECTED: AMD" in result.rendered_output

    finish = result.to_run_finish("trace-peer")
    assert finish.prompt_version is None
    assert finish.claims[-1].kind == "peer_guard_note"
    assert finish.claims[-1].text == "CROSS_TICKER_SOURCE_REJECTED: AMD"
    assert finish.claims[-1].guard_status == "rejected"
    assert finish.claims[-1].source_refs == []

    metadata = result.root_metadata()
    assert metadata["guard_notes"] == ["CROSS_TICKER_SOURCE_REJECTED: AMD"]
    assert metadata["error_codes"] == ["CROSS_TICKER_SOURCE_REJECTED: AMD"]
    assert metadata["cross_ticker_rejection_count"] == 1
    assert metadata["cross_ticker_leakage_count"] == 0


def test_interim_output_stays_fact_only_without_ranking_language() -> None:
    runner = RecordingRunner(
        packages={"NVDA": _package("NVDA"), "AMD": _package("AMD")},
    )

    result = asyncio.run(PeerResearchOrchestrator(runner.run).run(_request("NVDA", ["AMD"])))

    assert "winner" not in result.rendered_output.casefold()
    assert "rank" not in result.rendered_output.casefold()
    assert "preferred" not in result.rendered_output.casefold()
