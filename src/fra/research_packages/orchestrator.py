"""Serial explicit-peer orchestration over isolated single-ticker industry subruns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import uuid4

from pydantic import Field

from fra.domain import Intent, StrictModel, is_valid_us_listed_ticker
from fra.research_packages.comparability import compare_metrics
from fra.research_packages.models import (
    GuardedP2Report,
    GuardedResearchPackage,
    MetricComparison,
    MultiTickerResearchPackage,
    PeerResearchRequest,
    PeerScope,
)
from fra.skills.recipes import INDUSTRY_RESEARCH
from fra.storage.run_repositories import PersistedClaim, RunFinish

_CROSS_TICKER_SOURCE_REJECTED = "CROSS_TICKER_SOURCE_REJECTED"

SingleTickerRunner = Callable[[str, str], Awaitable[GuardedResearchPackage]]


class PeerScopeError(RuntimeError):
    """Stable peer-scope validation failure raised before runtime execution."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class PeerResearchResult(StrictModel):
    """Bounded multi-ticker result that satisfies the common application lifecycle."""

    run_id: str = Field(default="untracked", min_length=1)
    status: Literal["completed", "partial", "failed"]
    scope: PeerScope
    package: MultiTickerResearchPackage
    comparisons: list[MetricComparison] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    guarded_report: GuardedP2Report | None = None
    rendered_output: str
    source_policy_version: str | None = None

    def bind_run_id(self, run_id: str) -> PeerResearchResult:
        return self.model_copy(update={"run_id": run_id})

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        if self.guarded_report is not None:
            from fra.reporting.p2_guard import p2_persisted_claims

            claims = p2_persisted_claims(self.guarded_report)
        else:
            claims = _persisted_claims(
                self.package.packages,
                guard_notes=self.package.guard_notes,
            )
        return RunFinish(
            run_id=self.run_id,
            effective_intent=Intent.INDUSTRY_RESEARCH_REQUEST.value,
            status=self.status,
            corpus_scope=(
                list(self.guarded_report.provenance.corpus_versions)
                if self.guarded_report is not None
                else _corpus_scope(self.package.packages)
            ),
            prompt_version=(
                _single_prompt_version(self.guarded_report)
                if self.guarded_report is not None
                else None
            ),
            trace_id=trace_id,
            report_markdown=self.rendered_output,
            claims=claims,
        )

    def root_metadata(self) -> dict[str, object]:
        scope = self.guarded_report.scope if self.guarded_report is not None else self.scope
        cross_ticker_leakage_count = (
            self.guarded_report.cross_ticker_leakage_count
            if self.guarded_report is not None
            else 0
        )
        cross_ticker_rejection_count = (
            self.guarded_report.cross_ticker_rejection_count
            if self.guarded_report is not None
            else self.package.cross_ticker_leakage_count
        )
        metadata: dict[str, object] = {
            "effective_intent": Intent.INDUSTRY_RESEARCH_REQUEST.value,
            "primary_ticker": scope.primary_ticker,
            "peer_tickers": list(scope.peer_tickers),
            "peer_scope": scope.description,
            "missing_tickers": list(self.package.missing_tickers),
            "cross_ticker_leakage_count": cross_ticker_leakage_count,
            "cross_ticker_rejection_count": cross_ticker_rejection_count,
            "guard_notes": list(self.package.guard_notes),
            "error_codes": list(self.errors),
            "recipe_names": [INDUSTRY_RESEARCH.name.value] * len(self.package.packages),
            "recipe_versions": [INDUSTRY_RESEARCH.version] * len(self.package.packages),
            "comparison_count": len(self.comparisons),
            "corpus_scope": _corpus_scope(self.package.packages),
        }
        if self.guarded_report is not None:
            metadata["source_refs"] = [
                source.ref.encode() for source in self.guarded_report.retained_sources
            ]
            metadata["information_sufficiency"] = (
                self.guarded_report.information_sufficiency.value
            )
            metadata["guard_errors"] = list(self.guarded_report.guard_errors)
            metadata["prompt_versions"] = list(
                self.guarded_report.provenance.prompt_versions
            )
            metadata["requested_as_of_dates"] = [
                value.isoformat()
                for value in self.guarded_report.provenance.requested_as_of_dates
            ]
            metadata["evidence_cutoff_dates"] = [
                value.isoformat()
                for value in self.guarded_report.provenance.evidence_cutoff_dates
            ]
            metadata["source_policy_versions"] = list(
                self.guarded_report.provenance.source_policy_versions
            )
        if self.source_policy_version is not None:
            metadata["source_policy_version"] = self.source_policy_version
        return metadata


class PeerResearchOrchestrator:
    """Run the primary ticker and explicit peers serially through isolated subruns."""

    def __init__(
        self,
        single_ticker_runner: Callable[[str, str], Awaitable[GuardedResearchPackage]],
        *,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._single_ticker_runner = single_ticker_runner
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))

    async def run(self, request: PeerResearchRequest) -> PeerResearchResult:
        scope = validate_peer_scope(
            request.primary_ticker,
            request.peer_tickers,
            request.peer_scope,
        )
        packages: list[GuardedResearchPackage] = []
        missing_tickers: list[str] = []
        cross_ticker_leakage_count = 0
        guard_notes: list[str] = []

        for ticker in (scope.primary_ticker, *scope.peer_tickers):
            run_id = self._run_id_factory()
            try:
                package = await self._single_ticker_runner(
                    ticker,
                    request.question,
                    run_id=run_id,
                )
            except Exception:
                missing_tickers.append(ticker)
                if ticker == scope.primary_ticker:
                    return _failed_result(scope)
                continue

            leakage_count = _cross_ticker_leakage_count(package, expected_ticker=ticker)
            if leakage_count:
                cross_ticker_leakage_count += leakage_count
                missing_tickers.append(ticker)
                guard_notes.append(f"{_CROSS_TICKER_SOURCE_REJECTED}: {ticker}")
                if ticker == scope.primary_ticker:
                    return _failed_result(
                        scope,
                        cross_ticker_leakage_count=cross_ticker_leakage_count,
                        guard_notes=guard_notes,
                    )
                continue
            packages.append(package)

        status: Literal["completed", "partial", "failed"]
        if not packages:
            status = "failed"
            if scope.primary_ticker not in missing_tickers:
                missing_tickers.insert(0, scope.primary_ticker)
        elif missing_tickers:
            status = "partial"
        else:
            status = "completed"

        package = MultiTickerResearchPackage(
            primary_ticker=scope.primary_ticker,
            packages=packages,
            missing_tickers=missing_tickers,
            status=status,
            cross_ticker_leakage_count=cross_ticker_leakage_count,
            guard_notes=_unique(guard_notes),
        )
        comparisons = compare_metrics(packages)
        result = PeerResearchResult(
            status=status,
            scope=scope,
            package=package,
            comparisons=comparisons,
            errors=_unique(guard_notes),
            rendered_output="",
        )
        return result.model_copy(update={"rendered_output": _render_interim_output(result)})


def validate_peer_scope(
    primary_ticker: str,
    peer_tickers: list[str] | tuple[str, ...],
    description: str,
) -> PeerScope:
    """Validate explicit peers without discovering or ranking alternatives."""

    normalized_primary = primary_ticker.strip().upper()
    normalized_peers = tuple(peer.strip().upper() for peer in peer_tickers)
    normalized_description = description.strip()
    if not is_valid_us_listed_ticker(normalized_primary) or any(
        not is_valid_us_listed_ticker(peer) for peer in normalized_peers
    ):
        raise PeerScopeError(
            "INVALID_PEER_SCOPE",
            "primary and peer tickers must use valid US-listed ticker syntax",
        )
    if any(not peer for peer in normalized_peers):
        raise PeerScopeError(
            "INVALID_PEER_SCOPE",
            "peer_tickers must not contain blank tickers",
        )
    if len(normalized_peers) > 3:
        raise PeerScopeError(
            "PEER_LIMIT_EXCEEDED",
            "peer_tickers must contain at most three explicit peers",
        )
    if not normalized_description:
        raise PeerScopeError("INVALID_PEER_SCOPE", "peer_scope must not be blank")
    if normalized_primary in normalized_peers:
        raise PeerScopeError(
            "INVALID_PEER_SCOPE",
            "primary ticker cannot appear in peer_tickers",
        )
    if len(set(normalized_peers)) != len(normalized_peers):
        raise PeerScopeError("INVALID_PEER_SCOPE", "peer_tickers must be unique")
    return PeerScope(
        primary_ticker=normalized_primary,
        peer_tickers=normalized_peers,
        description=normalized_description,
    )


def _failed_result(
    scope: PeerScope,
    *,
    cross_ticker_leakage_count: int = 0,
    guard_notes: list[str] | None = None,
) -> PeerResearchResult:
    package = MultiTickerResearchPackage(
        primary_ticker=scope.primary_ticker,
        packages=[],
        missing_tickers=[scope.primary_ticker],
        status="failed",
        cross_ticker_leakage_count=cross_ticker_leakage_count,
        guard_notes=[] if guard_notes is None else _unique(guard_notes),
    )
    result = PeerResearchResult(
        status="failed",
        scope=scope,
        package=package,
        comparisons=[],
        errors=[] if guard_notes is None else _unique(guard_notes),
        rendered_output="",
    )
    return result.model_copy(update={"rendered_output": _render_interim_output(result)})


def _cross_ticker_leakage_count(
    package: GuardedResearchPackage,
    *,
    expected_ticker: str,
) -> int:
    count = 0
    if package.ticker != expected_ticker:
        count += 1
    for claim in package.claims:
        count += sum(1 for reference in claim.source_refs if reference.ticker != expected_ticker)
    for metric in package.financial_metrics:
        if metric.ticker != expected_ticker:
            count += 1
        count += sum(1 for reference in metric.source_refs if reference.ticker != expected_ticker)
    for source in package.filing_sources:
        if source.ticker != expected_ticker:
            count += 1
    for source in package.web_sources:
        if source.ticker != expected_ticker:
            count += 1
    return count


def _render_interim_output(result: PeerResearchResult) -> str:
    package_tickers = ", ".join(package.ticker for package in result.package.packages) or "none"
    missing = ", ".join(result.package.missing_tickers) or "none"
    comparison_lines = "\n".join(
        f"- {comparison.name}: {comparison.status.value}"
        for comparison in result.comparisons
    )
    if not comparison_lines:
        comparison_lines = "- none"
    guard_lines = "\n".join(f"- {note}" for note in result.package.guard_notes) or "- none"
    return (
        f"Peer comparison status: {result.status}\n"
        f"Primary ticker: {result.scope.primary_ticker}\n"
        f"Peer scope: {result.scope.description}\n"
        f"Successful packages: {package_tickers}\n"
        f"Missing tickers: {missing}\n"
        f"Cross-ticker leakage rejections: {result.package.cross_ticker_leakage_count}\n"
        "Guard notes:\n"
        f"{guard_lines}\n"
        "Comparable metrics:\n"
        f"{comparison_lines}\n"
        "Research assistance only; not investment advice."
    )


def _corpus_scope(packages: list[GuardedResearchPackage]) -> list[str]:
    values = [
        source.corpus_version
        for package in packages
        for source in package.filing_sources
        if source.ticker == package.ticker
    ]
    return list(dict.fromkeys(values))


def _single_prompt_version(report: GuardedP2Report) -> str | None:
    versions = report.provenance.prompt_versions
    return versions[0] if len(versions) == 1 else None


def _persisted_claims(
    packages: list[GuardedResearchPackage],
    *,
    guard_notes: list[str] | None = None,
) -> list[PersistedClaim]:
    claims: list[PersistedClaim] = []
    for package in packages:
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
                    confidence="high",
                    source_refs=list(metric.source_refs),
                    guard_status="retained",
                )
            )
    for note in guard_notes or []:
        claims.append(
            PersistedClaim(
                kind="peer_guard_note",
                text=note,
                confidence="high",
                source_refs=[],
                guard_status="rejected",
            )
        )
    return claims


def _metric_claim_text(metric: object) -> str:
    value = getattr(metric, "value")
    rendered_value = "missing" if value is None else str(value)
    return (
        f"{getattr(metric, 'ticker')} {getattr(metric, 'name')} {rendered_value} "
        f"{getattr(metric, 'unit')} for {getattr(metric, 'period_end')}"
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
