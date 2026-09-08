"""Deterministic research-quality screening over guarded research packages."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import Field

from financial_evidence_agent.application import current_research_context
from financial_evidence_agent.context import BudgetAuthority, BudgetLimits
from financial_evidence_agent.domain import (
    ClaimKind,
    Intent,
    SourceRef,
    SourceRefKind,
    StrictModel,
)
from financial_evidence_agent.graph.models import (
    Dependencies,
    SkillRunPlan,
    SkillRunResult,
)
from financial_evidence_agent.graph.nodes import (
    _execute_skill_recipe,
    _FatalP1Error,
    _trace_skill_result,
)
from financial_evidence_agent.memory.research import (
    build_research_memory_hints,
    scope_research_memory_store,
)
from financial_evidence_agent.memory.session import (
    build_session_memory_hints,
    load_session_memory_for_ticker,
)
from financial_evidence_agent.prompts import current_prompt_version
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedP2Report,
    GuardedResearchPackage,
    PackageClaim,
    ResearchQualityDecision,
    ResearchQualityResult,
)
from financial_evidence_agent.safety.router import (
    contains_quality_screen_out_of_scope_request,
    route_request,
)
from financial_evidence_agent.skills.models import ResearchFacet, SkillName
from financial_evidence_agent.skills.recipes import (
    COMPANY_DEEP_RESEARCH,
    RESEARCH_QUALITY_SCREEN,
)
from financial_evidence_agent.skills.schemas import (
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
)
from financial_evidence_agent.storage.run_repositories import PersistedClaim, RunFinish

_STALE_SOURCE_MAX_AGE_DAYS = 365
_POSITIVE_SUPPORT_FACETS = frozenset(
    {
        ResearchFacet.COMPANY_OVERVIEW,
        ResearchFacet.BUSINESS_MODEL,
        ResearchFacet.COMPETITIVE_POSITION,
        ResearchFacet.BULL_CASE,
    }
)
_COUNTER_FACETS = frozenset({ResearchFacet.BEAR_CASE, ResearchFacet.RISKS})
_PRIMARY_SOURCE_KINDS = frozenset({SourceRefKind.FILING, SourceRefKind.WEB})
logger = logging.getLogger(__name__)
_OUT_OF_SCOPE_REASON = (
    "Out of scope: research-quality screening does not rank securities, choose trades, "
    "set price targets, manage a portfolio, or select peers automatically."
)


def _utc_today() -> date:
    return datetime.now(UTC).date()


class QualitySubrunProtocolError(RuntimeError):
    """Internal protocol failure for deterministic quality-screen preparation."""


class QualityResearchResult(StrictModel):
    """Application-bound quality-screen result with the common persistence lifecycle."""

    run_id: str = Field(default="untracked", min_length=1)
    status: Literal["completed", "declined"] = "completed"
    ticker: str = Field(min_length=1, max_length=10)
    package: GuardedResearchPackage | None = None
    quality: ResearchQualityResult
    guarded_report: GuardedP2Report | None = None
    rendered_output: str
    effective_date: date
    max_source_age_days: int = Field(ge=1, le=3650)
    source_policy_version: str | None = None
    requested_as_of: date | None = None

    def bind_run_id(self, run_id: str) -> QualityResearchResult:
        return self.model_copy(update={"run_id": run_id})

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        if self.guarded_report is not None:
            from financial_evidence_agent.reporting.p2_guard import p2_persisted_claims

            claims = p2_persisted_claims(self.guarded_report)
            corpus_scope = [
                source.corpus_version
                for package in self.guarded_report.packages
                for source in package.filing_sources
                if source.ticker == package.ticker
            ]
        else:
            claims = _persisted_claims(self)
            corpus_scope = []
        return RunFinish(
            run_id=self.run_id,
            effective_intent=Intent.RESEARCH_QUALITY_SCREEN_REQUEST.value,
            status=self.status,
            corpus_scope=list(dict.fromkeys(corpus_scope)),
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
        package_recipes = (
            () if self.package is None else self.package.provenance.recipes
        )
        recipe_names = [
            *(recipe.name.value for recipe in package_recipes),
            SkillName.RESEARCH_QUALITY_SCREEN.value,
        ]
        recipe_versions = [
            *(recipe.version for recipe in package_recipes),
            RESEARCH_QUALITY_SCREEN.version,
        ]
        metadata = {
            "effective_intent": Intent.RESEARCH_QUALITY_SCREEN_REQUEST.value,
            "quality_decision": self.quality.decision.value,
            "quality_effective_date": self.effective_date.isoformat(),
            "quality_max_source_age_days": self.max_source_age_days,
            "recipe_names": recipe_names,
            "recipe_versions": recipe_versions,
            "source_refs": (
                [source.ref.encode() for source in self.guarded_report.retained_sources]
                if self.guarded_report is not None
                else [reference.encode() for reference in self.quality.source_refs]
            ),
            "coverage": None if self.package is None else self.package.coverage,
        }
        if self.guarded_report is not None:
            metadata["information_sufficiency"] = self.guarded_report.information_sufficiency.value
            metadata["guard_errors"] = list(self.guarded_report.guard_errors)
            metadata["cross_ticker_leakage_count"] = self.guarded_report.cross_ticker_leakage_count
            metadata["cross_ticker_rejection_count"] = (
                self.guarded_report.cross_ticker_rejection_count
            )
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
        else:
            metadata["cross_ticker_leakage_count"] = 0
            metadata["cross_ticker_rejection_count"] = 0
        return metadata


@dataclass(frozen=True, slots=True)
class QualityResearchRuntime:
    """Runtime that reuses a guarded package or prepares one via a single company recipe."""

    dependencies: Dependencies | None = None
    guarded_package: GuardedResearchPackage | None = None
    current_date_factory: Callable[[], date] = _utc_today
    max_source_age_days: int = _STALE_SOURCE_MAX_AGE_DAYS
    run_context_factory: object | None = None
    scope_resolver: object | None = None
    source_policy_version: str | None = None

    def execute(self, command, decision) -> QualityResearchResult:
        if decision.intent is not Intent.RESEARCH_QUALITY_SCREEN_REQUEST:
            raise ValueError("quality runtime requires research_quality_screen_request")
        context = current_research_context()
        dependencies = self.dependencies
        if context is not None and dependencies is not None:
            if not context.budget.configured:
                context.budget.configure(BudgetLimits.aggregate((COMPANY_DEEP_RESEARCH,)))
            dependencies = replace(
                dependencies,
                trace_run=context.trace,
                budget=context.budget,
                session_memory_store=context.session_memory_store,
                research_memory_store=scope_research_memory_store(
                    context.research_memory_store,
                    dependencies.web_evidence_validator,
                ),
            )
            if callable(self.run_context_factory):
                scoped = self.run_context_factory(context)
                dependencies = replace(
                    scoped,
                    trace_run=context.trace,
                    budget=context.budget,
                    session_memory_store=context.session_memory_store,
                    research_memory_store=scope_research_memory_store(
                        context.research_memory_store,
                        scoped.web_evidence_validator,
                    ),
                )
        if callable(self.scope_resolver):
            command = self.scope_resolver(
                command,
                None if dependencies is None else dependencies.budget,
            )
        if dependencies is not None:
            collector = dependencies.skill_collector
            with_scope = getattr(collector, "with_filing_scope", None)
            if callable(with_scope):
                dependencies = replace(
                    dependencies,
                    skill_collector=with_scope(
                        command.corpus_version,
                        command.filing_ids,
                        scope_error=command.scope_error,
                    ),
                )
        return run_quality_screen(
            command.ticker.strip().upper(),
            command.request,
            dependencies=dependencies,
            guarded_package=self.guarded_package,
            run_id=context.run_id if context is not None else "untracked",
            effective_date=self.current_date_factory(),
            max_source_age_days=self.max_source_age_days,
            source_policy_version=self.source_policy_version,
            session_id=command.session_id,
            corpus_version=command.corpus_version,
            requested_as_of=command.as_of_date,
        )


def classify_quality_scope(
    request: str,
    *,
    ticker: str | None = None,
) -> ResearchQualityResult | None:
    """Return the deterministic out-of-scope decision, or ``None`` for safe requests."""

    if not contains_quality_screen_out_of_scope_request(request, ticker=ticker):
        return None
    return ResearchQualityResult(
        decision=ResearchQualityDecision.OUT_OF_SCOPE,
        reasons=[_OUT_OF_SCOPE_REASON],
        source_refs=[],
    )


def run_quality_screen(
    ticker: str,
    request: str,
    *,
    dependencies: Dependencies | None = None,
    guarded_package: GuardedResearchPackage | None = None,
    run_id: str = "untracked",
    effective_date: date | None = None,
    max_source_age_days: int = _STALE_SOURCE_MAX_AGE_DAYS,
    source_policy_version: str | None = None,
    session_id: str | None = None,
    corpus_version: str | None = None,
    requested_as_of: date | None = None,
) -> QualityResearchResult:
    """Run deterministic quality screening over one guarded package."""

    normalized_ticker = ticker.strip().upper()
    if max_source_age_days <= 0:
        raise ValueError("max_source_age_days must be positive")
    resolved_effective_date = effective_date or _utc_today()
    package = (
        None
        if guarded_package is None
        else _with_requested_as_of(guarded_package, requested_as_of)
    )
    with _quality_observation(
        name="quality.screen",
        kind="chain",
        metadata={"reused_package": package is not None},
    ) as observation:
        quality = classify_quality_scope(request, ticker=normalized_ticker)
        if quality is None:
            if package is None:
                if dependencies is None:
                    raise ValueError(
                        "dependencies are required when no guarded package is supplied"
                    )
                package = run_company_deep_research_subrun(
                    normalized_ticker,
                    request,
                    dependencies,
                    run_id=run_id,
                    session_id=session_id,
                    corpus_version=corpus_version,
                    requested_as_of=requested_as_of,
                )
            if package.ticker != normalized_ticker:
                raise QualitySubrunProtocolError(
                    "guarded package ticker does not match quality request"
                )
            quality = screen_research_quality(
                package,
                request,
                effective_date=resolved_effective_date,
                max_source_age_days=max_source_age_days,
            )
        if observation is not None:
            observation.update(
                output={
                    "decision": quality.decision.value,
                    "source_ref_count": len(quality.source_refs),
                }
            )
    with _quality_observation(
        name="quality.render",
        kind="chain",
        metadata={"decision": quality.decision.value},
    ) as observation:
        rendered_output = render_quality_markdown(
            normalized_ticker,
            quality,
            package=package,
        )
        if observation is not None:
            observation.update(
                output={
                    "status": (
                        "declined"
                        if quality.decision is ResearchQualityDecision.OUT_OF_SCOPE
                        else "completed"
                    ),
                    "rendered_length": len(rendered_output),
                }
            )
    return QualityResearchResult(
        run_id=run_id,
        status=(
            "declined"
            if quality.decision is ResearchQualityDecision.OUT_OF_SCOPE
            else "completed"
        ),
        ticker=normalized_ticker,
        package=package,
        quality=quality,
        rendered_output=rendered_output,
        effective_date=resolved_effective_date,
        max_source_age_days=max_source_age_days,
        source_policy_version=source_policy_version,
        requested_as_of=requested_as_of,
    )


def run_company_deep_research_subrun(
    ticker: str,
    request: str,
    dependencies: Dependencies,
    *,
    run_id: str = "untracked",
    session_id: str | None = None,
    corpus_version: str | None = None,
    requested_as_of: date | None = None,
) -> GuardedResearchPackage:
    """Execute exactly one immutable company recipe and return its guarded package."""

    normalized_ticker = ticker.strip().upper()
    decision = route_request(normalized_ticker, request)
    if decision is not None and decision.intent in {
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
        Intent.UNSAFE_SOURCE_REQUEST,
    }:
        return _refused_package(
            normalized_ticker,
            decision.intent.value,
            requested_as_of=requested_as_of,
        )
    memory = load_session_memory_for_ticker(
        dependencies.session_memory_store,
        session_id,
        normalized_ticker,
    )
    research_memories = ()
    if dependencies.research_memory_store is not None:
        try:
            research_memories = tuple(
                dependencies.research_memory_store.search(
                    normalized_ticker,
                    request,
                    current_corpus_version=corpus_version,
                    limit=3,
                )
            )
        except Exception:
            logger.warning("research memory load failed")
    memory_hints = (
        *build_session_memory_hints(memory),
        *build_research_memory_hints(research_memories),
    )
    if dependencies.skill_run_repository is None:
        raise QualitySubrunProtocolError("quality subrun requires a skill-run repository")
    if dependencies.budget is None:
        authority = BudgetAuthority()
        authority.configure(BudgetLimits.aggregate((COMPANY_DEEP_RESEARCH,)))
        dependencies = replace(dependencies, budget=authority)
    plan = SkillRunPlan(
        run_id=dependencies.skill_run_repository.start(
            application_run_id=run_id,
            ticker=normalized_ticker,
            recipe_name=COMPANY_DEEP_RESEARCH.name.value,
            recipe_version=COMPANY_DEEP_RESEARCH.version,
            recipe_snapshot=COMPANY_DEEP_RESEARCH.model_dump(mode="json"),
        ),
        recipe=COMPANY_DEEP_RESEARCH,
    )
    state = {
        "ticker": normalized_ticker,
        "thesis": " ".join(request.split()),
        "node_trace": [],
        "recent_turns": memory.turns,
        "session_summary": memory.summary,
        "memory_hints": memory_hints,
        "report_as_of": requested_as_of,
    }
    try:
        result = _execute_skill_recipe(state, dependencies, plan)
    except _FatalP1Error as error:
        raise QualitySubrunProtocolError("company-deep-research subrun failed closed") from error
    _trace_skill_result(dependencies, result)
    return build_company_deep_research_package(
        result,
        ticker=normalized_ticker,
        source_policy_version=(
            dependencies.web_evidence_validator.policy_version
            if dependencies.web_evidence_validator is not None
            else None
        ),
        requested_as_of=requested_as_of,
    )


def build_company_deep_research_package(
    result: SkillRunResult,
    *,
    ticker: str,
    source_policy_version: str | None = None,
    requested_as_of: date | None = None,
) -> GuardedResearchPackage:
    """Convert a single guarded company recipe result into a retained package."""

    if result.recipe_name is not SkillName.COMPANY_DEEP_RESEARCH:
        raise QualitySubrunProtocolError("quality workflow executed the wrong recipe")
    if result.guarded_memo is None:
        return GuardedResearchPackage(
            ticker=ticker,
            claims=[],
            financial_metrics=[],
            filing_sources=[],
            web_sources=[],
            provenance=_empty_quality_package_provenance(
                recipe_name=SkillName.COMPANY_DEEP_RESEARCH,
                recipe_version=result.recipe_version,
                source_policy_version=source_policy_version,
                requested_as_of=requested_as_of,
            ),
            evidence_dates=[],
            coverage="insufficient",
            information_gaps=["Insufficient evidence for company deep research."],
            guard_notes=_unique(result.errors),
        )

    guarded = result.guarded_memo
    provenance = guarded.provenance
    if requested_as_of is not None:
        provenance = provenance.model_copy(
            update={"requested_as_of_dates": (requested_as_of,)}
        )
    return GuardedResearchPackage(
        ticker=guarded.ticker,
        claims=[
            PackageClaim(
                facet=section.facet,
                kind=claim.kind,
                text=claim.text,
                confidence=claim.confidence,
                source_refs=[
                    *(
                        SourceRef(
                            ticker=guarded.ticker,
                            kind=SourceRefKind.FILING,
                            source_id=source_id,
                        )
                        for source_id in claim.evidence_chunk_ids
                    ),
                    *(
                        SourceRef(
                            ticker=guarded.ticker,
                            kind=SourceRefKind.WEB,
                            source_id=source_id,
                        )
                        for source_id in claim.web_evidence_ids
                    ),
                ],
            )
            for section in guarded.memo.sections
            for claim in section.claims
        ],
        financial_metrics=[
            ComparableMetric(
                ticker=guarded.ticker,
                name=point.name,
                value=point.value,
                period_start=point.period_start,
                period_end=point.period_end,
                currency=point.currency,
                unit=point.unit,
                definition=point.definition,
                observation_id=point.observation_id,
                source_refs=[source.source_ref for source in point.source_provenance],
                source_provenance=list(point.source_provenance),
                verification_status=point.verification_status,
                status=_metric_status(point.verification_status),
                limitation=_metric_limitation(
                    point.verification_status,
                    point.discrepancy_note,
                ),
            )
            for point in guarded.memo.data_points
        ],
        filing_sources=list(guarded.filing_sources),
        web_sources=list(guarded.web_sources),
        provenance=provenance,
        evidence_dates=_evidence_dates(result),
        coverage=_coverage(guarded.memo.information_sufficiency),
        information_gaps=list(guarded.memo.information_gaps),
        guard_notes=_unique([*result.errors, *guarded.guard_errors]),
    )


def screen_research_quality(
    package: GuardedResearchPackage,
    request: str,
    *,
    effective_date: date | None = None,
    max_source_age_days: int = _STALE_SOURCE_MAX_AGE_DAYS,
) -> ResearchQualityResult:
    """Return one deterministic quality decision over a guarded package."""

    if max_source_age_days <= 0:
        raise ValueError("max_source_age_days must be positive")
    scope = classify_quality_scope(request, ticker=package.ticker)
    if scope is not None:
        return scope

    resolved_effective_date = effective_date or _utc_today()
    support_refs = _support_refs(package)
    counter_refs = _counter_refs(package)
    stale = _stale_evidence_dates(
        package,
        effective_date=resolved_effective_date,
        max_source_age_days=max_source_age_days,
    )
    source_quality_issue = _source_quality_issue(package)
    comparability_issue = _comparability_issue(package)
    reasons: list[str] = []

    if package.coverage != "complete":
        reasons.append(f"Coverage is {package.coverage}; the retained package is not complete.")
    if not support_refs:
        reasons.append("No retained supporting evidence claim is present.")
    if not counter_refs:
        reasons.append("No retained counterevidence claim is present.")
    if stale:
        rendered_dates = ", ".join(item.isoformat() for item in stale)
        reasons.append(f"Required source dates are stale: {rendered_dates}.")
    if source_quality_issue is not None:
        reasons.append(source_quality_issue)
    if comparability_issue is not None:
        reasons.append(comparability_issue)

    if reasons:
        return ResearchQualityResult(
            decision=ResearchQualityDecision.INSUFFICIENT_INFORMATION,
            reasons=reasons,
            source_refs=_unique_refs([*support_refs, *counter_refs, *_metric_refs(package)]),
        )

    support_ref = support_refs[0]
    counter_ref = counter_refs[0]
    quality_reasons = [
        f"Complete retained coverage includes supporting evidence {support_ref.encode()}.",
        f"Counterevidence is retained via {counter_ref.encode()}.",
    ]
    metric_refs = _metric_refs(package)
    if metric_refs:
        quality_reasons.append(
            "Comparable retained metrics cite "
            + ", ".join(reference.encode() for reference in metric_refs)
            + "."
        )
    return ResearchQualityResult(
        decision=ResearchQualityDecision.WORTH_FURTHER_RESEARCH,
        reasons=quality_reasons,
        source_refs=_unique_refs([*support_refs, *counter_refs, *metric_refs]),
    )


def render_quality_markdown(
    ticker: str,
    quality: ResearchQualityResult,
    *,
    package: GuardedResearchPackage | None,
) -> str:
    """Render the interim fixed quality-screen report until the unified P2 composer lands."""

    lines = [
        f"# Research quality screen — {ticker}",
        "",
        f"- Decision: {quality.decision.value}",
        f"- Recipe: {RESEARCH_QUALITY_SCREEN.name.value}",
        f"- Recipe version: {RESEARCH_QUALITY_SCREEN.version}",
        (
            "- Package recipes: "
            + (
                ", ".join(
                    recipe.name.value for recipe in package.provenance.recipes
                )
                if package is not None
                else "none"
            )
        ),
        f"- Coverage: {package.coverage if package is not None else 'not_evaluated'}",
        "",
        "## Scope",
        "",
        "This screen evaluates only the quality of the retained research package.",
        "",
        "## Reasons",
        "",
        *[f"- {reason}" for reason in quality.reasons],
        "",
        "## Source references",
        "",
        *(
            [f"- {reference.encode()}" for reference in quality.source_refs]
            if quality.source_refs
            else ["- No retained source references were required."]
        ),
        "",
        "> Research assistance only; not investment advice.",
    ]
    return "\n".join(lines) + "\n"


def _persisted_claims(result: QualityResearchResult) -> list[PersistedClaim]:
    decision_refs = list(result.quality.source_refs)
    claims = [
        PersistedClaim(
            kind="research_quality_decision",
            text=f"{result.ticker} research quality decision: {result.quality.decision.value}.",
            confidence="high",
            source_refs=decision_refs,
            guard_status="retained",
        )
    ]
    for reason in result.quality.reasons:
        claims.append(
            PersistedClaim(
                kind="research_quality_reason",
                text=reason,
                confidence="high",
                source_refs=[
                    reference
                    for reference in result.quality.source_refs
                    if reference.encode() in reason
                ],
                guard_status="retained",
            )
        )
    return claims


def _support_refs(package: GuardedResearchPackage) -> list[SourceRef]:
    return _unique_refs(
        [
            reference
            for claim in package.claims
            if claim.kind is ClaimKind.VERIFIED_FACT and claim.facet in _POSITIVE_SUPPORT_FACETS
            for reference in claim.source_refs
            if reference.kind in _PRIMARY_SOURCE_KINDS
        ]
    )


def _counter_refs(package: GuardedResearchPackage) -> list[SourceRef]:
    return _unique_refs(
        [
            reference
            for claim in package.claims
            if claim.facet in _COUNTER_FACETS
            for reference in claim.source_refs
            if reference.kind in _PRIMARY_SOURCE_KINDS
        ]
    )


def _metric_refs(package: GuardedResearchPackage) -> list[SourceRef]:
    return _unique_refs(
        [
            reference
            for metric in package.financial_metrics
            if metric.status is ComparabilityStatus.COMPARABLE
            for reference in metric.source_refs
        ]
    )


def _stale_evidence_dates(
    package: GuardedResearchPackage,
    *,
    effective_date: date,
    max_source_age_days: int,
) -> list[date]:
    cutoff = effective_date.toordinal() - max_source_age_days
    return [
        evidence_date
        for evidence_date in package.evidence_dates
        if evidence_date.toordinal() < cutoff
    ]


def _source_quality_issue(package: GuardedResearchPackage) -> str | None:
    if package.filing_sources:
        return None
    if any(source.source_tier.value == "primary" for source in package.web_sources):
        return None
    if package.web_sources:
        return "Retained evidence lacks a filing source or any primary-tier web source."
    return "No retained filing or web evidence source is present."


def _comparability_issue(package: GuardedResearchPackage) -> str | None:
    limited = [
        metric
        for metric in package.financial_metrics
        if metric.status is not ComparabilityStatus.COMPARABLE
    ]
    if not limited:
        return None
    rendered = ", ".join(f"{metric.name}={metric.status.value}" for metric in limited)
    return f"Metric comparability remains limited: {rendered}."


def _coverage(value: InformationSufficiency) -> str:
    if value is InformationSufficiency.SUFFICIENT:
        return "complete"
    if value is InformationSufficiency.PARTIAL:
        return "partial"
    return "insufficient"


def _metric_status(value: VerificationStatus) -> ComparabilityStatus:
    if value in {VerificationStatus.VERIFIED, VerificationStatus.SINGLE_SOURCE}:
        return ComparabilityStatus.COMPARABLE
    if value is VerificationStatus.DISCREPANCY:
        return ComparabilityStatus.DISCREPANCY
    if value is VerificationStatus.NOT_COMPARABLE:
        return ComparabilityStatus.NOT_COMPARABLE
    return ComparabilityStatus.MISSING


def _metric_limitation(
    status: VerificationStatus,
    note: str | None,
) -> str | None:
    if status is VerificationStatus.VERIFIED:
        return None
    if status is VerificationStatus.SINGLE_SOURCE:
        return "Only one independent canonical source was retained."
    return note or "Metric was not retained as directly comparable."


def _evidence_dates(skill_run: SkillRunResult) -> list[date]:
    guarded = skill_run.guarded_memo
    assert guarded is not None
    dates = [source.filed_at for source in guarded.filing_sources]
    dates.extend(
        source.published_at.date() if source.published_at is not None else source.fetched_at.date()
        for source in guarded.web_sources
    )
    return sorted(set(dates))


def _refused_package(
    ticker: str,
    refusal_code: str,
    *,
    requested_as_of: date | None = None,
) -> GuardedResearchPackage:
    return GuardedResearchPackage(
        ticker=ticker,
        claims=[],
        financial_metrics=[],
        filing_sources=[],
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(),
            requested_as_of_dates=(
                () if requested_as_of is None else (requested_as_of,)
            ),
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
        ),
        zero_recipe_outcome="refused",
        evidence_dates=[],
        coverage="insufficient",
        information_gaps=[],
        guard_notes=[refusal_code],
    )


def _empty_quality_package_provenance(
    *,
    recipe_name: SkillName,
    recipe_version: str,
    source_policy_version: str | None,
    requested_as_of: date | None,
) -> ReportProvenance:
    prompt_version = current_prompt_version()
    return ReportProvenance(
        recipes=(RecipeProvenance(name=recipe_name, version=recipe_version),),
        source_policy_versions=(
            () if source_policy_version is None else (source_policy_version,)
        ),
        prompt_versions=(() if prompt_version is None else (prompt_version,)),
        requested_as_of_dates=(
            () if requested_as_of is None else (requested_as_of,)
        ),
        information_sufficiency=InformationSufficiency.INSUFFICIENT,
    )


def _with_requested_as_of(
    package: GuardedResearchPackage,
    requested_as_of: date | None,
) -> GuardedResearchPackage:
    if requested_as_of is None:
        return package
    provenance = package.provenance.model_copy(
        update={"requested_as_of_dates": (requested_as_of,)}
    )
    return package.model_copy(update={"provenance": provenance})


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _unique_refs(values: list[SourceRef]) -> list[SourceRef]:
    unique: dict[str, SourceRef] = {}
    for value in values:
        unique[value.encode()] = value
    return list(unique.values())


def _single_prompt_version(report: GuardedP2Report) -> str | None:
    versions = report.provenance.prompt_versions
    return versions[0] if len(versions) == 1 else None


@contextmanager
def _quality_observation(
    *,
    name: str,
    kind: str,
    metadata: dict[str, object],
) -> Iterator[object | None]:
    context = current_research_context()
    if context is None:
        yield None
        return
    with context.trace.observation(name=name, kind=kind, metadata=metadata) as observation:
        yield observation
