"""Single-ticker industry package conversion over the controlled P1 machinery."""

from __future__ import annotations

from datetime import date

from financial_evidence_agent.domain import Intent, SourceRef, SourceRefKind
from financial_evidence_agent.graph.models import Dependencies, ResearchResult, SkillRunResult
from financial_evidence_agent.prompts import current_prompt_version
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    PackageClaim,
)
from financial_evidence_agent.skills.models import SkillName
from financial_evidence_agent.skills.schemas import (
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    VerificationStatus,
)


class IndustrySubrunProtocolError(RuntimeError):
    """Internal protocol failure for direct industry helper execution."""


def run_research(
    ticker: str,
    request: str,
    dependencies: Dependencies,
    *,
    intent: Intent,
    run_id: str = "untracked",
) -> ResearchResult:
    """Delegate lazily to avoid a module cycle with the graph workflow."""
    from financial_evidence_agent.graph.workflow import run_research as workflow_run_research

    return workflow_run_research(
        ticker,
        request,
        dependencies,
        intent=intent,
        run_id=run_id,
    )


def run_industry_subrun(
    ticker: str,
    request: str,
    dependencies: Dependencies,
    *,
    run_id: str = "untracked",
) -> GuardedResearchPackage:
    """Execute exactly the frozen industry recipe and return its guarded package."""
    result = run_research(
        ticker,
        request,
        dependencies,
        intent=Intent.INDUSTRY_RESEARCH_REQUEST,
        run_id=run_id,
    )
    return build_industry_package(result)


def build_industry_package(result: ResearchResult) -> GuardedResearchPackage:
    """Convert one guarded industry result into a namespaced retained-source package."""
    if _is_rule_refusal(result.decision.intent):
        return _refused_package(result)
    if result.decision.intent is not Intent.INDUSTRY_RESEARCH_REQUEST:
        raise IndustrySubrunProtocolError(
            f"unexpected industry subrun intent: {result.decision.intent.value}"
        )

    skill_runs = list(result.skill_runs)
    if len(skill_runs) > 1:
        raise IndustrySubrunProtocolError(
            "industry workflow must not execute more than one recipe"
        )

    skill_run = skill_runs[0] if skill_runs else None
    if skill_run is not None and skill_run.recipe_name is not SkillName.INDUSTRY_RESEARCH:
        raise IndustrySubrunProtocolError("industry workflow executed the wrong recipe")

    if skill_run is None:
        return _failed_package(result)

    if skill_run.guarded_memo is None:
        return GuardedResearchPackage(
            ticker=result.ticker,
            claims=[],
            financial_metrics=[],
            filing_sources=[],
            web_sources=[],
            provenance=_empty_package_provenance(
                recipe_name=SkillName.INDUSTRY_RESEARCH,
                recipe_version=skill_run.recipe_version,
                source_policy_version=result.source_policy_version,
                report_as_of=result.report_as_of,
            ),
            evidence_dates=[],
            coverage="insufficient",
            information_gaps=["Insufficient evidence for industry research."],
            guard_notes=_unique(
                [*skill_run.errors, *result.errors]
            ),
        )

    guarded = skill_run.guarded_memo
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
        provenance=guarded.provenance,
        evidence_dates=_evidence_dates(skill_run),
        coverage=_coverage(guarded.memo.information_sufficiency),
        information_gaps=list(guarded.memo.information_gaps),
        guard_notes=_unique([*skill_run.errors, *guarded.guard_errors]),
    )


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
    return note or "Metric was not retained in the guarded industry package."


def _evidence_dates(skill_run: SkillRunResult) -> list[date]:
    guarded = skill_run.guarded_memo
    assert guarded is not None
    dates = [source.filed_at for source in guarded.filing_sources]
    dates.extend(
        source.published_at.date() if source.published_at is not None else source.fetched_at.date()
        for source in guarded.web_sources
    )
    return sorted(set(dates))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_rule_refusal(intent: Intent) -> bool:
    return intent in {
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
        Intent.UNSAFE_SOURCE_REQUEST,
    }


def _refused_package(result: ResearchResult) -> GuardedResearchPackage:
    code = result.decision.intent.value
    return GuardedResearchPackage(
        ticker=result.ticker,
        claims=[],
        financial_metrics=[],
        filing_sources=[],
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(),
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
        ),
        zero_recipe_outcome="refused",
        evidence_dates=[],
        coverage="insufficient",
        information_gaps=[],
        guard_notes=[code],
    )


def _failed_package(result: ResearchResult) -> GuardedResearchPackage:
    return GuardedResearchPackage(
        ticker=result.ticker,
        claims=[],
        financial_metrics=[],
        filing_sources=[],
        web_sources=[],
        provenance=ReportProvenance(
            requested_as_of_dates=(
                () if result.report_as_of is None else (result.report_as_of,)
            ),
            recipes=(),
            information_sufficiency=InformationSufficiency.INSUFFICIENT,
        ),
        zero_recipe_outcome="failed",
        evidence_dates=[],
        coverage="insufficient",
        information_gaps=[],
        guard_notes=_unique(result.errors) or ["INDUSTRY_RESEARCH_NOT_EXECUTED"],
    )


def _empty_package_provenance(
    *,
    recipe_name: SkillName,
    recipe_version: str,
    source_policy_version: str | None,
    report_as_of: date | None,
) -> ReportProvenance:
    prompt_version = current_prompt_version()
    return ReportProvenance(
        recipes=(RecipeProvenance(name=recipe_name, version=recipe_version),),
        source_policy_versions=(
            () if source_policy_version is None else (source_policy_version,)
        ),
        prompt_versions=(() if prompt_version is None else (prompt_version,)),
        requested_as_of_dates=(() if report_as_of is None else (report_as_of,)),
        information_sufficiency=InformationSufficiency.INSUFFICIENT,
    )
