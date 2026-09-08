"""Strict P2 contracts for guarded research packages and comparisons."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "ComparableMetric": (
        "financial_evidence_agent.research_packages.models",
        "ComparableMetric",
    ),
    "ComparabilityStatus": (
        "financial_evidence_agent.research_packages.models",
        "ComparabilityStatus",
    ),
    "GuardedResearchPackage": (
        "financial_evidence_agent.research_packages.models",
        "GuardedResearchPackage",
    ),
    "MetricComparison": (
        "financial_evidence_agent.research_packages.models",
        "MetricComparison",
    ),
    "MultiTickerResearchPackage": (
        "financial_evidence_agent.research_packages.models",
        "MultiTickerResearchPackage",
    ),
    "PackageClaim": (
        "financial_evidence_agent.research_packages.models",
        "PackageClaim",
    ),
    "PeerResearchOrchestrator": (
        "financial_evidence_agent.research_packages.orchestrator",
        "PeerResearchOrchestrator",
    ),
    "PeerResearchRequest": (
        "financial_evidence_agent.research_packages.models",
        "PeerResearchRequest",
    ),
    "PeerResearchResult": (
        "financial_evidence_agent.research_packages.orchestrator",
        "PeerResearchResult",
    ),
    "PeerScope": ("financial_evidence_agent.research_packages.models", "PeerScope"),
    "PeerScopeError": (
        "financial_evidence_agent.research_packages.orchestrator",
        "PeerScopeError",
    ),
    "QualityResearchResult": (
        "financial_evidence_agent.research_packages.quality",
        "QualityResearchResult",
    ),
    "QualityResearchRuntime": (
        "financial_evidence_agent.research_packages.quality",
        "QualityResearchRuntime",
    ),
    "ResearchQualityDecision": (
        "financial_evidence_agent.research_packages.models",
        "ResearchQualityDecision",
    ),
    "ResearchQualityResult": (
        "financial_evidence_agent.research_packages.models",
        "ResearchQualityResult",
    ),
    "ResolvedSource": (
        "financial_evidence_agent.research_packages.models",
        "ResolvedSource",
    ),
    "classify_quality_scope": (
        "financial_evidence_agent.research_packages.quality",
        "classify_quality_scope",
    ),
    "compare_metric_pair": (
        "financial_evidence_agent.research_packages.comparability",
        "compare_metric_pair",
    ),
    "compare_metrics": (
        "financial_evidence_agent.research_packages.comparability",
        "compare_metrics",
    ),
    "run_company_deep_research_subrun": (
        "financial_evidence_agent.research_packages.quality",
        "run_company_deep_research_subrun",
    ),
    "run_industry_subrun": (
        "financial_evidence_agent.research_packages.industry",
        "run_industry_subrun",
    ),
    "run_quality_screen": (
        "financial_evidence_agent.research_packages.quality",
        "run_quality_screen",
    ),
    "screen_research_quality": (
        "financial_evidence_agent.research_packages.quality",
        "screen_research_quality",
    ),
    "validate_peer_scope": (
        "financial_evidence_agent.research_packages.orchestrator",
        "validate_peer_scope",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    """Load public package exports lazily to avoid submodule import cycles."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
