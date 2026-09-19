"""Research packages, comparisons, orchestration and quality screening."""

from fra.research_packages.comparability import (
    compare_metric_pair,
    compare_metrics,
)
from fra.research_packages.industry import (
    run_industry_subrun,
)
from fra.research_packages.models import (
    ComparabilityStatus,
    ComparableMetric,
    GuardedResearchPackage,
    MetricComparison,
    MultiTickerResearchPackage,
    PackageClaim,
    PeerResearchRequest,
    PeerScope,
    ResearchQualityDecision,
    ResearchQualityResult,
    ResolvedSource,
)
from fra.research_packages.orchestrator import (
    PeerResearchOrchestrator,
    PeerResearchResult,
    PeerScopeError,
    validate_peer_scope,
)
from fra.research_packages.quality import (
    QualityResearchResult,
    QualityResearchRuntime,
    classify_quality_scope,
    run_company_deep_research_subrun,
    run_quality_screen,
    screen_research_quality,
)

__all__ = [
    "ComparableMetric",
    "ComparabilityStatus",
    "GuardedResearchPackage",
    "MetricComparison",
    "MultiTickerResearchPackage",
    "PackageClaim",
    "PeerResearchOrchestrator",
    "PeerResearchRequest",
    "PeerResearchResult",
    "PeerScope",
    "PeerScopeError",
    "QualityResearchResult",
    "QualityResearchRuntime",
    "ResearchQualityDecision",
    "ResearchQualityResult",
    "ResolvedSource",
    "classify_quality_scope",
    "compare_metric_pair",
    "compare_metrics",
    "run_company_deep_research_subrun",
    "run_industry_subrun",
    "run_quality_screen",
    "screen_research_quality",
    "validate_peer_scope",
]
