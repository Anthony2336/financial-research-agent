"""Strict contracts for guarded P2 research packages and peer comparisons."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from fra.domain import (
    ClaimKind,
    Confidence,
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    StrictModel,
)
from fra.skills.models import ResearchFacet
from fra.skills.schemas import (
    FinancialSourceProvenance,
    InformationSufficiency,
    ReportProvenance,
    VerificationStatus,
    canonical_financial_text,
    financial_observation_id,
)
from fra.web_evidence.source_policy import PolicyValidatedWebEvidence


def _normalize_ticker(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


def _duplicate_values(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return duplicates


_ZERO_RECIPE_REFUSAL_CODES = frozenset(
    {
        "prohibited_advice",
        "prompt_injection",
        "unsafe_source_request",
    }
)


class ResearchQualityDecision(StrEnum):
    """Deterministic quality-screen outcomes for one guarded package."""

    WORTH_FURTHER_RESEARCH = "worth_further_research"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    OUT_OF_SCOPE = "out_of_scope"


class ComparabilityStatus(StrEnum):
    """Whether a metric can be compared without transformation or guesswork."""

    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    MISSING = "missing"
    DISCREPANCY = "discrepancy"


class PackageClaim(StrictModel):
    """One guarded claim carried into a P2 package using namespaced source refs."""

    facet: ResearchFacet
    kind: ClaimKind
    text: str = Field(min_length=1)
    confidence: Confidence
    source_refs: list[SourceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_source_refs_for_verified_fact(self) -> PackageClaim:
        if self.kind is ClaimKind.VERIFIED_FACT and not self.source_refs:
            raise ValueError("verified facts require at least one source ref")
        return self


class ComparableMetric(StrictModel):
    """One exact reported metric with explicit comparability status and namespaced sources."""

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    name: str = Field(min_length=1)
    value: Decimal | None
    period_start: date | None
    period_end: date
    currency: str | None
    unit: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    observation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_refs: list[SourceRef] = Field(default_factory=list)
    source_provenance: list[FinancialSourceProvenance] = Field(default_factory=list)
    verification_status: VerificationStatus
    status: ComparabilityStatus
    limitation: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return _normalize_ticker(value)

    @model_validator(mode="after")
    def validate_metric(self) -> ComparableMetric:
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start must not be after period_end")
        if self.observation_id != financial_observation_id(
            ticker=self.ticker,
            name=self.name,
            period_start=self.period_start,
            period_end=self.period_end,
            currency=self.currency,
            unit=self.unit,
            definition=self.definition,
        ):
            raise ValueError("observation_id does not match the canonical financial identity")
        if self.status is ComparabilityStatus.COMPARABLE and self.value is None:
            raise ValueError("comparable metrics require a value")
        if self.status is ComparabilityStatus.MISSING and self.value is not None:
            raise ValueError("missing metrics cannot carry a value")
        if (
            self.status
            in {
                ComparabilityStatus.NOT_COMPARABLE,
                ComparabilityStatus.MISSING,
                ComparabilityStatus.DISCREPANCY,
            }
            and not self.limitation
        ):
            raise ValueError("limitation is required when a metric is not directly comparable")
        if self.status is not ComparabilityStatus.MISSING and not self.source_refs:
            raise ValueError("non-missing metrics require at least one source ref")
        if self.source_refs != [source.source_ref for source in self.source_provenance]:
            raise ValueError("metric source provenance must match exact source refs in order")
        expected_status = {
            VerificationStatus.VERIFIED: ComparabilityStatus.COMPARABLE,
            VerificationStatus.SINGLE_SOURCE: ComparabilityStatus.COMPARABLE,
            VerificationStatus.DISCREPANCY: ComparabilityStatus.DISCREPANCY,
            VerificationStatus.NOT_COMPARABLE: ComparabilityStatus.NOT_COMPARABLE,
            VerificationStatus.MISSING: ComparabilityStatus.MISSING,
        }[self.verification_status]
        if self.status is not expected_status:
            raise ValueError("metric comparability status must preserve verification status")
        return self


class MetricComparison(StrictModel):
    """A comparison view over one metric across one or more guarded packages."""

    name: str = Field(min_length=1)
    observations: list[ComparableMetric] = Field(min_length=1)
    status: ComparabilityStatus
    delta: Decimal | None = None
    limitation: str | None = None

    @model_validator(mode="after")
    def validate_observations(self) -> MetricComparison:
        if (
            self.status is ComparabilityStatus.COMPARABLE
            and any(
                canonical_financial_text(observation.name)
                != canonical_financial_text(self.name)
                for observation in self.observations
            )
        ):
            raise ValueError("every observation name must match the metric comparison name")
        if (
            self.status
            in {
                ComparabilityStatus.NOT_COMPARABLE,
                ComparabilityStatus.MISSING,
                ComparabilityStatus.DISCREPANCY,
            }
            and not self.limitation
        ):
            raise ValueError("limitation is required when a comparison is not comparable")
        if self.status is not ComparabilityStatus.COMPARABLE and self.delta is not None:
            raise ValueError("only comparable metrics can retain a delta")
        return self


class GuardedResearchPackage(StrictModel):
    """One ticker's guarded P2 research state with retained evidence only."""

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    claims: list[PackageClaim] = Field(default_factory=list)
    financial_metrics: list[ComparableMetric] = Field(default_factory=list)
    filing_sources: list[EvidenceChunk] = Field(default_factory=list)
    web_sources: list[PolicyValidatedWebEvidence] = Field(default_factory=list)
    provenance: ReportProvenance
    zero_recipe_outcome: Literal["failed", "refused"] | None = None
    evidence_dates: list[date] = Field(default_factory=list)
    coverage: Literal["complete", "partial", "insufficient"]
    information_gaps: list[str] = Field(default_factory=list)
    guard_notes: list[str] = Field(default_factory=list)

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return _normalize_ticker(value)

    @model_validator(mode="after")
    def validate_package_scope(self) -> GuardedResearchPackage:
        if not self.provenance.recipes:
            if self.coverage != "insufficient":
                raise ValueError("zero-recipe packages must use insufficient coverage")
            if any(
                (
                    self.claims,
                    self.financial_metrics,
                    self.filing_sources,
                    self.web_sources,
                    self.evidence_dates,
                )
            ):
                raise ValueError("packages without executed recipes cannot retain evidence")
            if self.information_gaps:
                raise ValueError("zero-recipe packages cannot report information gaps")
            if self.zero_recipe_outcome is None:
                raise ValueError("zero-recipe packages require an explicit failed/refused outcome")
            if self.zero_recipe_outcome == "refused":
                if (
                    len(self.guard_notes) != 1
                    or self.guard_notes[0] not in _ZERO_RECIPE_REFUSAL_CODES
                ):
                    raise ValueError(
                        "zero-recipe refusals must encode exactly one approved refusal code"
                    )
            elif not self.guard_notes or any(
                note in _ZERO_RECIPE_REFUSAL_CODES for note in self.guard_notes
            ):
                raise ValueError(
                    "zero-recipe failures require non-refusal guard notes"
                )
        elif self.zero_recipe_outcome is not None:
            raise ValueError("executed-recipe packages cannot set zero_recipe_outcome")
        filing_source_keys = [
            SourceRef(ticker=self.ticker, kind=SourceRefKind.FILING, source_id=source.id).encode()
            for source in self.filing_sources
        ]
        web_source_keys = [
            SourceRef(ticker=self.ticker, kind=SourceRefKind.WEB, source_id=source.id).encode()
            for source in self.web_sources
        ]
        if duplicates := _duplicate_values(filing_source_keys):
            raise ValueError(f"duplicate filing source: {sorted(duplicates)[0]}")
        if duplicates := _duplicate_values(web_source_keys):
            raise ValueError(f"duplicate web source: {sorted(duplicates)[0]}")
        retained_source_refs = set(filing_source_keys)
        retained_source_refs.update(web_source_keys)
        exact_source_refs = tuple(
            SourceRef.decode(value, expected_ticker=self.ticker)
            for value in (*filing_source_keys, *web_source_keys)
        )
        if self.provenance.source_refs != exact_source_refs:
            raise ValueError("package provenance source refs must match retained sources")
        corpus_versions = tuple(
            dict.fromkeys(source.corpus_version for source in self.filing_sources)
        )
        if self.provenance.corpus_versions != corpus_versions:
            raise ValueError("package provenance corpora must match retained filing sources")
        expected_sufficiency = {
            "complete": InformationSufficiency.SUFFICIENT,
            "partial": InformationSufficiency.PARTIAL,
            "insufficient": InformationSufficiency.INSUFFICIENT,
        }[self.coverage]
        if self.provenance.information_sufficiency is not expected_sufficiency:
            raise ValueError("package provenance sufficiency must match coverage")
        for source in self.filing_sources:
            if source.ticker.strip().upper() != self.ticker:
                raise ValueError("filing source ticker does not match package ticker")
        for source in self.web_sources:
            if source.ticker.strip().upper() != self.ticker:
                raise ValueError("web source ticker does not match package ticker")
        for claim in self.claims:
            for source_ref in claim.source_refs:
                self._validate_source_ref(source_ref, retained_source_refs)
        for metric in self.financial_metrics:
            if metric.ticker != self.ticker:
                raise ValueError("financial metric ticker does not match package ticker")
            for source_ref in metric.source_refs:
                self._validate_source_ref(source_ref, retained_source_refs)
        return self

    def _validate_source_ref(
        self,
        source_ref: SourceRef,
        retained_source_refs: set[str],
    ) -> None:
        if source_ref.ticker != self.ticker:
            raise ValueError("source ref ticker does not match package ticker")
        if source_ref.kind not in {SourceRefKind.FILING, SourceRefKind.WEB}:
            raise ValueError("source ref kind is not allowed for the retained source set")
        if source_ref.encode() not in retained_source_refs:
            raise ValueError("source ref is absent from the retained source set")


class ResearchQualityResult(StrictModel):
    """Deterministic quality-screen result with explicit retained-source support."""

    decision: ResearchQualityDecision
    reasons: list[str] = Field(min_length=1)
    source_refs: list[SourceRef] = Field(default_factory=list)


class PeerResearchRequest(StrictModel):
    """Explicit, bounded peer-comparison request supplied by the caller."""

    primary_ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    peer_tickers: tuple[str, ...] = ()
    peer_scope: str = Field(min_length=1)
    question: str = Field(min_length=1)

    @field_validator("primary_ticker", mode="before")
    @classmethod
    def normalize_primary_ticker(cls, value: object) -> object:
        return _normalize_ticker(value)

    @field_validator("peer_tickers", mode="before")
    @classmethod
    def normalize_peer_tickers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(_normalize_ticker(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_peers(self) -> PeerResearchRequest:
        if any(not ticker for ticker in self.peer_tickers):
            raise ValueError("peer_tickers must not contain blank tickers")
        if len(self.peer_tickers) > 3:
            raise ValueError("peer_tickers must contain at most three tickers")
        if self.primary_ticker in self.peer_tickers:
            raise ValueError("primary ticker cannot appear in peer_tickers")
        if len(set(self.peer_tickers)) != len(self.peer_tickers):
            raise ValueError("peer_tickers must be unique")
        return self


class PeerScope(StrictModel):
    """Resolved, bounded peer scope after deterministic validation."""

    primary_ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    peer_tickers: tuple[str, ...]
    description: str = Field(min_length=1)

    @field_validator("primary_ticker", mode="before")
    @classmethod
    def normalize_primary_ticker(cls, value: object) -> object:
        return _normalize_ticker(value)

    @field_validator("peer_tickers", mode="before")
    @classmethod
    def normalize_peer_tickers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(_normalize_ticker(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_peers(self) -> PeerScope:
        if any(not ticker for ticker in self.peer_tickers):
            raise ValueError("peer_tickers must not contain blank tickers")
        if len(self.peer_tickers) > 3:
            raise ValueError("peer_tickers must contain at most three tickers")
        if self.primary_ticker in self.peer_tickers:
            raise ValueError("primary ticker cannot appear in peer_tickers")
        if len(set(self.peer_tickers)) != len(self.peer_tickers):
            raise ValueError("peer_tickers must be unique")
        return self


class MultiTickerResearchPackage(StrictModel):
    """Bounded multi-ticker output that keeps each guarded package isolated."""

    primary_ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    packages: list[GuardedResearchPackage] = Field(default_factory=list)
    missing_tickers: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"]
    cross_ticker_leakage_count: int = Field(default=0, ge=0)
    guard_notes: list[str] = Field(default_factory=list)

    @field_validator("primary_ticker", mode="before")
    @classmethod
    def normalize_primary_ticker(cls, value: object) -> object:
        return _normalize_ticker(value)

    @field_validator("missing_tickers", mode="before")
    @classmethod
    def normalize_missing_tickers(cls, value: object) -> object:
        if isinstance(value, list):
            return [_normalize_ticker(item) for item in value]
        return value

    @model_validator(mode="after")
    def validate_tickers(self) -> MultiTickerResearchPackage:
        package_tickers = [package.ticker for package in self.packages]
        if len(set(package_tickers)) != len(package_tickers):
            raise ValueError("duplicate package ticker")
        if len(set(self.missing_tickers)) != len(self.missing_tickers):
            raise ValueError("missing_tickers must be unique")
        overlapping_tickers = set(package_tickers).intersection(self.missing_tickers)
        if self.primary_ticker in overlapping_tickers:
            raise ValueError("primary ticker cannot appear in both packages and missing_tickers")
        if overlapping_tickers:
            raise ValueError("package tickers cannot also appear in missing_tickers")
        primary_package_count = package_tickers.count(self.primary_ticker)
        if self.status in {"completed", "partial"} and primary_package_count != 1:
            raise ValueError("completed and partial statuses require exactly one primary package")
        if self.status == "failed":
            if self.primary_ticker not in self.missing_tickers:
                raise ValueError(
                    "failed multi-ticker package must include the primary ticker "
                    "in missing_tickers"
                )
            if self.packages:
                raise ValueError("failed multi-ticker package cannot carry peer-only packages")
        return self


class ResolvedSource(StrictModel):
    """A rendered, namespaced source pointer with human-facing metadata."""

    ref: SourceRef
    title: str = Field(min_length=1)
    source_url: HttpUrl
    published_or_filed_at: datetime | date
    source_kind: SourceKind
    source_tier: SourceTier
    canonical_source_identity: str = Field(min_length=1, max_length=2_500)


class GuardedP2Report(StrictModel):
    """Final guarded P2 view model rendered and persisted by the application."""

    scope: PeerScope
    packages: list[GuardedResearchPackage] = Field(default_factory=list)
    comparisons: list[MetricComparison] = Field(default_factory=list)
    quality: ResearchQualityResult | None = None
    retained_sources: list[ResolvedSource] = Field(default_factory=list)
    provenance: ReportProvenance
    information_sufficiency: InformationSufficiency
    cross_ticker_rejection_count: int = Field(default=0, ge=0)
    cross_ticker_leakage_count: int = Field(default=0, ge=0)
    guard_errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retained_source_keys(self) -> GuardedP2Report:
        source_keys = [source.ref.encode() for source in self.retained_sources]
        if duplicates := _duplicate_values(source_keys):
            raise ValueError(f"duplicate retained source: {sorted(duplicates)[0]}")
        if self.provenance.source_refs != tuple(
            source.ref for source in self.retained_sources
        ):
            raise ValueError("P2 provenance source refs must match retained sources")
        if self.provenance.information_sufficiency is not self.information_sufficiency:
            raise ValueError("P2 provenance sufficiency must match the guarded report")
        return self
