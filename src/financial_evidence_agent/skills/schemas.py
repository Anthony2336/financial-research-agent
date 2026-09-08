"""Structured output contracts for P1 research recipes."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Annotated
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from financial_evidence_agent.domain import (
    Claim,
    EvidenceChunk,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    canonicalize_source_url,
)
from financial_evidence_agent.skills.models import ResearchFacet, SkillName
from financial_evidence_agent.web_evidence.source_policy import PolicyValidatedWebEvidence


class InformationSufficiency(StrEnum):
    """How completely a recipe's required facets are supported."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class VerificationStatus(StrEnum):
    """Reconciliation result for one reported financial value."""

    VERIFIED = "verified"
    SINGLE_SOURCE = "single_source"
    DISCREPANCY = "discrepancy"
    NOT_COMPARABLE = "not_comparable"
    MISSING = "missing"


class FinancialDataPoint(BaseModel):
    """One analyst-supplied exact financial observation awaiting deterministic guard."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    value: Decimal | None
    currency: str | None
    unit: str = Field(min_length=1)
    period_start: date | None
    period_end: date
    definition: str = Field(min_length=1)
    source_ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_period(self) -> FinancialDataPoint:
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start must not be after period_end")
        return self


class FinancialSourceProvenance(BaseModel):
    """Guard-derived provenance for one exact retained financial source object."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    source_ref: SourceRef
    source_kind: SourceKind
    source_tier: SourceTier
    canonical_source_identity: str = Field(min_length=1, max_length=2_500)

    @model_validator(mode="after")
    def validate_source_namespace(self) -> FinancialSourceProvenance:
        if self.source_ref.kind not in {SourceRefKind.FILING, SourceRefKind.WEB}:
            raise ValueError("financial sources require a filing or web storage namespace")
        if (
            self.source_ref.kind is SourceRefKind.FILING
            and self.source_kind is not SourceKind.FILING
        ):
            raise ValueError("filing storage requires filing origin kind")
        return self


def financial_observation_id(
    *,
    ticker: str,
    name: str,
    period_start: date | None,
    period_end: date,
    currency: str | None,
    unit: str,
    definition: str,
) -> str:
    """Return one canonical exact-identity digest that deliberately excludes value."""

    fields = (
        ticker.strip().upper(),
        *financial_metric_identity_components(
            name=name,
            period_start=period_start,
            period_end=period_end,
            currency=currency,
            unit=unit,
            definition=definition,
        ),
    )
    return sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def canonical_financial_text(value: str) -> str:
    """Normalize presentation-only Unicode, case, and whitespace differences."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def financial_metric_identity_components(
    *,
    name: str,
    period_start: date | None,
    period_end: date,
    currency: str | None,
    unit: str,
    definition: str,
) -> tuple[str, ...]:
    """Return the canonical cross-ticker identity used by guard and comparability."""

    return (
        canonical_financial_text(name),
        "" if period_start is None else period_start.isoformat(),
        period_end.isoformat(),
        "" if currency is None else currency.strip().upper(),
        canonical_financial_text(unit),
        canonical_financial_text(definition),
    )


_DASHED_ACCESSION = re.compile(r"(?<!\d)(\d{10})-(\d{2})-(\d{6})(?!\d)")
_COMPACT_ACCESSION = re.compile(r"(?<!\d)(\d{10})(\d{2})(\d{6})(?!\d)")


def canonical_financial_source_identity(
    *,
    storage_kind: SourceRefKind,
    source_kind: SourceKind,
    source_url: str,
    content_hash: str | None,
    accession_no: str | None = None,
) -> str:
    """Return a conservative document/authority identity for independence counting.

    Exact source refs and canonical URLs remain available for reproduction. Independence is
    intentionally stricter: SEC filing/web aliases share an accession identity, while other
    web rows from one publisher with identical content share an authority-content identity.
    Tracking and meaningful query parameters therefore remain in the exact URL without letting
    duplicate publisher content masquerade as an independent document.
    """

    if accession_no is not None:
        return f"filing-accession:{accession_no}"
    canonical_url = canonicalize_source_url(source_url)
    parsed = urlsplit(str(canonical_url))
    host = (parsed.hostname or "").lower()
    if source_kind is SourceKind.FILING or host == "sec.gov" or host.endswith(".sec.gov"):
        recovered = _sec_accession_from_url(str(canonical_url))
        if recovered is not None:
            return f"filing-accession:{recovered}"
    if storage_kind is SourceRefKind.FILING:
        raise ValueError("filing source identity requires an SEC accession")
    if content_hash:
        return f"web-authority-content:{host}:{content_hash.casefold()}"
    return f"web-document:{canonical_url}"


def _sec_accession_from_url(value: str) -> str | None:
    decoded = unquote(value)
    if match := _DASHED_ACCESSION.search(decoded):
        return "-".join(match.groups())
    if match := _COMPACT_ACCESSION.search(decoded):
        return "-".join(match.groups())
    return None


class GuardedFinancialDataPoint(FinancialDataPoint):
    """One retained financial observation with only guard-derived final metadata."""

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    observation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_provenance: list[FinancialSourceProvenance] = Field(min_length=1)
    verification_status: VerificationStatus
    discrepancy_note: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_guarded_observation(self) -> GuardedFinancialDataPoint:
        expected_id = financial_observation_id(
            ticker=self.ticker,
            name=self.name,
            period_start=self.period_start,
            period_end=self.period_end,
            currency=self.currency,
            unit=self.unit,
            definition=self.definition,
        )
        if self.observation_id != expected_id:
            raise ValueError("observation_id does not match the canonical financial identity")
        source_ids = [source.source_ref.source_id for source in self.source_provenance]
        if source_ids != self.source_ids:
            raise ValueError("source provenance must resolve every exact source id in order")
        if any(source.source_ref.ticker != self.ticker for source in self.source_provenance):
            raise ValueError("financial source ticker does not match observation ticker")
        if self.value is None and self.verification_status is not VerificationStatus.MISSING:
            raise ValueError("missing exact values must use missing verification status")
        if self.value is not None and self.verification_status is VerificationStatus.MISSING:
            raise ValueError("present exact values cannot use missing verification status")
        if (
            self.verification_status
            in {VerificationStatus.DISCREPANCY, VerificationStatus.NOT_COMPARABLE}
            and not self.discrepancy_note
        ):
            raise ValueError("discrepancy_note is required for conflicting financial data")
        return self


class RecipeProvenance(BaseModel):
    """One frozen recipe identity actually executed for a guarded report."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    name: SkillName
    version: str = Field(min_length=1)


class ReportProvenance(BaseModel):
    """Structured reproducibility inputs retained independently of Markdown."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    recipes: tuple[RecipeProvenance, ...]
    source_policy_versions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    corpus_versions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    prompt_versions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    requested_as_of_dates: tuple[date, ...] = ()
    evidence_cutoff_dates: tuple[date, ...] = ()
    information_sufficiency: InformationSufficiency
    source_refs: tuple[SourceRef, ...] = ()

    @model_validator(mode="after")
    def require_unique_reproducibility_values(self) -> ReportProvenance:
        values = (
            [(recipe.name.value, recipe.version) for recipe in self.recipes],
            list(self.source_policy_versions),
            list(self.corpus_versions),
            list(self.prompt_versions),
            list(self.requested_as_of_dates),
            list(self.evidence_cutoff_dates),
            [source.encode() for source in self.source_refs],
        )
        if any(len(items) != len(set(items)) for items in values):
            raise ValueError("report provenance values must be unique and ordered")
        return self


class SkillResearchSection(BaseModel):
    """Claims produced for one declared recipe facet."""

    model_config = ConfigDict(extra="forbid")

    facet: ResearchFacet
    claims: list[Claim]


class SkillResearchMemo(BaseModel):
    """Version-bound structured output from a P1 research recipe."""

    model_config = ConfigDict(extra="forbid")

    recipe_name: SkillName = Field(frozen=True)
    recipe_version: str = Field(min_length=1, frozen=True)
    research_question: str = Field(min_length=1)
    sections: list[SkillResearchSection]
    data_points: list[FinancialDataPoint] = Field(default_factory=list)
    information_sufficiency: InformationSufficiency
    information_gaps: list[str]
    confidence: Decimal = Field(ge=0, le=1)


class GuardedSkillResearchMemo(SkillResearchMemo):
    """A recipe memo whose financial observations carry guard-derived metadata."""

    data_points: list[GuardedFinancialDataPoint] = Field(default_factory=list)


class GuardedSkillMemo(BaseModel):
    """A P1 memo plus only the cited sources retained by its guard."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    memo: GuardedSkillResearchMemo
    filing_sources: list[EvidenceChunk]
    web_sources: list[PolicyValidatedWebEvidence]
    provenance: ReportProvenance
    guard_errors: list[str]

    @model_validator(mode="after")
    def validate_report_provenance(self) -> GuardedSkillMemo:
        if self.provenance.information_sufficiency is not self.memo.information_sufficiency:
            raise ValueError("report provenance sufficiency must match the guarded memo")
        if self.provenance.recipes != (
            RecipeProvenance(name=self.memo.recipe_name, version=self.memo.recipe_version),
        ):
            raise ValueError("report provenance recipe must match the guarded memo")
        expected_corpora = tuple(
            dict.fromkeys(source.corpus_version for source in self.filing_sources)
        )
        if self.provenance.corpus_versions != expected_corpora:
            raise ValueError("report provenance corpora must match retained filing sources")
        expected_refs = tuple(
            [
                *(
                    SourceRef(
                        ticker=self.ticker,
                        kind=SourceRefKind.FILING,
                        source_id=source.id,
                    )
                    for source in self.filing_sources
                ),
                *(
                    SourceRef(
                        ticker=self.ticker,
                        kind=SourceRefKind.WEB,
                        source_id=source.id,
                    )
                    for source in self.web_sources
                ),
            ]
        )
        if self.provenance.source_refs != expected_refs:
            raise ValueError("report provenance source refs must match retained sources")
        return self
