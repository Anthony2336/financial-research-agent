"""Shared domain contracts for routing, retrieval, and reporting."""

import re
from datetime import date, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

FilingForm = Literal["10-K", "10-Q", "8-K"]
DEFAULT_RESEARCH_FORMS: tuple[FilingForm, ...] = ("10-K", "10-Q", "8-K")
_US_LISTED_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def is_valid_us_listed_ticker(value: str) -> bool:
    """Return whether a normalized symbol fits the bounded US ticker syntax."""
    return _US_LISTED_TICKER.fullmatch(value.strip().upper()) is not None


class StrictModel(BaseModel):
    """Base model that rejects fields outside the documented contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Intent(StrEnum):
    """Allowed outcomes from the request router."""

    RESEARCH_REQUEST = "research_request"
    COMPANY_PROFILE_REQUEST = "company_profile_request"
    EARNINGS_REVIEW_REQUEST = "earnings_review_request"
    MARKET_SNAPSHOT_REQUEST = "market_snapshot_request"
    INDUSTRY_RESEARCH_REQUEST = "industry_research_request"
    RESEARCH_QUALITY_SCREEN_REQUEST = "research_quality_screen_request"
    PROHIBITED_ADVICE = "prohibited_advice"
    UNSAFE_SOURCE_REQUEST = "unsafe_source_request"
    AMBIGUOUS = "ambiguous"
    PROMPT_INJECTION = "prompt_injection"


class SourceKind(StrEnum):
    """Origin category for web evidence."""

    FILING = "filing"
    ISSUER_IR = "issuer_ir"
    AUTHORITATIVE_WEB = "authoritative_web"


class SourceRefKind(StrEnum):
    """Persistence namespace for source-addressable evidence corpora."""

    FILING = "filing"
    WEB = "web"
    MARKET_SNAPSHOT = "market_snapshot"
    MARKET_BAR = "market_bar"


class SourceRef(StrictModel):
    """One resolved ticker/type/id reference with a stable storage encoding."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    kind: SourceRefKind
    source_id: str = Field(min_length=1, max_length=512)

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    def encode(self) -> str:
        """Encode a resolved key without losing ticker or source kind."""
        return f"{self.ticker}:{self.kind.value}:{self.source_id}"

    @classmethod
    def decode(cls, value: str, *, expected_ticker: str | None = None) -> "SourceRef":
        """Decode a resolved key and optionally enforce its owning ticker."""
        parts = value.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("source reference must contain ticker, kind, and source id")
        reference = cls(ticker=parts[0], kind=parts[1], source_id=parts[2])
        if expected_ticker is not None and reference.ticker != expected_ticker.strip().upper():
            raise ValueError("source reference ticker does not match expected ticker")
        return reference


class UnresolvedSourceRef(StrictModel):
    """Typed fallback for a legacy reference whose corpus cannot be proven."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    source_id: str = Field(min_length=1, max_length=512)
    reason: Literal["legacy_unresolved", "cross_ticker"]

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    def storage_value(self) -> str:
        """Return the explicit marker used by the data backfill migration."""
        return f"{self.ticker}:unresolved:{self.source_id}"


SourceReference = SourceRef | UnresolvedSourceRef


def decode_stored_source_ref(value: str, *, ticker: str) -> SourceReference:
    """Read resolved keys and turn every legacy or mismatched key into a typed fallback."""
    normalized_ticker = ticker.strip().upper()
    parts = value.split(":", maxsplit=2)
    if len(parts) == 3 and parts[1] == "unresolved" and parts[0] == normalized_ticker:
        return UnresolvedSourceRef(
            ticker=normalized_ticker,
            source_id=parts[2],
            reason="legacy_unresolved",
        )
    try:
        return SourceRef.decode(value, expected_ticker=normalized_ticker)
    except ValueError:
        if len(parts) == 3 and parts[1] in {kind.value for kind in SourceRefKind}:
            return UnresolvedSourceRef(
                ticker=normalized_ticker,
                source_id=parts[2],
                reason="cross_ticker",
            )
        return UnresolvedSourceRef(
            ticker=normalized_ticker,
            source_id=value,
            reason="legacy_unresolved",
        )


class SourceTier(StrEnum):
    """Reliability tier for web evidence."""

    PRIMARY = "primary"
    AUTHORITATIVE_SECONDARY = "authoritative_secondary"


class ClaimKind(StrEnum):
    """Grounding category for a report claim."""

    VERIFIED_FACT = "verified_fact"
    INFERENCE = "inference"
    OPEN_QUESTION = "open_question"


class Confidence(StrEnum):
    """Coarse confidence label used in structured output."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RouterDecision(StrictModel):
    """Structured result produced by deterministic or model routing."""

    intent: Intent
    reason: str = Field(min_length=1, max_length=200)


class IntentRoutingErrorCode(StrEnum):
    """Stable structured-routing failures safe to expose at command boundaries."""

    FAST_ROUTE_INVALID = "FAST_ROUTE_INVALID"
    INVALID_PEER_SCOPE = "INVALID_PEER_SCOPE"
    PEER_LIMIT_EXCEEDED = "PEER_LIMIT_EXCEEDED"


class IntentRoutingError(RuntimeError):
    """Fail-closed structured-router boundary error."""

    def __init__(self, code: IntentRoutingErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class ResearchQuestion(StrictModel):
    """One research angle with explicit support and challenge queries."""

    question: str = Field(min_length=1, max_length=500)
    support_query: str = Field(min_length=1, max_length=500)
    challenge_query: str = Field(min_length=1, max_length=500)
    period: str | None = Field(default=None, max_length=100)
    forms: list[FilingForm] = Field(default_factory=list)


class EvidenceChunk(StrictModel):
    """A citable span from one filing in the current corpus."""

    id: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    corpus_version: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https://")
    form: Literal["10-K", "10-Q", "8-K"]
    filed_at: date
    accession_no: str = Field(min_length=1)
    section: str = Field(min_length=1)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_raw_span(self) -> "EvidenceChunk":
        if self.raw_end <= self.raw_start:
            raise ValueError("raw_end must be greater than raw_start")
        return self


class WebEvidence(StrictModel):
    """A citable document retrieved from an approved web source."""

    id: str
    ticker: str
    title: str
    content: str
    source_url: HttpUrl
    source_kind: SourceKind
    source_tier: SourceTier
    published_at: datetime | None
    fetched_at: datetime
    content_hash: str

    @field_validator("fetched_at")
    @classmethod
    def require_timezone_aware_fetched_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must include timezone information")
        return value


def canonicalize_source_url(value: HttpUrl | str) -> HttpUrl:
    """Return one fragment-free canonical HTTPS URL or fail closed."""
    url = value if isinstance(value, HttpUrl) else HttpUrl(value)
    if url.scheme != "https" or url.username is not None or url.password is not None:
        raise ValueError("source URL must use HTTPS and omit user information")
    parsed = urlsplit(str(url))
    if parsed.port not in {None, 443} or not parsed.hostname:
        raise ValueError("source URL must use the default HTTPS service")
    host = url.host or parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    canonical = urlunsplit(("https", host.lower(), parsed.path or "/", parsed.query, ""))
    return HttpUrl(canonical)


def content_addressed_web_evidence_id(
    ticker: str,
    source_url: HttpUrl | str,
    content_hash: str,
) -> str:
    """Return the shared ticker/URL/content-hash evidence identity."""
    canonical_url = canonicalize_source_url(source_url)
    return sha256(f"{ticker.strip().upper()}\n{canonical_url}\n{content_hash}".encode()).hexdigest()


class Claim(StrictModel):
    """One fact, inference, or open question in a research memo."""

    kind: ClaimKind
    text: str = Field(min_length=1, max_length=2_000)
    confidence: Confidence
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    web_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_verified_fact(self) -> "Claim":
        if (
            self.kind is ClaimKind.VERIFIED_FACT
            and not self.evidence_chunk_ids
            and not self.web_evidence_ids
        ):
            raise ValueError("verified facts require at least one evidence id")
        return self


class ResearchMemo(StrictModel):
    """Structured analyst output before citation validation and rendering."""

    research_question: str = Field(min_length=1, max_length=500)
    supporting_claims: list[Claim] = Field(default_factory=list)
    counter_claims: list[Claim] = Field(default_factory=list)
    inferences: list[Claim] = Field(default_factory=list)
    open_questions: list[Claim] = Field(default_factory=list)
    information_sufficiency: Literal["A", "B", "C"]
    confidence: Confidence
