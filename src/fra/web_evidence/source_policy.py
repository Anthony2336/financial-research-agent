"""Frozen URL ownership policy and persisted web-evidence revalidation."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

from fra.domain import (
    SourceKind,
    SourceTier,
    WebEvidence,
    canonicalize_source_url,
    content_addressed_web_evidence_id,
)

DEFAULT_STANDARD_AUTHORITY_DOMAINS = frozenset(
    {"sec.gov", "www.sec.gov", "reuters.com", "www.reuters.com"}
)
DEFAULT_INDUSTRY_AUTHORITY_DOMAINS = frozenset(
    {"ftc.gov", "justice.gov", "commerce.gov", "federalregister.gov"}
)
_SOURCE_POLICY_SCHEMA_VERSION = "1"


class SourcePolicyError(ValueError):
    """A URL does not belong to an approved source for the requested issuer."""

    code: Literal["SOURCE_NOT_ALLOWED"] = "SOURCE_NOT_ALLOWED"


class SourcePolicy(BaseModel):
    """Classify HTTPS URLs only after exact issuer/domain ownership validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issuer_domains: Mapping[str, frozenset[str]]
    authority_domains: frozenset[str] = DEFAULT_STANDARD_AUTHORITY_DOMAINS
    policy_schema_version: str = _SOURCE_POLICY_SCHEMA_VERSION

    @field_validator("issuer_domains")
    @classmethod
    def normalize_issuer_domains(
        cls, domains: Mapping[str, frozenset[str]]
    ) -> Mapping[str, frozenset[str]]:
        return {
            ticker.strip().upper(): frozenset(_normalize_domain(domain) for domain in values)
            for ticker, values in domains.items()
        }

    @field_validator("authority_domains")
    @classmethod
    def normalize_authority_domains(cls, domains: frozenset[str]) -> frozenset[str]:
        return frozenset(_normalize_domain(domain) for domain in domains)

    def model_post_init(self, __context: object) -> None:
        del __context
        object.__setattr__(
            self,
            "issuer_domains",
            MappingProxyType(dict(self.issuer_domains)),
        )

    @property
    def version(self) -> str:
        """Return a stable digest of the exact immutable policy snapshot."""
        snapshot = json.dumps(
            {
                "authority_domains": sorted(self.authority_domains),
                "issuer_domains": {
                    ticker: sorted(domains)
                    for ticker, domains in sorted(self.issuer_domains.items())
                },
                "policy_schema_version": self.policy_schema_version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(snapshot.encode()).hexdigest()

    def provider_domains(self, ticker: str) -> tuple[str, ...]:
        """Return the exact frozen domains a search provider may query for one ticker."""
        issuer_hosts = self.issuer_domains.get(ticker.strip().upper(), frozenset())
        return tuple(sorted(self.authority_domains | issuer_hosts))

    def classify(self, *, ticker: str, url: HttpUrl) -> tuple[SourceKind, SourceTier]:
        """Return provenance type and tier, or reject an unowned source URL."""
        if url.scheme != "https" or url.username is not None or url.password is not None:
            raise SourcePolicyError("source URL must use HTTPS and omit user information")
        try:
            explicit_port = urlsplit(str(url)).port
        except ValueError as error:
            raise SourcePolicyError("source URL has an invalid port") from error
        if explicit_port not in {None, 443}:
            raise SourcePolicyError("source URL must use the default HTTPS port")
        host = (url.host or "").lower()
        if not host:
            raise SourcePolicyError("source URL must include a host")

        issuer_hosts = self.issuer_domains.get(ticker.strip().upper(), frozenset())
        if any(_is_host_or_descendant(host, domain) for domain in issuer_hosts):
            return SourceKind.ISSUER_IR, SourceTier.PRIMARY

        matched_authority = next(
            (
                domain
                for domain in self.authority_domains
                if _is_host_or_descendant(host, domain)
            ),
            None,
        )
        if matched_authority is not None:
            if _is_host_or_descendant(host, "sec.gov"):
                return SourceKind.FILING, SourceTier.PRIMARY
            return SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY
        raise SourcePolicyError(f"source host is not approved for {ticker.strip().upper()}")


class PolicyValidatedWebEvidence(WebEvidence):
    """A persisted canonical web row validated under one frozen policy."""

    policy_version: str
    canonical_url: HttpUrl


class WebEvidenceReader(Protocol):
    """Read boundary needed for final persisted-row revalidation."""

    def get_many(self, evidence_ids: Sequence[str]) -> list[WebEvidence]:
        """Return canonical rows in caller order, omitting unknown IDs."""


class WebEvidenceValidationError(ValueError):
    """Stable final-guard failure for a web source."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class PersistedWebEvidenceValidator:
    """Revalidate MCP web evidence against policy and its canonical persisted row."""

    def __init__(self, policy: SourcePolicy, repository: WebEvidenceReader) -> None:
        self._policy = policy
        self._repository = repository

    @property
    def policy(self) -> SourcePolicy:
        return self._policy

    @property
    def policy_version(self) -> str:
        return self._policy.version

    def validate(self, *, ticker: str, evidence: WebEvidence) -> PolicyValidatedWebEvidence:
        """Return renderable evidence only after every provenance check succeeds."""
        normalized_ticker = ticker.strip().upper()
        if evidence.ticker.strip().upper() != normalized_ticker:
            raise WebEvidenceValidationError("WEB_SOURCE_TICKER_MISMATCH")
        published_at = evidence.published_at
        fetched_at = evidence.fetched_at
        if (
            published_at is None
            or published_at.tzinfo is None
            or published_at.utcoffset() is None
            or fetched_at.tzinfo is None
            or fetched_at.utcoffset() is None
            or published_at.astimezone(UTC) > fetched_at.astimezone(UTC)
        ):
            raise WebEvidenceValidationError("WEB_SOURCE_TIME_INVALID")
        try:
            canonical_url = canonicalize_source_url(evidence.source_url)
            source_kind, source_tier = self._policy.classify(
                ticker=normalized_ticker,
                url=canonical_url,
            )
        except (SourcePolicyError, ValueError) as error:
            raise WebEvidenceValidationError("WEB_SOURCE_POLICY_REJECTED") from error
        if (
            str(canonical_url) != str(evidence.source_url)
            or evidence.source_kind is not source_kind
            or evidence.source_tier is not source_tier
        ):
            raise WebEvidenceValidationError("WEB_SOURCE_POLICY_REJECTED")

        content_hash = sha256(evidence.content.encode()).hexdigest()
        expected_id = content_addressed_web_evidence_id(
            normalized_ticker,
            canonical_url,
            content_hash,
        )
        canonical = self._repository.get_many([evidence.id])
        if (
            evidence.content_hash != content_hash
            or evidence.id != expected_id
            or len(canonical) != 1
            or canonical[0] != evidence
        ):
            raise WebEvidenceValidationError("WEB_SOURCE_CANONICAL_MISMATCH")
        return PolicyValidatedWebEvidence(
            **evidence.model_dump(),
            policy_version=self.policy_version,
            canonical_url=canonical_url,
        )


def configured_primary_ir_domain(
    ticker: str,
    issuer_domains: Mapping[str, frozenset[str]],
) -> str | None:
    """Resolve exactly one configured issuer-owned HTTPS host for persistence."""
    normalized_ticker = ticker.strip().upper()
    policy = SourcePolicy(issuer_domains=issuer_domains)
    domains = policy.issuer_domains.get(normalized_ticker, frozenset())
    if not domains:
        return None
    if len(domains) != 1:
        raise ValueError(
            f"configured issuer domains for {normalized_ticker} must contain "
            "exactly one primary host"
        )
    domain = next(iter(domains))
    source_kind, source_tier = policy.classify(
        ticker=normalized_ticker,
        url=HttpUrl(f"https://{domain}/"),
    )
    if source_kind is not SourceKind.ISSUER_IR or source_tier is not SourceTier.PRIMARY:
        raise ValueError("configured issuer domain failed ownership validation")
    return domain


def build_standard_source_policy(
    *,
    issuer_domains: Mapping[str, frozenset[str]],
) -> SourcePolicy:
    """Build the fixed P1 and market profile: SEC, Reuters, and verified issuer IR only."""
    return SourcePolicy(
        issuer_domains=issuer_domains,
        authority_domains=DEFAULT_STANDARD_AUTHORITY_DOMAINS,
    )


def build_industry_source_policy(
    *,
    issuer_domains: Mapping[str, frozenset[str]],
    industry_authority_domains: frozenset[str] = DEFAULT_INDUSTRY_AUTHORITY_DOMAINS,
) -> SourcePolicy:
    """Build the industry profile with the fixed standard core plus approved government domains."""
    return SourcePolicy(
        issuer_domains=issuer_domains,
        authority_domains=DEFAULT_STANDARD_AUTHORITY_DOMAINS | industry_authority_domains,
    )


def _normalize_domain(domain: str) -> str:
    normalized = domain.strip().lower().rstrip(".")
    if not normalized or "://" in normalized or "/" in normalized or "@" in normalized:
        raise ValueError("source domains must be bare host names")
    return normalized


def _is_host_or_descendant(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")
