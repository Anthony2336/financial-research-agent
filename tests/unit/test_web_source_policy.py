"""Security boundaries for issuer-owned and authoritative web sources."""

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import HttpUrl

from financial_evidence_agent.domain import (
    SourceKind,
    SourceTier,
    WebEvidence,
    content_addressed_web_evidence_id,
)
from financial_evidence_agent.web_evidence.source_policy import (
    DEFAULT_INDUSTRY_AUTHORITY_DOMAINS,
    DEFAULT_STANDARD_AUTHORITY_DOMAINS,
    PersistedWebEvidenceValidator,
    SourcePolicy,
    SourcePolicyError,
    WebEvidenceValidationError,
    build_industry_source_policy,
    build_standard_source_policy,
    configured_primary_ir_domain,
)


@pytest.fixture
def policy() -> SourcePolicy:
    return SourcePolicy(
        issuer_domains={
            "NVDA": frozenset({"investor.nvidia.com"}),
            "AMD": frozenset({"ir.amd.com"}),
        }
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://investor.nvidia.com/news", (SourceKind.ISSUER_IR, SourceTier.PRIMARY)),
        (
            "https://news.investor.nvidia.com/releases",
            (SourceKind.ISSUER_IR, SourceTier.PRIMARY),
        ),
        (
            "https://www.reuters.com/technology/nvidia",
            (SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY),
        ),
        ("https://www.sec.gov/Archives/example", (SourceKind.FILING, SourceTier.PRIMARY)),
    ],
)
def test_classify_accepts_exact_hosts_and_descendants(
    policy: SourcePolicy, url: str, expected: tuple[SourceKind, SourceTier]
) -> None:
    """Removing boundary-aware host matching would reject valid subdomains or mis-tier sources."""
    assert policy.classify(ticker="nvda", url=HttpUrl(url)) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://www.reuters.com/article",
        "https://reuters.com.example.org/article",
        "https://www.reuters.com@example.org/article",
        "https://user:password@www.reuters.com/article",
    ],
)
def test_classify_rejects_non_https_lookalike_and_user_info_urls(
    policy: SourcePolicy, url: str
) -> None:
    """Scheme, suffix, or user-info confusion must not bypass the exact-host allowlist."""
    with pytest.raises(SourcePolicyError) as raised:
        policy.classify(ticker="NVDA", url=HttpUrl(url))

    assert raised.value.code == "SOURCE_NOT_ALLOWED"


def test_classify_rejects_another_issuers_ir_domain(policy: SourcePolicy) -> None:
    """An allowlisted IR host is still invalid when it is not owned by the requested ticker."""
    with pytest.raises(SourcePolicyError) as raised:
        policy.classify(ticker="NVDA", url=HttpUrl("https://ir.amd.com/news"))

    assert raised.value.code == "SOURCE_NOT_ALLOWED"


def test_classify_rejects_nonstandard_https_ports(policy: SourcePolicy) -> None:
    """An allowlisted hostname must not authorize an arbitrary service port."""
    with pytest.raises(SourcePolicyError):
        policy.classify(ticker="NVDA", url=HttpUrl("https://www.sec.gov:8443/private"))


def test_classify_honors_configured_authority_domains() -> None:
    """Replacing the default authority set must not silently hard-code Reuters-only behavior."""
    policy = SourcePolicy(
        issuer_domains={}, authority_domains=frozenset({"news.authority.example"})
    )

    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://updates.news.authority.example/story")
    ) == (SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY)


def test_policy_snapshot_is_deeply_immutable_and_has_a_stable_version() -> None:
    """Mutating the caller or exposed mapping must not change a running allowlist/version."""
    configured = {"NVDA": frozenset({"investor.nvidia.com"})}
    policy = SourcePolicy(issuer_domains=configured)
    version = policy.version

    configured["NVDA"] = frozenset({"evil.example"})
    with pytest.raises(TypeError):
        policy.issuer_domains["NVDA"] = frozenset({"evil.example"})  # type: ignore[index]

    assert policy.version == version
    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://investor.nvidia.com/results")
    ) == (SourceKind.ISSUER_IR, SourceTier.PRIMARY)
    with pytest.raises(SourcePolicyError):
        policy.classify(ticker="NVDA", url=HttpUrl("https://evil.example/results"))


def test_standard_source_policy_keeps_the_p1_and_market_authority_set_frozen() -> None:
    """The standard profile must remain SEC plus Reuters plus verified issuer ownership only."""
    policy = build_standard_source_policy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )

    assert policy.authority_domains == DEFAULT_STANDARD_AUTHORITY_DOMAINS
    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://www.reuters.com/technology/nvidia")
    ) == (SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY)
    with pytest.raises(SourcePolicyError):
        policy.classify(ticker="NVDA", url=HttpUrl("https://www.ftc.gov/news-events"))


def test_industry_source_policy_adds_the_versioned_government_profile_without_mutating_standard(
) -> None:
    """Industry policy broadens only the approved regulator/government set and changes version."""
    standard = build_standard_source_policy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )
    industry = build_industry_source_policy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )

    assert standard.authority_domains == DEFAULT_STANDARD_AUTHORITY_DOMAINS
    assert industry.authority_domains == (
        DEFAULT_STANDARD_AUTHORITY_DOMAINS | DEFAULT_INDUSTRY_AUTHORITY_DOMAINS
    )
    assert industry.version != standard.version
    assert industry.classify(
        ticker="NVDA", url=HttpUrl("https://www.ftc.gov/news-events")
    ) == (SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY)
    assert industry.provider_domains("NVDA") == (
        "commerce.gov",
        "federalregister.gov",
        "ftc.gov",
        "investor.nvidia.com",
        "justice.gov",
        "reuters.com",
        "sec.gov",
        "www.reuters.com",
        "www.sec.gov",
    )
    assert "investor.nvidia.com" not in industry.provider_domains("AMD")


def test_industry_source_policy_allows_admin_replacement_of_only_the_government_subset() -> None:
    """Admin overrides may replace the government list but cannot drop SEC or Reuters."""
    policy = build_industry_source_policy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})},
        industry_authority_domains=frozenset({"gao.gov"}),
    )

    assert policy.authority_domains == (DEFAULT_STANDARD_AUTHORITY_DOMAINS | {"gao.gov"})
    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://www.gao.gov/reports")
    ) == (SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY)
    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://www.sec.gov/Archives/example")
    ) == (SourceKind.FILING, SourceTier.PRIMARY)


def test_configured_primary_ir_domain_requires_one_ticker_owned_bare_host() -> None:
    assert configured_primary_ir_domain(
        "nvda",
        {"NVDA": frozenset({"investor.nvidia.com"})},
    ) == "investor.nvidia.com"
    assert configured_primary_ir_domain("NVDA", {}) is None

    with pytest.raises(ValueError, match="exactly one"):
        configured_primary_ir_domain(
            "NVDA",
            {"NVDA": frozenset({"investor.nvidia.com", "ir.nvidia.com"})},
        )
    with pytest.raises(ValueError, match="bare host"):
        configured_primary_ir_domain(
            "NVDA",
            {"NVDA": frozenset({"https://investor.nvidia.com/news"})},
        )


@pytest.mark.parametrize(
    "published_at",
    [None, datetime(2026, 8, 1), datetime(2026, 8, 3, tzinfo=UTC)],
)
def test_final_policy_guard_rejects_legacy_or_invalid_publication_time(
    published_at: datetime | None,
) -> None:
    content = "Canonical dated issuer evidence."
    content_hash = sha256(content.encode()).hexdigest()
    evidence = WebEvidence(
        id=content_addressed_web_evidence_id(
            "NVDA",
            "https://investor.nvidia.com/results",
            content_hash,
        ),
        ticker="NVDA",
        title="Results",
        content=content,
        source_url="https://investor.nvidia.com/results",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=published_at,
        fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
        content_hash=content_hash,
    )

    class Reader:
        def get_many(self, evidence_ids):
            return [evidence] if evidence_ids == [evidence.id] else []

    validator = PersistedWebEvidenceValidator(
        SourcePolicy(issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}),
        Reader(),
    )

    with pytest.raises(WebEvidenceValidationError) as raised:
        validator.validate(ticker="NVDA", evidence=evidence)

    assert raised.value.code == "WEB_SOURCE_TIME_INVALID"
