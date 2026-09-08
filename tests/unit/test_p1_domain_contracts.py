"""P1 contracts for informational routing and web evidence."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    Intent,
    RouterDecision,
    SourceKind,
    SourceTier,
    WebEvidence,
)


@pytest.mark.parametrize(
    ("intent", "expected_json"),
    [
        (
            Intent.COMPANY_PROFILE_REQUEST,
            '{"intent":"company_profile_request","reason":"deterministic match"}',
        ),
        (
            Intent.EARNINGS_REVIEW_REQUEST,
            '{"intent":"earnings_review_request","reason":"deterministic match"}',
        ),
    ],
)
def test_p1_intents_serialize_deterministically(intent: Intent, expected_json: str) -> None:
    """Changing either wire value would break downstream structured routing consumers."""
    decision = RouterDecision(intent=intent, reason="deterministic match")

    assert decision.model_dump_json() == expected_json


def test_web_evidence_rejects_naive_fetched_timestamp() -> None:
    """Naive fetch times lose the ordering needed to audit web evidence freshness."""
    with pytest.raises(ValidationError, match="fetched_at must include timezone information"):
        WebEvidence(
            id="web-1",
            ticker="NVDA",
            title="Quarterly results",
            content="Revenue rose year over year.",
            source_url="https://investor.nvidia.com/results",
            source_kind=SourceKind.ISSUER_IR,
            source_tier=SourceTier.PRIMARY,
            published_at=datetime(2026, 5, 28, tzinfo=UTC),
            fetched_at=datetime(2026, 5, 29, 9, 30),
            content_hash="sha256:abc123",
        )


def test_verified_fact_requires_sec_or_web_evidence_id() -> None:
    """Removing either citation collection must not allow an uncited fact through."""
    with pytest.raises(ValidationError, match="verified facts require at least one evidence id"):
        Claim(
            kind=ClaimKind.VERIFIED_FACT,
            text="Revenue increased.",
            confidence=Confidence.HIGH,
        )
