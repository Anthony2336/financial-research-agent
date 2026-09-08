"""Typed source-reference contracts shared by current and future result models."""

import pytest

import financial_evidence_agent.domain as domain
from financial_evidence_agent.application import _persisted_claim
from financial_evidence_agent.domain import Claim, ClaimKind, Confidence


def test_filing_and_web_with_the_same_id_encode_to_distinct_keys() -> None:
    """Removing the source namespace would make two different corpora indistinguishable."""
    assert hasattr(domain, "SourceRef")
    filing = domain.SourceRef(
        ticker="nvda", kind=domain.SourceRefKind.FILING, source_id="shared-id"
    )
    web = domain.SourceRef(
        ticker="NVDA", kind=domain.SourceRefKind.WEB, source_id="shared-id"
    )

    assert filing.encode() == "NVDA:filing:shared-id"
    assert web.encode() == "NVDA:web:shared-id"
    assert filing.encode() != web.encode()


def test_source_ref_decode_rejects_cross_ticker_relabeling() -> None:
    """A namespaced AMD source cannot be decoded as evidence for an NVDA result."""
    assert hasattr(domain, "SourceRef")
    with pytest.raises(ValueError, match="ticker"):
        domain.SourceRef.decode("AMD:filing:chunk-1", expected_ticker="NVDA")


def test_market_source_ref_round_trips_provider_scoped_id() -> None:
    """Market repository IDs retain their provider/feed identity after namespacing."""
    reference = domain.SourceRef(
        ticker="NVDA",
        kind=domain.SourceRefKind.MARKET_SNAPSHOT,
        source_id="alpaca:iex:NVDA:2026-08-31T14:00:00Z",
    )

    encoded = reference.encode()

    assert encoded == (
        "NVDA:market_snapshot:alpaca:iex:NVDA:2026-08-31T14:00:00Z"
    )
    assert domain.SourceRef.decode(encoded, expected_ticker="NVDA") == reference


def test_repository_decoder_returns_typed_unresolved_legacy_fallback() -> None:
    """A naked pre-backfill ID remains visibly unresolved instead of becoming web evidence."""
    assert hasattr(domain, "decode_stored_source_ref")
    reference = domain.decode_stored_source_ref("legacy-source", ticker="NVDA")

    assert reference == domain.UnresolvedSourceRef(
        ticker="NVDA",
        source_id="legacy-source",
        reason="legacy_unresolved",
    )
    assert reference.storage_value() == "NVDA:unresolved:legacy-source"


def test_persisted_claim_namespaces_filing_and_web_ids_separately() -> None:
    """Application finalization must not flatten equal filing/web IDs into one string."""
    claim = Claim(
        kind=ClaimKind.VERIFIED_FACT,
        text="Two independently sourced facts agree.",
        confidence=Confidence.HIGH,
        evidence_chunk_ids=["shared-id"],
        web_evidence_ids=["shared-id"],
    )

    persisted = _persisted_claim(claim, ticker="NVDA")

    assert [reference.encode() for reference in persisted.source_refs] == [
        "NVDA:filing:shared-id",
        "NVDA:web:shared-id",
    ]
