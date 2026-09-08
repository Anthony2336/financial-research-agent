"""Peer-scope validation coverage for explicit industry comparisons."""

from __future__ import annotations

import pytest

from financial_evidence_agent.research_packages.orchestrator import (
    PeerScopeError,
    validate_peer_scope,
)


@pytest.mark.parametrize(
    ("primary", "peers", "scope", "code"),
    [
        ("NVDA", ["NVDA"], "US semiconductors", "INVALID_PEER_SCOPE"),
        ("NVDA", ["AMD", "AMD"], "US semiconductors", "INVALID_PEER_SCOPE"),
        ("NVDA", ["AMD", "INTC", "AVGO", "QCOM"], "US semiconductors", "PEER_LIMIT_EXCEEDED"),
        ("NVDA", [""], "US semiconductors", "INVALID_PEER_SCOPE"),
        ("NVDA", ["   "], "US semiconductors", "INVALID_PEER_SCOPE"),
        ("NVDA", ["AMD"], "", "INVALID_PEER_SCOPE"),
        ("NVDA", ["AMD"], "   ", "INVALID_PEER_SCOPE"),
        ("password=x", ["AMD"], "US semiconductors", "INVALID_PEER_SCOPE"),
        ("NVDA", ["password=x"], "US semiconductors", "INVALID_PEER_SCOPE"),
    ],
)
def test_peer_scope_rejects_invalid_sets(
    primary: str,
    peers: list[str],
    scope: str,
    code: str,
) -> None:
    with pytest.raises(PeerScopeError) as raised:
        validate_peer_scope(primary, peers, scope)

    assert raised.value.code == code


def test_peer_scope_normalizes_and_preserves_explicit_order() -> None:
    scope = validate_peer_scope(" nvda ", ["amd", " Avgo "], " Datacenter accelerators ")

    assert scope.primary_ticker == "NVDA"
    assert scope.peer_tickers == ("AMD", "AVGO")
    assert scope.description == "Datacenter accelerators"
