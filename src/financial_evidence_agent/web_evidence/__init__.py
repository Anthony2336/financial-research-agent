"""Allowlisted web search, normalization, and source-policy boundaries."""

from financial_evidence_agent.web_evidence.gateway import (
    AllowlistedWebGateway,
    WebGatewayError,
)
from financial_evidence_agent.web_evidence.providers import (
    HttpxRedirectResolver,
    RawSearchHit,
    RedirectResolver,
    SearchProvider,
)
from financial_evidence_agent.web_evidence.source_policy import SourcePolicy, SourcePolicyError

__all__ = [
    "AllowlistedWebGateway",
    "HttpxRedirectResolver",
    "RawSearchHit",
    "RedirectResolver",
    "SearchProvider",
    "SourcePolicy",
    "SourcePolicyError",
    "WebGatewayError",
]
