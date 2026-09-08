"""Protocol tests for the P1 allowlisted-web MCP tools."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from fastmcp import Client
from pydantic import HttpUrl
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from financial_evidence_agent.domain import SourceKind, SourceTier, WebEvidence
from financial_evidence_agent.mcp_server.server import create_server
from financial_evidence_agent.retrieval.hybrid import HashEmbeddingProvider, HybridRetriever
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.models import WebEvidenceRecord
from financial_evidence_agent.storage.repositories import FilingRepository
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.gateway import AllowlistedWebGateway
from financial_evidence_agent.web_evidence.providers import (
    HttpxRedirectResolver,
    RawSearchHit,
    SearchProviderError,
)
from financial_evidence_agent.web_evidence.source_policy import SourcePolicy, SourcePolicyError


class FakeProvider:
    def __init__(self, hits: list[RawSearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        self.calls += 1
        return self.hits


class IdentityRedirectResolver:
    async def resolve(self, url):
        return url

    async def resolve_with_policy(self, url, *, ticker, source_policy):
        source_policy.classify(ticker=ticker, url=url)
        return url


class ErrorProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        raise self.error


class HangingProvider:
    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        await asyncio.sleep(1)
        return []


class SequencedHttpTransport:
    def __init__(self, outcome: int | Exception) -> None:
        self.outcome = outcome
        self.calls = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return httpx.Response(self.outcome, request=request)


def _server_with_provider(
    provider,
    *,
    timeout_seconds: float = 10.0,
    redirect_resolver=None,
):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    filing_repository = FilingRepository(engine)
    web_repository = WebEvidenceRepository(engine)
    source_policy = SourcePolicy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )
    gateway = AllowlistedWebGateway(
        provider=provider,
        redirect_resolver=redirect_resolver or IdentityRedirectResolver(),
        source_policy=source_policy,
        repository=web_repository,
        timeout_seconds=timeout_seconds,
    )
    return create_server(
        filing_repository,
        HybridRetriever(filing_repository, HashEmbeddingProvider()),
        web_gateway=gateway,
        web_repository=web_repository,
        source_policy=source_policy,
    )


@pytest.fixture
def web_server():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    filing_repository = FilingRepository(engine)
    web_repository = WebEvidenceRepository(engine)
    provider = FakeProvider(
        [
            RawSearchHit(
                title="NVIDIA update",
                url="https://investor.nvidia.com/news",
                excerpt="A source excerpt.",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]
    )
    source_policy = SourcePolicy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )
    gateway = AllowlistedWebGateway(
        provider=provider,
        redirect_resolver=IdentityRedirectResolver(),
        source_policy=source_policy,
        repository=web_repository,
    )
    server = create_server(
        filing_repository,
        HybridRetriever(filing_repository, HashEmbeddingProvider()),
        web_gateway=gateway,
        web_repository=web_repository,
        source_policy=source_policy,
    )
    return server, engine, provider


async def test_mcp_registers_p1_tools_without_renaming_p0_tools(web_server) -> None:
    """Adding P1 tools must preserve every established P0 discovery name."""
    server, _, _ = web_server
    async with Client(server) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "resolve_company",
        "fetch_recent_filings",
        "hybrid_search_filings",
        "get_source_spans",
        "search_allowlisted_web",
        "search_authoritative_events",
        "get_web_evidence",
        "get_market_snapshot",
        "get_market_bars",
    }


async def test_mcp_search_and_get_return_persisted_read_only_evidence(web_server) -> None:
    """The retrieval tool must return the persisted snapshot without mutating it."""
    server, engine, _ = web_server
    async with Client(server) as client:
        searched = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "nvda", "query": "company update", "max_results": 3},
        )
        assert searched.structured_content is not None
        evidence_id = searched.structured_content["evidence"][0]["id"]
        with Session(engine) as session:
            before = session.scalar(select(func.count()).select_from(WebEvidenceRecord))
        fetched = await client.call_tool("get_web_evidence", {"evidence_ids": [evidence_id]})
        with Session(engine) as session:
            after = session.scalar(select(func.count()).select_from(WebEvidenceRecord))

    assert fetched.structured_content is not None
    assert fetched.structured_content["evidence"][0]["id"] == evidence_id
    assert before == after == 1


async def test_mcp_event_search_uses_canonical_policy_and_time_window(web_server) -> None:
    """Event context cannot use a synthetic row, untimed source, or caller URL."""
    server, _, provider = web_server
    async with Client(server) as client:
        result = await client.call_tool(
            "search_authoritative_events",
            {
                "ticker": "NVDA",
                "query": "NVDA authoritative event",
                "window_start": "2026-07-30T00:00:00Z",
                "window_end": "2026-08-02T00:00:00Z",
                "max_results": 3,
            },
        )
        extra = await client.call_tool(
            "search_authoritative_events",
            {
                "ticker": "NVDA",
                "query": "NVDA event",
                "window_start": "2026-07-30T00:00:00Z",
                "window_end": "2026-08-02T00:00:00Z",
                "domain": "example.com",
            },
            raise_on_error=False,
        )

    assert result.structured_content is not None
    assert result.structured_content["error"] is None
    assert len(result.structured_content["evidence"]) == 1
    assert provider.calls == 1
    assert extra.is_error


def test_market_event_policy_allows_only_sec_issuer_ir_and_reuters() -> None:
    policy = SourcePolicy(
        issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
    )

    assert policy.classify(
        ticker="NVDA", url=HttpUrl("https://www.reuters.com/article")
    )[0] is SourceKind.AUTHORITATIVE_WEB
    with pytest.raises(SourcePolicyError):
        policy.classify(ticker="NVDA", url=HttpUrl("https://www.treasury.gov/news"))


async def test_mcp_validates_per_call_caps_and_exposes_typed_source_errors(web_server) -> None:
    """Invalid bounds and rejected sources must be distinguishable to protocol callers."""
    server, _, _ = web_server
    async with Client(server) as client:
        oversized = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "NVDA", "query": "company update", "max_results": 4},
            raise_on_error=False,
        )
        rejected = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "AMD", "query": "company update", "max_results": 1},
            raise_on_error=False,
        )
        extra = await client.call_tool(
            "get_web_evidence",
            {"evidence_ids": ["id"], "unexpected": True},
            raise_on_error=False,
        )

    assert oversized.is_error
    assert not rejected.is_error
    assert rejected.structured_content is not None
    assert rejected.structured_content["error"] is None
    assert rejected.structured_content["evidence"] == []
    assert extra.is_error


@pytest.mark.parametrize("invalid_max_results", ["3", 3.0, True])
async def test_mcp_rejects_coercible_max_results_before_gateway_execution(
    web_server, invalid_max_results: object
) -> None:
    """FastMCP signature coercion must not bypass the strict internal request contract."""
    server, _, provider = web_server

    async with Client(server) as client:
        result = await client.call_tool(
            "search_allowlisted_web",
            {
                "ticker": "NVDA",
                "query": "company update",
                "max_results": invalid_max_results,
            },
            raise_on_error=False,
        )

    assert result.is_error
    assert provider.calls == 0


async def test_mcp_registers_web_tools_and_reports_provider_unavailable_without_key() -> None:
    """Tool discovery and persisted reads must not depend on live provider configuration."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    filing_repository = FilingRepository(engine)
    web_repository = WebEvidenceRepository(engine)
    snapshot = web_repository.upsert(
        WebEvidence(
            id="ignored",
            ticker="NVDA",
            title="Persisted source",
            content="Persisted excerpt.",
            source_url="https://www.reuters.com/persisted",
            source_kind=SourceKind.AUTHORITATIVE_WEB,
            source_tier=SourceTier.AUTHORITATIVE_SECONDARY,
            published_at=datetime(2026, 7, 31, tzinfo=UTC),
            fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
            content_hash="persisted-hash",
        )
    )
    server = create_server(
        filing_repository,
        HybridRetriever(filing_repository, HashEmbeddingProvider()),
        web_repository=web_repository,
    )

    async with Client(server) as client:
        tools = await client.list_tools()
        search = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "NVDA", "query": "company update"},
        )
        fetched = await client.call_tool(
            "get_web_evidence", {"evidence_ids": [snapshot.id]}
        )

    assert {"search_allowlisted_web", "get_web_evidence"}.issubset(
        {tool.name for tool in tools}
    )
    assert search.structured_content is not None
    assert search.structured_content["error"]["code"] == "WEB_PROVIDER_UNAVAILABLE"
    assert fetched.structured_content is not None
    assert fetched.structured_content["evidence"][0]["id"] == snapshot.id


@pytest.mark.parametrize(
    ("provider", "timeout_seconds", "expected_code"),
    [
        (
            ErrorProvider(SearchProviderError("bad credentials", status_code=401)),
            10.0,
            "WEB_PROVIDER_ERROR",
        ),
        (HangingProvider(), 0.001, "WEB_PROVIDER_TIMEOUT"),
    ],
)
async def test_mcp_returns_structured_provider_error_codes(
    provider, timeout_seconds: float, expected_code: str
) -> None:
    """Recovery logic must inspect a literal field rather than parse tool error prose."""
    server = _server_with_provider(provider, timeout_seconds=timeout_seconds)

    async with Client(server) as client:
        result = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "NVDA", "query": "company update"},
        )

    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == expected_code


@pytest.mark.parametrize(
    ("outcome", "expected_code", "expected_attempts"),
    [
        (httpx.ReadTimeout("read timed out"), "WEB_PROVIDER_TIMEOUT", 3),
        (429, "WEB_PROVIDER_ERROR", 3),
        (503, "WEB_PROVIDER_ERROR", 3),
        (404, "WEB_PROVIDER_ERROR", 1),
    ],
)
async def test_mcp_returns_structured_resolver_codes_with_exact_attempt_caps(
    outcome: int | Exception,
    expected_code: str,
    expected_attempts: int,
) -> None:
    """Resolver recovery must remain machine-readable at the real protocol boundary."""
    sequence = SequencedHttpTransport(outcome)
    resolver = HttpxRedirectResolver(
        timeout_seconds=0.1,
        transport=httpx.MockTransport(sequence.handle),
    )
    provider = FakeProvider(
        [
            RawSearchHit(
                title="Reuters update",
                url="https://www.reuters.com/article",
                excerpt="A source excerpt.",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]
    )
    server = _server_with_provider(
        provider,
        timeout_seconds=0.1,
        redirect_resolver=resolver,
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "search_allowlisted_web",
            {"ticker": "NVDA", "query": "company update"},
        )

    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == expected_code
    assert sequence.calls == expected_attempts
