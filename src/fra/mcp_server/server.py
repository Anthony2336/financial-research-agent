"""FastMCP stdio entry point for the read-only P0 filing tools."""

from fastmcp import FastMCP
from sqlalchemy import create_engine

from fra.config import Settings
from fra.mcp_server.market_tools import (
    MarketGatewayFactory,
    register_market_tools,
)
from fra.mcp_server.tools import register_tools
from fra.mcp_server.web_tools import register_web_tools
from fra.retrieval.hybrid import BgeM3EmbeddingProvider, HybridRetriever
from fra.storage.cache import build_cache
from fra.storage.market_repositories import MarketDataRepository
from fra.storage.repositories import FilingRepository
from fra.storage.web_repositories import WebEvidenceRepository
from fra.web_evidence.gateway import AllowlistedWebGateway
from fra.web_evidence.providers import (
    HttpxRedirectResolver,
    TavilySearchProvider,
)
from fra.web_evidence.source_policy import SourcePolicy


def create_server(
    repository: FilingRepository,
    retriever: HybridRetriever,
    *,
    web_gateway: AllowlistedWebGateway | None = None,
    web_repository: WebEvidenceRepository | None = None,
    market_gateway_factory: MarketGatewayFactory | None = None,
    market_repository: MarketDataRepository | None = None,
    market_data_max_bars: int = 5,
    source_policy: SourcePolicy | None = None,
) -> FastMCP:
    """Create an injectable server without connecting to databases or loading models."""
    mcp = FastMCP("Financial-Research-Agent")
    register_tools(mcp, repository, retriever)
    resolved_web_repository = web_repository or WebEvidenceRepository(repository.engine)
    register_web_tools(mcp, web_gateway, resolved_web_repository, source_policy)
    resolved_market_repository = market_repository or MarketDataRepository(repository.engine)
    register_market_tools(
        mcp,
        market_gateway_factory,
        resolved_market_repository,
        max_bars=market_data_max_bars,
    )
    return mcp


def main() -> None:
    """Compatibility entry point delegating runtime assembly to ``bootstrap``."""
    from fra.bootstrap import build_mcp_server

    server = build_mcp_server(
        Settings(),
        engine_factory=create_engine,
        repository_factory=FilingRepository,
        web_repository_factory=WebEvidenceRepository,
        cache_factory=build_cache,
        embedding_provider_factory=BgeM3EmbeddingProvider,
        retriever_factory=HybridRetriever,
        provider_factory=TavilySearchProvider,
        redirect_resolver_factory=HttpxRedirectResolver,
        source_policy_factory=SourcePolicy,
        gateway_factory=AllowlistedWebGateway,
        server_factory=create_server,
    )
    server.run(transport="stdio")
