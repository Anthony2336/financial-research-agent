"""Warm-cache acceptance coverage over the production cache boundaries."""

from __future__ import annotations

from datetime import UTC, date, datetime
from time import perf_counter

import pytest
from pydantic import HttpUrl
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from financial_evidence_agent.domain import WebEvidence
from financial_evidence_agent.market_data.gateway import MarketDataGateway
from financial_evidence_agent.market_data.models import MarketBar, MarketSnapshot
from financial_evidence_agent.retrieval.hybrid import HybridRetriever
from financial_evidence_agent.retrieval.ingest import ingest_sec
from financial_evidence_agent.retrieval.sec import SecFilingCandidate, SecFilingDocument
from financial_evidence_agent.storage.cache import InMemoryTtlJsonCache, MarketDataJsonCache
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.repositories import FilingRepository
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.gateway import AllowlistedWebGateway
from financial_evidence_agent.web_evidence.providers import RawSearchHit
from financial_evidence_agent.web_evidence.source_policy import SourcePolicy

NOW = datetime(2026, 8, 31, 14, 1, tzinfo=UTC)
_SEC_DOCUMENT = b"""<!doctype html><html><body>
<h2>MD&amp;A</h2><p>Data center revenue grew because customer demand increased.</p>
<h2>Risk Factors</h2><p>Customer concentration may cause results to fluctuate.</p>
</body></html>"""


class _ControlledSecGateway:
    def __init__(self, candidate: SecFilingCandidate) -> None:
        self.candidate = candidate
        self.document_calls = 0

    def list_filings(self, ticker: str, user_agent: str) -> list[SecFilingCandidate]:
        del ticker, user_agent
        return [self.candidate]

    def fetch_document(
        self, filing: SecFilingCandidate, user_agent: str
    ) -> SecFilingDocument:
        del user_agent
        self.document_calls += 1
        return SecFilingDocument(
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                f"{filing.accession_no.replace('-', '')}/{filing.primary_document}"
            ),
            raw_bytes=_SEC_DOCUMENT,
        )


class _CountingEmbeddings:
    version = "warm-cache-embedding-v1"
    dimensions = 1024

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, *([0.0] * 1023)] for _ in texts]


class _CountingReranker:
    version = "warm-cache-reranker-v1"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, *, query: str, evidence, limit: int):
        del query
        self.calls += 1
        return list(evidence[:limit])


class _ControlledMarketProvider:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.bar_calls = 0

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        self.snapshot_calls += 1
        return MarketSnapshot(
            id="warm-cache-snapshot",
            provider="alpaca",
            feed="iex",
            coverage="IEX-only",
            symbol=symbol,
            exchange="IEX",
            currency="USD",
            price="123.45",
            open="122.00",
            day_high="124.00",
            day_low="121.50",
            previous_close="121.00",
            as_of=NOW,
            fetched_at=NOW,
            market_status="closed",
            raw_payload_hash="a" * 64,
        )

    async def get_bars(self, symbol: str, *, interval: str, limit: int) -> list[MarketBar]:
        del limit
        self.bar_calls += 1
        return [
            MarketBar(
                id="warm-cache-bar",
                provider="alpaca",
                feed="iex",
                coverage="IEX-only",
                symbol=symbol,
                exchange="IEX",
                currency="USD",
                interval=interval,
                timestamp=datetime(2026, 8, 28, 4, tzinfo=UTC),
                open="120.00",
                high="124.00",
                low="119.00",
                close="123.45",
                volume=1000,
                fetched_at=NOW,
                raw_payload_hash="b" * 64,
            )
        ]


class _ControlledWebProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        self.calls += 1
        return [
            RawSearchHit(
                title="NVIDIA update",
                url=HttpUrl("https://www.reuters.com/technology/nvidia"),
                excerpt="Reported revenue evidence.",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]


class _ControlledRedirectResolver:
    async def resolve_with_policy(
        self,
        url: HttpUrl,
        *,
        ticker: str,
        source_policy: SourcePolicy,
    ) -> HttpUrl:
        source_policy.classify(ticker=ticker, url=url)
        return url


@pytest.mark.asyncio
async def test_warm_cache_skips_external_and_retrieval_work_with_diagnostic_timing() -> None:
    """Removing any valid cache reuse makes the second controlled operation cross a boundary."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = FilingRepository(engine)
    web_repository = WebEvidenceRepository(engine)
    cache = InMemoryTtlJsonCache(clock=lambda: 0.0)
    candidate = SecFilingCandidate(
        ticker="NVDA",
        resolved_cik="1045810",
        cik="1045810",
        legal_name="NVIDIA CORPORATION",
        accession_no="0001045810-26-000041",
        form="10-Q",
        filed_at=date(2026, 5, 28),
        primary_document="nvda-20260528.htm",
    )
    sec_gateway = _ControlledSecGateway(candidate)
    embeddings = _CountingEmbeddings()
    reranker = _CountingReranker()
    market_provider = _ControlledMarketProvider()
    web_provider = _ControlledWebProvider()
    market_cache = MarketDataJsonCache(cache, clock=lambda: NOW)

    async def run_operation():
        summary = ingest_sec(
            "NVDA",
            ["10-Q"],
            None,
            repository=repository,
            user_agent="tester@example.com",
            gateway=sec_gateway,
            embedding_provider=embeddings,
            cache=cache,
        )
        retrieval = HybridRetriever(
            repository,
            embeddings,
            reranker=reranker,
            cache=cache,
        ).search_with_metrics("NVDA", "data center revenue", summary.corpus_version, k=1)
        market = await MarketDataGateway(
            market_provider,
            cache=market_cache,
            clock=lambda: NOW,
        ).fetch("NVDA")
        web = await AllowlistedWebGateway(
            provider=web_provider,
            redirect_resolver=_ControlledRedirectResolver(),
            source_policy=SourcePolicy(issuer_domains={}),
            repository=web_repository,
            cache=cache,
            clock=lambda: NOW,
        ).search(ticker="NVDA", query="data center revenue")
        return summary, retrieval, market, web

    await run_operation()
    started = perf_counter()
    summary, retrieval, market, web = await run_operation()
    elapsed_seconds = perf_counter() - started

    print(
        "warm-cache timing diagnostic: "
        f"{elapsed_seconds:.3f}s (documented controlled target: <60s)"
    )
    assert summary.ticker == "NVDA"
    assert retrieval.metrics.cache_hit is True
    assert retrieval.evidence
    assert market.snapshot.symbol == "NVDA"
    assert web and isinstance(web[0], WebEvidence)
    assert sec_gateway.document_calls == 1
    assert embeddings.calls == 2
    assert reranker.calls == 1
    assert market_provider.snapshot_calls == 1
    assert market_provider.bar_calls == 1
    assert web_provider.calls == 1
