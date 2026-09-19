"""Integration coverage for MCP-only P1 evidence collection and composition."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from fastmcp.client.client import CallToolResult
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fra import bootstrap as bootstrap_module
from fra.bootstrap import build_p1_runtime
from fra.config import Settings
from fra.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchQuestion,
    SourceKind,
    SourceTier,
    WebEvidence,
)
from fra.graph import FastMCPToolClient, run_research
from fra.graph.models import SkillAnalysisInput, SkillPlanningInput
from fra.mcp_server.client_adapters import MCPFilingSearch
from fra.mcp_server.server import create_server
from fra.retrieval.collector import EvidenceCollector
from fra.retrieval.hybrid import HashEmbeddingProvider, HybridRetriever
from fra.retrieval.ingest import ingest_fixture
from fra.skills.models import (
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    WebUsagePolicy,
)
from fra.skills.schemas import (
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)
from fra.storage.database import create_schema
from fra.storage.repositories import FilingRepository


def _seed_fixture(database_url: str) -> None:
    engine = create_engine(database_url)
    create_schema(engine)
    ingest_fixture(
        Path("tests/fixtures/nvda_10q.html"),
        "NVDA",
        "10-Q",
        FilingRepository(engine),
    )


class RecordingDelegate:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, dict(arguments)))
        return self._delegate.call_tool(name, arguments)


async def test_collector_parses_real_in_process_fastmcp_envelopes() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = FilingRepository(engine)
    assert ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    server = create_server(repository, HybridRetriever(repository, HashEmbeddingProvider()))
    client = RecordingDelegate(FastMCPToolClient(server))
    recipe = ResearchRecipe(
        name="earnings_review",
        version="test",
        accepted_intents=frozenset({Intent.EARNINGS_REVIEW_REQUEST}),
        allowed_tools=frozenset({"hybrid_search_filings"}),
        source_policy=(SourceKind.FILING,),
        required_facets=(ResearchFacet.EARNINGS_CHANGE,),
        budget=RecipeBudget(
            max_questions=1,
            max_local_results_per_query=2,
            max_retrieval_rounds=1,
            max_web_calls=0,
            max_web_results=0,
            max_planner_output_tokens=350,
            max_analysis_output_tokens=1_200,
            max_repair_output_tokens=600,
            max_evidence_tokens=3_000,
        ),
        web_usage_policy=WebUsagePolicy.NONE,
        input_schema="ResearchInput",
        output_schema="ResearchMemo",
        guard_profile="strict_citation",
    )

    bundle = await EvidenceCollector(local_search=MCPFilingSearch(client)).collect(
        ticker="NVDA",
        recipe=recipe,
        questions=[
            ResearchQuestion(
                question="What changed?",
                support_query="data center demand",
                challenge_query="growth risks",
            )
        ],
    )

    assert bundle.coverage.complete is True
    assert bundle.filing_evidence
    assert [name for name, _ in client.calls] == [
        "hybrid_search_filings",
        "hybrid_search_filings",
    ]


class RecordingMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.web_called = False

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, dict(arguments)))
        if name == "search_allowlisted_web":
            self.web_called = True
            return _tool_result(
                {"evidence": [_web().model_dump(mode="json")], "error": None}
            )
        assert name == "hybrid_search_filings"
        query = str(arguments["query"])
        if not self.web_called and "facet=information_gaps" in query:
            return _tool_result(
                {
                    "chunks": [],
                    "error": {"code": "EMPTY_RETRIEVAL", "message": "no match"},
                }
            )
        return _tool_result(
            {"chunks": [_chunk().model_dump(mode="json")], "error": None}
        )


def _tool_result(payload: dict[str, object]) -> CallToolResult:
    return CallToolResult(
        content=[],
        structured_content=payload,
        meta=None,
        is_error=False,
    )


def _chunk() -> EvidenceChunk:
    return EvidenceChunk(
        id="mcp-filing",
        ticker="NVDA",
        corpus_version="NVDA-v1",
        content="Filing evidence for the requested facet.",
        source_url="https://www.sec.gov/Archives/nvda.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=40,
    )


def _web() -> WebEvidence:
    return WebEvidence(
        id="mcp-web",
        ticker="NVDA",
        title="Issuer update",
        content="Allowlisted issuer evidence for the challenge target.",
        source_url="https://investor.nvidia.com/news",
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, tzinfo=UTC),
        content_hash="sha256:mcp-web",
    )


class Planner:
    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        return [
            ResearchQuestion(
                question=f"Review {request.recipe.name.value}.",
                support_query="supporting disclosure",
                challenge_query="challenging disclosure",
            )
        ]


class Analyst:
    async def analyze(self, *, request: SkillAnalysisInput, evidence) -> SkillResearchMemo:
        filing_source_id = evidence.filing_evidence[0].id
        web_source_id = evidence.web_evidence[0].id if evidence.web_evidence else None
        return SkillResearchMemo(
            recipe_name=request.recipe.name,
            recipe_version=request.recipe.version,
            research_question=f"Review {request.recipe.name.value}.",
            sections=[
                SkillResearchSection(
                    facet=facet,
                    claims=[
                        Claim(
                            kind=ClaimKind.VERIFIED_FACT,
                            text=f"Source-backed {facet.value} finding.",
                            confidence=Confidence.HIGH,
                            evidence_chunk_ids=(
                                []
                                if facet is ResearchFacet.INFORMATION_GAPS and web_source_id
                                else [filing_source_id]
                            ),
                            web_evidence_ids=(
                                [web_source_id]
                                if facet is ResearchFacet.INFORMATION_GAPS and web_source_id
                                else []
                            ),
                        )
                    ],
                )
                for facet in request.recipe.required_facets
            ],
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            information_gaps=[],
            confidence=Decimal("0.9"),
        )


class SkillRuns:
    def __init__(self) -> None:
        self._next = 0

    def start(self, **kwargs: object) -> str:
        del kwargs
        self._next += 1
        return f"skill-run-{self._next}"

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: list[str],
        errors: list[str],
    ) -> None:
        del run_id, status, source_ids, errors


def test_p1_workflow_uses_mcp_for_local_retry_and_web_without_direct_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_mcp = RecordingMCP()
    client_constructions: list[object] = []
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'p1-mcp.sqlite3'}",
        redis_url=None,
        fast_model="fast-test",
        analyst_model="analyst-test",
        openai_api_key=SecretStr("model-key"),
        tavily_api_key=SecretStr("web-key"),
        _env_file=None,
    )
    _seed_fixture(settings.database_url)
    monkeypatch.setattr(
        bootstrap_module,
        "BgeM3EmbeddingProvider",
        lambda model_name, *, cache_dir=None: HashEmbeddingProvider(),
    )
    def build_recording_client(server: object) -> RecordingMCP:
        client_constructions.append(server)
        return recording_mcp

    monkeypatch.setattr(bootstrap_module, "FastMCPToolClient", build_recording_client)
    runtime = build_p1_runtime(settings, ticker="NVDA")
    assert len(client_constructions) == 1
    assert runtime.dependencies.mcp_client is recording_mcp

    def fail_direct_access(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("P1 collector bypassed MCP")

    monkeypatch.setattr(FilingRepository, "latest_corpus_version", fail_direct_access)
    monkeypatch.setattr(HybridRetriever, "search", fail_direct_access)
    monkeypatch.setattr(
        bootstrap_module._LazyAllowlistedWebGateway,
        "search",
        fail_direct_access,
    )
    dependencies = replace(
        runtime.dependencies,
        skill_planner=Planner(),
        skill_analyst=Analyst(),
        skill_run_repository=SkillRuns(),
    )

    result = run_research(
        "NVDA",
        "Review the latest earnings.",
        dependencies,
        intent=Intent.EARNINGS_REVIEW_REQUEST,
    )

    names = [name for name, _ in recording_mcp.calls]
    assert names.count("hybrid_search_filings") > 1
    assert names.count("search_allowlisted_web") == 1
    assert result.status in {"completed", "partial"}, (
        result.errors,
        [(run.recipe_name, run.status, run.errors) for run in result.skill_runs],
    )
