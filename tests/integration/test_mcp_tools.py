"""Integration coverage for the read-only FastMCP filing tools."""

import asyncio
import os
import sys
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fra.domain import ResearchQuestion
from fra.graph.models import FastMCPToolClient
from fra.mcp_server.client_adapters import MCPFilingSearch
from fra.mcp_server.server import create_server
from fra.mcp_server.tools import (
    FetchRecentFilingsResponse,
    FilingOutput,
)
from fra.retrieval.collector import (
    THESIS_COLLECTION_POLICY,
    CollectionErrorCode,
    EvidenceCollectionError,
    EvidenceCollector,
)
from fra.retrieval.hybrid import (
    HashEmbeddingProvider,
    HybridRetriever,
)
from fra.retrieval.indexing import EmbeddingIndexer
from fra.retrieval.ingest import ingest_fixture
from fra.retrieval.rerank import (
    LazyFlashRankReranker,
    RerankerModelUnavailableError,
)
from fra.storage.database import create_schema
from fra.storage.models import (
    Base,
    Chunk,
    Company,
    CorpusFiling,
    Filing,
    ResearchCorpus,
)
from fra.storage.repositories import ChunkToStore, FilingRepository


def _data(result):
    """Return FastMCP's parsed structured result from the real protocol client."""
    assert result.structured_content is not None
    return result.structured_content


def _store_filing(
    repository: FilingRepository,
    *,
    ticker: str,
    form: str,
    filed_at: date,
    marker: str,
) -> str:
    """Persist a small filing with a distinct corpus version for protocol tests."""
    content = f"{marker} disclosure text"
    return repository.store_filing(
        ticker=ticker,
        form=form,
        accession_no=f"0000000000-{marker}",
        filed_at=filed_at,
        source_url=f"https://www.sec.gov/Archives/edgar/data/{marker}",
        raw_text=content,
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
        chunks=[
            ChunkToStore(
                section="Other Disclosure",
                chunk_index=0,
                content=content,
                token_count=len(content.split()),
                raw_start=0,
                raw_end=len(content),
            )
        ],
    )


def _database_state(engine) -> dict[str, tuple[tuple[object, ...], ...]]:
    """Capture every persisted P0 table so read paths cannot hide mutations."""
    with engine.connect() as connection:
        return {
            table.name: tuple(
                tuple(row) for row in connection.execute(select(table).order_by(*table.c))
            )
            for table in Base.metadata.sorted_tables
        }


@pytest.fixture
def engine():
    """Create a fixture corpus with two issuers for scope tests."""
    database_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(database_engine)
    return database_engine


@pytest.fixture
def repository(engine) -> FilingRepository:
    """Populate both issuers from the deterministic local filing fixture."""
    fixture = Path("tests/fixtures/nvda_10q.html")
    repository = FilingRepository(engine)
    embedding_provider = HashEmbeddingProvider()
    assert (
        ingest_fixture(
            fixture,
            "NVDA",
            "10-Q",
            repository,
            embedding_provider=embedding_provider,
        )
        == "NVDA-v1"
    )
    assert (
        ingest_fixture(
            fixture,
            "AMD",
            "10-Q",
            repository,
            embedding_provider=embedding_provider,
        )
        == "AMD-v1"
    )
    return repository


@pytest.fixture
def mcp_server(repository: FilingRepository):
    """Build the in-memory FastMCP server without a model download."""
    return create_server(repository, HybridRetriever(repository, HashEmbeddingProvider()))


@pytest.fixture
async def mcp_client(mcp_server):
    """Exercise the real in-memory MCP protocol, rather than tool functions directly."""
    async with Client(mcp_server) as client:
        yield client


async def test_mcp_preserves_four_p0_tools_alongside_p1_web_tools(
    mcp_client: Client,
) -> None:
    """P1 discovery must retain every established P0 tool name."""
    tools = await mcp_client.list_tools()

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


async def test_mcp_search_returns_source_metadata(mcp_client: Client) -> None:
    """Search results must remain citable through their persisted SEC metadata."""
    result = await mcp_client.call_tool(
        "hybrid_search_filings",
        {"ticker": "NVDA", "query": "data center demand", "filing_ids": [], "k": 5},
    )

    data = _data(result)
    assert data["chunks"]
    assert data["chunks"][0]["source_url"].startswith("https://www.sec.gov/")
    assert data["chunks"][0]["id"]
    assert all(chunk["ticker"] == "NVDA" for chunk in data["chunks"])


def test_fetch_recent_filings_response_requires_exactly_one_valid_outcome() -> None:
    filing = {
        "id": "filing-1",
        "ticker": "NVDA",
        "form": "10-Q",
        "filed_at": "2026-05-20",
        "source_url": "https://www.sec.gov/Archives/filing-1.htm",
        "accession_no": "0001045810-26-000001",
        "corpus_version": "NVDA-v1",
    }

    success = FetchRecentFilingsResponse.model_validate(
        {"corpus_version": "NVDA-v1", "filings": [filing], "error": None}
    )
    failure = FetchRecentFilingsResponse.model_validate(
        {
            "corpus_version": None,
            "filings": [],
            "error": {"code": "NO_FILINGS", "message": "none"},
        }
    )

    assert success.error is None and success.filings
    assert failure.error is not None and failure.filings == []
    with pytest.raises(ValidationError):
        FetchRecentFilingsResponse.model_validate(
            {
                "corpus_version": "NVDA-v1",
                "filings": [filing],
                "error": {"code": "NO_FILINGS", "message": "contradictory"},
            }
        )
    with pytest.raises(ValidationError):
        FetchRecentFilingsResponse.model_validate(
            {"corpus_version": None, "filings": [filing], "error": None}
        )


@pytest.mark.parametrize("blank_version", ["", "   "])
def test_fetch_recent_filings_contract_rejects_blank_top_level_or_nested_version(
    blank_version: str,
) -> None:
    filing = {
        "id": "filing-1",
        "ticker": "NVDA",
        "form": "10-Q",
        "filed_at": "2026-05-20",
        "source_url": "https://www.sec.gov/Archives/filing-1.htm",
        "accession_no": "0001045810-26-000001",
        "corpus_version": "NVDA-v1",
    }

    with pytest.raises(ValidationError):
        FetchRecentFilingsResponse.model_validate(
            {
                "corpus_version": blank_version,
                "filings": [filing],
                "error": None,
            }
        )
    with pytest.raises(ValidationError):
        FilingOutput.model_validate({**filing, "corpus_version": blank_version})


async def test_fetch_recent_filings_tool_returns_only_valid_outcome_envelopes(
    mcp_client: Client,
) -> None:
    success = await mcp_client.call_tool("fetch_recent_filings", {"ticker": "NVDA"})
    failure = await mcp_client.call_tool("fetch_recent_filings", {"ticker": "MISSING"})

    success_value = FetchRecentFilingsResponse.model_validate(_data(success))
    failure_value = FetchRecentFilingsResponse.model_validate(_data(failure))

    assert success_value.error is None
    assert success_value.corpus_version is not None
    assert success_value.corpus_version.strip() == success_value.corpus_version
    assert success_value.filings
    assert all(
        filing.corpus_version.strip() == filing.corpus_version
        for filing in success_value.filings
    )
    assert failure_value.error is not None
    assert failure_value.corpus_version is None
    assert failure_value.filings == []


async def test_mcp_search_preserves_typed_reranker_dependency_failure(
    repository: FilingRepository,
) -> None:
    class UnavailableRetriever:
        def search_with_metrics(self, *args, **kwargs):
            del args, kwargs
            raise RerankerModelUnavailableError("private model cache detail")

    server = create_server(repository, UnavailableRetriever())
    async with Client(server) as client:
        result = await client.call_tool(
            "hybrid_search_filings",
            {"ticker": "NVDA", "query": "data center demand", "filing_ids": [], "k": 1},
            raise_on_error=False,
        )

    assert result.is_error is False
    assert _data(result)["error"] == {
        "code": "RERANKER_MODEL_UNAVAILABLE",
        "message": "Configured FlashRank assets are unavailable",
    }
    assert "private model cache detail" not in repr(result)


def test_reranker_execution_failure_preserves_full_mcp_collection_chain(
    repository: FilingRepository,
) -> None:
    class ConstructedRanker:
        def rerank(self, request):
            del request
            raise OSError("private ONNX execution detail")

    retriever = HybridRetriever(
        repository,
        HashEmbeddingProvider(),
        reranker=LazyFlashRankReranker(
            "test-model",
            ranker_factory=lambda *, model_name, cache_dir: ConstructedRanker(),
            asset_validator=lambda model_name, cache_dir: None,
        ),
    )
    server = create_server(repository, retriever)
    collector = EvidenceCollector(
        local_search=MCPFilingSearch(FastMCPToolClient(server)),
        corpus_version="NVDA-v1",
        filing_ids=tuple(
            filing.id for filing in repository.list_recent_filings("NVDA", forms=[], limit=4)
        ),
    )
    questions = [
        ResearchQuestion(
            question="Does disclosed demand support growth?",
            support_query="data center demand",
            challenge_query="data center risks",
        )
    ]

    with pytest.raises(EvidenceCollectionError) as caught:
        asyncio.run(
            collector.collect(
                ticker="NVDA",
                recipe=THESIS_COLLECTION_POLICY,
                questions=questions,
            )
        )

    assert caught.value.code is CollectionErrorCode.DEPENDENCY_ERROR
    assert caught.value.detail == (
        "local search: RERANKER_MODEL_UNAVAILABLE: configured FlashRank assets are unavailable"
    )
    assert "private ONNX execution detail" not in str(caught.value)


async def test_mcp_reindexes_provider_mismatch_before_dense_search(
    repository: FilingRepository,
) -> None:
    """The MCP boundary must replace hash vectors before issuing a BGE-space query."""
    hash_provider = HashEmbeddingProvider()
    EmbeddingIndexer(repository, hash_provider).ensure_indexed("NVDA", "NVDA-v1")

    class RecordingBge:
        version = "sentence-transformers:BAAI/bge-m3"
        dimensions = 1024

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(texts)
            return [[1.0, *([0.0] * 1023)] for _ in texts]

    provider = RecordingBge()
    server = create_server(repository, HybridRetriever(repository, provider))
    async with Client(server) as client:
        result = await client.call_tool(
            "hybrid_search_filings",
            {"ticker": "NVDA", "query": "data center demand", "filing_ids": [], "k": 1},
        )

    assert not result.is_error
    assert len(provider.batches) == 2
    assert provider.batches[0] == [
        chunk.content for chunk in repository.list_chunks("NVDA", "NVDA-v1")
    ]
    with Session(repository.engine) as session:
        versions = set(
            session.scalars(
                select(Chunk.embedding_model)
                .join(Chunk.filing)
                .join(Filing.company)
                .where(Company.ticker == "NVDA", Filing.corpus_version == "NVDA-v1")
            ).all()
        )
    assert versions == {provider.version}


async def test_mcp_stdio_entrypoint_lists_tools_and_fetches_fixture_filing(
    tmp_path: Path,
) -> None:
    """The package entry point must serve the real stdio MCP protocol."""
    database_path = tmp_path / "mcp.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    assert ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "fra.mcp_server"],
        env={**os.environ, "DATABASE_URL": database_url},
        cwd=str(Path.cwd()),
        keep_alive=False,
        log_file=tmp_path / "mcp.stderr",
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        result = await client.call_tool("fetch_recent_filings", {"ticker": "NVDA"})

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
    assert _data(result)["filings"][0]["source_url"].startswith("https://www.sec.gov/")


async def test_mcp_excludes_persisted_non_allowlisted_forms(
    mcp_client: Client, repository: FilingRepository, engine
) -> None:
    """Valid omitted/empty form calls never materialize unsupported persisted forms."""
    assert (
        _store_filing(
            repository,
            ticker="NVDA",
            form="20-F",
            filed_at=date(2026, 1, 1),
            marker="20f",
        )
        == "NVDA-v2"
    )
    with Session(engine) as session:
        unsupported_filing_id = session.scalar(select(Filing.id).where(Filing.form == "20-F"))
        unsupported_chunk_id = session.scalar(
            select(Chunk.id).join(Chunk.filing).where(Filing.form == "20-F")
        )
    assert unsupported_filing_id is not None
    assert unsupported_chunk_id is not None

    omitted = await mcp_client.call_tool("fetch_recent_filings", {"ticker": "NVDA"})
    empty_forms = await mcp_client.call_tool(
        "fetch_recent_filings", {"ticker": "NVDA", "forms": []}
    )
    selected = await mcp_client.call_tool(
        "hybrid_search_filings",
        {"ticker": "NVDA", "query": "data center demand", "filing_ids": [], "k": 1},
    )
    explicit_unsupported = await mcp_client.call_tool(
        "hybrid_search_filings",
        {
            "ticker": "NVDA",
            "query": "data center demand",
            "filing_ids": [unsupported_filing_id],
            "k": 1,
        },
        raise_on_error=False,
    )
    source = await mcp_client.call_tool(
        "get_source_spans", {"chunk_ids": [unsupported_chunk_id]}, raise_on_error=False
    )

    assert all(filing["form"] == "10-Q" for filing in _data(omitted)["filings"])
    assert all(filing["form"] == "10-Q" for filing in _data(empty_forms)["filings"])
    assert _data(selected)["chunks"][0]["form"] == "10-Q"
    assert _data(explicit_unsupported)["error"]["code"] == "NO_FILINGS"
    assert _data(source)["error"]["code"] == "EMPTY_RETRIEVAL"


async def test_mcp_explicitly_rejects_persisted_def_14a_proxy(
    mcp_client: Client,
    repository: FilingRepository,
    engine,
) -> None:
    """Unsupported proxy filings must remain outside every filing read."""
    assert (
        _store_filing(
            repository,
            ticker="NVDA",
            form="DEF 14A",
            filed_at=date(2026, 1, 2),
            marker="def14a",
        )
        == "NVDA-v2"
    )
    with Session(engine) as session:
        proxy_chunk_id = session.scalar(
            select(Chunk.id).join(Chunk.filing).where(Filing.form == "DEF 14A")
        )
    assert proxy_chunk_id is not None

    explicit = await mcp_client.call_tool(
        "fetch_recent_filings",
        {"ticker": "NVDA", "forms": ["DEF 14A"]},
        raise_on_error=False,
    )
    omitted = await mcp_client.call_tool("fetch_recent_filings", {"ticker": "NVDA"})
    source = await mcp_client.call_tool(
        "get_source_spans",
        {"chunk_ids": [proxy_chunk_id]},
        raise_on_error=False,
    )

    assert explicit.is_error
    assert all(filing["form"] != "DEF 14A" for filing in _data(omitted)["filings"])
    assert _data(source)["error"]["code"] == "EMPTY_RETRIEVAL"

    assert (
        _store_filing(
            repository,
            ticker="FOREIGN",
            form="20-F",
            filed_at=date(2026, 1, 1),
            marker="foreign20f",
        )
        == "FOREIGN-v1"
    )
    unsupported_only_filings = await mcp_client.call_tool(
        "fetch_recent_filings", {"ticker": "FOREIGN"}, raise_on_error=False
    )
    unsupported_only_search = await mcp_client.call_tool(
        "hybrid_search_filings",
        {"ticker": "FOREIGN", "query": "annual report", "filing_ids": [], "k": 1},
        raise_on_error=False,
    )

    assert _data(unsupported_only_filings)["error"]["code"] == "NO_FILINGS"
    assert _data(unsupported_only_search)["error"]["code"] == "NO_FILINGS"


def test_latest_corpus_uses_version_sequence_not_filed_date(
    repository: FilingRepository,
) -> None:
    """An older historical filing ingested later is the selected current corpus."""
    assert (
        _store_filing(
            repository,
            ticker="NVDA",
            form="10-Q",
            filed_at=date(2024, 1, 1),
            marker="older",
        )
        == "NVDA-v2"
    )

    assert repository.latest_corpus_version("NVDA") == "NVDA-v2"


async def test_mcp_selected_filing_ids_constrain_the_ticker_corpus(
    mcp_client: Client, repository: FilingRepository
) -> None:
    """A supplied filing from another ticker cannot expand or select NVDA's corpus."""
    amd_filing_id = repository.list_recent_filings("AMD", forms=[], limit=1)[0].id

    result = await mcp_client.call_tool(
        "hybrid_search_filings",
        {
            "ticker": "NVDA",
            "query": "data center demand",
            "filing_ids": [amd_filing_id],
            "k": 1,
        },
        raise_on_error=False,
    )

    assert _data(result)["error"]["code"] == "NO_FILINGS"


async def test_mcp_resolves_forms_and_cutoff_to_an_existing_snapshot_subset(
    mcp_client: Client, repository: FilingRepository
) -> None:
    """Research scope returns one immutable version plus only eligible member IDs."""
    quarterly_id = repository.list_recent_filings("NVDA", forms=[], limit=1)[0].id
    current_content = "current report subset marker"
    repository.store_filing(
        ticker="NVDA",
        form="8-K",
        accession_no="nvda-current-subset",
        filed_at=date(2026, 6, 2),
        source_url="https://www.sec.gov/nvda-current-subset",
        raw_text=current_content,
        content_hash=sha256(current_content.encode()).hexdigest(),
        chunks=[ChunkToStore("Other Disclosure", 0, current_content, 4, 0, len(current_content))],
    )
    with Session(repository.engine) as session:
        current_id = session.scalar(
            select(Filing.id).where(Filing.accession_no == "nvda-current-subset")
        )
    assert current_id is not None

    older_matching_form = await mcp_client.call_tool(
        "fetch_recent_filings",
        {
            "ticker": "NVDA",
            "forms": ["10-Q"],
            "as_of_date": "2099-01-01",
        },
    )
    assert _data(older_matching_form)["corpus_version"] == "NVDA-v1"
    assert [filing["id"] for filing in _data(older_matching_form)["filings"]] == [
        quarterly_id
    ]

    version = repository.create_corpus("NVDA", [quarterly_id, current_id], date(2026, 6, 2))

    historical = await mcp_client.call_tool(
        "fetch_recent_filings",
        {
            "ticker": "NVDA",
            "forms": ["10-Q", "8-K"],
            "as_of_date": "2025-12-31",
        },
    )
    assert _data(historical)["corpus_version"] == "NVDA-v1"
    assert [filing["id"] for filing in _data(historical)["filings"]] == [quarterly_id]

    fetched = await mcp_client.call_tool(
        "fetch_recent_filings",
        {
            "ticker": "NVDA",
            "forms": ["10-Q"],
            "as_of_date": "2099-01-01",
        },
    )
    data = _data(fetched)

    assert data["corpus_version"] == version
    assert [filing["id"] for filing in data["filings"]] == [quarterly_id]
    selected = await mcp_client.call_tool(
        "hybrid_search_filings",
        {
            "ticker": "NVDA",
            "query": "data center demand current report subset marker",
            "corpus_version": version,
            "filing_ids": [quarterly_id],
            "k": 8,
        },
    )
    assert {chunk["form"] for chunk in _data(selected)["chunks"]} == {"10-Q"}


async def test_mcp_rejects_inconsistent_explicit_version_and_filing_ids(
    mcp_client: Client, repository: FilingRepository
) -> None:
    """An ID cannot be imported from a different ticker or snapshot into explicit scope."""
    amd_filing_id = repository.list_recent_filings("AMD", forms=[], limit=1)[0].id

    result = await mcp_client.call_tool(
        "hybrid_search_filings",
        {
            "ticker": "NVDA",
            "query": "data center demand",
            "corpus_version": "NVDA-v1",
            "filing_ids": [amd_filing_id],
            "k": 1,
        },
        raise_on_error=False,
    )

    assert _data(result)["error"]["code"] == "NO_FILINGS"


async def test_mcp_fails_closed_for_ambiguous_duplicate_memberships(
    mcp_client: Client,
    repository: FilingRepository,
) -> None:
    """Corrupt duplicate membership sets must not be resolved by choosing a version."""
    filing = repository.list_recent_filings("NVDA", forms=[], limit=1)[0]
    with Session(repository.engine) as session, session.begin():
        company_id = session.scalar(select(Company.id).where(Company.ticker == "NVDA"))
        assert company_id is not None
        duplicate = ResearchCorpus(
            id="corrupt-duplicate-corpus",
            company_id=company_id,
            version="NVDA-v2",
            membership_hash="f" * 64,
            as_of_date=filing.filed_at,
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
        session.add(duplicate)
        session.add(CorpusFiling(corpus_id=duplicate.id, filing_id=filing.id))

    result = await mcp_client.call_tool(
        "hybrid_search_filings",
        {
            "ticker": "NVDA",
            "query": "data center demand",
            "filing_ids": [filing.id],
            "k": 1,
        },
        raise_on_error=False,
    )

    assert _data(result)["error"]["code"] == "NO_FILINGS"


async def test_mcp_returns_machine_readable_errors_and_validates_constraints(
    mcp_client: Client,
) -> None:
    """Invalid scopes are structured responses and schema limits reject oversized calls."""
    unknown = await mcp_client.call_tool(
        "resolve_company", {"ticker": "ZZZZ"}, raise_on_error=False
    )
    empty = await mcp_client.call_tool(
        "hybrid_search_filings",
        {"ticker": "NVDA", "query": "***", "filing_ids": [], "k": 1},
        raise_on_error=False,
    )
    too_many = await mcp_client.call_tool(
        "fetch_recent_filings",
        {"ticker": "NVDA", "forms": ["10-Q"], "limit": 5},
        raise_on_error=False,
    )

    assert _data(unknown)["error"]["code"] == "UNSUPPORTED_TICKER"
    assert _data(empty)["error"]["code"] == "EMPTY_RETRIEVAL"
    assert too_many.is_error


async def test_mcp_tools_are_read_only(
    mcp_client: Client, engine, repository: FilingRepository
) -> None:
    """Each tool call must leave every persisted table's row count unchanged."""
    chunk_id = repository.list_chunks("NVDA", "NVDA-v1")[0].id
    calls = [
        ("resolve_company", {"ticker": "NVDA"}),
        ("fetch_recent_filings", {"ticker": "NVDA"}),
        (
            "hybrid_search_filings",
            {"ticker": "NVDA", "query": "data center demand", "filing_ids": [], "k": 1},
        ),
        ("get_source_spans", {"chunk_ids": [chunk_id]}),
    ]

    for tool_name, arguments in calls:
        before = _database_state(engine)
        result = await mcp_client.call_tool(tool_name, arguments)

        assert not result.is_error
        assert _database_state(engine) == before
