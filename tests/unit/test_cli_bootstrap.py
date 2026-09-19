from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine
from typer.testing import CliRunner

import fra.bootstrap as bootstrap_module
from fra.bootstrap import (
    BootstrapConfigurationError,
    BootstrapErrorCode,
    UnsupportedTickerError,
    build_industry_runtime,
    build_p0_runtime,
    build_p1_runtime,
    build_quality_runtime,
)
from fra.cli import app
from fra.config import Settings
from fra.contracts import ResearchCommand, ResearchMode
from fra.mcp_server.tools import FilingOutput
from fra.retrieval.indexing import (
    EmbeddingModelUnavailableError,
    HashEmbeddingProvider,
)
from fra.retrieval.ingest import IngestSummary, ingest_fixture
from fra.storage.cache import (
    InMemoryTtlJsonCache,
    NoopJsonCache,
    RedisJsonCache,
)
from fra.storage.database import create_schema
from fra.storage.repositories import FilingRepository
from fra.storage.run_repositories import RunStart
from fra.web_evidence.source_policy import SourcePolicy

runner = CliRunner()


def test_root_help_lists_p0_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "research" in result.output
    assert "ingest" in result.output
    assert "eval" in result.output


def test_research_help_preserves_thesis_and_lists_p1_options() -> None:
    result = runner.invoke(app, ["research", "--help"])

    assert result.exit_code == 0
    assert "ticker" in result.output.lower()
    assert "--thesis" in result.output
    assert "--question" in result.output
    assert "--mode" in result.output
    assert "--session-id" in result.output
    assert "--forms" in result.output
    assert "--as-of-date" in result.output
    assert "--market" in result.output
    assert "company-profile" in result.output
    assert "earnings-review" in result.output
    assert "industry-research" in result.output
    assert "market-snapshot" in result.output
    assert "required" in result.output.lower()


def test_research_cli_normalizes_and_passes_the_product_scope(monkeypatch) -> None:
    import fra.cli as cli_module

    captured: dict[str, ResearchCommand] = {}

    class Application:
        def run(self, command: ResearchCommand):
            captured["command"] = command
            return SimpleNamespace(rendered_output="ok")

    monkeypatch.setattr(cli_module, "Settings", lambda: object())
    monkeypatch.setattr(
        cli_module, "build_research_application", lambda *args, **kwargs: Application()
    )

    result = runner.invoke(
        app,
        [
            "research",
            "nvda",
            "--thesis",
            "  A factual thesis with   normalized spacing.  ",
            "--session-id",
            " session-1 ",
            "--forms",
            " 10-q, 8-k ",
            "--as-of-date",
            "2099-01-01",
            "--market",
            "us",
        ],
    )

    assert result.exit_code == 0, result.output
    command = captured["command"]
    assert command.ticker == "NVDA"
    assert command.request == "A factual thesis with normalized spacing."
    assert command.session_id == "session-1"
    assert command.forms == ("10-Q", "8-K")
    assert command.as_of_date == date(2099, 1, 1)
    assert command.market == "US"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--forms", "10-Q,20-F"],
        ["--forms", "10-Q,,8-K"],
        ["--as-of-date", "2099-02-30"],
        ["--market", "CA"],
        ["--session-id", "   "],
    ],
)
def test_invalid_research_scope_stops_before_settings_or_runtime(
    monkeypatch, arguments: list[str]
) -> None:
    import fra.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: pytest.fail("invalid input constructed settings"),
    )
    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "A factual thesis long enough to validate.",
            *arguments,
        ],
    )

    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("length", "accepted"), [(19, False), (20, True), (500, True), (501, False)]
)
def test_research_cli_enforces_normalized_thesis_boundaries(
    monkeypatch, length: int, accepted: bool
) -> None:
    import fra.cli as cli_module

    calls: list[ResearchCommand] = []

    class Application:
        def run(self, command: ResearchCommand):
            calls.append(command)
            return SimpleNamespace(rendered_output="ok")

    monkeypatch.setattr(cli_module, "Settings", lambda: object())
    monkeypatch.setattr(
        cli_module, "build_research_application", lambda *args, **kwargs: Application()
    )
    result = runner.invoke(app, ["research", "NVDA", "--thesis", f"  {'x' * length}  "])

    assert result.exit_code == (0 if accepted else 2), result.output
    assert len(calls) == int(accepted)


def test_postgres_fixture_cli_uses_hash_embeddings_without_constructing_bge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit fixture path stays offline even when the target database is PostgreSQL."""
    import fra.cli as cli_module
    import fra.retrieval.indexing as indexing_module
    import fra.retrieval.ingest as ingest_module

    repository = SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    captured: dict[str, object] = {}

    def fail_bge(*args, **kwargs):
        raise AssertionError("fixture ingest constructed BGE")

    def fixture_summary(path, ticker, form, supplied_repository, *, embedding_provider):
        captured["provider"] = embedding_provider
        assert supplied_repository is repository
        return IngestSummary(
            ticker="NVDA",
            requested_forms=["10-Q"],
            selected_form="10-Q",
            accession_no="0001045810-26-000001",
            filed_at=date(2026, 5, 20),
            source_url="https://www.sec.gov/Archives/nvda.htm",
            content_hash="a" * 64,
            corpus_version="NVDA-v1",
            chunk_count=2,
        )

    monkeypatch.setattr(cli_module, "build_filing_repository", lambda settings: repository)
    monkeypatch.setattr(indexing_module, "BgeM3EmbeddingProvider", fail_bge)
    monkeypatch.setattr(ingest_module, "ingest_fixture_summary", fixture_summary)

    result = runner.invoke(
        app,
        [
            "ingest",
            "--fixture",
            "tests/fixtures/nvda_10q.html",
            "--ticker",
            "NVDA",
            "--form",
            "10-Q",
        ],
    )

    assert result.exit_code == 0, result.output
    provider = captured["provider"]
    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimensions == 1024


def _p1_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite+pysqlite:///:memory:",
        "redis_url": None,
        "fast_model": "fast-test",
        "analyst_model": "analyst-test",
        "openai_api_key": SecretStr("model-key"),
        "tavily_api_key": SecretStr("web-key"),
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "langfuse_host": None,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def _seed_fixture(database_url: str) -> None:
    engine = create_engine(database_url)
    create_schema(engine)
    ingest_fixture(
        Path("tests/fixtures/nvda_10q.html"),
        "NVDA",
        "10-Q",
        FilingRepository(engine),
    )


def test_p1_without_redis_uses_noop_cache(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'no-redis.sqlite3'}"
    _seed_fixture(database_url)
    runtime = build_p1_runtime(
        _p1_settings(
            database_url=database_url,
            redis_url=None,
        ),
        ticker="NVDA",
    )

    assert isinstance(runtime.cache, NoopJsonCache)


def test_p1_without_web_key_disables_optional_web_adapter(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'no-web.sqlite3'}"
    _seed_fixture(database_url)
    runtime = build_p1_runtime(
        _p1_settings(
            database_url=database_url,
            tavily_api_key=None,
            context_compressor_cache_dir="/approved/llmlingua",
            tokenizer_cache_dir="/approved/tokenizer",
        ),
        ticker="NVDA",
    )

    assert runtime.web_search is None
    assert runtime.dependencies.skill_repair_model is runtime.dependencies.skill_analyst
    compressor = runtime.dependencies.skill_analyst._context_compressor
    assert compressor._model_cache_dir == "/approved/llmlingua"
    assert compressor._tokenizer_cache_dir == "/approved/tokenizer"


def test_p1_freezes_one_source_policy_before_any_tool_call(tmp_path) -> None:
    """Gateway and final guard must share one eager immutable policy/version snapshot."""
    snapshots: list[SourcePolicy] = []

    def source_policy_factory(*, issuer_domains) -> SourcePolicy:
        snapshot = SourcePolicy(issuer_domains=issuer_domains)
        snapshots.append(snapshot)
        return snapshot

    settings = _p1_settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'frozen-policy.sqlite3'}",
        web_issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})},
    )
    _seed_fixture(settings.database_url)
    runtime = build_p1_runtime(
        settings,
        ticker="NVDA",
        source_policy_factory=source_policy_factory,
    )

    assert len(snapshots) == 1
    assert runtime.source_policy_version == snapshots[0].version
    assert runtime.dependencies.web_evidence_validator.policy is snapshots[0]


def test_industry_runtime_uses_the_expanded_policy_profile_without_mutating_standard(
    tmp_path,
) -> None:
    settings = _p1_settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'industry-policy.sqlite3'}",
        web_issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})},
    )
    _seed_fixture(settings.database_url)

    standard = build_p1_runtime(settings, ticker="NVDA")
    industry = build_industry_runtime(settings, ticker="NVDA")

    assert standard.dependencies.web_evidence_validator is not None
    assert industry.dependencies.web_evidence_validator is not None
    assert "ftc.gov" not in standard.dependencies.web_evidence_validator.policy.authority_domains
    assert "ftc.gov" in industry.dependencies.web_evidence_validator.policy.authority_domains
    assert (
        industry.source_policy_version
        != standard.dependencies.web_evidence_validator.policy.version
    )


def test_quality_runtime_uses_utc_date_provider_and_bounded_max_age(tmp_path) -> None:
    settings = _p1_settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'quality-policy.sqlite3'}",
        research_quality_max_source_age_days=180,
    )
    _seed_fixture(settings.database_url)

    runtime = build_quality_runtime(settings, ticker="NVDA")

    assert runtime.current_date_factory is bootstrap_module._utc_today
    assert runtime.max_source_age_days == 180


def test_live_runtimes_require_explicit_local_corpus_before_external_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    def forbidden(name: str):
        def _inner(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            raise AssertionError(f"unexpected {name}")

        return _inner

    monkeypatch.setattr(bootstrap_module, "build_cache", forbidden("build_cache"))
    monkeypatch.setattr(bootstrap_module, "create_server", forbidden("create_server"))
    settings = _p1_settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'missing-corpus.sqlite3'}",
    )

    for builder in (build_p1_runtime, build_industry_runtime, build_quality_runtime):
        with pytest.raises(UnsupportedTickerError) as error:
            builder(settings, ticker="NVDA")
        assert error.value.code is BootstrapErrorCode.UNSUPPORTED_TICKER

    assert calls == []


def test_p1_without_model_key_fails_before_constructing_runtime(monkeypatch) -> None:
    def fail_engine_construction(url: str) -> object:
        raise AssertionError(f"unexpected engine construction for {url}")

    monkeypatch.setattr(
        "fra.bootstrap.create_engine",
        fail_engine_construction,
    )

    with pytest.raises(BootstrapConfigurationError) as error:
        build_p1_runtime(_p1_settings(openai_api_key=None), ticker="NVDA")

    assert error.value.code is BootstrapErrorCode.P1_MODEL_CONFIGURATION_MISSING
    assert "OPENAI_API_KEY" in error.value.detail


def test_thesis_production_runtime_uses_documented_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Replacing any production dependency with its demo substitute must fail this test."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'thesis-production.sqlite3'}"
    _seed_fixture(database_url)
    constructed: dict[str, object] = {}

    def embedding_factory(model_name: str, *, cache_dir: str | None = None):
        constructed["embedding"] = (model_name, cache_dir)
        return HashEmbeddingProvider()

    def reranker_factory(
        model_name: str,
        *,
        cache_dir=None,
        ranker_factory=None,
        asset_validator=None,
    ):
        del ranker_factory, asset_validator
        constructed["reranker"] = (model_name, cache_dir)
        from fra.retrieval.rerank import IdentityReranker

        return IdentityReranker()

    cache = InMemoryTtlJsonCache()
    monkeypatch.setattr(bootstrap_module, "BgeM3EmbeddingProvider", embedding_factory)
    monkeypatch.setattr(bootstrap_module, "LazyFlashRankReranker", reranker_factory)
    monkeypatch.setattr(bootstrap_module, "build_cache", lambda settings: cache)

    runtime = build_p0_runtime(
        _p1_settings(
            database_url=database_url,
            redis_url="redis://cache.example:6379/0",
            offline_demo=False,
            tavily_api_key=None,
            reranker_cache_dir="/approved/reranker",
            context_compressor_cache_dir="/approved/llmlingua",
            tokenizer_cache_dir="/approved/tokenizer",
        ),
        ticker="NVDA",
    )

    assert constructed == {
        "embedding": ("BAAI/bge-m3", None),
        "reranker": ("ms-marco-MiniLM-L-12-v2", "/approved/reranker"),
    }
    assert runtime.cache is cache
    assert isinstance(runtime.dependencies.fast_model, bootstrap_module.OpenAIThesisFastModel)
    assert isinstance(
        runtime.dependencies.analyst_model,
        bootstrap_module.OpenAIThesisAnalystModel,
    )
    assert runtime.dependencies.thesis_repair_model is runtime.dependencies.analyst_model
    assert isinstance(
        runtime.dependencies.fast_model._context_compressor,
        bootstrap_module.LazyLLMLinguaCompressor,
    )
    assert (
        runtime.dependencies.fast_model._context_compressor
        is runtime.dependencies.analyst_model._context_compressor
    )
    assert (
        runtime.dependencies.fast_model._context_compressor._model_cache_dir
        == "/approved/llmlingua"
    )
    assert (
        runtime.dependencies.fast_model._context_compressor._tokenizer_cache_dir
        == "/approved/tokenizer"
    )
    assert runtime.dependencies.thesis_collector is not None
    assert runtime.dependencies.thesis_collection_policy.budget.max_retrieval_rounds == 2
    assert runtime.dependencies.thesis_collection_policy.budget.max_web_calls == 1
    assert runtime.execution_note != "deterministic offline demo"


@pytest.mark.parametrize(
    ("overrides", "missing_name"),
    [
        ({"openai_api_key": None}, "OPENAI_API_KEY"),
        ({"fast_model": None}, "FAST_MODEL"),
        ({"analyst_model": None}, "ANALYST_MODEL"),
    ],
)
def test_thesis_production_missing_configuration_fails_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    missing_name: str,
) -> None:
    monkeypatch.setattr(
        bootstrap_module,
        "create_engine",
        lambda url: pytest.fail(f"missing configuration constructed engine {url}"),
    )

    with pytest.raises(BootstrapConfigurationError) as caught:
        build_p0_runtime(
            _p1_settings(
                **{
                    "redis_url": "redis://cache.example:6379/0",
                    "offline_demo": False,
                    **overrides,
                }
            ),
            ticker="NVDA",
        )

    assert caught.value.code is BootstrapErrorCode.THESIS_CONFIGURATION_MISSING
    assert missing_name in caught.value.detail


def test_thesis_production_with_web_key_requires_tavily_dependency_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise RuntimeError("private import detail")

    monkeypatch.setattr(
        bootstrap_module,
        "ensure_tavily_dependency",
        unavailable,
        raising=False,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "create_engine",
        lambda url: pytest.fail(f"missing Tavily dependency constructed engine {url}"),
    )

    with pytest.raises(BootstrapConfigurationError) as caught:
        build_p0_runtime(
            _p1_settings(
                redis_url="redis://cache.example:6379/0",
                offline_demo=False,
                tavily_api_key=SecretStr("configured-web-key"),
            ),
            ticker="NVDA",
        )

    assert caught.value.code is BootstrapErrorCode.WEB_SEARCH_DEPENDENCY_UNAVAILABLE
    assert caught.value.detail == (
        "TAVILY_API_KEY is configured but the optional 'web-search' dependency is unavailable"
    )
    assert "private import detail" not in caught.value.detail


def test_thesis_cli_never_silently_substitutes_demo_for_missing_production_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'missing-thesis-config.sqlite3'}"
    _seed_fixture(database_url)
    monkeypatch.setenv("DATABASE_URL", database_url)
    for name in ("OPENAI_API_KEY", "FAST_MODEL", "ANALYST_MODEL", "REDIS_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OFFLINE_DEMO", "false")

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does disclosed demand support sustained revenue growth?",
        ],
    )

    assert result.exit_code == 2
    assert BootstrapErrorCode.THESIS_CONFIGURATION_MISSING.value in result.output
    assert "deterministic offline demo" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("client", "expected_error"),
    [
        (
            SimpleNamespace(
                call_tool=lambda name, arguments: (_ for _ in ()).throw(
                    RuntimeError("private transport detail")
                )
            ),
            "MCP_CALL_ERROR: filing scope resolution failed",
        ),
        (
            SimpleNamespace(
                call_tool=lambda name, arguments: SimpleNamespace(
                    is_error=True,
                    structured_content=None,
                )
            ),
            "MCP_PROTOCOL_ERROR: filing scope tool returned is_error",
        ),
        (
            SimpleNamespace(
                call_tool=lambda name, arguments: SimpleNamespace(
                    is_error=False,
                    structured_content={"filings": [{"malformed": True}]},
                )
            ),
            "MCP_RESPONSE_ERROR: invalid filing scope response",
        ),
    ],
)
def test_scope_resolver_distinguishes_call_protocol_and_response_failures(
    client: object,
    expected_error: str,
) -> None:
    command = ResearchCommand(
        ticker="NVDA",
        request="Does disclosed demand support sustained growth?",
        mode=ResearchMode.THESIS,
    )

    resolved = bootstrap_module._resolve_research_scope(client, command)

    assert resolved.corpus_version is None
    assert resolved.filing_ids == ()
    assert resolved.scope_error == expected_error
    assert "private transport detail" not in resolved.scope_error


def test_scope_resolver_keeps_valid_no_filings_as_an_ordinary_empty_scope() -> None:
    client = SimpleNamespace(
        call_tool=lambda name, arguments: SimpleNamespace(
            is_error=False,
            structured_content={
                "corpus_version": None,
                "filings": [],
                "error": {"code": "NO_FILINGS", "message": "private upstream detail"},
            },
        )
    )
    command = ResearchCommand(
        ticker="NVDA",
        request="Does disclosed demand support sustained growth?",
        mode=ResearchMode.THESIS,
    )

    resolved = bootstrap_module._resolve_research_scope(client, command)

    assert resolved.scope_error == "NO_FILINGS: no filings matched the requested scope"
    assert "private upstream detail" not in resolved.scope_error


@pytest.mark.parametrize(
    "structured_content",
    [
        {
            "corpus_version": "NVDA-v1",
            "filings": [
                {
                    "id": "filing-1",
                    "ticker": "NVDA",
                    "form": "10-Q",
                    "filed_at": "2026-05-20",
                    "source_url": "https://www.sec.gov/Archives/filing-1.htm",
                    "accession_no": "0001045810-26-000001",
                    "corpus_version": "NVDA-v1",
                }
            ],
            "error": {"code": "NO_FILINGS", "message": "contradictory payload"},
        },
        {
            "corpus_version": None,
            "filings": [
                {
                    "id": "filing-1",
                    "ticker": "NVDA",
                    "form": "10-Q",
                    "filed_at": "2026-05-20",
                    "source_url": "https://www.sec.gov/Archives/filing-1.htm",
                    "accession_no": "0001045810-26-000001",
                    "corpus_version": "NVDA-v1",
                }
            ],
            "error": None,
        },
    ],
)
def test_scope_resolver_rejects_contradictory_or_incomplete_success_envelopes(
    structured_content: dict[str, object],
) -> None:
    client = SimpleNamespace(
        call_tool=lambda name, arguments: SimpleNamespace(
            is_error=False,
            structured_content=structured_content,
        )
    )
    command = ResearchCommand(
        ticker="NVDA",
        request="Does disclosed demand support sustained growth?",
        mode=ResearchMode.THESIS,
    )

    resolved = bootstrap_module._resolve_research_scope(client, command)

    assert resolved.scope_error == "MCP_RESPONSE_ERROR: invalid filing scope response"
    assert resolved.corpus_version is None
    assert resolved.filing_ids == ()


@pytest.mark.parametrize(
    ("top_level_version", "nested_version"),
    [
        ("", "NVDA-v1"),
        ("   ", "NVDA-v1"),
        ("NVDA-v1", ""),
        ("NVDA-v1", "   "),
    ],
)
def test_scope_resolver_rejects_blank_top_level_or_nested_corpus_version(
    top_level_version: str,
    nested_version: str,
) -> None:
    client = SimpleNamespace(
        call_tool=lambda name, arguments: SimpleNamespace(
            is_error=False,
            structured_content={
                "corpus_version": top_level_version,
                "filings": [
                    {
                        "id": "filing-1",
                        "ticker": "NVDA",
                        "form": "10-Q",
                        "filed_at": "2026-05-20",
                        "source_url": "https://www.sec.gov/Archives/filing-1.htm",
                        "accession_no": "0001045810-26-000001",
                        "corpus_version": nested_version,
                    }
                ],
                "error": None,
            },
        )
    )
    command = ResearchCommand(
        ticker="NVDA",
        request="Does disclosed demand support sustained growth?",
        mode=ResearchMode.THESIS,
    )

    resolved = bootstrap_module._resolve_research_scope(client, command)

    assert resolved.scope_error == "MCP_RESPONSE_ERROR: invalid filing scope response"
    assert resolved.corpus_version is None
    assert resolved.filing_ids == ()


def test_scope_resolver_defensively_rejects_blank_version_after_validation_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupt = bootstrap_module.FetchRecentFilingsResponse.model_construct(
        corpus_version="   ",
        filings=[
            FilingOutput.model_construct(
                id="filing-1",
                ticker="NVDA",
                form="10-Q",
                filed_at="2026-05-20",
                source_url="https://www.sec.gov/Archives/filing-1.htm",
                accession_no="0001045810-26-000001",
                corpus_version="   ",
            )
        ],
        error=None,
    )
    monkeypatch.setattr(
        bootstrap_module.FetchRecentFilingsResponse,
        "model_validate",
        classmethod(lambda cls, value: corrupt),
    )
    client = SimpleNamespace(
        call_tool=lambda name, arguments: SimpleNamespace(
            is_error=False,
            structured_content={"bypassed": True},
        )
    )
    command = ResearchCommand(
        ticker="NVDA",
        request="Does disclosed demand support sustained growth?",
        mode=ResearchMode.THESIS,
    )

    resolved = bootstrap_module._resolve_research_scope(client, command)

    assert resolved.scope_error == "MCP_RESPONSE_ERROR: invalid filing scope response"
    assert resolved.corpus_version is None
    assert resolved.filing_ids == ()


def test_p0_offline_runtime_never_constructs_external_clients(monkeypatch, tmp_path) -> None:
    def fail_external_construction(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("P0 attempted to construct an external client")

    for name in (
        "build_trace_sink",
        "OpenAISkillPlannerModel",
        "OpenAISkillAnalystModel",
        "OpenAIThesisFastModel",
        "OpenAIThesisAnalystModel",
        "TavilySearchProvider",
    ):
        monkeypatch.setattr(
            f"fra.bootstrap.{name}",
            fail_external_construction,
        )

    cache = NoopJsonCache()
    monkeypatch.setattr(bootstrap_module, "build_cache", lambda settings: cache)

    runtime = build_p0_runtime(
        _p1_settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'p0.sqlite3'}",
            redis_url="redis://cache.example:6379/0",
            offline_demo=True,
        ),
        ticker="NVDA",
    )

    assert runtime.execution_note.startswith("deterministic offline demo")
    assert runtime.web_search is None
    assert runtime.cache is cache


def test_p0_offline_runtime_passes_configured_cache_to_hybrid_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Omitting this cache makes separately started offline CLI runs cold every time."""
    captured: dict[str, object] = {}

    class CapturingRetriever:
        def __init__(self, repository, embedding_provider, *, cache) -> None:
            captured["repository"] = repository
            captured["embedding_provider"] = embedding_provider
            captured["cache"] = cache

    monkeypatch.setattr(bootstrap_module, "HybridRetriever", CapturingRetriever)
    monkeypatch.setattr(bootstrap_module, "create_server", lambda repository, retriever: object())

    runtime = build_p0_runtime(
        _p1_settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'p0.sqlite3'}",
            redis_url="redis://cache.example:6379/0",
            offline_demo=True,
        ),
        ticker="NVDA",
    )

    assert isinstance(captured["cache"], RedisJsonCache)
    assert runtime.cache is captured["cache"]


def test_p0_offline_runtime_without_redis_uses_noop_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The offline demo must remain deterministic when no Redis service is configured."""
    captured: dict[str, object] = {}

    class CapturingRetriever:
        def __init__(self, repository, embedding_provider, *, cache) -> None:
            del repository, embedding_provider
            captured["cache"] = cache

    monkeypatch.setattr(bootstrap_module, "HybridRetriever", CapturingRetriever)
    monkeypatch.setattr(bootstrap_module, "create_server", lambda repository, retriever: object())

    runtime = build_p0_runtime(
        _p1_settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'p0.sqlite3'}",
            redis_url=None,
            offline_demo=True,
        ),
        ticker="NVDA",
    )

    assert isinstance(captured["cache"], NoopJsonCache)
    assert runtime.cache is captured["cache"]


def test_fixed_refusal_survives_unreachable_postgres_without_runtime_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusals still trace and best-effort persist, but never build a runtime."""

    class _Observation:
        trace_id = "trace-refusal"
        output: object | None = None

        def update(self, *, output=None, metadata=None) -> None:
            del metadata
            self.output = output

        def observation(self, *, name, kind, input=None, metadata=None):
            del name, kind, input, metadata

            class _Manager:
                def __enter__(self_nonlocal):
                    return _Observation()

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _Manager()

    class _TraceSink:
        def __init__(self) -> None:
            self.roots: list[_Observation] = []
            self.flush_calls = 0

        def run(self, *, run_id, input, metadata):
            del run_id, input, metadata
            sink = self

            class _Manager:
                def __enter__(self_nonlocal):
                    root = _Observation()
                    sink.roots.append(root)
                    return root

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _Manager()

        def flush(self) -> None:
            self.flush_calls += 1

    runtime_calls: list[str] = []
    db_calls: list[dict[str, object]] = []
    sink = _TraceSink()

    def forbidden_runtime(*args: object, **kwargs: object) -> object:
        del args, kwargs
        runtime_calls.append("called")
        raise AssertionError("fixed refusal built a runtime")

    def failing_engine(url: str, **kwargs: object) -> object:
        db_calls.append({"url": url, **kwargs})
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(bootstrap_module, "build_trace_sink", lambda settings: sink)
    monkeypatch.setattr(bootstrap_module, "create_engine", failing_engine)
    application = bootstrap_module.build_research_application(
        Settings(
            database_url="postgresql+psycopg://unreachable.invalid/db",
            refusal_persistence_connect_timeout_seconds=2,
            _env_file=None,
        ),
        p0_builder=forbidden_runtime,
        p1_builder=forbidden_runtime,
        industry_builder=forbidden_runtime,
        market_builder=forbidden_runtime,
        quality_builder=forbidden_runtime,
    )

    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="Should I buy NVDA?",
            mode=ResearchMode.AUTO,
        )
    )

    assert result.status == "refused"
    assert sink.flush_calls == 1
    assert len(sink.roots) == 1
    assert db_calls == [
        {
            "url": "postgresql+psycopg://unreachable.invalid/db",
            "connect_args": {"connect_timeout": 2},
        },
    ]
    assert runtime_calls == []


def test_lazy_run_repository_configures_short_postgres_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any best-effort PostgreSQL persistence connection must have a bounded timeout."""
    captured: dict[str, object] = {}

    class Writer:
        def start(self, value: object) -> None:
            captured["value"] = value

    def engine_factory(url: str, **kwargs: object) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bootstrap_module, "create_engine", engine_factory)
    monkeypatch.setattr(bootstrap_module, "_initialize_schema", lambda engine: None)
    monkeypatch.setattr(bootstrap_module, "ResearchRunRepository", lambda engine: Writer())
    settings = Settings(
        database_url="postgresql+psycopg://database.example/research",
        refusal_persistence_connect_timeout_seconds=2,
        _env_file=None,
    )

    bootstrap_module._LazyResearchRunRepository(settings).start(
        RunStart(run_id="run-1", ticker="NVDA", request="question")
    )

    assert captured["connect_args"] == {"connect_timeout": 2}


def test_lazy_company_resolver_reads_canonical_ticker_from_local_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production resolver must lazily reuse the populated local company table."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    FilingRepository(engine).upsert_company_metadata(
        ticker="NVDA",
        cik="0001045810",
        legal_name="NVIDIA Corporation",
        ir_domain="investor.nvidia.com",
    )
    engine_calls: list[str] = []
    schema_calls: list[object] = []

    def engine_factory(url: str):
        engine_calls.append(url)
        return engine

    monkeypatch.setattr(bootstrap_module, "create_engine", engine_factory)
    monkeypatch.setattr(
        bootstrap_module,
        "_initialize_schema",
        lambda value: schema_calls.append(value),
    )
    resolver = bootstrap_module._LazyCompanyResolver(
        Settings(database_url="sqlite+pysqlite:///:memory:", _env_file=None)
    )

    assert engine_calls == []
    assert resolver.resolve("nvda") == "NVDA"
    assert resolver.resolve("NOPE") is None
    assert engine_calls == ["sqlite+pysqlite:///:memory:"]
    assert schema_calls == [engine]


def test_lazy_company_resolver_returns_fixed_unavailable_without_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private = "password=private-password-value"

    def failing_engine(url: str) -> object:
        del url
        raise RuntimeError(private)

    monkeypatch.setattr(bootstrap_module, "create_engine", failing_engine)
    resolver = bootstrap_module._LazyCompanyResolver(
        Settings(database_url="sqlite+pysqlite:///:memory:", _env_file=None)
    )

    with caplog.at_level("WARNING"):
        assert resolver.resolve("NVDA") is None

    assert "company resolution unavailable" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


def test_research_cli_maps_embedding_model_unavailable_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing local BGE assets must become a stable CLI error, never a provider traceback."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'missing-embedding.sqlite3'}"
    _seed_fixture(database_url)
    monkeypatch.setenv("DATABASE_URL", database_url)

    def unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise EmbeddingModelUnavailableError(
            "EMBEDDING_MODEL_UNAVAILABLE: prefetch BAAI/bge-m3 before startup"
        )

    monkeypatch.setattr("fra.cli.build_p1_runtime", unavailable)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "company-profile",
            "--question",
            "Introduce the company",
        ],
    )

    assert result.exit_code == 2
    assert "EMBEDDING_MODEL_UNAVAILABLE" in result.output
    assert "Traceback" not in result.output


def test_thesis_production_runs_without_optional_redis(monkeypatch, tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'thesis-no-redis.sqlite3'}"
    _seed_fixture(database_url)
    monkeypatch.setattr(
        bootstrap_module, "BgeM3EmbeddingProvider", lambda *args, **kwargs: HashEmbeddingProvider()
    )
    runtime = build_p0_runtime(
        _p1_settings(
            database_url=database_url, redis_url=None, offline_demo=False, tavily_api_key=None
        ),
        ticker="NVDA",
    )
    assert isinstance(runtime.cache, NoopJsonCache)
    assert runtime.dependencies.thesis_collector is not None
    assert runtime.repository.latest_corpus_version("NVDA") is not None
