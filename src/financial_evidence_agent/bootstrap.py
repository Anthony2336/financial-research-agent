"""Central runtime composition for CLI and MCP entry points."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from threading import RLock
from typing import Any

import httpx
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import StaticPool

from financial_evidence_agent.application import (
    IntentRouter,
    NoopResearchRunWriter,
    ResearchApplication,
    ResearchCommand,
    ResearchExecutionContext,
    current_research_context,
)
from financial_evidence_agent.config import Settings
from financial_evidence_agent.context import (
    BudgetExhaustedError,
    BudgetGate,
    BudgetLimits,
    LazyLLMLinguaCompressor,
)
from financial_evidence_agent.domain import Intent, RouterDecision, SourceRef
from financial_evidence_agent.graph.demo_models import (
    DEMO_TICKER,
    DeterministicDemoAnalystModel,
    DeterministicDemoFastModel,
    bundled_nvda_fixture_path,
)
from financial_evidence_agent.graph.models import (
    Dependencies,
    FastMCPToolClient,
    FastMCPToolSession,
    MarketDependencies,
    MarketResearchResult,
    ResearchResult,
)
from financial_evidence_agent.market_data.alpaca import AlpacaMarketDataProvider
from financial_evidence_agent.market_data.gateway import (
    MarketDataGateway,
    MarketFetchWriter,
    NoopMarketFetchWriter,
)
from financial_evidence_agent.market_data.models import (
    MarketDataError,
    MarketDataErrorCode,
)
from financial_evidence_agent.mcp_server.client_adapters import (
    MCPFilingSearch,
    MCPWebSearch,
)
from financial_evidence_agent.mcp_server.market_tools import (
    RunScopedMarketFetchWriter,
    register_market_tools,
)
from financial_evidence_agent.mcp_server.server import create_server
from financial_evidence_agent.mcp_server.tools import FetchRecentFilingsResponse
from financial_evidence_agent.mcp_server.web_tools import register_web_tools
from financial_evidence_agent.memory.research import (
    ResearchMemory,
    ResearchMemoryKind,
    ResearchMemoryService,
    scope_research_memory_store,
)
from financial_evidence_agent.memory.session import SessionMemoryStore
from financial_evidence_agent.model_providers.openai import (
    OpenAIIntentRouter,
    OpenAISkillAnalystModel,
    OpenAISkillPlannerModel,
    OpenAIThesisAnalystModel,
    OpenAIThesisFastModel,
)
from financial_evidence_agent.observability import (
    NoopTraceSink,
    build_trace_sink,
    observe,
)
from financial_evidence_agent.research_packages.orchestrator import PeerResearchResult
from financial_evidence_agent.research_packages.quality import QualityResearchRuntime
from financial_evidence_agent.retrieval.collector import (
    THESIS_COLLECTION_POLICY,
    EvidenceCollector,
    WebSearch,
)
from financial_evidence_agent.retrieval.hybrid import (
    BgeM3EmbeddingProvider,
    HashEmbeddingProvider,
    HybridRetriever,
    RetrievalMetrics,
)
from financial_evidence_agent.retrieval.indexing import EmbeddingIndexer, EmbeddingProvider
from financial_evidence_agent.retrieval.ingest import ingest_fixture
from financial_evidence_agent.retrieval.rerank import LazyFlashRankReranker
from financial_evidence_agent.storage.cache import (
    CompositeJsonCache,
    MarketDataJsonCache,
    NoopJsonCache,
    build_cache,
    build_market_cache,
    build_session_cache,
)
from financial_evidence_agent.storage.database import (
    DatabaseMigrationRequiredError,
    create_schema,
    ensure_migrations_current,
)
from financial_evidence_agent.storage.market_repositories import MarketDataRepository
from financial_evidence_agent.storage.memory_repositories import ResearchMemoryRepository
from financial_evidence_agent.storage.repositories import FilingRepository
from financial_evidence_agent.storage.run_repositories import (
    ResearchRunRepository,
    RunFinish,
    RunStart,
    SourceFetchWrite,
)
from financial_evidence_agent.storage.web_repositories import (
    SkillRunRepository,
    WebEvidenceRepository,
)
from financial_evidence_agent.web_evidence.gateway import AllowlistedWebGateway
from financial_evidence_agent.web_evidence.providers import (
    HttpxRedirectResolver,
    TavilySearchProvider,
    ensure_tavily_dependency,
)
from financial_evidence_agent.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
    build_industry_source_policy,
    build_standard_source_policy,
)

logger = logging.getLogger(__name__)


class BootstrapErrorCode(StrEnum):
    """Stable startup failures safe to expose at command boundaries."""

    P1_MODEL_CONFIGURATION_MISSING = "P1_MODEL_CONFIGURATION_MISSING"
    THESIS_CONFIGURATION_MISSING = "THESIS_CONFIGURATION_MISSING"
    WEB_SEARCH_DEPENDENCY_UNAVAILABLE = "WEB_SEARCH_DEPENDENCY_UNAVAILABLE"
    FAST_MODEL_CONFIGURATION_MISSING = "FAST_MODEL_CONFIGURATION_MISSING"
    DATABASE_MIGRATION_REQUIRED = "DATABASE_MIGRATION_REQUIRED"
    UNSUPPORTED_TICKER = "UNSUPPORTED_TICKER"


class BootstrapError(RuntimeError):
    """Typed failure raised before a workflow can run with invalid dependencies."""

    def __init__(self, code: BootstrapErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class BootstrapConfigurationError(BootstrapError):
    """Fail-closed runtime configuration error."""


class UnsupportedTickerError(BootstrapError):
    """A ticker absent from the local, explicitly ingested corpus."""

    def __init__(self, ticker: str) -> None:
        normalized = ticker.strip().upper()
        super().__init__(
            BootstrapErrorCode.UNSUPPORTED_TICKER,
            f"No supported local company or filing corpus for {normalized}",
        )


class _LazyCompanyResolver:
    """Resolve supported tickers through one lazy local repository."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._repository: FilingRepository | None = None

    def resolve(self, ticker: str) -> str | None:
        try:
            company = self._get().get_company(ticker)
        except Exception:
            logger.warning("company resolution unavailable")
            return None
        return None if company is None else company.ticker

    def _get(self) -> FilingRepository:
        if self._repository is None:
            self._engine = create_engine(self._settings.database_url)
            _initialize_schema(self._engine)
            self._repository = FilingRepository(self._engine)
        return self._repository


class _LazyResearchRunRepository:
    """Delay database/schema access until after deterministic safety routing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._repository: ResearchRunRepository | None = None
        self._skill_repository: SkillRunRepository | None = None

    def start(self, value: RunStart) -> None:
        self._get().start(value)

    def finish(self, value: RunFinish) -> None:
        self._get().finish(value)

    def record_fetch(self, value: SourceFetchWrite) -> None:
        self._get().record_fetch(value)

    def start_refusal(self, value: RunStart) -> bool:
        """Best-effort refusal persistence still initializes the configured repository."""
        self._get().start(value)
        return True

    def finish_refusal(self, value: RunFinish) -> None:
        """Best-effort refusal persistence uses the same initialized repository contract."""
        self._get().finish(value)

    def _get(self) -> ResearchRunRepository:
        if self._repository is None:
            connect_args = (
                {"connect_timeout": (self._settings.refusal_persistence_connect_timeout_seconds)}
                if make_url(self._settings.database_url).get_backend_name() == "postgresql"
                else {}
            )
            self._engine = create_engine(
                self._settings.database_url,
                connect_args=connect_args,
            )
            _initialize_schema(self._engine)
            self._repository = ResearchRunRepository(self._engine)
        return self._repository

    def skill_runs(self) -> SkillRunRepository:
        """Return a skill writer backed by the same lazy application engine."""
        self._get()
        assert self._engine is not None
        if self._skill_repository is None:
            self._skill_repository = SkillRunRepository(self._engine)
        return self._skill_repository


class _LazySkillRunRepository:
    """Delay quality skill-run storage until the owning run has started."""

    def __init__(self, repositories: _LazyResearchRunRepository) -> None:
        self._repositories = repositories

    def start(self, **values: Any) -> str:
        return self._get().start(**values)

    def finish(self, run_id: str, **values: Any) -> None:
        self._get().finish(run_id, **values)

    def _get(self) -> SkillRunRepository:
        return self._repositories.skill_runs()


class _ResearchMemoryResources:
    """Thread-safe engine/provider cache with policy-isolated memory services."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = RLock()
        self._engine: Engine | None = None
        self._provider: EmbeddingProvider | None = None
        self._services: dict[str, ResearchMemoryService] = {}

    def service(self, validator: object | None) -> ResearchMemoryService:
        policy_key = _research_memory_policy_key(validator)
        with self._lock:
            existing = self._services.get(policy_key)
            if existing is not None:
                return existing
            if self._engine is None:
                self._engine = create_engine(self._settings.database_url)
                _initialize_schema(self._engine)
            if self._provider is None:
                self._provider = (
                    HashEmbeddingProvider()
                    if self._settings.offline_demo
                    else BgeM3EmbeddingProvider(
                        self._settings.embedding_model,
                        cache_dir=self._settings.embedding_cache_dir,
                    )
                )
            service = ResearchMemoryService(
                ResearchMemoryRepository(
                    self._engine,
                    web_evidence_validator=validator,
                ),
                self._provider,
                ttl_days=self._settings.research_memory_ttl_days,
            )
            self._services[policy_key] = service
            return service


def _research_memory_policy_key(validator: object | None) -> str:
    if validator is None:
        return "filing-only"
    version = getattr(validator, "policy_version", None)
    if isinstance(version, str) and version.strip():
        return f"policy:{version.strip()}"
    return f"validator:{type(validator).__qualname__}:{id(validator)}"


class _LazyResearchMemoryStore:
    """Policy-scoped view over shared lazy engine, provider, and service resources."""

    def __init__(
        self,
        settings: Settings,
        *,
        web_evidence_validator: object | None = None,
        resources: _ResearchMemoryResources | None = None,
    ) -> None:
        self._settings = settings
        self._web_evidence_validator = web_evidence_validator
        self._resources = resources or _ResearchMemoryResources(settings)

    def with_web_evidence_validator(
        self,
        validator: object | None,
    ) -> _LazyResearchMemoryStore:
        """Return a lazy store bound to one runtime's immutable current policy."""
        return _LazyResearchMemoryStore(
            self._settings,
            web_evidence_validator=validator,
            resources=self._resources,
        )

    def search(
        self,
        ticker: str,
        query: str,
        *,
        current_corpus_version: str | None = None,
        limit: int = 3,
    ) -> list[ResearchMemory]:
        return self._get().search(
            ticker,
            query,
            current_corpus_version=current_corpus_version,
            limit=limit,
        )

    def store_guarded(
        self,
        *,
        ticker: str,
        memory_kind: ResearchMemoryKind,
        summary: str,
        source_run_id: str,
        evidence_source_refs: tuple[SourceRef, ...],
        corpus_version: str,
        importance: float = 0.5,
    ) -> ResearchMemory:
        return self._get().store_guarded(
            ticker=ticker,
            memory_kind=memory_kind,
            summary=summary,
            source_run_id=source_run_id,
            evidence_source_refs=evidence_source_refs,
            corpus_version=corpus_version,
            importance=importance,
        )

    def _get(self) -> ResearchMemoryService:
        return self._resources.service(self._web_evidence_validator)


def _initialize_schema(engine: Engine) -> None:
    """Use isolated metadata creation only outside deployed PostgreSQL databases."""
    if engine.dialect.name != "postgresql":
        create_schema(engine)
        return
    try:
        ensure_migrations_current(engine)
    except DatabaseMigrationRequiredError as error:
        raise BootstrapConfigurationError(
            BootstrapErrorCode.DATABASE_MIGRATION_REQUIRED,
            str(error).split(": ", maxsplit=1)[1],
        ) from error


@dataclass(frozen=True, slots=True)
class ResearchRuntime:
    """Provider-neutral dependencies and truthful execution metadata for one CLI run."""

    dependencies: Dependencies
    repository: FilingRepository | None = None
    cache: CompositeJsonCache = field(default_factory=NoopJsonCache)
    web_search: WebSearch | None = None
    execution_note: str | None = None
    run_context_factory: Callable[[ResearchExecutionContext], Dependencies] | None = None
    scope_resolver: Callable[[ResearchCommand, BudgetGate | None], ResearchCommand] | None = None
    source_policy_version: str | None = None

    def apply_execution_note(
        self,
        rendered_output: str,
        *,
        is_p1: bool,
        has_guarded_memo: bool,
    ) -> str:
        return _render_with_runtime_note(
            rendered_output,
            self,
            is_p1=is_p1,
            has_guarded_memo=has_guarded_memo,
        )

    def execute(
        self,
        command: ResearchCommand,
        decision: RouterDecision,
    ) -> ResearchResult | PeerResearchResult:
        """Execute the selected graph and apply truthful runtime metadata."""
        from financial_evidence_agent.graph.nodes import dispatch_skill_recipes
        from financial_evidence_agent.graph.workflow import run_research

        context = current_research_context()
        dependencies = self.dependencies
        run_id = "untracked"
        if context is not None:
            policies = (
                (self.dependencies.thesis_collection_policy or THESIS_COLLECTION_POLICY,)
                if decision.intent is Intent.RESEARCH_REQUEST
                else dispatch_skill_recipes(decision.intent)
            )
            if not context.budget.configured:
                context.budget.configure(BudgetLimits.aggregate(policies))
            run_id = context.run_id
            dependencies = replace(
                self.dependencies,
                trace_run=context.trace,
                budget=context.budget,
                session_memory_store=context.session_memory_store,
                research_memory_store=scope_research_memory_store(
                    context.research_memory_store,
                    self.dependencies.web_evidence_validator,
                ),
            )
            if self.run_context_factory is not None:
                scoped = self.run_context_factory(context)
                dependencies = replace(
                    dependencies,
                    skill_collector=scoped.skill_collector,
                    thesis_collector=scoped.thesis_collector,
                )
        if self.scope_resolver is not None:
            command = self.scope_resolver(command, dependencies.budget)
        dependencies = _dependencies_with_filing_scope(dependencies, command)
        result = run_research(
            command.ticker.strip().upper(),
            command.request,
            dependencies,
            intent=decision.intent,
            run_id=run_id,
            corpus_version=command.corpus_version,
            filing_ids=command.filing_ids,
            scope_error=command.scope_error,
            session_id=command.session_id,
            report_as_of=command.as_of_date,
        )
        rendered_output = self.apply_execution_note(
            result.rendered_output,
            is_p1=decision.intent
            in {
                Intent.COMPANY_PROFILE_REQUEST,
                Intent.EARNINGS_REVIEW_REQUEST,
                Intent.INDUSTRY_RESEARCH_REQUEST,
            },
            has_guarded_memo=result.guarded_memo is not None,
        )
        return result.model_copy(
            update={
                "rendered_output": rendered_output,
                "source_policy_version": self.source_policy_version,
            }
        )


def _dependencies_with_filing_scope(
    dependencies: Dependencies,
    command: ResearchCommand,
) -> Dependencies:
    updates: dict[str, object] = {}
    for field_name in ("skill_collector", "thesis_collector"):
        collector = getattr(dependencies, field_name)
        with_scope = getattr(collector, "with_filing_scope", None)
        if callable(with_scope):
            updates[field_name] = with_scope(
                command.corpus_version,
                command.filing_ids,
                scope_error=command.scope_error,
            )
    return replace(dependencies, **updates) if updates else dependencies


def _resolve_research_scope(
    client,
    command: ResearchCommand,
    *,
    budget_gate: BudgetGate | None = None,
) -> ResearchCommand:
    arguments: dict[str, object] = {
        "ticker": command.ticker,
        "forms": list(command.forms),
        "limit": 4,
    }
    if command.as_of_date is not None:
        arguments["as_of_date"] = command.as_of_date.isoformat()
    try:
        if budget_gate is not None:
            budget_gate.consume(tool_calls=1)
        raw_response = client.call_tool("fetch_recent_filings", arguments)
    except BudgetExhaustedError as error:
        return _scope_resolution_failure(
            command,
            f"BUDGET_EXHAUSTED: {error.dimension}",
        )
    except Exception:
        return _scope_resolution_failure(
            command,
            "MCP_CALL_ERROR: filing scope resolution failed",
        )
    if getattr(raw_response, "is_error", True):
        return _scope_resolution_failure(
            command,
            "MCP_PROTOCOL_ERROR: filing scope tool returned is_error",
        )
    try:
        response = FetchRecentFilingsResponse.model_validate(
            getattr(raw_response, "structured_content", None)
        )
    except Exception:
        return _scope_resolution_failure(
            command,
            "MCP_RESPONSE_ERROR: invalid filing scope response",
        )
    if response.error is not None:
        if response.error.code == "NO_FILINGS":
            return _scope_resolution_failure(
                command,
                "NO_FILINGS: no filings matched the requested scope",
            )
        return _scope_resolution_failure(
            command,
            f"{response.error.code}: filing scope tool rejected the request",
        )
    if (
        response.corpus_version is None
        or not response.corpus_version.strip()
        or not response.filings
    ):
        if response.corpus_version is None and not response.filings:
            return _scope_resolution_failure(
                command,
                "NO_FILINGS: no filings matched the requested scope",
            )
        return _scope_resolution_failure(
            command,
            "MCP_RESPONSE_ERROR: invalid filing scope response",
        )
    filing_ids: list[str] = []
    for filing in response.filings:
        try:
            filed_at = date.fromisoformat(filing.filed_at)
        except ValueError:
            return _scope_resolution_failure(
                command,
                "MCP_RESPONSE_ERROR: invalid filing scope response",
            )
        if (
            filing.ticker != command.ticker
            or filing.form not in command.forms
            or not filing.corpus_version.strip()
            or filing.corpus_version != response.corpus_version
            or (command.as_of_date is not None and filed_at > command.as_of_date)
        ):
            return _scope_resolution_failure(
                command,
                "MCP_RESPONSE_ERROR: invalid filing scope response",
            )
        filing_ids.append(filing.id)
    if not filing_ids or len(filing_ids) != len(set(filing_ids)):
        return _scope_resolution_failure(
            command,
            "MCP_RESPONSE_ERROR: invalid filing scope response",
        )
    return command.model_copy(
        update={
            "corpus_version": response.corpus_version,
            "filing_ids": tuple(filing_ids),
            "scope_error": None,
        }
    )


def _scope_resolution_failure(command: ResearchCommand, detail: str) -> ResearchCommand:
    return command.model_copy(
        update={
            "corpus_version": None,
            "filing_ids": (),
            "scope_error": detail,
        }
    )


@dataclass(frozen=True, slots=True)
class MarketResearchRuntime:
    """One application-run market runtime with no model or research planner."""

    ticker: str
    dependencies: MarketDependencies

    def execute(
        self,
        command: ResearchCommand,
        decision: RouterDecision,
    ) -> MarketResearchResult:
        if decision.intent is not Intent.MARKET_SNAPSHOT_REQUEST:
            raise ValueError("market runtime requires market_snapshot_request")
        if command.ticker.strip().upper() != self.ticker:
            raise ValueError("market runtime ticker does not match command")
        from financial_evidence_agent.graph.market_workflow import run_market_workflow

        return run_market_workflow(command, self.dependencies)


class _ConfiguredIntentRouter:
    """Validate and construct the structured router only when AUTO needs it."""

    def __init__(
        self,
        settings: Settings,
        router_factory: Callable[..., IntentRouter] = OpenAIIntentRouter,
    ) -> None:
        self._settings = settings
        self._router_factory = router_factory
        self._router: IntentRouter | None = None

    def route(self, request: str) -> RouterDecision:
        if self._router is None:
            missing: list[str] = []
            if self._settings.openai_api_key is None:
                missing.append("OPENAI_API_KEY")
            if self._settings.fast_model is None:
                missing.append("FAST_MODEL")
            if missing:
                raise BootstrapConfigurationError(
                    BootstrapErrorCode.FAST_MODEL_CONFIGURATION_MISSING,
                    "AUTO routing requires configured model credentials: " + ", ".join(missing),
                )
            self._router = self._router_factory(
                model=self._settings.fast_model,
                api_key=self._settings.openai_api_key.get_secret_value(),
                token_cache_dir=self._settings.tokenizer_cache_dir,
            )
        return self._router.route(request)


class _ResearchRuntimeFactory:
    """Select P0 or P1 composition after the application resolves intent."""

    def __init__(
        self,
        settings: Settings,
        *,
        p0_builder: Callable[..., ResearchRuntime],
        p1_builder: Callable[..., ResearchRuntime],
        industry_builder: Callable[..., ResearchRuntime],
        market_builder: Callable[..., MarketResearchRuntime],
        quality_builder: Callable[..., QualityResearchRuntime],
    ) -> None:
        self._settings = settings
        self._p0_builder = p0_builder
        self._p1_builder = p1_builder
        self._industry_builder = industry_builder
        self._market_builder = market_builder
        self._quality_builder = quality_builder

    def build(
        self, command: ResearchCommand, intent: Intent
    ) -> ResearchRuntime | MarketResearchRuntime | QualityResearchRuntime:
        if intent is Intent.MARKET_SNAPSHOT_REQUEST:
            return self._market_builder(self._settings, ticker=command.ticker)
        if intent is Intent.INDUSTRY_RESEARCH_REQUEST:
            return self._industry_builder(self._settings, ticker=command.ticker)
        if intent is Intent.RESEARCH_QUALITY_SCREEN_REQUEST:
            return self._quality_builder(self._settings, ticker=command.ticker)
        if intent in {Intent.COMPANY_PROFILE_REQUEST, Intent.EARNINGS_REVIEW_REQUEST}:
            return self._p1_builder(self._settings, ticker=command.ticker)
        return self._p0_builder(self._settings, ticker=command.ticker)


class _LazyAllowlistedWebGateway:
    """Delay optional provider and HTTP client construction until fallback is used."""

    def __init__(
        self,
        *,
        api_key: str,
        repository: WebEvidenceRepository,
        cache: CompositeJsonCache,
        timeout_seconds: float,
        source_policy: SourcePolicy,
        provider_factory: Any = TavilySearchProvider,
        redirect_resolver_factory: Any = HttpxRedirectResolver,
        gateway_factory: Any = AllowlistedWebGateway,
    ) -> None:
        self._api_key = api_key
        self._repository = repository
        self._cache = cache
        self._timeout_seconds = timeout_seconds
        self._source_policy = source_policy
        self._provider_factory = provider_factory
        self._redirect_resolver_factory = redirect_resolver_factory
        self._gateway_factory = gateway_factory
        self._gateway: AllowlistedWebGateway | None = None

    async def search(self, *, ticker: str, query: str, max_results: int = 3):
        if self._gateway is None:
            self._gateway = self._gateway_factory(
                provider=self._provider_factory(self._api_key),
                redirect_resolver=self._redirect_resolver_factory(
                    timeout_seconds=self._timeout_seconds
                ),
                source_policy=self._source_policy,
                repository=self._repository,
                timeout_seconds=self._timeout_seconds,
                cache=self._cache,
                source_policy_version=self._source_policy.version,
            )
        return await self._gateway.search(
            ticker=ticker,
            query=query,
            max_results=max_results,
        )


def build_production_retriever(
    settings: Settings,
    repository: FilingRepository,
    embedding_provider: BgeM3EmbeddingProvider,
    cache: CompositeJsonCache,
    *,
    ranker_factory: Callable[..., object] | None = None,
    asset_validator: Callable[[str, str | None], None] | None = None,
    metrics_sink: Callable[[RetrievalMetrics], None] | None = None,
    retriever_factory: Callable[..., HybridRetriever] = HybridRetriever,
) -> HybridRetriever:
    """Compose the lazy FlashRank retriever used only by live runtime entry points."""
    return retriever_factory(
        repository,
        embedding_provider,
        reranker=LazyFlashRankReranker(
            settings.reranker_model,
            cache_dir=settings.reranker_cache_dir,
            ranker_factory=ranker_factory,
            asset_validator=asset_validator,
        ),
        cache=cache,
        metrics_sink=metrics_sink,
    )


def build_p0_runtime(settings: Settings, *, ticker: str) -> ResearchRuntime:
    """Build the production thesis workflow unless offline demo mode is explicit."""
    if settings.offline_demo:
        return _build_offline_p0_runtime(settings, ticker=ticker)

    _validate_thesis_configuration(settings)
    if settings.tavily_api_key is not None:
        try:
            ensure_tavily_dependency()
        except Exception:
            raise BootstrapConfigurationError(
                BootstrapErrorCode.WEB_SEARCH_DEPENDENCY_UNAVAILABLE,
                "TAVILY_API_KEY is configured but the optional 'web-search' dependency "
                "is unavailable",
            ) from None
    engine = create_engine(settings.database_url)
    _initialize_schema(engine)
    repository = FilingRepository(engine)
    normalized_ticker = ticker.strip().upper()
    corpus_version = repository.latest_corpus_version(normalized_ticker)
    if corpus_version is None:
        raise UnsupportedTickerError(normalized_ticker)

    source_policy = build_standard_source_policy(issuer_domains=settings.web_issuer_domains)
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    if engine.dialect.name == "postgresql":
        EmbeddingIndexer(repository, embedding_provider).ensure_indexed(
            normalized_ticker,
            corpus_version,
        )
    cache = build_cache(settings)
    if isinstance(cache, NoopJsonCache):
        raise BootstrapConfigurationError(
            BootstrapErrorCode.THESIS_CONFIGURATION_MISSING,
            "production thesis research requires a valid REDIS_URL",
        )
    retriever = build_production_retriever(
        settings,
        repository,
        embedding_provider,
        cache,
        metrics_sink=_trace_retrieval_metrics,
    )
    web_repository = WebEvidenceRepository(engine)
    web_validator = PersistedWebEvidenceValidator(source_policy, web_repository)
    gateway = (
        _LazyAllowlistedWebGateway(
            api_key=settings.tavily_api_key.get_secret_value(),
            repository=web_repository,
            cache=cache,
            timeout_seconds=settings.web_search_timeout_seconds,
            source_policy=source_policy,
        )
        if settings.tavily_api_key is not None
        else None
    )
    server = create_server(
        repository,
        retriever,
        web_gateway=gateway,  # type: ignore[arg-type]
        web_repository=web_repository,
        source_policy=source_policy,
    )
    mcp_client = FastMCPToolClient(server)
    noop_writer = NoopResearchRunWriter()
    web_search = (
        MCPWebSearch(mcp_client, source_fetch_writer=noop_writer, run_id="untracked")
        if gateway is not None
        else None
    )
    collector = EvidenceCollector(
        local_search=MCPFilingSearch(
            mcp_client,
            source_fetch_writer=noop_writer,
            run_id="untracked",
        ),
        web_search=web_search,
    )
    api_key = settings.openai_api_key.get_secret_value()
    context_compressor = LazyLLMLinguaCompressor(
        model_name=settings.context_compressor_model,
        model_cache_dir=settings.context_compressor_cache_dir,
        tokenizer_cache_dir=settings.tokenizer_cache_dir,
    )
    assert settings.fast_model is not None
    assert settings.analyst_model is not None
    thesis_analyst = OpenAIThesisAnalystModel(
        model=settings.analyst_model,
        api_key=api_key,
        context_compressor=context_compressor,
        token_cache_dir=settings.tokenizer_cache_dir,
    )
    dependencies = Dependencies(
        mcp_client=mcp_client,
        fast_model=OpenAIThesisFastModel(
            model=settings.fast_model,
            api_key=api_key,
            context_compressor=context_compressor,
            token_cache_dir=settings.tokenizer_cache_dir,
        ),
        analyst_model=thesis_analyst,
        thesis_repair_model=thesis_analyst,
        trace_sink=NoopTraceSink(),
        web_evidence_validator=web_validator,
        thesis_collector=collector,
        thesis_collection_policy=THESIS_COLLECTION_POLICY,
    )

    def bind_run(context: ResearchExecutionContext) -> Dependencies:
        run_web_search = (
            MCPWebSearch(
                mcp_client,
                source_fetch_writer=context.run_repository,
                run_id=context.run_id,
            )
            if gateway is not None
            else None
        )
        return replace(
            dependencies,
            trace_run=context.trace,
            thesis_collector=EvidenceCollector(
                local_search=MCPFilingSearch(
                    mcp_client,
                    source_fetch_writer=context.run_repository,
                    run_id=context.run_id,
                ),
                web_search=run_web_search,
            ),
        )

    return ResearchRuntime(
        dependencies=dependencies,
        repository=repository,
        cache=cache,
        web_search=web_search,
        execution_note=(
            None
            if web_search is not None
            else "production thesis (allowlisted web fallback unavailable; local evidence only)"
        ),
        run_context_factory=bind_run,
        scope_resolver=lambda command, budget: _resolve_research_scope(
            mcp_client,
            command,
            budget_gate=budget,
        ),
        source_policy_version=source_policy.version,
    )


def _build_offline_p0_runtime(settings: Settings, *, ticker: str) -> ResearchRuntime:
    """Build the explicitly selected deterministic thesis demo."""
    engine = create_engine(settings.database_url)
    _initialize_schema(engine)
    repository = FilingRepository(engine)
    cache = build_cache(settings)
    embedding_provider = HashEmbeddingProvider()
    normalized_ticker = ticker.strip().upper()
    existing_corpus = repository.latest_corpus_version(normalized_ticker)
    if existing_corpus is not None:
        EmbeddingIndexer(repository, embedding_provider).ensure_indexed(
            normalized_ticker,
            existing_corpus,
        )
    server = create_server(
        repository,
        HybridRetriever(repository, embedding_provider, cache=cache),
    )
    origin = (
        "pre-existing local corpus"
        if existing_corpus is not None
        else "no pre-existing local corpus"
    )
    dependencies = Dependencies(
        mcp_client=FastMCPToolClient(server),
        fast_model=DeterministicDemoFastModel(),
        analyst_model=DeterministicDemoAnalystModel(),
        trace_sink=NoopTraceSink(),
    )
    return ResearchRuntime(
        dependencies=dependencies,
        repository=repository,
        cache=cache,
        scope_resolver=lambda command, budget: _resolve_research_scope(
            dependencies.mcp_client,
            command,
            budget_gate=budget,
        ),
        execution_note=f"deterministic offline demo ({origin}; no model API calls).",
    )


def build_p1_runtime(
    settings: Settings,
    *,
    ticker: str,
    ranker_factory: Callable[..., object] | None = None,
    asset_validator: Callable[[str, str | None], None] | None = None,
    source_policy_factory: Any = build_standard_source_policy,
) -> ResearchRuntime:
    """Build live P1 dependencies after validating every required model setting."""
    _validate_p1_model_configuration(settings)
    engine = create_engine(settings.database_url)
    _initialize_schema(engine)
    repository = FilingRepository(engine)
    normalized_ticker = ticker.strip().upper()
    corpus_version = repository.latest_corpus_version(normalized_ticker)
    if corpus_version is None:
        raise UnsupportedTickerError(normalized_ticker)
    source_policy = source_policy_factory(issuer_domains=settings.web_issuer_domains)
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    if engine.dialect.name == "postgresql" and corpus_version is not None:
        EmbeddingIndexer(repository, embedding_provider).ensure_indexed(
            normalized_ticker,
            corpus_version,
        )

    cache = build_cache(settings)
    retriever = build_production_retriever(
        settings,
        repository,
        embedding_provider,
        cache,
        ranker_factory=ranker_factory,
        asset_validator=asset_validator,
        metrics_sink=_trace_retrieval_metrics,
    )
    web_repository = WebEvidenceRepository(engine)
    web_evidence_validator = PersistedWebEvidenceValidator(source_policy, web_repository)
    gateway: _LazyAllowlistedWebGateway | None = None
    web_search: WebSearch | None = None
    if settings.tavily_api_key is not None:
        gateway = _LazyAllowlistedWebGateway(
            api_key=settings.tavily_api_key.get_secret_value(),
            repository=web_repository,
            cache=cache,
            timeout_seconds=settings.web_search_timeout_seconds,
            source_policy=source_policy,
        )
    server = create_server(
        repository,
        retriever,
        web_gateway=gateway,  # type: ignore[arg-type]
        web_repository=web_repository,
        source_policy=source_policy,
    )
    mcp_client = FastMCPToolClient(server)
    noop_fetch_writer = NoopResearchRunWriter()
    if gateway is not None:
        web_search = MCPWebSearch(
            mcp_client,
            source_fetch_writer=noop_fetch_writer,
            run_id="untracked",
        )
    collector = EvidenceCollector(
        local_search=MCPFilingSearch(
            mcp_client,
            source_fetch_writer=noop_fetch_writer,
            run_id="untracked",
        ),
        web_search=web_search,
    )
    api_key = settings.openai_api_key.get_secret_value()
    context_compressor = LazyLLMLinguaCompressor(
        model_name=settings.context_compressor_model,
        model_cache_dir=settings.context_compressor_cache_dir,
        tokenizer_cache_dir=settings.tokenizer_cache_dir,
    )
    assert settings.fast_model is not None
    assert settings.analyst_model is not None
    skill_analyst = OpenAISkillAnalystModel(
        model=settings.analyst_model,
        api_key=api_key,
        context_compressor=context_compressor,
        token_cache_dir=settings.tokenizer_cache_dir,
    )
    dependencies = Dependencies(
        mcp_client=mcp_client,
        fast_model=DeterministicDemoFastModel(),
        analyst_model=DeterministicDemoAnalystModel(),
        trace_sink=NoopTraceSink(),
        skill_planner=OpenAISkillPlannerModel(
            model=settings.fast_model,
            api_key=api_key,
            context_compressor=context_compressor,
            token_cache_dir=settings.tokenizer_cache_dir,
        ),
        skill_collector=collector,
        skill_analyst=skill_analyst,
        skill_repair_model=skill_analyst,
        skill_run_repository=SkillRunRepository(engine),
        web_evidence_validator=web_evidence_validator,
    )

    def bind_run(context: ResearchExecutionContext) -> Dependencies:
        run_web_search = (
            MCPWebSearch(
                mcp_client,
                source_fetch_writer=context.run_repository,
                run_id=context.run_id,
            )
            if gateway is not None
            else None
        )
        return replace(
            dependencies,
            trace_run=context.trace,
            skill_collector=EvidenceCollector(
                local_search=MCPFilingSearch(
                    mcp_client,
                    source_fetch_writer=context.run_repository,
                    run_id=context.run_id,
                ),
                web_search=run_web_search,
            ),
        )

    return ResearchRuntime(
        dependencies=dependencies,
        repository=repository,
        cache=cache,
        web_search=web_search,
        execution_note=(
            None
            if web_search is not None
            else "allowlisted web fallback unavailable; local evidence only"
        ),
        run_context_factory=bind_run,
        scope_resolver=lambda command, budget: _resolve_research_scope(
            mcp_client,
            command,
            budget_gate=budget,
        ),
        source_policy_version=source_policy.version,
    )


def build_industry_runtime(
    settings: Settings,
    *,
    ticker: str,
    ranker_factory: Callable[..., object] | None = None,
) -> ResearchRuntime:
    """Build the single-ticker industry runtime with its expanded authority profile."""
    return build_p1_runtime(
        settings,
        ticker=ticker,
        ranker_factory=ranker_factory,
        source_policy_factory=lambda *, issuer_domains: build_industry_source_policy(
            issuer_domains=issuer_domains,
            industry_authority_domains=settings.web_industry_authority_domains,
        ),
    )


def build_quality_runtime(
    settings: Settings,
    *,
    ticker: str,
    ranker_factory: Callable[..., object] | None = None,
) -> QualityResearchRuntime:
    """Build the deterministic quality-screen runtime over one company subrun."""

    base_runtime = build_p1_runtime(
        settings,
        ticker=ticker,
        ranker_factory=ranker_factory,
    )
    return QualityResearchRuntime(
        dependencies=base_runtime.dependencies,
        current_date_factory=_utc_today,
        max_source_age_days=settings.research_quality_max_source_age_days,
        run_context_factory=base_runtime.run_context_factory,
        scope_resolver=base_runtime.scope_resolver,
        source_policy_version=base_runtime.source_policy_version,
    )


def _utc_today() -> date:
    return datetime.now(UTC).date()


def build_research_application(
    settings: Settings,
    *,
    p0_builder: Callable[..., ResearchRuntime] | None = None,
    p1_builder: Callable[..., ResearchRuntime] | None = None,
    industry_builder: Callable[..., ResearchRuntime] | None = None,
    market_builder: Callable[..., MarketResearchRuntime] | None = None,
    quality_builder: Callable[..., QualityResearchRuntime] | None = None,
    intent_router_factory: Callable[..., IntentRouter] = OpenAIIntentRouter,
) -> ResearchApplication:
    """Compose lazy routing and runtime selection behind the application boundary."""
    run_repository = _LazyResearchRunRepository(settings)
    return ResearchApplication(
        _ConfiguredIntentRouter(settings, intent_router_factory),
        _ResearchRuntimeFactory(
            settings,
            p0_builder=p0_builder or build_p0_runtime,
            p1_builder=p1_builder or build_p1_runtime,
            industry_builder=industry_builder or build_industry_runtime,
            market_builder=market_builder or build_market_runtime,
            quality_builder=quality_builder or build_quality_runtime,
        ),
        run_repository=run_repository,
        trace_sink_factory=lambda: build_trace_sink(settings),
        quality_date_factory=_utc_today,
        quality_max_source_age_days=settings.research_quality_max_source_age_days,
        session_memory_store=SessionMemoryStore(build_session_cache(settings)),
        research_memory_store=_LazyResearchMemoryStore(settings),
        skill_run_repository=_LazySkillRunRepository(run_repository),
        company_resolver=_LazyCompanyResolver(settings),
    )


def build_market_runtime(
    settings: Settings,
    *,
    ticker: str,
    engine_factory: Any = create_engine,
    cache_factory: Any = build_market_cache,
    provider_factory: Any = AlpacaMarketDataProvider,
    gateway_factory: Any = MarketDataGateway,
    http_client_factory: Any = httpx.AsyncClient,
) -> MarketResearchRuntime:
    """Build a fresh persisted market-only server and gateway for one application run."""
    if settings.alpaca_api_key_id is None or settings.alpaca_api_secret_key is None:
        raise MarketDataError(
            MarketDataErrorCode.CONFIGURATION_MISSING,
            "Alpaca key ID and secret key are required; tool_calls=0",
        )

    normalized_ticker = ticker.strip().upper()
    context = current_research_context()
    run_writer = context.run_repository if context is not None else NoopResearchRunWriter()
    run_id = context.run_id if context is not None else "untracked"
    engine: Engine = engine_factory(settings.database_url)
    if isinstance(engine, Engine):
        _initialize_schema(engine)
    cache = cache_factory(settings)
    if isinstance(cache, NoopJsonCache):
        cache = build_market_cache(settings)
    http_client = http_client_factory()
    fetch_writer = RunScopedMarketFetchWriter(run_writer, run_id=run_id)
    market_repository = MarketDataRepository(engine)
    gateway = build_market_gateway(
        settings,
        cache=cache,
        http_client=http_client,
        fetch_writer=fetch_writer,
        provider_factory=provider_factory,
        gateway_factory=gateway_factory,
    )
    server = FastMCP("financial-evidence-agent-market")
    source_policy = build_industry_source_policy(
        issuer_domains=settings.web_issuer_domains,
        industry_authority_domains=settings.web_industry_authority_domains,
    )
    web_repository = WebEvidenceRepository(engine)
    web_gateway = (
        _LazyAllowlistedWebGateway(
            api_key=settings.tavily_api_key.get_secret_value(),
            repository=web_repository,
            cache=cache,
            timeout_seconds=settings.web_search_timeout_seconds,
            source_policy=source_policy,
        )
        if settings.tavily_api_key is not None
        else None
    )
    register_market_tools(
        server,
        lambda: gateway,
        market_repository,
        max_bars=settings.market_data_max_bars,
    )
    register_web_tools(server, web_gateway, web_repository, source_policy)
    return MarketResearchRuntime(
        ticker=normalized_ticker,
        dependencies=MarketDependencies(
            mcp_client=FastMCPToolClient(server),
            max_bars=settings.market_data_max_bars,
            max_staleness_seconds=settings.market_data_max_staleness_seconds,
            context_window_days=settings.market_context_window_days,
            abnormal_move_threshold=settings.market_abnormal_move_threshold,
            web_evidence_validator=PersistedWebEvidenceValidator(source_policy, web_repository),
            bundle_writer=market_repository,
            session_factory=lambda: FastMCPToolSession(
                server,
                resources=(http_client, cache),
            ),
        ),
    )


def build_eval_runtime(settings: Settings | None = None) -> ResearchRuntime:
    """Build the in-memory deterministic evaluation runtime."""
    del settings
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = FilingRepository(engine)
    embedding_provider = HashEmbeddingProvider()
    with bundled_nvda_fixture_path() as fixture_path:
        ingest_fixture(
            fixture_path,
            DEMO_TICKER,
            "10-Q",
            repository,
            embedding_provider=embedding_provider,
        )
    server = create_server(
        repository,
        HybridRetriever(repository, embedding_provider),
    )
    return ResearchRuntime(
        dependencies=Dependencies(
            mcp_client=FastMCPToolClient(server),
            fast_model=DeterministicDemoFastModel(),
            analyst_model=DeterministicDemoAnalystModel(),
            trace_sink=NoopTraceSink(),
        ),
        repository=repository,
        execution_note="deterministic offline evaluation",
    )


def build_filing_repository(settings: Settings) -> FilingRepository:
    """Build the schema-backed filing repository used by the ingest command."""
    engine = create_engine(settings.database_url)
    _initialize_schema(engine)
    return FilingRepository(engine)


def build_market_gateway(
    settings: Settings,
    *,
    cache: CompositeJsonCache,
    http_client: httpx.AsyncClient,
    fetch_writer: MarketFetchWriter,
    provider_factory: Any = AlpacaMarketDataProvider,
    gateway_factory: Any = MarketDataGateway,
) -> MarketDataGateway:
    """Build one fresh run-scoped gateway with concrete cache/fetch boundaries."""
    if settings.alpaca_api_key_id is None or settings.alpaca_api_secret_key is None:
        raise MarketDataError(
            MarketDataErrorCode.CONFIGURATION_MISSING,
            "Alpaca market-data credentials are required",
        )
    provider = provider_factory(
        key_id=settings.alpaca_api_key_id.get_secret_value(),
        secret_key=settings.alpaca_api_secret_key.get_secret_value(),
        client=http_client,
        trading_environment=settings.alpaca_trading_environment,
        timeout_seconds=settings.market_data_timeout_seconds,
        fetch_writer=fetch_writer,
    )
    return gateway_factory(
        provider,
        max_bars=settings.market_data_max_bars,
        max_staleness_seconds=settings.market_data_max_staleness_seconds,
        cache=MarketDataJsonCache(
            cache,
            max_bars=settings.market_data_max_bars,
            max_staleness_seconds=settings.market_data_max_staleness_seconds,
        ),
    )


def build_mcp_server(
    settings: Settings,
    *,
    engine_factory: Any = create_engine,
    repository_factory: Any = FilingRepository,
    web_repository_factory: Any = WebEvidenceRepository,
    cache_factory: Any = build_cache,
    embedding_provider_factory: Any = BgeM3EmbeddingProvider,
    retriever_factory: Any = HybridRetriever,
    provider_factory: Any = TavilySearchProvider,
    redirect_resolver_factory: Any = HttpxRedirectResolver,
    source_policy_factory: Any = SourcePolicy,
    gateway_factory: Any = AllowlistedWebGateway,
    market_provider_factory: Any = AlpacaMarketDataProvider,
    market_gateway_factory: Any = MarketDataGateway,
    market_http_client_factory: Any = httpx.AsyncClient,
    server_factory: Any = create_server,
) -> FastMCP:
    """Compose the stdio server with web tools registered even when web is disabled."""
    engine: Engine = engine_factory(settings.database_url)
    if isinstance(engine, Engine):
        _initialize_schema(engine)
    repository = repository_factory(engine)
    web_repository = web_repository_factory(engine)
    cache = cache_factory(settings)
    retriever = build_production_retriever(
        settings,
        repository,
        embedding_provider_factory(
            settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        ),
        cache,
        retriever_factory=retriever_factory,
    )
    source_policy = source_policy_factory(issuer_domains=settings.web_issuer_domains)
    web_gateway = None
    if settings.tavily_api_key is not None:
        web_gateway = _LazyAllowlistedWebGateway(
            api_key=settings.tavily_api_key.get_secret_value(),
            repository=web_repository,
            cache=cache,
            timeout_seconds=settings.web_search_timeout_seconds,
            source_policy=source_policy,
            provider_factory=provider_factory,
            redirect_resolver_factory=redirect_resolver_factory,
            gateway_factory=gateway_factory,
        )
    market_gateway_builder = None
    if settings.alpaca_api_key_id is not None and settings.alpaca_api_secret_key is not None:
        market_http_client = market_http_client_factory()
        market_fetch_writer = NoopMarketFetchWriter()

        def market_gateway_builder():
            return build_market_gateway(
                settings,
                cache=cache,
                http_client=market_http_client,
                fetch_writer=market_fetch_writer,
                provider_factory=market_provider_factory,
                gateway_factory=market_gateway_factory,
            )

    return server_factory(
        repository,
        retriever,
        web_gateway=web_gateway,
        web_repository=web_repository,
        market_gateway_factory=market_gateway_builder,
        market_repository=MarketDataRepository(engine),
        market_data_max_bars=settings.market_data_max_bars,
        source_policy=source_policy,
    )


def mcp_main() -> None:
    """Start the centrally composed read-only MCP stdio server."""
    build_mcp_server(Settings()).run(transport="stdio")


def _validate_p1_model_configuration(settings: Settings) -> None:
    missing: list[str] = []
    if settings.openai_api_key is None:
        missing.append("OPENAI_API_KEY")
    if settings.fast_model is None:
        missing.append("FAST_MODEL")
    if settings.analyst_model is None:
        missing.append("ANALYST_MODEL")
    if missing:
        raise BootstrapConfigurationError(
            BootstrapErrorCode.P1_MODEL_CONFIGURATION_MISSING,
            "P1 requires configured model credentials: " + ", ".join(missing),
        )


def _validate_thesis_configuration(settings: Settings) -> None:
    missing: list[str] = []
    if settings.openai_api_key is None:
        missing.append("OPENAI_API_KEY")
    if settings.fast_model is None:
        missing.append("FAST_MODEL")
    if settings.analyst_model is None:
        missing.append("ANALYST_MODEL")
    if not settings.redis_url:
        missing.append("REDIS_URL")
    if missing:
        raise BootstrapConfigurationError(
            BootstrapErrorCode.THESIS_CONFIGURATION_MISSING,
            "production thesis research requires configured dependencies: "
            + ", ".join(missing),
        )


def _render_with_runtime_note(
    rendered_output: str,
    runtime: ResearchRuntime,
    *,
    is_p1: bool,
    has_guarded_memo: bool,
) -> str:
    if runtime.execution_note is None:
        return rendered_output
    if is_p1:
        return f"> Information gap: {runtime.execution_note}.\n\n{rendered_output}"
    mode_note = f"> Execution mode: {runtime.execution_note}"
    if has_guarded_memo:
        return rendered_output.replace("\n\n", f"\n\n{mode_note}\n", 1)
    return f"{mode_note}\n\n{rendered_output}"


def _trace_retrieval_metrics(metrics: RetrievalMetrics) -> None:
    metadata = {
        "sparse_candidates": metrics.sparse_candidates,
        "dense_candidates": metrics.dense_candidates,
        "fused_candidates": metrics.fused_candidates,
        "fused_ids": list(metrics.fused_ids),
        "reranker_version": metrics.reranker_version,
        "retained_ids": list(metrics.retained_ids),
        "retained_scores": [list(value) for value in metrics.retained_scores],
        "cache_hit": metrics.cache_hit,
    }
    with observe(name="retrieval.hybrid", kind="retriever", metadata=metadata):
        pass
