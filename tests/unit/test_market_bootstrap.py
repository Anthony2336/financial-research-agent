"""Market gateway composition contracts."""

from decimal import Decimal

from pydantic import SecretStr

from fra.bootstrap import (
    build_market_gateway,
    build_market_runtime,
    build_mcp_server,
)
from fra.config import Settings
from fra.market_data.gateway import NoopMarketFetchWriter
from fra.storage.cache import (
    InMemoryTtlJsonCache,
    MarketDataJsonCache,
    NoopJsonCache,
)
from fra.storage.market_repositories import MarketDataRepository
from fra.web_evidence.source_policy import (
    DEFAULT_STANDARD_AUTHORITY_DOMAINS,
)


class CapturingProvider:
    instances: list["CapturingProvider"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class CapturingGateway:
    instances: list["CapturingGateway"] = []

    def __init__(self, provider, **kwargs) -> None:
        self.provider = provider
        self.kwargs = kwargs
        self.instances.append(self)


def _settings() -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        redis_url=None,
        alpaca_api_key_id=SecretStr("key-id"),
        alpaca_api_secret_key=SecretStr("secret-key"),
        market_data_max_bars=7,
        market_data_max_staleness_seconds=120,
        market_abnormal_move_threshold="0.075",
        _env_file=None,
    )


def test_build_market_gateway_injects_concrete_cache_and_explicit_fetch_writer() -> None:
    """Market application composition must provide a working cache."""
    CapturingProvider.instances.clear()
    CapturingGateway.instances.clear()
    settings = _settings()
    raw_cache = NoopJsonCache()
    writer = NoopMarketFetchWriter()
    http_client = object()

    gateway = build_market_gateway(
        settings,
        cache=raw_cache,
        http_client=http_client,
        fetch_writer=writer,
        provider_factory=CapturingProvider,
        gateway_factory=CapturingGateway,
    )

    assert gateway is CapturingGateway.instances[0]
    assert gateway.provider.kwargs["client"] is http_client
    assert gateway.provider.kwargs["fetch_writer"] is writer
    assert isinstance(gateway.kwargs["cache"], MarketDataJsonCache)
    assert gateway.kwargs["max_bars"] == 7
    assert gateway.kwargs["max_staleness_seconds"] == 120


def test_standalone_mcp_composes_fresh_market_gateway_factory_per_runtime_call() -> None:
    """The long-lived stdio server must not retain one gateway's run-local memo forever."""
    CapturingProvider.instances.clear()
    CapturingGateway.instances.clear()
    captured: dict[str, object] = {}
    raw_cache = NoopJsonCache()

    class Repository:
        def __init__(self, engine) -> None:
            self.engine = engine

    class Retriever:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    def server_factory(*args, **kwargs):
        del args
        captured.update(kwargs)
        return object()

    server = build_mcp_server(
        _settings(),
        engine_factory=lambda url: object(),
        repository_factory=Repository,
        web_repository_factory=lambda engine: object(),
        cache_factory=lambda settings: raw_cache,
        embedding_provider_factory=lambda *args, **kwargs: object(),
        retriever_factory=Retriever,
        market_provider_factory=CapturingProvider,
        market_gateway_factory=CapturingGateway,
        market_http_client_factory=lambda: object(),
        server_factory=server_factory,
    )

    factory = captured["market_gateway_factory"]
    first = factory()
    second = factory()
    assert server is not None
    assert first is not second
    assert len(CapturingGateway.instances) == 2
    assert isinstance(captured["market_repository"], MarketDataRepository)
    assert captured["market_data_max_bars"] == 7


def test_market_application_runtime_replaces_noop_cache_with_bounded_memory_cache() -> None:
    """The application route remains concretely cached without a Redis URL."""
    CapturingProvider.instances.clear()
    CapturingGateway.instances.clear()

    build_market_runtime(
        _settings(),
        ticker="NVDA",
        cache_factory=lambda settings: NoopJsonCache(),
        provider_factory=CapturingProvider,
        gateway_factory=CapturingGateway,
        http_client_factory=object,
    )

    cache = CapturingGateway.instances[0].kwargs["cache"]
    assert isinstance(cache, MarketDataJsonCache)
    assert isinstance(cache._cache, InMemoryTtlJsonCache)


def test_market_runtime_uses_standard_plus_configured_authority_domains_for_context() -> None:
    runtime = build_market_runtime(
        Settings(
            database_url="sqlite+pysqlite:///:memory:",
            alpaca_api_key_id=SecretStr("key-id"),
            alpaca_api_secret_key=SecretStr("secret-key"),
            web_issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})},
            web_industry_authority_domains=frozenset({"gao.gov"}),
            _env_file=None,
        ),
        ticker="NVDA",
        cache_factory=lambda settings: NoopJsonCache(),
        provider_factory=CapturingProvider,
        gateway_factory=CapturingGateway,
        http_client_factory=object,
    )

    policy = runtime.dependencies.web_evidence_validator.policy
    assert policy.authority_domains == (DEFAULT_STANDARD_AUTHORITY_DOMAINS | {"gao.gov"})
    assert runtime.dependencies.abnormal_move_threshold == Decimal("0.05")


def test_market_runtime_freezes_the_configured_exact_decimal_move_threshold() -> None:
    runtime = build_market_runtime(
        _settings(),
        ticker="NVDA",
        cache_factory=lambda settings: NoopJsonCache(),
        provider_factory=CapturingProvider,
        gateway_factory=CapturingGateway,
        http_client_factory=object,
    )

    assert runtime.dependencies.abnormal_move_threshold == Decimal("0.075")
