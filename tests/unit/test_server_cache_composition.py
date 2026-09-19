from pydantic import SecretStr

from fra.config import Settings
from fra.mcp_server import server as server_module
from fra.storage.cache import NoopJsonCache, RedisJsonCache


def _run_main(
    monkeypatch,
    *,
    redis_url: str | None,
    provider_must_stay_lazy: bool = False,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        redis_url=redis_url,
        tavily_api_key=SecretStr("test-key"),
        _env_file=None,
    )

    class CapturingGateway:
        def __init__(self, *, cache, **kwargs) -> None:
            captured["gateway_cache"] = cache

    class CapturingRetriever:
        def __init__(
            self,
            repository,
            embedding_provider,
            *,
            cache,
            reranker,
            metrics_sink,
        ) -> None:
            del repository, embedding_provider, metrics_sink
            captured["retriever_cache"] = cache
            captured["reranker_version"] = reranker.version

    class FakeServer:
        def run(self, *, transport: str) -> None:
            captured["transport"] = transport

    monkeypatch.setattr(server_module, "Settings", lambda: settings)
    monkeypatch.setattr(server_module, "create_engine", lambda url: object())
    def provider(api_key: str) -> object:
        del api_key
        if provider_must_stay_lazy:
            raise AssertionError("optional web provider was constructed during startup")
        return object()

    def create_server(*args, **kwargs):
        del args
        captured["web_gateway"] = kwargs.get("web_gateway")
        return FakeServer()

    monkeypatch.setattr(server_module, "TavilySearchProvider", provider)
    monkeypatch.setattr(server_module, "AllowlistedWebGateway", CapturingGateway)
    monkeypatch.setattr(server_module, "HybridRetriever", CapturingRetriever)
    monkeypatch.setattr(server_module, "create_server", create_server)

    server_module.main()
    return captured


def test_main_injects_one_configured_cache_into_live_consumers(monkeypatch) -> None:
    """A configured Redis URL must reach both live cache consumers as one adapter."""
    captured = _run_main(monkeypatch, redis_url="redis://cache.example:6379/0")

    cache = captured["retriever_cache"]
    assert isinstance(cache, RedisJsonCache)
    assert getattr(captured["web_gateway"], "_cache") is cache
    assert captured["transport"] == "stdio"
    assert captured["reranker_version"] == (
        "flashrank-risk-v2:ms-marco-MiniLM-L-12-v2"
    )


def test_main_injects_one_noop_cache_when_redis_is_disabled(monkeypatch) -> None:
    """Disabled Redis must compose both consumers with the same deterministic no-op."""
    captured = _run_main(monkeypatch, redis_url=None)

    cache = captured["retriever_cache"]
    assert isinstance(cache, NoopJsonCache)
    assert getattr(captured["web_gateway"], "_cache") is cache


def test_compatibility_main_does_not_construct_optional_web_provider(monkeypatch) -> None:
    captured = _run_main(
        monkeypatch,
        redis_url=None,
        provider_must_stay_lazy=True,
    )

    assert captured["web_gateway"] is not None
    assert captured["transport"] == "stdio"
