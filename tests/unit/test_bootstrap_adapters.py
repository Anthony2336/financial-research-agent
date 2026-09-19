"""Production bootstrap adapter tests."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Lock
from types import SimpleNamespace

import fra.bootstrap as bootstrap_module
from fra.bootstrap import build_production_retriever
from fra.config import Settings
from fra.domain import SourceRef, SourceRefKind
from fra.memory.research import ResearchMemoryKind
from fra.storage.cache import NoopJsonCache


def test_production_retriever_uses_configured_lazy_flashrank_reranker() -> None:
    """Replacing the live adapter with identity reranking would make P1 ranking a no-op."""

    class _EmbeddingProvider:
        version = "fake-embedding-v1"
        dimensions = 1

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    retriever = build_production_retriever(
        Settings(reranker_model="test-flashrank", _env_file=None),
        repository=object(),
        embedding_provider=_EmbeddingProvider(),
        cache=NoopJsonCache(),
        ranker_factory=lambda *, model_name, cache_dir: object(),
    )

    assert retriever.reranker_version == "flashrank-risk-v2:test-flashrank"


def test_lazy_research_memory_store_binds_explicit_current_web_policy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    engine = object()
    validator = object()

    class _Service:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            captured["search"] = (args, kwargs)
            return []

    def repository_factory(bound_engine, *, web_evidence_validator=None):
        captured["engine"] = bound_engine
        captured["validator"] = web_evidence_validator
        return object()

    def service_factory(repository, provider, *, ttl_days):
        captured["repository"] = repository
        captured["provider"] = provider
        captured["ttl_days"] = ttl_days
        return _Service()

    monkeypatch.setattr(bootstrap_module, "create_engine", lambda url: engine)
    monkeypatch.setattr(bootstrap_module, "_initialize_schema", lambda value: None)
    monkeypatch.setattr(
        bootstrap_module,
        "ResearchMemoryRepository",
        repository_factory,
    )
    monkeypatch.setattr(bootstrap_module, "ResearchMemoryService", service_factory)
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        offline_demo=True,
        research_memory_ttl_days=45,
        _env_file=None,
    )

    scoped = bootstrap_module._LazyResearchMemoryStore(
        settings
    ).with_web_evidence_validator(validator)
    assert scoped.search("NVDA", "query") == []

    assert captured["engine"] is engine
    assert captured["validator"] is validator
    assert captured["ttl_days"] == 45


def test_scoped_lazy_memory_store_reuses_thread_safe_resources_by_policy(
    monkeypatch,
) -> None:
    counters = {"engine": 0, "provider": 0, "repository": 0, "service": 0}
    counter_lock = Lock()
    services_by_policy: dict[str, list[object]] = {}

    @dataclass(frozen=True)
    class Validator:
        policy_version: str

    class Service:
        def __init__(self, policy_key: str) -> None:
            self.policy_key = policy_key
            self.calls: list[str] = []

        def search(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            self.calls.append("search")
            return []

        def store_guarded(self, **values: object) -> object:
            del values
            self.calls.append("store")
            return object()

    engine = object()
    provider = object()

    def engine_factory(url: str) -> object:
        del url
        with counter_lock:
            counters["engine"] += 1
        return engine

    def provider_factory(model_name: str, *, cache_dir=None) -> object:
        del model_name, cache_dir
        time.sleep(0.02)
        with counter_lock:
            counters["provider"] += 1
        return provider

    def repository_factory(bound_engine, *, web_evidence_validator=None):
        assert bound_engine is engine
        policy_key = (
            "filing-only"
            if web_evidence_validator is None
            else web_evidence_validator.policy_version
        )
        with counter_lock:
            counters["repository"] += 1
        return policy_key

    def service_factory(repository, bound_provider, *, ttl_days):
        assert bound_provider is provider
        assert ttl_days == 90
        service = Service(repository)
        with counter_lock:
            counters["service"] += 1
            services_by_policy.setdefault(repository, []).append(service)
        return service

    monkeypatch.setattr(bootstrap_module, "create_engine", engine_factory)
    monkeypatch.setattr(bootstrap_module, "_initialize_schema", lambda value: None)
    monkeypatch.setattr(bootstrap_module, "BgeM3EmbeddingProvider", provider_factory)
    monkeypatch.setattr(bootstrap_module, "ResearchMemoryRepository", repository_factory)
    monkeypatch.setattr(bootstrap_module, "ResearchMemoryService", service_factory)
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        offline_demo=False,
        research_memory_ttl_days=90,
        _env_file=None,
    )
    root = bootstrap_module._LazyResearchMemoryStore(settings)
    standard_a = root.with_web_evidence_validator(Validator("standard-v1"))
    standard_b = root.with_web_evidence_validator(Validator("standard-v1"))
    industry = root.with_web_evidence_validator(Validator("industry-v1"))

    assert standard_a.search("NVDA", "query") == []
    standard_b.store_guarded(
        ticker="NVDA",
        memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
        summary="Guarded issuer disclosure.",
        source_run_id="run-1",
        evidence_source_refs=(
            SourceRef(
                ticker="NVDA",
                kind=SourceRefKind.FILING,
                source_id="chunk-1",
            ),
        ),
        corpus_version="NVDA-v1",
    )
    assert industry.search("NVDA", "query") == []
    assert root.search("NVDA", "query") == []
    assert root.with_web_evidence_validator(None).search("NVDA", "query") == []

    concurrent = root.with_web_evidence_validator(Validator("concurrent-v1"))
    start = Barrier(8)

    def search(_: int) -> list[object]:
        start.wait(timeout=5)
        return concurrent.search("NVDA", "query")

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(search, range(8))) == [[]] * 8

    assert counters == {
        "engine": 1,
        "provider": 1,
        "repository": 4,
        "service": 4,
    }
    assert len(services_by_policy["standard-v1"]) == 1
    assert services_by_policy["standard-v1"][0].calls == ["search", "store"]
    assert len(services_by_policy["industry-v1"]) == 1
    assert len(services_by_policy["filing-only"]) == 1
    assert len(services_by_policy["concurrent-v1"]) == 1


def test_policy_scoped_services_share_one_serialized_bge_provider(
    monkeypatch,
) -> None:
    provider_constructions = 0
    model_factory_calls = 0
    encode_calls = 0
    active_encodes = 0
    max_active_encodes = 0
    engine_calls = 0
    repositories: list[object] = []
    observed_embeddings: list[list[float]] = []
    counters_lock = Lock()
    real_provider = bootstrap_module.BgeM3EmbeddingProvider

    @dataclass(frozen=True)
    class Validator:
        policy_version: str

    class DetectingSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal model_factory_calls
            del args, kwargs
            time.sleep(0.02)
            with counters_lock:
                model_factory_calls += 1

        def encode(self, texts: list[str], *, normalize_embeddings: bool):
            nonlocal active_encodes, encode_calls, max_active_encodes
            assert normalize_embeddings is True
            with counters_lock:
                active_encodes += 1
                encode_calls += 1
                max_active_encodes = max(max_active_encodes, active_encodes)
            try:
                time.sleep(0.01)
                return [[float(len(text)), *([0.0] * 1023)] for text in texts]
            finally:
                with counters_lock:
                    active_encodes -= 1

    class Repository:
        def __init__(self, validator: object | None) -> None:
            self.validator = validator

        def search(
            self,
            ticker: str,
            query_embedding: list[float],
            limit: int = 3,
            *,
            embedding_model: str,
            current_corpus_version: str | None = None,
        ) -> list[object]:
            del ticker, limit, embedding_model, current_corpus_version
            observed_embeddings.append(query_embedding)
            return []

        def store_guarded(self, **values: object) -> object:
            observed_embeddings.append(values["embedding"])  # type: ignore[arg-type]
            return values

    def engine_factory(url: str) -> object:
        nonlocal engine_calls
        del url
        engine_calls += 1
        return object()

    def provider_factory(model_name: str, *, cache_dir=None):
        nonlocal provider_constructions
        provider_constructions += 1
        return real_provider(model_name, cache_dir=cache_dir)

    def repository_factory(engine: object, *, web_evidence_validator=None):
        del engine
        repository = Repository(web_evidence_validator)
        repositories.append(repository)
        return repository

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=DetectingSentenceTransformer),
    )
    monkeypatch.setattr(bootstrap_module, "create_engine", engine_factory)
    monkeypatch.setattr(bootstrap_module, "_initialize_schema", lambda value: None)
    monkeypatch.setattr(bootstrap_module, "BgeM3EmbeddingProvider", provider_factory)
    monkeypatch.setattr(bootstrap_module, "ResearchMemoryRepository", repository_factory)
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        offline_demo=False,
        _env_file=None,
    )
    root = bootstrap_module._LazyResearchMemoryStore(settings)
    standard = root.with_web_evidence_validator(Validator("standard-v1"))
    industry = root.with_web_evidence_validator(Validator("industry-v1"))
    source_ref = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.FILING,
        source_id="chunk-1",
    )
    start = Barrier(8)

    def call(index: int) -> object:
        start.wait(timeout=5)
        if index % 2 == 0:
            return standard.search("NVDA", f"query-{index}")
        return industry.store_guarded(
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary=f"Issuer disclosure {index}.",
            source_run_id="run-1",
            evidence_source_refs=(source_ref,),
            corpus_version="NVDA-v1",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(8)))

    assert results[::2] == [[]] * 4
    assert all(isinstance(result, dict) for result in results[1::2])
    assert engine_calls == 1
    assert provider_constructions == 1
    assert len(repositories) == 2
    assert model_factory_calls == 1
    assert encode_calls == 8
    assert max_active_encodes == 1
    assert len(observed_embeddings) == 8
    assert all(len(vector) == 1024 for vector in observed_embeddings)
