"""Unit tests for deterministic, offline hybrid retrieval."""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fra.domain import EvidenceChunk
from fra.retrieval.hybrid import (
    HashEmbeddingProvider,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from fra.retrieval.indexing import (
    BgeM3EmbeddingProvider,
    EmbeddingModelUnavailableError,
)
from fra.retrieval.rerank import RerankResult


def _chunk(
    chunk_id: str,
    content: str,
    *,
    ticker: str = "NVDA",
    corpus_version: str = "NVDA-v1",
) -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker=ticker,
        corpus_version=corpus_version,
        content=content,
        source_url="https://example.test/filing",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        accession_no="0001045810-25-000041",
        section="MD&A",
        raw_start=0,
        raw_end=len(content),
    )


class _UnscopedRepository:
    """Deliberately returns every chunk to prove retrieval enforces its own scope."""

    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = chunks

    def list_chunks(self, ticker: str, corpus_version: str) -> list[EvidenceChunk]:
        return self._chunks

    def dense_search(
        self,
        *,
        ticker: str,
        corpus_version: str,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[EvidenceChunk]:
        del embedding_model, query_embedding
        return [
            chunk
            for chunk in self._chunks
            if chunk.ticker == ticker and chunk.corpus_version == corpus_version
        ][:limit]


class _DensePrefersBeta:
    """A deterministic provider that ranks beta before alpha."""

    version = "dense-prefers-beta-v1"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if text == "common alpha" else [1.0, 0.0] for text in texts]


class _FakeReranker:
    """Records its one candidate pool and returns a deliberate evidence order."""

    version = "fake-v1"

    def __init__(self, result: list[EvidenceChunk]) -> None:
        self._result = result
        self.calls: list[tuple[str, list[EvidenceChunk], int]] = []

    def rerank(
        self, *, query: str, evidence: list[EvidenceChunk], limit: int
    ) -> list[EvidenceChunk]:
        self.calls.append((query, evidence, limit))
        return self._result


class _FalsyReranker(_FakeReranker):
    """A valid adapter may use falsiness for unrelated internal state."""

    def __bool__(self) -> bool:
        return False


class _RepositoryDenseRanking(_UnscopedRepository):
    def __init__(self, chunks: list[EvidenceChunk], dense: list[EvidenceChunk]) -> None:
        super().__init__(chunks)
        self._dense = dense
        self.dense_calls: list[tuple[str, str, str, list[float], int]] = []

    def dense_search(
        self,
        *,
        ticker: str,
        corpus_version: str,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[EvidenceChunk]:
        self.dense_calls.append(
            (ticker, corpus_version, embedding_model, query_embedding, limit)
        )
        return self._dense[:limit]


class _RecordingQueryEmbeddings:
    version = "query-v1"
    dimensions = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[1.0, 0.0] for _ in texts]


def test_rrf_rewards_documents_present_in_both_rankings() -> None:
    """Dropping one ranking's contribution would stop shared evidence from winning."""
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])

    assert result[0] == "b"


def test_rrf_breaks_equal_scores_by_document_id() -> None:
    """Nondeterministic equal-score ordering would make citations fluctuate."""
    result = reciprocal_rank_fusion([["z"], ["a"]])

    assert result == ["a", "z"]


def test_rrf_duplicates_do_not_change_fused_result() -> None:
    """A duplicate rank entry must not displace later unique evidence."""
    deduplicated = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])

    assert reciprocal_rank_fusion([["a", "a", "b"], ["a", "c", "a"]]) == deduplicated


def test_hash_embeddings_are_stable_and_have_a_fixed_dimension() -> None:
    """Using Python's randomized hash would make offline retrieval vary by process."""
    provider = HashEmbeddingProvider(dimensions=8)

    first = provider.embed(["data center demand", "risk"])
    second = provider.embed(["data center demand", "risk"])

    assert first == second
    assert all(len(vector) == 8 for vector in first)


def test_hash_embedding_defaults_match_persisted_vector_contract() -> None:
    """Repository-backed hash embeddings must use the production vector shape and tag."""
    provider = HashEmbeddingProvider()

    assert provider.dimensions == 1024
    assert provider.version == "hash-1024-v1"
    assert len(provider.embed(["revenue"])[0]) == 1024


def test_bge_provider_loads_only_the_configured_model_on_first_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing live dependencies must not import or load model assets eagerly."""
    calls: list[tuple[str, str | None, bool]] = []

    class FakeSentenceTransformer:
        def __init__(
            self,
            model_name: str,
            *,
            cache_folder: str | None = None,
            local_files_only: bool,
        ) -> None:
            calls.append((model_name, cache_folder, local_files_only))

        def encode(self, texts: list[str], *, normalize_embeddings: bool):
            assert normalize_embeddings is True
            return [[1.0, *([0.0] * 1023)] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = BgeM3EmbeddingProvider("BAAI/bge-m3", cache_dir="/models/cache")

    assert calls == []
    assert provider.version == "sentence-transformers:BAAI/bge-m3"
    assert provider.dimensions == 1024
    assert len(provider.embed(["query"])[0]) == 1024
    assert calls == [("BAAI/bge-m3", "/models/cache", True)]


def test_bge_provider_maps_unavailable_assets_to_stable_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or corrupt cached weights must not leak provider-specific exceptions."""
    class UnavailableSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise OSError("corrupt cache")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=UnavailableSentenceTransformer),
    )

    with pytest.raises(
        EmbeddingModelUnavailableError,
        match=r"EMBEDDING_MODEL_UNAVAILABLE.*prefetch",
    ):
        BgeM3EmbeddingProvider("BAAI/bge-m3").embed(["query"])


def test_bge_provider_serializes_concurrent_first_load_and_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0
    encode_calls = 0
    active_encodes = 0
    max_active_encodes = 0
    counters_lock = Lock()

    class DetectingSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal factory_calls
            del args, kwargs
            time.sleep(0.02)
            with counters_lock:
                factory_calls += 1

        def encode(self, texts: list[str], *, normalize_embeddings: bool):
            nonlocal active_encodes, encode_calls, max_active_encodes
            assert normalize_embeddings is True
            with counters_lock:
                active_encodes += 1
                encode_calls += 1
                max_active_encodes = max(max_active_encodes, active_encodes)
            try:
                time.sleep(0.01)
                return [
                    [float(len(text)), *([0.0] * 1023)]
                    for text in texts
                ]
            finally:
                with counters_lock:
                    active_encodes -= 1

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=DetectingSentenceTransformer),
    )
    provider = BgeM3EmbeddingProvider("BAAI/bge-m3")
    start = Barrier(8)

    def embed(index: int) -> list[list[float]]:
        start.wait(timeout=5)
        return provider.embed([f"query-{index}"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(embed, range(8)))

    assert factory_calls == 1
    assert encode_calls == 8
    assert max_active_encodes == 1
    assert [result[0][0] for result in results] == [7.0] * 8
    assert all(len(result[0]) == 1024 for result in results)


def test_search_embeds_only_the_query_and_consumes_repository_dense_ranking() -> None:
    """A cache miss must not re-embed any persisted document content."""
    alpha = _chunk("a-alpha", "common alpha")
    beta = _chunk("b-beta", "common beta")
    repository = _RepositoryDenseRanking([alpha, beta], [beta, alpha])
    embeddings = _RecordingQueryEmbeddings()
    retriever = HybridRetriever(repository, embeddings, candidate_pool_size=2)

    retriever.search("nvda", "common", "NVDA-v1", k=1)

    assert embeddings.batches == [["common"]]
    assert repository.dense_calls == [
        ("NVDA", "NVDA-v1", "query-v1", [1.0, 0.0], 2)
    ]


def test_search_filters_ticker_and_corpus_before_ranking() -> None:
    """An unscoped repository result must never influence or leak into this search."""
    expected = _chunk("nvda-current", "data center demand accelerated computing")
    retriever = HybridRetriever(
        _UnscopedRepository(
            [
                expected,
                _chunk(
                    "amd-leak",
                    "data center demand accelerated computing",
                    ticker="AMD",
                    corpus_version="AMD-v1",
                ),
                _chunk(
                    "nvda-old",
                    "data center demand accelerated computing",
                    corpus_version="NVDA-v0",
                ),
            ]
        ),
        HashEmbeddingProvider(),
    )

    result = retriever.search("nvda", "data center demand", "NVDA-v1")

    assert result == [expected]


def test_search_returns_empty_for_blank_query_or_empty_scope() -> None:
    """Ranking an empty query or corpus would otherwise yield arbitrary evidence."""
    retriever = HybridRetriever(_UnscopedRepository([]), HashEmbeddingProvider())

    assert retriever.search("NVDA", "", "NVDA-v1") == []
    assert retriever.search("NVDA", "demand", "NVDA-v1") == []


def test_retrieval_cache_read_failure_returns_ranked_evidence_without_exception_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache read outage must fall back to retrieval without a traceback leak."""
    private = "password=private-password-value"
    expected = _chunk("nvda-current", "revenue demand accelerated")

    class FailingReadCache:
        def get_json_sync(self, key: str) -> None:
            del key
            raise RuntimeError(private)

        def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
            del key, value, ttl_seconds

    retriever = HybridRetriever(
        _UnscopedRepository([expected]),
        HashEmbeddingProvider(),
        cache=FailingReadCache(),
    )

    with caplog.at_level(logging.WARNING):
        result = retriever.search("NVDA", "revenue", "NVDA-v1")

    assert result == [expected]
    assert "retrieval cache read failed" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


def test_retrieval_cache_write_failure_returns_ranked_evidence_without_exception_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache write outage must retain ranked evidence without a traceback leak."""
    private = "password=private-password-value"
    expected = _chunk("nvda-current", "revenue demand accelerated")

    class FailingWriteCache:
        def get_json_sync(self, key: str) -> None:
            del key
            return None

        def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
            del key, value, ttl_seconds
            raise RuntimeError(private)

    retriever = HybridRetriever(
        _UnscopedRepository([expected]),
        HashEmbeddingProvider(),
        cache=FailingWriteCache(),
    )

    with caplog.at_level(logging.WARNING):
        result = retriever.search("NVDA", "revenue", "NVDA-v1")

    assert result == [expected]
    assert "retrieval cache write failed" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


def test_search_handles_a_scoped_corpus_with_no_indexable_tokens() -> None:
    """Punctuation-only evidence must not make BM25 divide by zero."""
    punctuation_only = _chunk("punctuation", "!!!")
    retriever = HybridRetriever(_UnscopedRepository([punctuation_only]), HashEmbeddingProvider())

    assert retriever.search("NVDA", "demand", "NVDA-v1") == [punctuation_only]


@pytest.mark.parametrize("query", ["alpha", "common"])
def test_search_keeps_non_positive_bm25_candidates(query: str) -> None:
    """Zero or negative BM25 scores still belong in the sparse Top-100 ranking."""
    alpha = _chunk("a-alpha", "common alpha")
    beta = _chunk("z-beta", "common beta")
    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta], [beta, alpha]),
        _DensePrefersBeta(),
    )

    assert retriever.search("NVDA", query, "NVDA-v1", k=1) == [alpha]


def test_search_rejects_non_positive_k_and_caps_results_at_eight() -> None:
    """Ignoring the public result bound could overrun P0's evidence budget."""
    chunks = [_chunk(f"chunk-{index}", f"data center demand marker {index}") for index in range(9)]
    retriever = HybridRetriever(_UnscopedRepository(chunks), HashEmbeddingProvider())

    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("NVDA", "data center demand", "NVDA-v1", k=0)

    assert len(retriever.search("NVDA", "data center demand", "NVDA-v1", k=99)) == 8


def test_search_passes_the_rrf_candidate_pool_once_and_uses_reranker_order() -> None:
    """Bypassing reranking or passing raw rankings would make the fake order ineffective."""
    alpha = _chunk("a-alpha", "common alpha")
    beta = _chunk("b-beta", "common beta")
    gamma = _chunk("c-gamma", "common gamma")
    reranker = _FakeReranker([gamma, beta])
    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta, gamma], [beta, gamma, alpha]),
        _DensePrefersBeta(),
        reranker=reranker,
        candidate_pool_size=2,
    )

    result = retriever.search("NVDA", "alpha", "NVDA-v1", k=1)

    assert result == [gamma]
    assert reranker.calls == [("alpha", [beta, gamma], 2)]


def test_search_ignores_unknown_evidence_returned_by_a_reranker() -> None:
    """An adapter must not inject evidence that was absent from the scoped RRF pool."""
    alpha = _chunk("a-alpha", "common alpha")
    beta = _chunk("b-beta", "common beta")
    unknown = _chunk("unknown", "untrusted evidence")
    reranker = _FakeReranker([unknown, beta])
    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta], [beta, alpha]),
        _DensePrefersBeta(),
        reranker=reranker,
    )

    result = retriever.search("NVDA", "alpha", "NVDA-v1")

    assert result == [beta]


def test_search_uses_an_injected_falsy_reranker() -> None:
    """Truthiness fallback would silently discard a valid adapter's ranking."""
    alpha = _chunk("a-alpha", "common alpha")
    beta = _chunk("b-beta", "common beta")
    reranker = _FalsyReranker([beta, alpha])
    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta], [beta, alpha]),
        _DensePrefersBeta(),
        reranker=reranker,
    )

    result = retriever.search("NVDA", "alpha", "NVDA-v1", k=1)

    assert result == [beta]
    assert reranker.calls == [("alpha", [alpha, beta], 2)]


def test_search_emits_immutable_metrics_for_its_own_reranked_candidate_pool() -> None:
    """Shared last-run state would mix IDs or counts across concurrent retrieval calls."""
    alpha = _chunk("chunk-1", "common alpha")
    beta = _chunk("chunk-2", "common beta")
    metrics = []

    class _ScoringReranker:
        version = "scoring-v1"

        def rerank(self, *, query: str, evidence, limit: int) -> RerankResult:
            del query, limit
            return RerankResult(evidence=[evidence[1], evidence[0]], scores={"chunk-2": 0.9})

    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta], [beta, alpha]),
        _DensePrefersBeta(),
        reranker=_ScoringReranker(),
        metrics_sink=metrics.append,
    )

    result = retriever.search("NVDA", "common", "NVDA-v1", k=2)

    assert [chunk.id for chunk in result] == ["chunk-2", "chunk-1"]
    assert len(metrics) == 1
    assert metrics[0].sparse_candidates == 2
    assert metrics[0].dense_candidates == 2
    assert metrics[0].fused_candidates == 2
    assert metrics[0].fused_ids == ("chunk-1", "chunk-2")
    assert metrics[0].reranker_version == "scoring-v1"
    assert metrics[0].retained_ids == ("chunk-2", "chunk-1")
    assert metrics[0].retained_scores == (("chunk-2", 0.9),)
    assert metrics[0].scores == metrics[0].retained_scores
    with pytest.raises(ValidationError, match="frozen"):
        metrics[0].reranker_version = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        metrics[0].retained_ids.append("other")
    with pytest.raises(TypeError):
        metrics[0].retained_scores[0] = ("other", 0.1)


def test_search_with_metrics_returns_its_own_per_call_envelope() -> None:
    """MCP callers need cache/order metrics without reading shared last-run state."""
    alpha = _chunk("chunk-1", "common alpha")
    beta = _chunk("chunk-2", "common beta")
    retriever = HybridRetriever(
        _RepositoryDenseRanking([alpha, beta], [beta, alpha]),
        _DensePrefersBeta(),
    )

    result = retriever.search_with_metrics("NVDA", "common", "NVDA-v1", k=2)

    assert [chunk.id for chunk in result.evidence] == ["chunk-1", "chunk-2"]
    assert result.metrics.fused_ids == ("chunk-1", "chunk-2")
    assert result.metrics.cache_hit is False


def test_sparse_index_is_reused_for_distinct_queries_without_stale_content(monkeypatch) -> None:
    """Query changes reuse corpus preparation; content changes must rebuild it."""
    import fra.retrieval.hybrid as hybrid

    builds = []
    real_bm25 = hybrid.BM25Okapi

    def build_index(corpus):
        builds.append(corpus)
        return real_bm25(corpus)

    monkeypatch.setattr(hybrid, "BM25Okapi", build_index)
    chunks = [
        _chunk("reuse-a", "reusable sparse test revenue growth"),
        _chunk("reuse-b", "reusable sparse test operational risk"),
        _chunk("reuse-c", "reusable sparse test liquidity cash"),
    ]
    first = hybrid._sparse_ranking(chunks, ["revenue"])
    second = hybrid._sparse_ranking(chunks, ["risk"])
    assert first[0] == "reuse-a"
    assert second[0] == "reuse-b"
    assert len(builds) == 1

    changed = [chunks[0].model_copy(update={"content": "new risk exposure"}), *chunks[1:]]
    hybrid._sparse_ranking(changed, ["risk"])
    assert len(builds) == 2
    # Metadata/IDs are taken from the current scope, never from the cached index.
    other_scope = [c.model_copy(update={"id": f"other-{c.id}", "ticker": "MSFT"}) for c in chunks]
    assert hybrid._sparse_ranking(other_scope, ["revenue"])[0] == "other-reuse-a"
    assert len(builds) == 2
