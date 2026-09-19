"""Deterministic sparse and dense retrieval over one filing corpus."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache
from hashlib import sha256
from typing import Literal

from pydantic import ConfigDict, TypeAdapter
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from fra.domain import EvidenceChunk, StrictModel
from fra.retrieval.indexing import (
    BgeM3EmbeddingProvider,
    EmbeddingIndexer,
    EmbeddingProvider,
    HashEmbeddingProvider,
)
from fra.retrieval.rerank import (
    IdentityReranker,
    Reranker,
    RerankResult,
    reserve_challenge_evidence,
)
from fra.storage.cache import NoopJsonCache, SyncJsonCache
from fra.storage.repositories import FilingRepository

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_RANKING_LIMIT = 100
_MAX_RESULTS = 8
_CACHE_TTL_SECONDS = 3_600
_EVIDENCE_LIST = TypeAdapter(list[EvidenceChunk])

logger = logging.getLogger(__name__)


class RetrievalMetrics(StrictModel):
    """Immutable retrieval facts emitted to the runtime-owned trace adapter."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    sparse_candidates: int
    dense_candidates: int
    fused_candidates: int
    fused_ids: tuple[str, ...] = ()
    reranker_version: str
    cache_hit: bool = False
    retained_ids: tuple[str, ...] = ()
    retained_scores: tuple[tuple[str, float], ...] = ()

    @property
    def scores(self) -> tuple[tuple[str, float], ...]:
        """Compatibility alias for the explicit retained-score field."""
        return self.retained_scores


class RetrievalResult(StrictModel):
    """One call's evidence and metrics without shared mutable run state."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    evidence: tuple[EvidenceChunk, ...]
    metrics: RetrievalMetrics


__all__ = [
    "BgeM3EmbeddingProvider",
    "HashEmbeddingProvider",
    "HybridRetriever",
    "RetrievalMetrics",
    "RetrievalResult",
]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked identifiers, breaking equal scores by identifier ascending."""
    if k <= 0:
        raise ValueError("k must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, document_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[document_id] += 1 / (k + rank)
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))


class HybridRetriever:
    """Retrieve filing chunks using independent sparse and dense rankings."""

    def __init__(
        self,
        repository: FilingRepository,
        embedding_provider: EmbeddingProvider,
        *,
        reranker: Reranker | None = None,
        candidate_pool_size: int = _RANKING_LIMIT,
        cache: SyncJsonCache | None = None,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
        metrics_sink: Callable[[RetrievalMetrics], None] | None = None,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._reranker = IdentityReranker() if reranker is None else reranker
        self._candidate_pool_size = candidate_pool_size
        self._cache = NoopJsonCache() if cache is None else cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._metrics_sink = metrics_sink

    @property
    def reranker_version(self) -> str:
        """Expose the concrete cache-visible reranker identity for composition checks."""
        return self._reranker.version

    def ensure_indexed(self, ticker: str, corpus_version: str) -> int:
        """Ensure this retriever's provider owns every stored vector in the corpus."""
        return EmbeddingIndexer(self._repository, self._embedding_provider).ensure_indexed(
            ticker,
            corpus_version,
        )

    def search(
        self,
        ticker: str,
        query: str,
        corpus_version: str,
        k: int = _MAX_RESULTS,
        *,
        filing_ids: Sequence[str] | None = None,
        evidence_side: Literal["support", "challenge"] | None = None,
    ) -> list[EvidenceChunk]:
        """Return at most eight chunks from exactly one ticker and corpus version."""
        return list(
            self.search_with_metrics(
                ticker,
                query,
                corpus_version,
                k,
                filing_ids=filing_ids,
                evidence_side=evidence_side,
            ).evidence
        )

    def search_with_metrics(
        self,
        ticker: str,
        query: str,
        corpus_version: str,
        k: int = _MAX_RESULTS,
        *,
        filing_ids: Sequence[str] | None = None,
        evidence_side: Literal["support", "challenge"] | None = None,
    ) -> RetrievalResult:
        """Return evidence with immutable metrics owned by this retrieval call."""
        if k <= 0:
            raise ValueError("k must be positive")
        query_tokens = _tokenize(query)
        if not query_tokens:
            return self._result([])

        normalized_ticker = ticker.upper()
        normalized_filing_ids = _normalized_filing_ids(filing_ids)
        result_limit = min(k, _MAX_RESULTS)
        cache_key = _retrieval_cache_key(
            corpus_version=corpus_version,
            ticker=normalized_ticker,
            query=query,
            limit=k,
            reranker_version=self._reranker.version,
            embedding_version=self._embedding_provider.version,
            filing_ids=normalized_filing_ids,
            evidence_side=evidence_side,
        )
        cached = self._cached_evidence(
            cache_key,
            ticker=normalized_ticker,
            corpus_version=corpus_version,
            limit=result_limit,
            filing_ids=normalized_filing_ids,
        )
        if cached is not None:
            return self._result(
                cached,
                fused_ids=[chunk.id for chunk in cached],
                cache_hit=True,
            )

        if self._supports_persisted_indexing():
            with self._repository.embedding_write_context(normalized_ticker) as session:
                EmbeddingIndexer(
                    self._repository,
                    self._embedding_provider,
                ).ensure_indexed_in_session(session, normalized_ticker, corpus_version)
                return self._uncached_search(
                    normalized_ticker=normalized_ticker,
                    query=query,
                    query_tokens=query_tokens,
                    corpus_version=corpus_version,
                    result_limit=result_limit,
                    cache_key=cache_key,
                    filing_ids=normalized_filing_ids,
                    evidence_side=evidence_side,
                    repository_session=session,
                )
        return self._uncached_search(
            normalized_ticker=normalized_ticker,
            query=query,
            query_tokens=query_tokens,
            corpus_version=corpus_version,
            result_limit=result_limit,
            cache_key=cache_key,
            filing_ids=normalized_filing_ids,
            evidence_side=evidence_side,
        )

    def _supports_persisted_indexing(self) -> bool:
        return self._embedding_provider.dimensions == 1024 and all(
            callable(getattr(self._repository, name, None))
            for name in (
                "embedding_write_context",
                "list_chunks_requiring_embedding_in_session",
                "store_embeddings_in_session",
                "list_chunks_in_session",
                "dense_search_in_session",
            )
        )

    def _uncached_search(
        self,
        *,
        normalized_ticker: str,
        query: str,
        query_tokens: list[str],
        corpus_version: str,
        result_limit: int,
        cache_key: str,
        filing_ids: tuple[str, ...] | None,
        evidence_side: Literal["support", "challenge"] | None,
        repository_session: Session | None = None,
    ) -> RetrievalResult:
        """Index/query one corpus while the caller retains any required corpus lock."""

        if repository_session is not None:
            scoped_chunks = self._repository.list_chunks_in_session(
                repository_session,
                normalized_ticker,
                corpus_version,
                filing_ids=filing_ids,
            )
        elif filing_ids is None:
            scoped_chunks = self._repository.list_chunks(normalized_ticker, corpus_version)
        else:
            scoped_chunks = self._repository.list_chunks(
                normalized_ticker,
                corpus_version,
                filing_ids=filing_ids,
            )
        chunks = [
            chunk
            for chunk in scoped_chunks
            if chunk.ticker.upper() == normalized_ticker and chunk.corpus_version == corpus_version
        ]
        if not chunks:
            return self._result([])

        sparse_ranking = _sparse_ranking(chunks, query_tokens)
        query_vector = self._embedding_provider.embed([query])[0]
        dense_arguments = {
            "ticker": normalized_ticker,
            "corpus_version": corpus_version,
            "embedding_model": self._embedding_provider.version,
            "query_embedding": query_vector,
            "limit": self._candidate_pool_size,
        }
        if filing_ids is not None:
            dense_arguments["filing_ids"] = filing_ids
        dense = (
            self._repository.dense_search_in_session(
                repository_session,
                **dense_arguments,
            )
            if repository_session is not None
            else self._repository.dense_search(**dense_arguments)
        )
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        dense_ranking = [chunk.id for chunk in dense if chunk.id in chunks_by_id]
        fused_ids = reciprocal_rank_fusion([sparse_ranking, dense_ranking])
        candidate_pool = [
            chunks_by_id[chunk_id] for chunk_id in fused_ids[: self._candidate_pool_size]
        ]
        reranked = self._reranker.rerank(
            query=query,
            evidence=candidate_pool,
            limit=len(candidate_pool),
        )
        rerank_result = (
            reranked
            if isinstance(reranked, RerankResult)
            else RerankResult(evidence=list(reranked))
        )
        candidate_by_id = {chunk.id: chunk for chunk in candidate_pool}
        ranked = _validated_reranked_evidence(
            rerank_result.evidence,
            candidate_by_id,
            len(candidate_pool),
        )
        result = reserve_challenge_evidence(
            ranked,
            result_limit,
            evidence_side=evidence_side,
        )
        retrieval_result = self._result(
            result,
            sparse_candidates=len(sparse_ranking),
            dense_candidates=len(dense_ranking),
            fused_candidates=len(candidate_pool),
            fused_ids=[chunk.id for chunk in candidate_pool],
            retained_scores=[
                (chunk.id, rerank_result.scores[chunk.id])
                for chunk in result
                if chunk.id in rerank_result.scores
            ],
        )
        if result:
            try:
                self._cache.set_json_sync(
                    cache_key,
                    [chunk.model_dump(mode="json") for chunk in result],
                    ttl_seconds=self._cache_ttl_seconds,
                )
            except Exception:
                logger.warning("retrieval cache write failed")
        return retrieval_result

    def _result(
        self,
        evidence: Sequence[EvidenceChunk],
        *,
        sparse_candidates: int = 0,
        dense_candidates: int = 0,
        fused_candidates: int = 0,
        fused_ids: Sequence[str] = (),
        retained_scores: Sequence[tuple[str, float]] = (),
        cache_hit: bool = False,
    ) -> RetrievalResult:
        metrics = RetrievalMetrics(
            sparse_candidates=sparse_candidates,
            dense_candidates=dense_candidates,
            fused_candidates=fused_candidates,
            fused_ids=tuple(fused_ids),
            reranker_version=self._reranker.version,
            cache_hit=cache_hit,
            retained_ids=tuple(chunk.id for chunk in evidence),
            retained_scores=tuple(retained_scores),
        )
        if self._metrics_sink is not None:
            self._metrics_sink(metrics)
        return RetrievalResult(evidence=tuple(evidence), metrics=metrics)

    def _cached_evidence(
        self,
        key: str,
        *,
        ticker: str,
        corpus_version: str,
        limit: int,
        filing_ids: tuple[str, ...] | None,
    ) -> list[EvidenceChunk] | None:
        try:
            value = self._cache.get_json_sync(key)
            if not isinstance(value, list) or not value:
                return None
            evidence = _EVIDENCE_LIST.validate_python(value)
        except Exception:
            logger.warning("retrieval cache read failed")
            return None
        if (
            len(evidence) > limit
            or len({chunk.id for chunk in evidence}) != len(evidence)
            or any(
                chunk.ticker.upper() != ticker or chunk.corpus_version != corpus_version
                for chunk in evidence
            )
        ):
            logger.warning("retrieval cache value rejected for %s", key)
            return None
        if filing_ids is not None:
            try:
                scoped_chunks = self._repository.list_chunks(
                    ticker,
                    corpus_version,
                    filing_ids=filing_ids,
                )
            except TypeError:
                logger.warning("retrieval cache scope could not be verified for %s", key)
                return None
            allowed_chunk_ids = {chunk.id for chunk in scoped_chunks}
            if any(chunk.id not in allowed_chunk_ids for chunk in evidence):
                logger.warning("retrieval cache filing scope rejected for %s", key)
                return None
        scoped_reader = getattr(self._repository, "list_chunks_in_corpus_by_ids", None)
        if callable(scoped_reader):
            canonical = scoped_reader(
                ticker,
                corpus_version,
                [chunk.id for chunk in evidence],
            )
        else:
            canonical = self._repository.list_chunks_by_ids([chunk.id for chunk in evidence])
        if len(canonical) != len(evidence) or any(
            cached_chunk != stored_chunk
            for cached_chunk, stored_chunk in zip(evidence, canonical, strict=True)
        ):
            logger.warning("retrieval cache provenance mismatch for %s", key)
            return None
        return canonical


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _retrieval_cache_key(
    *,
    corpus_version: str,
    ticker: str,
    query: str,
    limit: int,
    reranker_version: str,
    embedding_version: str,
    filing_ids: tuple[str, ...] | None = None,
    evidence_side: Literal["support", "challenge"] | None = None,
) -> str:
    query_hash = sha256(query.encode("utf-8")).hexdigest()
    base = (
        f"retrieval:{corpus_version}:{ticker}:{query_hash}:{limit}:"
        f"{reranker_version}:{embedding_version}"
    )
    if evidence_side is not None:
        base = f"{base}:side-{evidence_side}"
    if filing_ids is None:
        return base
    scope_hash = sha256("\n".join(filing_ids).encode("utf-8")).hexdigest()
    return f"{base}:filings-{scope_hash}"


def _normalized_filing_ids(
    filing_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if filing_ids is None:
        return None
    normalized = tuple(sorted(filing_id.strip() for filing_id in filing_ids))
    if not normalized or any(not filing_id for filing_id in normalized):
        raise ValueError("filing_ids must contain at least one non-empty id")
    if len(set(normalized)) != len(normalized):
        raise ValueError("filing_ids must be unique")
    return normalized


@lru_cache(maxsize=8)
def _sparse_index(contents: tuple[str, ...]) -> BM25Okapi | None:
    """Reuse corpus tokenization and BM25 statistics across query rewrites.

    Only text is cached: identifiers and provenance always come from the current
    repository scope. A changed corpus produces a different key. The LRU retains
    at most eight corpora, including their term statistics.
    """
    tokens = [_tokenize(content) for content in contents]
    return BM25Okapi(tokens) if any(tokens) else None


def _sparse_ranking(chunks: Sequence[EvidenceChunk], query_tokens: list[str]) -> list[str]:
    index = _sparse_index(tuple(chunk.content for chunk in chunks))
    if index is None:
        return sorted(chunk.id for chunk in chunks)[:_RANKING_LIMIT]
    scores = index.get_scores(query_tokens)
    return [
        chunk.id
        for chunk, _ in sorted(
            zip(chunks, scores, strict=True), key=lambda item: (-item[1], item[0].id)
        )
    ][:_RANKING_LIMIT]


def _validated_reranked_evidence(
    reranked: Sequence[EvidenceChunk],
    candidate_by_id: dict[str, EvidenceChunk],
    limit: int,
) -> list[EvidenceChunk]:
    """Keep only unique evidence that originated in the scoped RRF candidate pool."""
    result: list[EvidenceChunk] = []
    seen_ids: set[str] = set()
    for chunk in reranked:
        if chunk.id in candidate_by_id and chunk.id not in seen_ids:
            result.append(candidate_by_id[chunk.id])
            seen_ids.add(chunk.id)
        if len(result) == limit:
            break
    return result
