"""Citation-bound long-term research-memory values and embedding service."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import ConfigDict, Field, field_validator, model_validator

from fra.context import MemoryHint
from fra.domain import SourceRef, SourceRefKind, StrictModel
from fra.memory.privacy import contains_private_financial_or_secret
from fra.retrieval.indexing import (
    PERSISTED_EMBEDDING_DIMENSIONS,
    EmbeddingProvider,
)

logger = logging.getLogger(__name__)
_SUMMARY_CREDENTIAL = re.compile(
    r"\b(?:password|passcode|api[_ -]?(?:key|secret)|client[_ -]?secret|"
    r"access[_ -]?token|refresh[_ -]?token|session[_ -]?token|bearer[_ -]?token|"
    r"tokens?|credentials?|private[_ -]?key)\b|\bauthorization\s*:\s*bearer\b|"
    r"\bbearer\s+(?=[A-Z0-9._~-]{8,}\b)(?=[A-Z0-9._~-]*[._~-])"
    r"[A-Z0-9._~-]+|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:密码|密碼|口令|API密钥|API密鑰|API秘密|客户端密钥|客戶端密鑰|"
    r"访问令牌|訪問令牌|刷新令牌|会话令牌|會話令牌|令牌|私钥|私鑰|凭据|憑據)",
    re.IGNORECASE,
)
_SUMMARY_ACCOUNT_NUMBER = re.compile(
    r"\b(?:brokerage\s+)?account\s*(?:number|no\.?|#)\s*"
    r"(?:(?:is|equals?)\b|[:=])?\s*[A-Z0-9-]{3,}|"
    r"(?:券商|经纪|經紀)?(?:账户|帳戶)(?:号码|號碼|号|號)\s*"
    r"(?:是|为|為|[:=：])?\s*[A-Z0-9-]{3,}",
    re.IGNORECASE,
)
class ResearchMemoryKind(StrEnum):
    """The only durable research-hint categories."""

    RESEARCH_SUMMARY = "research_summary"
    COUNTEREVIDENCE = "counterevidence"
    OPEN_QUESTION = "open_question"
    SOURCE_POINTER = "source_pointer"


class ResearchMemory(StrictModel):
    """One frozen pointer/summary that has no citation authority of its own."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    scope_key: str = Field(min_length=1, max_length=64)
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    memory_kind: ResearchMemoryKind
    summary: str = Field(min_length=1, max_length=1_200)
    source_run_id: str = Field(min_length=1, max_length=64)
    evidence_source_refs: tuple[SourceRef, ...] = Field(min_length=1, max_length=20)
    corpus_version: str = Field(min_length=1, max_length=32)
    embedding: tuple[float, ...] = Field(
        min_length=PERSISTED_EMBEDDING_DIMENSIONS,
        max_length=PERSISTED_EMBEDDING_DIMENSIONS,
        exclude=True,
        repr=False,
    )
    embedding_model: str = Field(min_length=1, max_length=255)
    importance: float = Field(ge=0, le=1)
    created_at: datetime
    expires_at: datetime
    stale: bool = False

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("created_at", "expires_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("research memory times must include timezone information")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_binding(self) -> ResearchMemory:
        if self.scope_key != f"ticker:{self.ticker}":
            raise ValueError("scope_key must preserve the normalized ticker identity")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if not is_research_memory_summary_eligible(self.summary, self.ticker):
            raise ValueError("research memory summary failed privacy eligibility")
        encoded = [reference.encode() for reference in self.evidence_source_refs]
        if len(encoded) != len(set(encoded)):
            raise ValueError("evidence_source_refs must be unique")
        if any(reference.ticker != self.ticker for reference in self.evidence_source_refs):
            raise ValueError("evidence source reference ticker must match memory ticker")
        if any(
            reference.kind
            not in {
                SourceRefKind.FILING,
                SourceRefKind.WEB,
            }
            for reference in self.evidence_source_refs
        ):
            raise ValueError("research memories require filing or web evidence references")
        return self


class ResearchMemoryRepositoryProtocol(Protocol):
    """Persistence contract consumed by the embedding service."""

    def store_guarded(
        self,
        *,
        ticker: str,
        memory_kind: ResearchMemoryKind,
        summary: str,
        source_run_id: str,
        evidence_source_refs: tuple[SourceRef, ...],
        corpus_version: str,
        embedding: list[float],
        embedding_model: str,
        importance: float,
        created_at: datetime,
        expires_at: datetime,
    ) -> ResearchMemory:
        """Store one independently validated guarded memory."""

    def search(
        self,
        ticker: str,
        query_embedding: list[float],
        limit: int = 3,
        *,
        embedding_model: str,
        current_corpus_version: str | None = None,
    ) -> list[ResearchMemory]:
        """Return at most three valid semantic hints."""


class ResearchMemorySearch(Protocol):
    """Graph-facing query boundary; it never exposes evidence bodies."""

    def search(
        self,
        ticker: str,
        query: str,
        *,
        current_corpus_version: str | None = None,
        limit: int = 3,
    ) -> list[ResearchMemory]:
        """Return typed long-term hints for query expansion only."""


class ResearchMemoryStore(ResearchMemorySearch, Protocol):
    """Application write plus graph read boundary for long-term hints."""

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
        """Persist one final guarded candidate after local run completion."""


class ResearchMemoryService:
    """Embed bounded summaries/queries and delegate durable validation."""

    def __init__(
        self,
        repository: ResearchMemoryRepositoryProtocol,
        embedding_provider: EmbeddingProvider,
        *,
        ttl_days: int = 90,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not 1 <= ttl_days <= 3650:
            raise ValueError("research memory TTL must be between 1 and 3650 days")
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._ttl_days = ttl_days
        self._clock = clock

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
        """Embed and store one bounded post-guard value with configured expiry."""
        normalized_summary = " ".join(summary.split())
        if not normalized_summary or len(normalized_summary) > 1_200:
            raise ValueError("research memory summary must contain 1 to 1200 characters")
        if not is_research_memory_summary_eligible(normalized_summary, ticker):
            raise ValueError("research memory summary failed privacy eligibility")
        created_at = self._aware_now()
        return self._repository.store_guarded(
            ticker=ticker,
            memory_kind=memory_kind,
            summary=normalized_summary,
            source_run_id=source_run_id,
            evidence_source_refs=evidence_source_refs,
            corpus_version=corpus_version,
            embedding=self._embed(normalized_summary),
            embedding_model=self._embedding_provider.version,
            importance=importance,
            created_at=created_at,
            expires_at=created_at + timedelta(days=self._ttl_days),
        )

    def search(
        self,
        ticker: str,
        query: str,
        *,
        current_corpus_version: str | None = None,
        limit: int = 3,
    ) -> list[ResearchMemory]:
        """Embed one query and return repository-validated pointers only."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []
        return self._repository.search(
            ticker,
            self._embed(normalized_query),
            limit=limit,
            current_corpus_version=current_corpus_version,
            embedding_model=self._embedding_provider.version,
        )

    def _embed(self, value: str) -> list[float]:
        if self._embedding_provider.dimensions != PERSISTED_EMBEDDING_DIMENSIONS:
            raise ValueError("research memory embeddings must have 1024 dimensions")
        if not self._embedding_provider.version.strip():
            raise ValueError("research memory embedding model must not be blank")
        vectors = self._embedding_provider.embed([value])
        if len(vectors) != 1 or len(vectors[0]) != PERSISTED_EMBEDDING_DIMENSIONS:
            raise ValueError("research memory provider returned an invalid embedding shape")
        return [float(item) for item in vectors[0]]

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("research memory clock must return an aware datetime")
        return value.astimezone(UTC)


def scope_research_memory_store(
    store: ResearchMemoryStore | None,
    web_evidence_validator: object | None,
) -> ResearchMemoryStore | None:
    """Bind one runtime's current web policy when the store supports scoping."""
    if store is None:
        return None
    binder = getattr(store, "with_web_evidence_validator", None)
    if not callable(binder):
        return store
    try:
        scoped = binder(web_evidence_validator)
    except Exception:
        logger.warning("research memory policy binding failed")
        return None
    return scoped


def is_research_memory_summary_eligible(summary: str, ticker: str) -> bool:
    """Reject generated personal-finance/private values without blocking issuer facts."""
    normalized = " ".join(summary.split())
    if not normalized:
        return False
    if contains_private_financial_or_secret(normalized, current_ticker=ticker):
        return False
    if _SUMMARY_CREDENTIAL.search(normalized) or _SUMMARY_ACCOUNT_NUMBER.search(normalized):
        return False
    return True


def build_research_memory_hints(
    memories: tuple[ResearchMemory, ...] | list[ResearchMemory],
) -> tuple[MemoryHint, ...]:
    """Render at most three identified, explicitly non-evidentiary planner hints."""
    hints: list[MemoryHint] = []
    for rank, memory in enumerate(memories[:3]):
        freshness = "stale-corpus" if memory.stale else "current-corpus"
        pointers = " | ".join(reference.encode() for reference in memory.evidence_source_refs)
        hints.append(
            MemoryHint(
                text=(
                    f"Long-term {memory.memory_kind.value} hint ({freshness}); query expansion "
                    f"only, rehydrate current evidence before use: {memory.summary}; "
                    f"source pointers={pointers}"
                ),
                score=float(3 - rank) + memory.importance,
                identity="research",
                pointer_id=memory.id,
            )
        )
    return tuple(hints)
