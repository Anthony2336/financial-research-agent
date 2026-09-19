"""Provider-versioned document embedding indexing."""

import re
from hashlib import sha256
from threading import RLock
from typing import Protocol

from sqlalchemy.orm import Session

from fra.storage.repositories import FilingRepository

PERSISTED_EMBEDDING_DIMENSIONS = 1024
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    """Produces versioned dense vectors without changing input order."""

    @property
    def version(self) -> str:
        """Return the stable provider and model revision identifier."""

    @property
    def dimensions(self) -> int:
        """Return the number of values in every produced vector."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text without changing its order."""


class EmbeddingIndexer:
    """Persist only missing or provider-stale vectors for one filing corpus."""

    def __init__(self, repository: FilingRepository, provider: EmbeddingProvider) -> None:
        self._repository = repository
        self._provider = provider

    def ensure_indexed(self, ticker: str, corpus_version: str) -> int:
        """Embed and atomically store every stale chunk, returning the write count."""
        with self._repository.embedding_write_context(ticker) as session:
            return self.ensure_indexed_in_session(session, ticker, corpus_version)

    def ensure_indexed_in_session(
        self,
        session: Session,
        ticker: str,
        corpus_version: str,
    ) -> int:
        """Recheck and index in the caller's protected ticker transaction."""
        if self._provider.dimensions != PERSISTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "repository-backed embeddings must have "
                f"{PERSISTED_EMBEDDING_DIMENSIONS} dimensions"
            )
        missing = self._repository.list_chunks_requiring_embedding_in_session(
            session,
            ticker,
            corpus_version,
            self._provider.version,
        )
        if not missing:
            return 0
        vectors = self._provider.embed([chunk.content for chunk in missing])
        self._repository.store_embeddings_in_session(
            session,
            missing,
            vectors,
            self._provider.version,
        )
        return len(missing)


class HashEmbeddingProvider:
    """Deterministic embedding provider for offline and in-memory use."""

    def __init__(self, dimensions: int = PERSISTED_EMBEDDING_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def version(self) -> str:
        return f"hash-{self._dimensions}-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Map normalized tokens to signed hash buckets without network access."""
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in _TOKEN_PATTERN.findall(text.lower()):
                digest = sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], byteorder="big") % self._dimensions
                vector[bucket] += 1.0 if digest[4] & 1 else -1.0
            vectors.append(vector)
        return vectors


class EmbeddingModelUnavailableError(RuntimeError):
    """Stable live-provider failure for missing or corrupt embedding assets."""


class BgeM3EmbeddingProvider:
    """Configured BGE provider whose sentence-transformers model loads lazily."""

    def __init__(self, model_name: str = "BAAI/bge-m3", *, cache_dir: str | None = None) -> None:
        if not model_name.strip():
            raise ValueError("embedding model must not be blank")
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: object | None = None
        self._model_lock = RLock()

    @property
    def version(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimensions(self) -> int:
        return PERSISTED_EMBEDDING_DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Load configured cached assets on demand and return plain Python vectors."""
        if not texts:
            return []
        try:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self._model_name,
                        cache_folder=self._cache_dir,
                        local_files_only=True,
                    )
                embeddings = self._model.encode(  # type: ignore[union-attr]
                    texts,
                    normalize_embeddings=True,
                )
            vectors = [[float(value) for value in vector] for vector in embeddings]
            if len(vectors) != len(texts) or any(
                len(vector) != self.dimensions for vector in vectors
            ):
                raise ValueError("configured model returned an invalid embedding shape")
            return vectors
        except EmbeddingModelUnavailableError:
            raise
        except Exception as error:
            raise EmbeddingModelUnavailableError(
                "EMBEDDING_MODEL_UNAVAILABLE: could not load or run "
                f"{self._model_name}. prefetch the configured model into "
                "EMBEDDING_CACHE_DIR before startup; see README.md."
            ) from error
