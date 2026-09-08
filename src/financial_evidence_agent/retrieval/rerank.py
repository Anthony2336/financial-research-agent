"""Injectable reranker adapters for filing evidence."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from financial_evidence_agent.domain import EvidenceChunk, StrictModel

if TYPE_CHECKING:
    from flashrank import Ranker

_FLASHRANK_MANIFEST = ".financial-evidence-agent-manifest.json"
_FLASHRANK_REQUIRED_FILES = {
    "ms-marco-MiniLM-L-12-v2": (
        "flashrank-MiniLM-L-12-v2_Q.onnx",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    )
}
_LOCAL_ASSET_ERROR = (
    "RERANKER_MODEL_UNAVAILABLE: configured local FlashRank assets are missing or "
    "corrupt; run the protected prefetch before production research."
)


class Reranker(Protocol):
    """Reorders a bounded evidence candidate pool for one query."""

    @property
    def version(self) -> str:
        """Return an identifier suitable for retrieval cache keys."""

    def rerank(
        self, *, query: str, evidence: Sequence[EvidenceChunk], limit: int
    ) -> RerankResult:
        """Return the most relevant evidence in descending rank order."""


class RerankResult(StrictModel):
    """One reranking result with optional scores from the external ranker."""

    evidence: list[EvidenceChunk]
    scores: dict[str, float] = Field(default_factory=dict)


class IdentityReranker:
    """Offline fallback that retains the deterministic RRF candidate order."""

    version = "identity-risk-v2"

    def rerank(
        self, *, query: str, evidence: Sequence[EvidenceChunk], limit: int
    ) -> RerankResult:
        return RerankResult(evidence=list(evidence[:limit]))


class RerankerModelUnavailableError(RuntimeError):
    """Stable production failure for missing or corrupt FlashRank assets."""


class FlashRankReranker:
    """Adapt FlashRank output passage IDs back to the input evidence objects."""

    def __init__(self, ranker: Ranker, *, model_name: str) -> None:
        self._ranker = ranker
        self._model_name = model_name

    @property
    def version(self) -> str:
        return f"flashrank-risk-v2:{self._model_name}"

    def rerank(
        self, *, query: str, evidence: Sequence[EvidenceChunk], limit: int
    ) -> RerankResult:
        from flashrank import RerankRequest

        evidence_by_id = {chunk.id: chunk for chunk in evidence}
        results = self._ranker.rerank(
            RerankRequest(
                query=query,
                passages=[{"id": chunk.id, "text": chunk.content} for chunk in evidence],
            )
        )
        ranked_evidence: list[EvidenceChunk] = []
        scores: dict[str, float] = {}
        seen_ids: set[str] = set()
        scope = (
            (evidence[0].ticker, evidence[0].corpus_version)
            if evidence
            else None
        )
        for result in results:
            chunk_id = result.get("id")
            if chunk_id not in evidence_by_id or chunk_id in seen_ids:
                continue
            chunk = evidence_by_id[chunk_id]
            if scope is not None and (chunk.ticker, chunk.corpus_version) != scope:
                continue
            seen_ids.add(chunk_id)
            ranked_evidence.append(chunk)
            score = result.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                scores[chunk_id] = float(score)
        retained = ranked_evidence[:limit]
        return RerankResult(
            evidence=retained,
            scores={chunk.id: scores[chunk.id] for chunk in retained if chunk.id in scores},
        )


def _default_ranker_factory(*, model_name: str, cache_dir: str) -> Ranker:
    from flashrank import Ranker

    return Ranker(model_name=model_name, cache_dir=cache_dir)


def validate_local_flashrank_assets(model_name: str, cache_dir: str | None) -> None:
    """Verify the approved FlashRank 0.2.10 layout and prefetch manifest locally."""
    required = _FLASHRANK_REQUIRED_FILES.get(model_name)
    if required is None or not cache_dir:
        raise RerankerModelUnavailableError(_LOCAL_ASSET_ERROR)
    try:
        cache_root = Path(cache_dir).resolve(strict=True)
        model_dir = (cache_root / model_name).resolve(strict=True)
        model_dir.relative_to(cache_root)
        manifest_value = json.loads((model_dir / _FLASHRANK_MANIFEST).read_text("utf-8"))
        if not isinstance(manifest_value, dict):
            raise ValueError
        hashes = manifest_value.get("files")
        if (
            manifest_value.get("schema_version") != 1
            or manifest_value.get("model_name") != model_name
            or not isinstance(hashes, dict)
            or set(hashes) != set(required)
        ):
            raise ValueError
        for name in required:
            path = (model_dir / name).resolve(strict=True)
            path.relative_to(model_dir)
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError
            expected = hashes.get(name)
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or _file_sha256(path) != expected
            ):
                raise ValueError
            if name.endswith(".json") and not isinstance(
                json.loads(path.read_text("utf-8")), dict
            ):
                raise ValueError
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise RerankerModelUnavailableError(_LOCAL_ASSET_ERROR) from None


def write_flashrank_asset_manifest(model_name: str, cache_dir: str) -> Path:
    """Stamp hashes after the explicit protected prefetch has completed."""
    required = _FLASHRANK_REQUIRED_FILES.get(model_name)
    if required is None:
        raise ValueError("unsupported FlashRank model for protected prefetch")
    model_dir = Path(cache_dir) / model_name
    files: dict[str, str] = {}
    for name in required:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("FlashRank prefetch did not create the required asset layout")
        if name.endswith(".json") and not isinstance(
            json.loads(path.read_text("utf-8")), dict
        ):
            raise ValueError("FlashRank prefetch created an invalid JSON asset")
        files[name] = _file_sha256(path)
    manifest = model_dir / _FLASHRANK_MANIFEST
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "model_name": model_name, "files": files},
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_challenge_evidence(
    ranked: Sequence[EvidenceChunk],
    limit: int,
    *,
    evidence_side: str | None,
) -> list[EvidenceChunk]:
    """Reserve the highest-ranked canonical Risk Factors candidate within one scope."""
    if limit <= 0 or not ranked:
        return []
    scope = (ranked[0].ticker, ranked[0].corpus_version)
    unique = list(
        {
            chunk.id: chunk
            for chunk in ranked
            if (chunk.ticker, chunk.corpus_version) == scope
        }.values()
    )
    retained = unique[:limit]
    if evidence_side != "challenge":
        return retained
    if any(_is_risk_factor(chunk) for chunk in retained):
        return retained
    risk = next((chunk for chunk in unique if _is_risk_factor(chunk)), None)
    if risk is not None:
        retained[-1] = risk
    return retained


def _is_risk_factor(chunk: EvidenceChunk) -> bool:
    return " ".join(chunk.section.casefold().split()) == "risk factors"


class LazyFlashRankReranker:
    """Create FlashRank only when a non-empty production search requires it."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        ranker_factory: Callable[..., object] | None = None,
        asset_validator: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker_factory = (
            ranker_factory if ranker_factory is not None else _default_ranker_factory
        )
        self._asset_validator = (
            asset_validator if asset_validator is not None else validate_local_flashrank_assets
        )
        self._delegate: FlashRankReranker | None = None
        self._delegate_lock = Lock()

    @property
    def version(self) -> str:
        return f"flashrank-risk-v2:{self._model_name}"

    def rerank(
        self, *, query: str, evidence: Sequence[EvidenceChunk], limit: int
    ) -> RerankResult:
        if not evidence:
            return RerankResult(evidence=[])
        delegate = self._delegate
        if delegate is None:
            with self._delegate_lock:
                delegate = self._delegate
                if delegate is None:
                    try:
                        self._asset_validator(self._model_name, self._cache_dir)
                        ranker = self._ranker_factory(
                            model_name=self._model_name,
                            cache_dir=self._cache_dir,
                        )
                    except RerankerModelUnavailableError:
                        raise
                    except Exception as error:
                        raise RerankerModelUnavailableError(
                            "RERANKER_MODEL_UNAVAILABLE: could not load FlashRank model "
                            f"{self._model_name}. prefetch or install the configured reranker "
                            "assets before startup; see README.md."
                        ) from error
                    delegate = FlashRankReranker(ranker, model_name=self._model_name)  # type: ignore[arg-type]
                    self._delegate = delegate
        try:
            return delegate.rerank(query=query, evidence=evidence, limit=limit)
        except RerankerModelUnavailableError:
            raise
        except (ImportError, OSError, RuntimeError) as error:
            raise RerankerModelUnavailableError(
                "RERANKER_MODEL_UNAVAILABLE: could not run FlashRank model "
                f"{self._model_name}. prefetch or install the configured reranker assets "
                "before startup; see README.md."
            ) from error
