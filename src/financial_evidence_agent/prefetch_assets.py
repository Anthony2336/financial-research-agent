"""Explicit network-enabled provisioning for protected model-backed smokes only."""

from __future__ import annotations

from os import environ
from pathlib import Path

from financial_evidence_agent.config import Settings
from financial_evidence_agent.retrieval.rerank import write_flashrank_asset_manifest


def main() -> None:
    """Download each approved asset into its configured cache and stamp local integrity."""
    settings = Settings()
    embedding_cache = _required_cache(settings.embedding_cache_dir, "EMBEDDING_CACHE_DIR")
    tokenizer_cache = _required_cache(settings.tokenizer_cache_dir, "TOKENIZER_CACHE_DIR")
    compressor_cache = _required_cache(
        settings.context_compressor_cache_dir,
        "CONTEXT_COMPRESSOR_CACHE_DIR",
    )
    reranker_cache = _required_cache(settings.reranker_cache_dir, "RERANKER_CACHE_DIR")
    model_names = _required_model_names(settings.fast_model, settings.analyst_model)

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from tiktoken import encoding_for_model

    SentenceTransformer(
        settings.embedding_model,
        cache_folder=str(embedding_cache),
        local_files_only=False,
    )
    previous_tiktoken_cache = environ.get("TIKTOKEN_CACHE_DIR")
    environ["TIKTOKEN_CACHE_DIR"] = str(tokenizer_cache)
    try:
        for model_name in (*model_names, "gpt-3.5-turbo"):
            encoding_for_model(model_name)
    finally:
        if previous_tiktoken_cache is None:
            environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            environ["TIKTOKEN_CACHE_DIR"] = previous_tiktoken_cache
    snapshot_download(
        repo_id=settings.context_compressor_model,
        cache_dir=str(compressor_cache),
        local_files_only=False,
    )

    from flashrank import Ranker

    Ranker(
        model_name=settings.reranker_model,
        cache_dir=str(reranker_cache),
    )
    write_flashrank_asset_manifest(settings.reranker_model, str(reranker_cache))


def _required_cache(value: str | None, setting_name: str) -> Path:
    if not value:
        raise RuntimeError(f"{setting_name} is required for protected asset prefetch")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _required_model_names(fast_model: str | None, analyst_model: str | None) -> tuple[str, ...]:
    if not fast_model or not analyst_model:
        raise RuntimeError("FAST_MODEL and ANALYST_MODEL are required for tokenizer prefetch")
    return tuple(dict.fromkeys((fast_model, analyst_model)))
