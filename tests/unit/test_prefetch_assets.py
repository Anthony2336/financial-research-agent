"""Offline contract for the explicitly invoked protected asset provisioner."""

from pathlib import Path
from types import SimpleNamespace

import fra.prefetch_assets as prefetch_module


def test_prefetch_command_populates_every_configured_cache_without_provider_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    settings = SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        embedding_cache_dir=str(tmp_path / "embedding"),
        tokenizer_cache_dir=str(tmp_path / "tokenizer"),
        context_compressor_model="microsoft/llmlingua-model",
        context_compressor_cache_dir=str(tmp_path / "compressor"),
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_cache_dir=str(tmp_path / "reranker"),
        fast_model="gpt-5-mini",
        analyst_model="gpt-5",
    )
    monkeypatch.setattr(prefetch_module, "Settings", lambda: settings)

    class SentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.append(("embedding", (model_name, kwargs)))

    def encoding_for_model(model_name: str) -> object:
        calls.append(("tokenizer", model_name))
        return object()

    def snapshot_download(**kwargs: object) -> str:
        calls.append(("compressor", kwargs))
        return str(tmp_path / "compressor" / "snapshot")

    class Ranker:
        def __init__(self, *, model_name: str, cache_dir: str) -> None:
            calls.append(("reranker", (model_name, cache_dir)))
            model_dir = Path(cache_dir) / model_name
            model_dir.mkdir(parents=True)
            payloads = {
                "flashrank-MiniLM-L-12-v2_Q.onnx": b"onnx",
                "config.json": b"{}",
                "tokenizer_config.json": b"{}",
                "special_tokens_map.json": b"{}",
                "tokenizer.json": b"{}",
            }
            for name, payload in payloads.items():
                (model_dir / name).write_bytes(payload)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=SentenceTransformer),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "tiktoken",
        SimpleNamespace(encoding_for_model=encoding_for_model),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "flashrank",
        SimpleNamespace(Ranker=Ranker),
    )
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "prior-cache")

    prefetch_module.main()

    assert calls == [
        (
            "embedding",
            (
                "BAAI/bge-m3",
                {
                    "cache_folder": str(tmp_path / "embedding"),
                    "local_files_only": False,
                },
            ),
        ),
        ("tokenizer", "gpt-5-mini"),
        ("tokenizer", "gpt-5"),
        ("tokenizer", "gpt-3.5-turbo"),
        (
            "compressor",
            {
                "repo_id": "microsoft/llmlingua-model",
                "cache_dir": str(tmp_path / "compressor"),
                "local_files_only": False,
            },
        ),
        (
            "reranker",
            ("ms-marco-MiniLM-L-12-v2", str(tmp_path / "reranker")),
        ),
    ]
    assert __import__("os").environ["TIKTOKEN_CACHE_DIR"] == "prior-cache"
    assert (
        tmp_path
        / "reranker"
        / "ms-marco-MiniLM-L-12-v2"
        / ".financial-evidence-agent-manifest.json"
    ).is_file()
