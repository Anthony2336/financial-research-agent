"""Deterministic context allocation, compression, and run-budget contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha1, sha256
from threading import Barrier

import pytest

from fra.context import (
    BudgetAuthority,
    BudgetExhaustedError,
    BudgetLimits,
    BudgetState,
    ContextBudgetError,
    ContextBuilder,
    ContextCompressionError,
    ContextCompressionErrorCode,
    ContextEvidence,
    ContextLimits,
    LazyLLMLinguaCompressor,
    LazyTiktokenTokenCounter,
    MemoryHint,
    TokenCounterError,
    TokenEncodingAsset,
)
from fra.skills.recipes import P1_RECIPES


def _token_asset(payload: bytes = b"local encoding") -> TokenEncodingAsset:
    return TokenEncodingAsset(
        encoding_name="test_encoding",
        cache_url="https://assets.invalid/test_encoding.tiktoken",
        expected_hash=sha256(payload).hexdigest(),
        pattern=r"\w+",
        special_tokens={},
    )


def _write_token_asset(tmp_path, asset: TokenEncodingAsset, payload: bytes) -> None:
    cache_key = sha1(asset.cache_url.encode()).hexdigest()
    (tmp_path / cache_key).write_bytes(payload)


def _marker_tokens(value: str) -> int:
    return value.count("TKN")


def _evidence(
    evidence_id: str,
    *,
    score: float,
    body: str,
) -> ContextEvidence:
    return ContextEvidence(
        evidence_id=evidence_id,
        source_type="filing",
        ticker="NVDA",
        body=body,
        score=score,
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/filing.htm",
        date="2026-05-28",
        accession_no="0001045810-26-000041",
        section="MD&A",
        raw_start=120,
        raw_end=240,
    )


class _RecordingCompressor:
    def __init__(self, compressed: str) -> None:
        self.compressed = compressed
        self.calls: list[tuple[str, int]] = []

    def compress(self, body: str, *, max_tokens: int) -> str:
        self.calls.append((body, max_tokens))
        return self.compressed


def test_total_overflow_drops_low_score_evidence_before_memory_hints() -> None:
    """Dropping a memory hint first would violate the frozen allocation priority."""
    context = ContextBuilder(
        control="control",
        limits=ContextLimits(
            max_control_tokens=500,
            max_task_tokens=500,
            max_memory_tokens=300,
            max_evidence_tokens=10,
            max_total_tokens=7,
        ),
        token_counter=_marker_tokens,
    ).build(
        task="task",
        memory_hints=[MemoryHint(text="TKN TKN", score=1.0)],
        evidence=[
            _evidence("high", score=0.9, body="TKN TKN TKN TKN"),
            _evidence("low", score=0.1, body="TKN TKN TKN TKN"),
        ],
    )

    assert [item.evidence_id for item in context.evidence] == ["high"]
    assert [hint.text for hint in context.memory_hints] == ["TKN TKN"]
    assert context.dropped_evidence_ids == ("low",)


def test_counts_cover_exact_rendered_message_areas_and_total_payload() -> None:
    """Container tags or metadata omitted from accounting could exceed provider limits."""
    count = len
    context = ContextBuilder(
        control="SYSTEM🙂",
        limits=ContextLimits(
            max_control_tokens=100,
            max_task_tokens=100,
            max_memory_tokens=200,
            max_evidence_tokens=2_000,
            max_total_tokens=2_500,
        ),
        token_counter=count,
    ).build(
        task="任务🙂 uncommon-ASCII-~|^`",
        memory_hints=[MemoryHint(text="线索🙂", score=1.0)],
        evidence=[
            _evidence(
                "证据🙂-1",
                score=1.0,
                body="收入为 $44.10 于 2026-05-28 🙂",
            )
        ],
    )

    assert context.control_tokens == count(context.control)
    assert context.task_tokens == count(context.render_task())
    assert context.memory_tokens == count(context.render_memory())
    assert context.evidence_tokens == count(context.render_evidence())
    assert context.total_tokens == count(context.control) + count(context.render())
    assert context.task_tokens <= 100
    assert context.memory_tokens <= 200
    assert context.evidence_tokens <= 2_000
    assert context.total_tokens <= 2_500


def test_large_protected_metadata_is_counted_and_drops_whole_evidence_block() -> None:
    evidence = _evidence("metadata-heavy", score=1.0, body="support")
    evidence = evidence.model_copy(update={"source_url": "https://sec.gov/" + "x" * 800})
    context = ContextBuilder(
        control="system",
        limits=ContextLimits(max_evidence_tokens=200),
        token_counter=len,
    ).build(task="task", memory_hints=[], evidence=[evidence])

    assert context.evidence == ()
    assert context.dropped_evidence_ids == ("metadata-heavy",)
    assert context.evidence_tokens == len(context.render_evidence())
    assert "metadata-heavy" not in context.render()


@pytest.mark.parametrize(
    "text",
    [
        "纯中文证据边界",
        "emoji🙂🚀🧪",
        "uncommon ASCII ~|^`{}[]\\",
        "混合🙂 44.10 2026-05-28",
    ],
)
def test_lazy_tiktoken_counter_uses_exact_configured_model_encoding(
    text: str,
    tmp_path,
) -> None:
    calls: list[tuple[str, str]] = []
    payload = b"local encoding"
    asset = _token_asset(payload)
    _write_token_asset(tmp_path, asset, payload)

    class Encoding:
        def encode(self, value: str) -> list[int]:
            return list(value.encode("utf-8"))

    counter = LazyTiktokenTokenCounter(
        model_name="gpt-5",
        cache_dir=str(tmp_path),
        model_assets={"gpt-5": asset},
        encoding_factory=lambda selected, path: calls.append(
            (selected.encoding_name, str(path))
        )
        or Encoding(),
    )

    assert counter(text) == len(text.encode("utf-8"))
    assert counter(text) == len(text.encode("utf-8"))
    assert calls == [("test_encoding", str(tmp_path / sha1(asset.cache_url.encode()).hexdigest()))]


def test_lazy_tiktoken_counter_cache_miss_never_calls_network_capable_factory(
    tmp_path,
) -> None:
    calls: list[str] = []
    asset = _token_asset()

    def network_capable(selected: TokenEncodingAsset, path: object) -> object:
        del selected, path
        calls.append("network")
        raise AssertionError("cache miss reached network-capable resolver")

    counter = LazyTiktokenTokenCounter(
        model_name="gpt-5",
        cache_dir=str(tmp_path),
        model_assets={"gpt-5": asset},
        encoding_factory=network_capable,
    )

    with pytest.raises(TokenCounterError) as caught:
        counter("evidence")

    assert calls == []
    assert str(tmp_path) not in str(caught.value)
    assert "Prefetch" in caught.value.detail


def test_lazy_tiktoken_counter_rejects_unknown_model_before_cache_or_factory(
    tmp_path,
) -> None:
    calls: list[str] = []
    counter = LazyTiktokenTokenCounter(
        model_name="private/model/path",
        cache_dir=str(tmp_path),
        encoding_factory=lambda asset, path: calls.append("called"),
    )

    with pytest.raises(TokenCounterError) as caught:
        counter("evidence")

    assert calls == []
    assert "private/model/path" not in str(caught.value)
    assert "recognized OpenAI model" in caught.value.detail


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4o-mini-2024-07-18",
    ],
)
def test_local_tokenizer_recognizes_strict_dated_mini_nano_models(
    model_name: str,
    tmp_path,
) -> None:
    counter = LazyTiktokenTokenCounter(
        model_name=model_name,
        cache_dir=str(tmp_path),
        encoding_factory=lambda asset, path: pytest.fail("cache miss reached factory"),
    )

    with pytest.raises(TokenCounterError) as caught:
        counter("evidence")

    assert "Prefetch" in caught.value.detail
    assert "recognized OpenAI model" not in caught.value.detail


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini-latest",
        "gpt-4o-mini-2024-7-18",
        "gpt-4o-mini-2024-07-18-extra",
        "gpt-4o-micro-2024-07-18",
        "private/gpt-4o-mini-2024-07-18",
        "gpt-5-mini.fake",
    ],
)
def test_local_tokenizer_rejects_dated_mini_nano_near_misses(
    model_name: str,
    tmp_path,
) -> None:
    counter = LazyTiktokenTokenCounter(
        model_name=model_name,
        cache_dir=str(tmp_path),
        encoding_factory=lambda asset, path: pytest.fail("unknown model reached factory"),
    )

    with pytest.raises(TokenCounterError) as caught:
        counter("evidence")

    assert "recognized OpenAI model" in caught.value.detail
    assert model_name not in str(caught.value)


def test_lazy_tiktoken_counter_rejects_corrupt_cached_asset_safely(tmp_path) -> None:
    asset = _token_asset(b"expected")
    _write_token_asset(tmp_path, asset, b"corrupt")
    counter = LazyTiktokenTokenCounter(
        model_name="gpt-5",
        cache_dir=str(tmp_path),
        model_assets={"gpt-5": asset},
        encoding_factory=lambda selected, path: pytest.fail("corrupt cache reached factory"),
    )

    with pytest.raises(TokenCounterError) as caught:
        counter("evidence")

    assert str(tmp_path) not in str(caught.value)
    assert "hash" in caught.value.detail.lower()


def test_provider_independent_fallback_is_utf8_byte_upper_bound() -> None:
    text = "纯中文🙂🚀 uncommon~|^`"
    context = ContextBuilder(
        control=text,
        limits=ContextLimits(
            max_control_tokens=1_000,
            max_task_tokens=1_000,
            max_memory_tokens=1_000,
            max_evidence_tokens=1_000,
            max_total_tokens=5_000,
        ),
    ).build(task=text, memory_hints=[], evidence=[])

    assert context.control_tokens == len(text.encode("utf-8"))
    assert context.control_tokens >= len(text)
    assert context.total_tokens == len(context.control.encode("utf-8")) + len(
        context.render().encode("utf-8")
    )


def test_compression_changes_only_body_and_preserves_canonical_citation_metadata() -> None:
    """Compression must never rewrite source identity, numbers, dates, or raw boundaries."""
    original = "Revenue was $44.10 on 2026-05-28. TKN TKN TKN TKN TKN"
    compressed = "Revenue was $44.10 on 2026-05-28. TKN"
    compressor = _RecordingCompressor(compressed)
    context = ContextBuilder(
        control="control",
        compressor=compressor,
        limits=ContextLimits(max_evidence_tokens=2),
        token_counter=_marker_tokens,
    ).build(
        task="task",
        memory_hints=[],
        evidence=[_evidence("sec-1", score=1.0, body=original)],
    )

    assert len(context.evidence) == 1
    retained = context.evidence[0]
    assert retained.body == compressed
    assert retained.original_body == original
    assert retained.evidence_id == "sec-1"
    assert retained.ticker == "NVDA"
    assert retained.source_url == ("https://www.sec.gov/Archives/edgar/data/1045810/filing.htm")
    assert retained.date == "2026-05-28"
    assert retained.accession_no == "0001045810-26-000041"
    assert retained.section == "MD&A"
    assert (retained.raw_start, retained.raw_end) == (120, 240)
    assert compressor.calls == [(original, 2)]
    rendered = context.render()
    assert 'untrusted="true"' in rendered
    assert 'evidence_id="sec-1"' in rendered
    assert "$44.10" in rendered and "2026-05-28" in rendered
    assert rendered.count("<evidence ") == rendered.count("</evidence>") == 1


def test_compressed_body_that_loses_numeric_support_is_dropped() -> None:
    """Retaining unreadable or numerically altered compression would break grounding."""
    context = ContextBuilder(
        control="control",
        compressor=_RecordingCompressor("generic TKN"),
        limits=ContextLimits(max_evidence_tokens=2),
        token_counter=_marker_tokens,
    ).build(
        task="task",
        memory_hints=[],
        evidence=[
            _evidence(
                "sec-1",
                score=1.0,
                body="Revenue was $44.10 on 2026-05-28. TKN TKN TKN",
            )
        ],
    )

    assert context.evidence == ()
    assert context.dropped_evidence_ids == ("sec-1",)


@pytest.mark.parametrize(
    "compressed",
    [
        "Revenue improved $44.10 2026-05-28 margin 55% TKN",
        "Revenue $44.11 2026-05-28 margin 55% TKN",
        "Revenue $44.10 $44.10 2026-05-28 margin 55% TKN",
        "Revenue $44.10 margin 55% TKN",
        "margin 55% Revenue $44.10 2026-05-28 TKN",
        "Revenue $44.10 2026-05-29 margin 55% TKN",
    ],
    ids=["added", "changed-number", "repeated", "lost-date", "reordered", "date-mutation"],
)
def test_compression_must_be_ordered_extractive_and_preserve_all_numbers(
    compressed: str,
) -> None:
    original = "Revenue was $44.10 on 2026-05-28 and margin was 55%. TKN TKN TKN"
    context = ContextBuilder(
        control="control",
        compressor=_RecordingCompressor(compressed),
        limits=ContextLimits(max_evidence_tokens=2),
        token_counter=_marker_tokens,
    ).build(
        task="task",
        memory_hints=[],
        evidence=[_evidence("sec-1", score=1.0, body=original)],
    )

    assert context.evidence == ()
    assert context.dropped_evidence_ids == ("sec-1",)


def test_control_and_task_are_rejected_instead_of_truncated() -> None:
    """Truncating control or task constraints could remove a safety boundary."""
    with pytest.raises(ContextBudgetError, match="control"):
        ContextBuilder(
            control="TKN TKN",
            limits=ContextLimits(max_control_tokens=1),
            token_counter=_marker_tokens,
        )

    builder = ContextBuilder(
        control="control",
        limits=ContextLimits(max_task_tokens=1),
        token_counter=_marker_tokens,
    )
    with pytest.raises(ContextBudgetError, match="task"):
        builder.build(task="TKN TKN", memory_hints=[], evidence=[])


def test_lazy_llmlingua_compressor_defers_loading_and_types_missing_package() -> None:
    """Offline construction must not import or download the production compressor model."""
    calls: list[tuple[str, str | None]] = []

    def missing_factory(model_name: str, device_map: str | None) -> object:
        calls.append((model_name, device_map))
        raise ModuleNotFoundError("No module named 'llmlingua'")

    compressor = LazyLLMLinguaCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map="cpu",
        compressor_factory=missing_factory,
    )

    assert calls == []
    with pytest.raises(ContextCompressionError) as caught:
        compressor.compress("evidence body", max_tokens=10)

    assert caught.value.code is ContextCompressionErrorCode.PACKAGE_UNAVAILABLE
    assert "llmlingua" in caught.value.detail.lower()
    assert calls == [("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", "cpu")]


def test_default_llmlingua_load_resolves_only_approved_local_assets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fra.context as context_module

    model_path = tmp_path / "models" / "snapshot"
    model_path.mkdir(parents=True)
    tokenizer_path = tmp_path / "tokens" / "asset"
    tokenizer_path.parent.mkdir()
    tokenizer_path.write_bytes(b"verified-tokenizer")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        context_module,
        "_resolve_local_hf_snapshot",
        lambda model_name, cache_dir: (
            calls.append(("model", model_name, cache_dir)) or model_path
        ),
    )
    monkeypatch.setattr(
        context_module,
        "_validated_tokenizer_cache_path",
        lambda model_name, cache_dir: (
            calls.append(("tokenizer", model_name, cache_dir)) or tokenizer_path
        ),
    )

    class Delegate:
        def compress_prompt(self, *args: object, **kwargs: object) -> dict[str, str]:
            del args, kwargs
            return {"compressed_prompt": "retained evidence"}

    monkeypatch.setattr(
        context_module,
        "_build_llmlingua_compressor",
        lambda local_model_path, device_map, tokenizer_cache_dir: (
            calls.append(
                ("construct", local_model_path, device_map, tokenizer_cache_dir)
            )
            or Delegate()
        ),
    )
    compressor = LazyLLMLinguaCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        model_cache_dir=str(tmp_path / "models"),
        tokenizer_cache_dir=str(tmp_path / "tokens"),
    )

    assert compressor.compress("evidence", max_tokens=5) == "retained evidence"
    assert calls == [
        (
            "model",
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            str(tmp_path / "models"),
        ),
        ("tokenizer", "gpt-3.5-turbo", str(tmp_path / "tokens")),
        (
            "construct",
            model_path,
            "cpu",
            str(tmp_path / "tokens"),
        ),
    ]


def test_default_llmlingua_missing_local_assets_fails_typed_without_construction(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fra.context as context_module

    constructor_calls: list[object] = []
    monkeypatch.setattr(
        context_module,
        "_resolve_local_hf_snapshot",
        lambda model_name, cache_dir: (_ for _ in ()).throw(
            OSError("private missing cache path")
        ),
    )
    monkeypatch.setattr(
        context_module,
        "_build_llmlingua_compressor",
        lambda *args: constructor_calls.append(args),
    )
    compressor = LazyLLMLinguaCompressor(
        model_name="configured-model",
        model_cache_dir=str(tmp_path / "models"),
        tokenizer_cache_dir=str(tmp_path / "tokens"),
    )

    with pytest.raises(ContextCompressionError) as caught:
        compressor.compress("evidence", max_tokens=5)

    assert caught.value.code is ContextCompressionErrorCode.MODEL_UNAVAILABLE
    assert "private missing cache path" not in str(caught.value)
    assert constructor_calls == []


def test_huggingface_snapshot_resolution_is_explicitly_local_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fra.context as context_module

    cache_dir = tmp_path / "models"
    snapshot = cache_dir / "models--approved" / "snapshots" / "commit"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    resolved = context_module._resolve_local_hf_snapshot(  # noqa: SLF001
        "approved/model",
        str(cache_dir),
    )

    assert resolved == snapshot.resolve()
    assert calls == [
        {
            "repo_id": "approved/model",
            "cache_dir": str(cache_dir.resolve()),
            "local_files_only": True,
        }
    ]


def test_llmlingua_constructor_receives_local_path_and_offline_model_flags(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import sys
    from types import SimpleNamespace

    import fra.context as context_module

    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    tokenizer_cache = tmp_path / "tokens"
    tokenizer_cache.mkdir()
    calls: list[dict[str, object]] = []

    def constructor(**kwargs: object) -> object:
        calls.append({**kwargs, "tiktoken_cache": os.environ.get("TIKTOKEN_CACHE_DIR")})
        return object()

    monkeypatch.setitem(
        sys.modules,
        "llmlingua",
        SimpleNamespace(PromptCompressor=constructor),
    )

    context_module._build_llmlingua_compressor(  # noqa: SLF001
        model_path,
        "cpu",
        str(tokenizer_cache),
    )

    assert calls == [
        {
            "model_name": str(model_path),
            "device_map": "cpu",
            "model_config": {
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "use_llmlingua2": True,
            "tiktoken_cache": str(tokenizer_cache),
        }
    ]


def test_lazy_llmlingua_compressor_types_model_load_failure_without_private_detail() -> None:
    def unavailable_factory(model_name: str, device_map: str | None) -> object:
        del model_name, device_map
        raise OSError("private local model path")

    compressor = LazyLLMLinguaCompressor(
        model_name="missing-model",
        compressor_factory=unavailable_factory,
    )

    with pytest.raises(ContextCompressionError) as caught:
        compressor.compress("evidence body", max_tokens=10)

    assert caught.value.code is ContextCompressionErrorCode.MODEL_UNAVAILABLE
    assert "missing-model" not in caught.value.detail
    assert "private local model path" not in str(caught.value)


def test_llmlingua_runtime_programming_failure_is_not_reclassified() -> None:
    failure = RuntimeError("programming invariant failed")

    def broken_factory(model_name: str, device_map: str | None) -> object:
        del model_name, device_map
        raise failure

    compressor = LazyLLMLinguaCompressor(
        model_name="configured-model",
        compressor_factory=broken_factory,
    )

    with pytest.raises(RuntimeError) as caught:
        compressor.compress("evidence body", max_tokens=10)

    assert caught.value is failure


def test_llmlingua_compression_runtime_failure_is_not_reclassified() -> None:
    failure = RuntimeError("compression invariant failed")

    class Delegate:
        def compress_prompt(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise failure

    compressor = LazyLLMLinguaCompressor(
        model_name="configured-model",
        compressor_factory=lambda model_name, device_map: Delegate(),
    )

    with pytest.raises(RuntimeError) as caught:
        compressor.compress("evidence body", max_tokens=10)

    assert caught.value is failure


def test_lazy_llmlingua_compressor_uses_the_real_list_context_contract() -> None:
    """Passing a bare string would make LLMLingua treat characters as context items."""
    calls: list[tuple[list[str], int, bool]] = []

    class Delegate:
        def compress_prompt(
            self,
            context: list[str],
            *,
            target_token: int,
            force_reserve_digit: bool,
        ) -> dict[str, str]:
            calls.append((context, target_token, force_reserve_digit))
            return {"compressed_prompt": "retained support"}

    compressor = LazyLLMLinguaCompressor(
        model_name="cached-model",
        compressor_factory=lambda model_name, device_map: Delegate(),
    )

    assert compressor.compress("canonical evidence body", max_tokens=17) == (
        "retained support"
    )
    assert calls == [(["canonical evidence body"], 17, True)]


def test_budget_state_enforces_every_run_count_without_incrementing_after_exhaustion() -> None:
    """A rejected operation must not mutate counts or permit later budget overrun."""
    state = BudgetState(
        limits=BudgetLimits(
            max_planner_calls=1,
            max_analysis_calls=1,
            max_repair_calls=1,
            max_tool_calls=3,
            max_retrieval_rounds=2,
            max_web_calls=1,
        ),
        started_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    state.consume(planner_calls=1, tool_calls=2, retrieval_rounds=2, web_calls=1)
    state.consume(analysis_calls=1, repair_calls=1, tool_calls=1)
    before = state.model_dump()

    with pytest.raises(BudgetExhaustedError) as caught:
        state.consume(analysis_calls=1)

    assert caught.value.dimension == "analysis_calls"
    assert state.model_dump() == before


def test_aggregate_budget_has_one_shared_repair_allowance_for_all_policies() -> None:
    assert BudgetLimits.aggregate(P1_RECIPES[:2]).max_repair_calls == 1
    assert BudgetLimits.aggregate(()).max_repair_calls == 0


def test_hierarchical_budget_gate_is_atomic_under_concurrency() -> None:
    authority = BudgetAuthority()
    shared_limits = BudgetLimits(
        max_planner_calls=1,
        max_analysis_calls=0,
        max_repair_calls=0,
        max_tool_calls=0,
        max_retrieval_rounds=0,
        max_web_calls=0,
    )
    authority.configure(shared_limits)
    gates = [authority.child(shared_limits), authority.child(shared_limits)]
    barrier = Barrier(3)
    side_effects: list[int] = []

    def attempt(index: int) -> str:
        barrier.wait()
        try:
            gates[index].consume(planner_calls=1)
        except BudgetExhaustedError:
            return "exhausted"
        side_effects.append(index)
        return "success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, index) for index in range(2)]
        barrier.wait()
        results = [future.result() for future in futures]

    assert sorted(results) == ["exhausted", "success"]
    assert len(side_effects) == 1
    assert authority.state.planner_calls == 1
    assert sum(gate.local.planner_calls for gate in gates) == 1
    assert gates[0].shared is gates[1].shared is authority


def test_local_budget_rejection_rolls_back_shared_state() -> None:
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=1,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=0,
            max_retrieval_rounds=0,
            max_web_calls=0,
        )
    )
    gate = authority.child(
        BudgetLimits(
            max_planner_calls=0,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=0,
            max_retrieval_rounds=0,
            max_web_calls=0,
        )
    )

    with pytest.raises(BudgetExhaustedError) as caught:
        gate.consume(planner_calls=1)

    assert caught.value.dimension == "planner_calls"
    assert authority.state.planner_calls == 0
    assert gate.local.planner_calls == 0
