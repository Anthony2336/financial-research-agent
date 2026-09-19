"""Unit tests for offline reranker adapters."""

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

import fra.retrieval.rerank as rerank_module
from fra.domain import EvidenceChunk
from fra.retrieval.rerank import (
    FlashRankReranker,
    IdentityReranker,
    LazyFlashRankReranker,
    reserve_challenge_evidence,
)


def _chunk(
    chunk_id: str,
    content: str,
    *,
    section: str = "MD&A",
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
        section=section,
        raw_start=0,
        raw_end=len(content),
    )


class _FakeRanker:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def rerank(self, request: object) -> list[dict[str, str | float]]:
        self.requests.append(request)
        return [
            {"id": "beta", "text": "second passage", "score": 0.9},
            {"id": "alpha", "text": "first passage", "score": 0.2},
        ]


class _OrderedRanker:
    def __init__(self, *ids: str) -> None:
        self.ids = ids

    def rerank(self, request: object) -> list[dict[str, str | float]]:
        del request
        return [
            {"id": chunk_id, "text": chunk_id, "score": 1.0 - index / 10}
            for index, chunk_id in enumerate(self.ids)
        ]


def _accept_test_assets(model_name: str, cache_dir: str | None) -> None:
    del model_name, cache_dir


def _write_flashrank_cache(cache_dir: Path) -> Path:
    model_dir = cache_dir / "ms-marco-MiniLM-L-12-v2"
    model_dir.mkdir(parents=True)
    payloads = {
        "flashrank-MiniLM-L-12-v2_Q.onnx": b"locked onnx payload",
        "config.json": b'{"pad_token_id": 0}',
        "tokenizer_config.json": b'{"model_max_length": 512, "pad_token": "[PAD]"}',
        "special_tokens_map.json": b'{"unk_token": "[UNK]"}',
        "tokenizer.json": b'{"version": "1.0"}',
    }
    for name, payload in payloads.items():
        (model_dir / name).write_bytes(payload)
    (model_dir / ".financial-evidence-agent-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "ms-marco-MiniLM-L-12-v2",
                "files": {
                    name: sha256(payload).hexdigest()
                    for name, payload in payloads.items()
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model_dir


def test_identity_reranker_preserves_candidate_order_up_to_limit() -> None:
    """Identity fallback must retain deterministic P0 ordering without a model."""
    alpha = _chunk("alpha", "first passage")
    beta = _chunk("beta", "second passage")

    result = IdentityReranker().rerank(query="query", evidence=[alpha, beta], limit=1)

    assert result.evidence == [alpha]
    assert result.scores == {}


def test_identity_reranker_reserves_risk_factors_when_the_limit_would_omit_them() -> None:
    support = _chunk("support", "support passage")
    risk = _chunk("risk", "risk passage", section="Risk Factors")

    ranked = IdentityReranker().rerank(
        query="query",
        evidence=[support, risk],
        limit=2,
    )
    result = reserve_challenge_evidence(
        ranked.evidence,
        1,
        evidence_side="challenge",
    )

    assert result == [risk]


def test_flashrank_reranker_maps_ranked_passage_ids_to_original_chunks() -> None:
    """Returning copied or mismatched chunks would corrupt evidence provenance."""
    alpha = _chunk("alpha", "first passage")
    beta = _chunk("beta", "second passage")
    ranker = _FakeRanker()
    reranker = FlashRankReranker(ranker, model_name="test-model")

    result = reranker.rerank(query="market demand", evidence=[alpha, beta], limit=1)

    assert result.evidence == [beta]
    assert result.evidence[0] is beta
    assert result.scores == {"beta": 0.9}
    assert len(ranker.requests) == 1
    request = ranker.requests[0]
    assert request.query == "market demand"
    assert request.passages == [
        {"id": "alpha", "text": "first passage"},
        {"id": "beta", "text": "second passage"},
    ]


def test_flashrank_reserves_highest_ranked_in_scope_risk_without_exceeding_limit() -> None:
    support = _chunk("support", "top support")
    second = _chunk("second", "second support")
    lower_risk = _chunk("lower-risk", "lower risk", section="Risk Factors")
    highest_risk = _chunk("highest-risk", "highest risk", section="Risk Factors")
    reranker = FlashRankReranker(
        _OrderedRanker("support", "second", "highest-risk", "lower-risk"),
        model_name="test-model",
    )

    ranked = reranker.rerank(
        query="query",
        evidence=[support, second, lower_risk, highest_risk],
        limit=4,
    )
    result = reserve_challenge_evidence(
        ranked.evidence,
        2,
        evidence_side="challenge",
    )

    assert result == [support, highest_risk]
    assert ranked.scores["support"] == 1.0
    assert ranked.scores["highest-risk"] == 0.8


def test_flashrank_risk_reservation_avoids_duplicates_and_cross_scope_candidates() -> None:
    support = _chunk("support", "top support")
    same_scope_risk = _chunk("risk", "same scope risk", section="Risk Factors")
    cross_ticker_risk = _chunk(
        "amd-risk",
        "other issuer risk",
        section="Risk Factors",
        ticker="AMD",
        corpus_version="AMD-v1",
    )
    reranker = FlashRankReranker(
        _OrderedRanker("support", "support", "amd-risk", "risk"),
        model_name="test-model",
    )

    result = reranker.rerank(
        query="query",
        evidence=[support, support, cross_ticker_risk, same_scope_risk],
        limit=2,
    )

    assert result.evidence == [support, same_scope_risk]
    assert len({item.id for item in result.evidence}) == 2


def test_flashrank_preserves_order_when_no_eligible_risk_exists() -> None:
    first = _chunk("first", "first")
    second = _chunk("second", "second")
    reranker = FlashRankReranker(
        _OrderedRanker("second", "first"),
        model_name="test-model",
    )

    result = reranker.rerank(query="query", evidence=[first, second], limit=1)

    assert result.evidence == [second]


def test_lazy_flashrank_constructs_the_external_ranker_only_for_nonempty_evidence() -> None:
    """Eager construction would download model assets during ordinary runtime assembly."""
    alpha = _chunk("alpha", "first passage")
    ranker = _FakeRanker()
    calls: list[str] = []

    def factory(*, model_name: str, cache_dir: str | None) -> _FakeRanker:
        assert cache_dir is None
        calls.append(model_name)
        return ranker

    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=factory,
        asset_validator=_accept_test_assets,
    )

    assert reranker.version == "flashrank-risk-v2:test-model"
    assert reranker.rerank(query="query", evidence=[], limit=1).evidence == []
    assert calls == []

    result = reranker.rerank(query="query", evidence=[alpha], limit=1)

    assert result.evidence == [alpha]
    assert calls == ["test-model"]


def test_missing_flashrank_assets_fail_before_the_real_constructor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []

    def forbidden_constructor(**kwargs: object) -> object:
        constructor_calls.append(kwargs)
        raise AssertionError("missing assets reached FlashRank download boundary")

    monkeypatch.setattr("flashrank.Ranker", forbidden_constructor)
    reranker = LazyFlashRankReranker(
        "ms-marco-MiniLM-L-12-v2",
        cache_dir=str(tmp_path),
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError) as caught:
        reranker.rerank(query="query", evidence=[_chunk("alpha", "first")], limit=1)

    assert str(caught.value) == (
        "RERANKER_MODEL_UNAVAILABLE: configured local FlashRank assets are missing or "
        "corrupt; run the protected prefetch before production research."
    )
    assert constructor_calls == []


def test_missing_flashrank_assets_fail_before_an_injected_factory(
    tmp_path: Path,
) -> None:
    """An injected factory must not silently permit a model download."""

    class RecordingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, model_name: str, cache_dir: str | None) -> _FakeRanker:
            assert model_name == "ms-marco-MiniLM-L-12-v2"
            assert cache_dir == str(tmp_path)
            self.calls += 1
            return _FakeRanker()

    factory = RecordingFactory()
    reranker = LazyFlashRankReranker(
        "ms-marco-MiniLM-L-12-v2",
        cache_dir=str(tmp_path),
        ranker_factory=factory,
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError) as caught:
        reranker.rerank(query="query", evidence=[_chunk("alpha", "first")], limit=1)

    assert str(caught.value) == (
        "RERANKER_MODEL_UNAVAILABLE: configured local FlashRank assets are missing or "
        "corrupt; run the protected prefetch before production research."
    )
    assert factory.calls == 0


def test_falsey_injected_factory_is_used_when_assets_are_explicitly_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthiness must not replace an explicitly supplied callable factory."""
    fallback_calls = 0

    class FalseyFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, *, model_name: str, cache_dir: str | None) -> _OrderedRanker:
            assert model_name == "test-model"
            assert cache_dir is None
            self.calls += 1
            return _OrderedRanker("alpha")

    def fallback(*, model_name: str, cache_dir: str | None) -> _OrderedRanker:
        nonlocal fallback_calls
        del model_name, cache_dir
        fallback_calls += 1
        return _OrderedRanker("alpha")

    factory = FalseyFactory()
    monkeypatch.setattr(rerank_module, "_default_ranker_factory", fallback)
    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=factory,
        asset_validator=_accept_test_assets,
    )

    result = reranker.rerank(
        query="query",
        evidence=[_chunk("alpha", "first")],
        limit=1,
    )

    assert [item.id for item in result.evidence] == ["alpha"]
    assert factory.calls == 1
    assert fallback_calls == 0


def test_falsey_explicit_asset_validator_is_used_before_the_injected_factory() -> None:
    """An explicit callable validator remains a narrow injection seam even if falsey."""

    class FalseyValidator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def __bool__(self) -> bool:
            return False

        def __call__(self, model_name: str, cache_dir: str | None) -> None:
            self.calls.append((model_name, cache_dir))

    factory_calls = 0

    def factory(*, model_name: str, cache_dir: str | None) -> _OrderedRanker:
        nonlocal factory_calls
        assert model_name == "test-model"
        assert cache_dir is None
        factory_calls += 1
        return _OrderedRanker("alpha")

    validator = FalseyValidator()
    result = LazyFlashRankReranker(
        "test-model",
        ranker_factory=factory,
        asset_validator=validator,
    ).rerank(
        query="query",
        evidence=[_chunk("alpha", "first")],
        limit=1,
    )

    assert [item.id for item in result.evidence] == ["alpha"]
    assert validator.calls == [("test-model", None)]
    assert factory_calls == 1


def test_falsey_factory_without_validator_still_fails_before_factory(
    tmp_path: Path,
) -> None:
    """Explicit factory semantics must not weaken production asset validation."""

    class FalseyFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, *, model_name: str, cache_dir: str | None) -> _FakeRanker:
            del model_name, cache_dir
            self.calls += 1
            return _FakeRanker()

    factory = FalseyFactory()
    reranker = LazyFlashRankReranker(
        "ms-marco-MiniLM-L-12-v2",
        cache_dir=str(tmp_path),
        ranker_factory=factory,
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError):
        reranker.rerank(
            query="query",
            evidence=[_chunk("alpha", "first")],
            limit=1,
        )

    assert factory.calls == 0


def test_corrupt_flashrank_asset_fails_hash_validation_before_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_flashrank_cache(tmp_path)
    (model_dir / "flashrank-MiniLM-L-12-v2_Q.onnx").write_bytes(b"corrupt")
    constructor_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "flashrank.Ranker",
        lambda **kwargs: constructor_calls.append(kwargs),
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError):
        LazyFlashRankReranker(
            "ms-marco-MiniLM-L-12-v2",
            cache_dir=str(tmp_path),
        ).rerank(query="query", evidence=[_chunk("alpha", "first")], limit=1)

    assert constructor_calls == []


def test_validated_flashrank_cache_is_passed_to_the_locked_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_flashrank_cache(tmp_path)
    calls: list[dict[str, object]] = []

    def constructor(**kwargs: object) -> _OrderedRanker:
        calls.append(kwargs)
        return _OrderedRanker("alpha")

    monkeypatch.setattr("flashrank.Ranker", constructor)
    result = LazyFlashRankReranker(
        "ms-marco-MiniLM-L-12-v2",
        cache_dir=str(tmp_path),
    ).rerank(query="query", evidence=[_chunk("alpha", "first")], limit=1)

    assert [item.id for item in result.evidence] == ["alpha"]
    assert calls == [
        {
            "model_name": "ms-marco-MiniLM-L-12-v2",
            "cache_dir": str(tmp_path),
        }
    ]


def test_lazy_flashrank_wraps_factory_failure_in_actionable_typed_error() -> None:
    alpha = _chunk("alpha", "first passage")

    def unavailable(*, model_name: str, cache_dir: str | None) -> object:
        assert cache_dir is None
        raise OSError(f"private cache failure for {model_name}")

    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=unavailable,
        asset_validator=_accept_test_assets,
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError) as caught:
        reranker.rerank(query="query", evidence=[alpha], limit=1)

    assert str(caught.value) == (
        "RERANKER_MODEL_UNAVAILABLE: could not load FlashRank model test-model. "
        "prefetch or install the configured reranker assets before startup; see README.md."
    )
    assert "private cache failure" not in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_lazy_flashrank_wraps_constructed_ranker_execution_failure() -> None:
    alpha = _chunk("alpha", "first passage")

    class ConstructedRanker:
        def rerank(self, request):
            del request
            raise OSError("private ONNX execution detail")

    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=lambda *, model_name, cache_dir: ConstructedRanker(),
        asset_validator=_accept_test_assets,
    )

    with pytest.raises(rerank_module.RerankerModelUnavailableError) as caught:
        reranker.rerank(query="query", evidence=[alpha], limit=1)

    assert str(caught.value) == (
        "RERANKER_MODEL_UNAVAILABLE: could not run FlashRank model test-model. "
        "prefetch or install the configured reranker assets before startup; see README.md."
    )
    assert "private ONNX execution detail" not in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_lazy_flashrank_preserves_delegate_value_error() -> None:
    alpha = _chunk("alpha", "first passage")

    class ContractRejectingRanker:
        def rerank(self, request):
            del request
            raise ValueError("scoped evidence contract rejected")

    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=lambda *, model_name, cache_dir: ContractRejectingRanker(),
        asset_validator=_accept_test_assets,
    )

    with pytest.raises(ValueError, match="scoped evidence contract rejected"):
        reranker.rerank(query="query", evidence=[alpha], limit=1)


def test_lazy_flashrank_constructs_one_delegate_for_concurrent_first_use() -> None:
    """An unsynchronized first-use check could start two model downloads at once."""
    alpha = _chunk("alpha", "first passage")
    ranker = _FakeRanker()
    first_factory_entered = Event()
    second_factory_entered = Event()
    allow_factory_return = Event()
    call_lock = Lock()
    factory_calls = 0
    results = []

    def factory(*, model_name: str, cache_dir: str | None) -> _FakeRanker:
        nonlocal factory_calls
        assert model_name == "test-model"
        assert cache_dir is None
        with call_lock:
            factory_calls += 1
            if factory_calls == 1:
                first_factory_entered.set()
            else:
                second_factory_entered.set()
        assert allow_factory_return.wait(timeout=1)
        return ranker

    reranker = LazyFlashRankReranker(
        "test-model",
        ranker_factory=factory,
        asset_validator=_accept_test_assets,
    )

    def rerank_once() -> None:
        results.append(reranker.rerank(query="query", evidence=[alpha], limit=1).evidence)

    first = Thread(target=rerank_once)
    second = Thread(target=rerank_once)
    first.start()
    assert first_factory_entered.wait(timeout=1)
    second.start()
    try:
        assert not second_factory_entered.wait(timeout=0.2)
    finally:
        allow_factory_return.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert factory_calls == 1
    assert results == [[alpha], [alpha]]
