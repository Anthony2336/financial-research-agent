"""Unit coverage for deterministic P2 evaluation contracts."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from financial_evidence_agent.domain import Intent, SourceRef, SourceRefKind
from financial_evidence_agent.evals.p2_runner import (
    P2EvalCase,
    P2EvalResult,
    load_p2_eval_cases,
    score_p2_eval,
)
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    ResearchQualityDecision,
)


def _write_dataset(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


def _result(
    *,
    result_id: str = "market-complete",
    kind: str = "market_complete",
    actual_intent: Intent = Intent.MARKET_SNAPSHOT_REQUEST,
    actual_status: str = "completed",
    actual_recipe_names: list[str] | None = None,
    actual_source_refs: list[str] | None = None,
    intent_matched: bool = True,
    recipe_matched: bool = True,
    market_metadata_valid: bool | None = True,
    freshness_handled_correctly: bool | None = True,
    source_ref_valid: bool = True,
    cross_ticker_leakage_count: int = 0,
    comparability_correct: bool | None = None,
    quality_decision_correct: bool | None = None,
    neutral_causality_safe: bool | None = None,
    source_policy_violations: int = 0,
    budget_violations: int = 0,
    passed: bool = True,
) -> P2EvalResult:
    return P2EvalResult(
        id=result_id,
        kind=kind,
        actual_intent=actual_intent,
        actual_status=actual_status,
        actual_recipe_names=[] if actual_recipe_names is None else actual_recipe_names,
        actual_source_refs=[] if actual_source_refs is None else actual_source_refs,
        intent_matched=intent_matched,
        recipe_matched=recipe_matched,
        market_metadata_valid=market_metadata_valid,
        freshness_handled_correctly=freshness_handled_correctly,
        source_ref_valid=source_ref_valid,
        cross_ticker_leakage_count=cross_ticker_leakage_count,
        comparability_correct=comparability_correct,
        quality_decision_correct=quality_decision_correct,
        neutral_causality_safe=neutral_causality_safe,
        source_policy_violations=source_policy_violations,
        budget_violations=budget_violations,
        passed=passed,
    )


def _industry_case() -> dict[str, object]:
    return {
        "id": "industry",
        "kind": "industry",
        "ticker": "NVDA",
        "request": "Describe the accelerator industry.",
        "mode": "industry-research",
        "expected_intent": "industry_research_request",
        "expected_status": "completed",
        "expected_recipe_names": ["industry_research"],
        "expected_source_refs": ["NVDA:filing:sec-nvda-industry-research"],
    }


def test_p2_dataset_contains_required_case_classes() -> None:
    cases = load_p2_eval_cases(Path("src/financial_evidence_agent/evals/p2_dataset.jsonl"))

    assert {case.kind for case in cases} >= {
        "market_complete",
        "market_stale",
        "market_unavailable",
        "market_context_unknown",
        "industry",
        "quality_out_of_scope",
        "peer_partial",
        "peer_not_comparable",
        "cross_ticker_attack",
        "advice_refusal",
    }


def test_p2_metrics_count_cross_ticker_leakage_as_failure() -> None:
    metrics = score_p2_eval([_result(cross_ticker_leakage_count=1, passed=False)])

    assert metrics.cross_ticker_leakage_count == 1
    assert metrics.pass_rate == Decimal("0")


def test_p2_case_is_strict_and_rejects_invalid_decimal_missing_refs_and_network(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"line 2"):
        load_p2_eval_cases(
            _write_dataset(
                tmp_path / "unknown.jsonl",
                [
                    _industry_case(),
                    {**_industry_case(), "id": "bad-extra", "unexpected": True},
                ],
            )
        )

    with pytest.raises(ValueError, match=r"invalid Decimal"):
        load_p2_eval_cases(
            _write_dataset(
                tmp_path / "bad-decimal.jsonl",
                [
                    {
                        **_industry_case(),
                        "id": "bad-decimal",
                        "expected_market_price": "nan",
                    }
                ],
            )
        )

    with pytest.raises(ValueError, match=r"expected_source_refs"):
        load_p2_eval_cases(
            _write_dataset(
                tmp_path / "missing-refs.jsonl",
                [{**_industry_case(), "id": "missing-refs", "expected_source_refs": []}],
            )
        )

    with pytest.raises(ValueError, match=r"network"):
        load_p2_eval_cases(
            _write_dataset(
                tmp_path / "network.jsonl",
                [{**_industry_case(), "id": "network", "network_access": True}],
            )
        )

    duplicate_line = json.dumps(_industry_case())
    duplicate_dataset = tmp_path / "duplicate.jsonl"
    duplicate_dataset.write_text(f"{duplicate_line}\n{duplicate_line}\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="duplicate id 'industry' at line 2; first defined at line 1",
    ):
        load_p2_eval_cases(duplicate_dataset)

    assert P2EvalCase.model_config["extra"] == "forbid"


def test_p2_metrics_micro_average_only_applicable_case_checks() -> None:
    metrics = score_p2_eval(
        [
            _result(
                result_id="market-complete",
                kind="market_complete",
                market_metadata_valid=True,
                freshness_handled_correctly=True,
                neutral_causality_safe=None,
                comparability_correct=None,
                quality_decision_correct=None,
            ),
            _result(
                result_id="peer-not-comparable",
                kind="peer_not_comparable",
                actual_intent=Intent.INDUSTRY_RESEARCH_REQUEST,
                actual_recipe_names=["industry_research", "industry_research"],
                recipe_matched=False,
                market_metadata_valid=None,
                freshness_handled_correctly=None,
                comparability_correct=False,
                quality_decision_correct=None,
                neutral_causality_safe=None,
                passed=False,
            ),
            _result(
                result_id="quality",
                kind="quality_out_of_scope",
                actual_intent=Intent.RESEARCH_QUALITY_SCREEN_REQUEST,
                actual_status="completed",
                market_metadata_valid=None,
                freshness_handled_correctly=None,
                comparability_correct=None,
                quality_decision_correct=True,
                neutral_causality_safe=None,
            ),
        ]
    )

    assert metrics.intent_accuracy == Decimal("1")
    assert metrics.recipe_accuracy == Decimal("0.6666666666666666666666666667")
    assert metrics.market_metadata_validity == Decimal("1")
    assert metrics.freshness_handling_accuracy == Decimal("1")
    assert metrics.source_ref_validity == Decimal("1")
    assert metrics.comparability_accuracy == Decimal("0")
    assert metrics.quality_decision_accuracy == Decimal("1")
    assert metrics.neutral_causality_accuracy == Decimal("1")
    assert metrics.pass_rate == Decimal("0.6666666666666666666666666667")


def test_p2_result_and_metrics_json_dump_use_string_decimals() -> None:
    case = P2EvalCase(
        id="market-complete",
        kind="market_complete",
        ticker="NVDA",
        request="What is the current price?",
        mode="market-snapshot",
        expected_intent=Intent.MARKET_SNAPSHOT_REQUEST,
        expected_status="completed",
        expected_source_refs=[
            SourceRef(
                ticker="NVDA",
                kind=SourceRefKind.MARKET_SNAPSHOT,
                source_id="snapshot-nvda",
            )
        ],
        expected_market_price=Decimal("123.450000000000000001"),
        expected_quality_decision=ResearchQualityDecision.OUT_OF_SCOPE,
        expected_comparison_statuses=[ComparabilityStatus.NOT_COMPARABLE],
    )
    payload = case.model_dump(mode="json")

    assert payload["expected_market_price"] == "123.450000000000000001"
