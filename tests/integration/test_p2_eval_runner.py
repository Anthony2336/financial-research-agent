"""Integration coverage for deterministic P2 evaluation and CLI wiring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from financial_evidence_agent import cli as cli_module
from financial_evidence_agent.application import ResearchCommand
from financial_evidence_agent.cli import app
from financial_evidence_agent.evals.p2_runner import (
    DeterministicP2ApplicationFactory,
    evaluate_p2_case,
    load_p2_eval_cases,
    run_p2_eval,
)

runner = CliRunner()


def test_run_p2_eval_is_deterministic_and_exercises_required_case_classes() -> None:
    dataset = Path("src/financial_evidence_agent/evals/p2_dataset.jsonl")

    summary = run_p2_eval(dataset)
    repeated = run_p2_eval(dataset)

    assert summary.model_dump_json(indent=2) == repeated.model_dump_json(indent=2)
    assert summary.case_count == 10
    assert summary.passed_count == 10
    assert summary.pass_rate == Decimal("1")
    assert len(summary.results) == summary.case_count

    results_by_id = {result.id: result for result in summary.results}
    assert results_by_id["market-complete"].actual_status == "completed"
    assert results_by_id["market-complete"].market_metadata_valid is True
    assert results_by_id["market-context-unknown"].neutral_causality_safe is True
    assert results_by_id["industry"].actual_recipe_names == ["industry_research"]
    assert results_by_id["quality-out-of-scope"].quality_decision_correct is True
    assert results_by_id["peer-partial"].actual_status == "partial"
    assert results_by_id["peer-not-comparable"].comparability_correct is True
    assert results_by_id["cross-ticker-attack"].cross_ticker_rejection_count == 1
    assert results_by_id["cross-ticker-attack"].cross_ticker_leakage_count == 0
    assert results_by_id["cross-ticker-attack"].passed is True
    assert results_by_id["advice-refusal"].actual_status == "refused"
    assert summary.metrics.intent_accuracy == Decimal("1")
    assert summary.metrics.recipe_accuracy == Decimal("1")
    assert summary.metrics.pass_rate == Decimal("1")
    assert summary.metrics.cross_ticker_leakage_count == 0
    assert summary.metrics.source_policy_violations == 0
    assert summary.metrics.budget_violations == 0


def test_eval_cli_suite_p2_emits_strict_json_and_string_decimal_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")

    result = runner.invoke(app, ["eval", "--suite", "p2"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["case_count"] == 10
    assert payload["passed_count"] == 10
    assert payload["metrics"]["intent_accuracy"] == "1"
    assert payload["metrics"]["recipe_accuracy"] == "1"
    assert payload["metrics"]["source_ref_validity"] == "1"
    assert payload["metrics"]["cross_ticker_leakage_count"] == 0
    assert payload["metrics"]["pass_rate"] == "1"
    attack = next(item for item in payload["results"] if item["id"] == "cross-ticker-attack")
    assert attack["cross_ticker_rejection_count"] == 1
    assert attack["cross_ticker_leakage_count"] == 0
    assert attack["passed"] is True
    assert payload["results"][0]["id"] == "market-complete"


def test_eval_cli_default_suite_preserves_legacy_fields_and_adds_p2_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-p0-key")

    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["case_count"] == 7
    assert payload["passed_count"] == 7
    assert payload["p1_case_count"] == 8
    assert payload["p1_passed_count"] == 8
    assert payload["p1_metrics"] == {
        "intent_accuracy": "1",
        "recipe_accuracy": "1",
        "facet_coverage": "0.6153846153846153846153846154",
        "citation_validity": "1",
        "unsupported_claim_rate": "0",
        "source_policy_violations": 0,
        "budget_violations": 0,
    }
    assert payload["p2_case_count"] == 10
    assert payload["p2_passed_count"] == 10
    assert payload["p2_metrics"]["intent_accuracy"] == "1"
    assert payload["p2_metrics"]["pass_rate"] == "1"
    assert len(payload["p2_results"]) == 10


def test_eval_cli_rejects_invalid_suite_dataset_combinations_before_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_dataset = tmp_path / "legacy.jsonl"
    legacy_dataset.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: pytest.fail("Settings() should not be constructed for invalid suite flags"),
    )

    result = runner.invoke(
        app,
        ["eval", "--suite", "p2", "--dataset", str(legacy_dataset)],
    )

    assert result.exit_code == 2
    assert "--dataset is only valid for the legacy P0/P1 dataset" in result.output


def test_eval_cli_p2_returns_one_after_printing_a_red_summary(tmp_path: Path) -> None:
    red_dataset = tmp_path / "red-p2.jsonl"
    source = Path("src/financial_evidence_agent/evals/p2_dataset.jsonl")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0]["expected_status"] = "failed"
    red_dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "--suite", "p2", "--p2-dataset", str(red_dataset)])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed_count"] < payload["case_count"]


def test_peer_eval_cases_record_real_p1_recipe_traces() -> None:
    cases = {
        case.id: case
        for case in load_p2_eval_cases(
            Path("src/financial_evidence_agent/evals/p2_dataset.jsonl")
        )
    }
    factory = DeterministicP2ApplicationFactory()

    partial = evaluate_p2_case(cases["peer-partial"], factory)
    not_comparable = evaluate_p2_case(cases["peer-not-comparable"], factory)

    partial_audit = factory.audit_for_case("peer-partial")
    comparable_audit = factory.audit_for_case("peer-not-comparable")
    assert partial.actual_status == "partial"
    assert not_comparable.actual_status == "completed"
    assert partial_audit is not None
    assert comparable_audit is not None
    assert _count_observations(partial_audit.trace_roots, "graph.p1_recipe") == 3
    assert _count_observations(comparable_audit.trace_roots, "graph.p1_recipe") == 2


def test_cross_ticker_attack_uses_real_quality_runtime_observations() -> None:
    case = next(
        case
        for case in load_p2_eval_cases(Path("src/financial_evidence_agent/evals/p2_dataset.jsonl"))
        if case.id == "cross-ticker-attack"
    )
    factory = DeterministicP2ApplicationFactory()
    application = factory.for_case(case)

    result = application.run(
        ResearchCommand(
            ticker=case.ticker,
            request=case.request,
            mode=case.mode,
            peer_tickers=case.peer_tickers,
            peer_scope=case.peer_scope,
            with_context=case.with_context,
        )
    )

    audit = factory.audit_for_case(case.id)
    assert audit is not None
    assert result.guarded_report is not None
    assert len(result.guarded_report.packages[0].claims) == 1
    assert any(
        "CLAIM_DROPPED[NVDA:1]: CROSS_TICKER_SOURCE_REJECTED" in error
        for error in result.guarded_report.guard_errors
    )
    names = _descendant_names(audit.trace_roots[0])
    assert "quality.screen" in names
    assert "quality.render" in names


def test_red_p2_eval_skips_langfuse_operations_after_printing_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    red_dataset = tmp_path / "red-p2.jsonl"
    source = Path("src/financial_evidence_agent/evals/p2_dataset.jsonl")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0]["expected_status"] = "failed"
    red_dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for name, value in (
        ("LANGFUSE_PUBLIC_KEY", "langfuse-public"),
        ("LANGFUSE_SECRET_KEY", "langfuse-secret"),
        ("LANGFUSE_HOST", "https://langfuse.invalid"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "financial_evidence_agent.observability.build_langfuse_operation_client",
        lambda settings: pytest.fail("red P2 eval must not construct a Langfuse client"),
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "--suite",
            "p2",
            "--p2-dataset",
            str(red_dataset),
            "--langfuse-dataset",
            "financial-evidence-p2",
            "--sync-langfuse-dataset",
            "--langfuse-experiment",
            "p2-red",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed_count"] < payload["case_count"]


def test_eval_cli_suite_p2_subprocess_stdout_is_byte_identical(tmp_path: Path) -> None:
    eval_script = Path(sys.executable).parent / "eval"
    environment = _offline_environment()
    environment["DATABASE_URL"] = f"sqlite+pysqlite:///{tmp_path / 'p2-eval.sqlite3'}"

    first = subprocess.run(
        [str(eval_script), "--suite", "p2"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )
    second = subprocess.run(
        [str(eval_script), "--suite", "p2"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert first.returncode == 0, first.stderr.decode()
    assert second.returncode == 0, second.stderr.decode()
    assert first.stdout == second.stdout


def _descendant_names(root: object) -> list[str]:
    names: list[str] = []
    pending = list(getattr(root, "children", []))
    while pending:
        child = pending.pop(0)
        names.append(getattr(child, "name"))
        pending.extend(getattr(child, "children", []))
    return names


def _count_observations(roots: list[object], name: str) -> int:
    return sum(_descendant_names(root).count(name) for root in roots)


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "OPENAI_API_KEY",
        "FAST_MODEL",
        "ANALYST_MODEL",
        "TAVILY_API_KEY",
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
    ):
        environment.pop(name, None)
    environment["UV_CACHE_DIR"] = "/tmp/financial-evidence-agent-uv-cache"
    return environment
