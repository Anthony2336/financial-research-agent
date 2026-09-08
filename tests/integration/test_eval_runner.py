"""Deterministic, offline evaluation contract tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from financial_evidence_agent.cli import app
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    SourceKind,
    SourceTier,
)
from financial_evidence_agent.evals import runner as eval_runner
from financial_evidence_agent.evals.runner import (
    EvalCase,
    P1EvalCase,
    evaluate_p1_cases,
    load_eval_cases,
    load_p1_eval_cases,
    run_eval,
)
from financial_evidence_agent.graph.models import Dependencies, ResearchResult, SkillRunResult
from financial_evidence_agent.retrieval.collector import EvidenceCollector
from financial_evidence_agent.skills.models import ResearchFacet, SkillName
from financial_evidence_agent.web_evidence.source_policy import SourcePolicy

cli_runner = CliRunner()


@dataclass
class Recorder:
    mcp_calls: int = 0
    analyst_calls: int = 0
    trace_events: list[str] = field(default_factory=list)


def _chunk(chunk_id: str, content: str, *, section: str = "MD&A") -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker="NVDA",
        corpus_version="NVDA-v1",
        content=content,
        source_url="https://www.sec.gov/Archives/example.htm",
        form="10-Q",
        filed_at=date(2025, 5, 28),
        accession_no="0001045810-25-000041",
        section=section,
        raw_start=10,
        raw_end=10 + len(content),
    )


SUPPORT = _chunk("support", "Revenue grew because data center demand increased.")
COUNTER = _chunk(
    "counter",
    "Customer concentration may cause operating results to fluctuate.",
    section="Risk Factors",
)


class FakeMCPClient:
    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        assert name == "hybrid_search_filings"
        self._recorder.mcp_calls += 1
        return {"chunks": [SUPPORT.model_dump(), COUNTER.model_dump()], "error": None}


class FakeFastModel:
    def route(self, thesis: str) -> RouterDecision:
        return RouterDecision(intent=Intent.RESEARCH_REQUEST, reason=f"route {thesis[:10]}")

    def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
        return [
            ResearchQuestion(
                question=thesis,
                support_query=f"{ticker} revenue growth",
                challenge_query=f"{ticker} risk concentration",
                forms=["10-Q"],
            )
        ]


class PartiallyGroundedAnalyst:
    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        del evidence
        self._recorder.analyst_calls += 1
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=SUPPORT.content,
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[SUPPORT.id],
                )
            ],
            counter_claims=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text="An unsupported counter claim.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=["missing"],
                )
            ],
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


class MisfiledFactAnalyst:
    """Return a factual claim in a section where the guard must reject it."""

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        del evidence
        return ResearchMemo(
            research_question=questions[0].question,
            inferences=[
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=SUPPORT.content,
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[SUPPORT.id],
                )
            ],
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


class UnequalCaseSizeAnalyst:
    """Produce one versus three draft facts to distinguish micro from macro metrics."""

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        del evidence
        supporting = Claim(
            kind=ClaimKind.VERIFIED_FACT,
            text=SUPPORT.content,
            confidence=Confidence.HIGH,
            evidence_chunk_ids=[SUPPORT.id],
        )
        counter_claims = []
        if "large" in questions[0].question:
            counter_claims = [
                Claim(
                    kind=ClaimKind.VERIFIED_FACT,
                    text=f"Unsupported fact {index}.",
                    confidence=Confidence.HIGH,
                    evidence_chunk_ids=[f"missing-{index}"],
                )
                for index in range(2)
            ]
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[supporting],
            counter_claims=counter_claims,
            information_sufficiency="A",
            confidence=Confidence.HIGH,
        )


class RecordingTraceSink:
    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def record(self, node: str, attributes: dict[str, object]) -> None:
        del attributes
        self._recorder.trace_events.append(node)


def _write_dataset(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


def _research_case() -> dict[str, object]:
    return {
        "case_id": "partial-grounding",
        "ticker": "NVDA",
        "thesis": "Does revenue growth have evidence and material risks?",
        "expected_intent": "research_request",
        "expected_status": "completed",
        "expected_facets": ["supporting_evidence", "counter_evidence"],
    }


def test_bundled_dataset_has_six_unique_bilingual_p0_cases() -> None:
    path = Path("src/financial_evidence_agent/evals/dataset.jsonl")

    cases = load_eval_cases(path)

    assert len(cases) >= 6
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.expected_intent for case in cases} >= {
        Intent.RESEARCH_REQUEST,
        Intent.PROHIBITED_ADVICE,
    }
    assert sum(case.expected_intent is Intent.PROHIBITED_ADVICE for case in cases) >= 2
    research_cases = [case for case in cases if case.expected_intent is Intent.RESEARCH_REQUEST]
    assert len(research_cases) >= 4
    assert any("counter_evidence" in case.expected_facets for case in cases)
    assert any("insufficient_evidence" in case.expected_facets for case in cases)
    assert any(
        case.ticker == "NVDA" and "insufficient_evidence" in case.expected_facets for case in cases
    )
    assert any(
        any("\u4e00" <= character <= "\u9fff" for character in case.thesis)
        for case in research_cases
    )
    assert any(case.thesis.isascii() for case in research_cases)


def test_bundled_dataset_has_all_required_strict_p1_cases() -> None:
    path = Path("src/financial_evidence_agent/evals/dataset.jsonl")

    cases = load_p1_eval_cases(path)

    assert len(cases) == 8
    assert len({case.id for case in cases}) == len(cases)
    assert {case.id for case in cases} == {
        "p1-company-profile",
        "p1-earnings-review",
        "p1-financial-discrepancy",
        "p1-insufficient-local-evidence",
        "p1-successful-web-fallback",
        "p1-disallowed-domain-fallback",
        "p1-prohibited-advice",
        "p1-prompt-injection",
    }
    company = next(case for case in cases if case.id == "p1-company-profile")
    earnings = next(case for case in cases if case.id == "p1-earnings-review")
    assert company.expected_recipes == [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    assert earnings.expected_recipes == [
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    assert ResearchFacet.BULL_CASE in company.required_facets
    assert ResearchFacet.BEAR_CASE in earnings.required_facets
    assert P1EvalCase.model_config["extra"] == "forbid"


def test_fatal_p1_graph_result_cannot_satisfy_expected_partial_case() -> None:
    """Mapping a fatal graph status to partial would let infrastructure failure pass eval."""
    recipes = [
        SkillName.COMPANY_DEEP_RESEARCH,
        SkillName.MANAGEMENT_AND_GOVERNANCE_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    case = P1EvalCase(
        id="fatal-mutation",
        ticker="NVDA",
        request="介绍 NVDA 公司",
        expected_intent=Intent.COMPANY_PROFILE_REQUEST,
        expected_recipes=recipes,
        required_facets=[],
        expected_guard_result="partial",
    )
    fatal_result = ResearchResult(
        status="failed",
        ticker="NVDA",
        thesis=case.request,
        decision=RouterDecision(
            intent=Intent.COMPANY_PROFILE_REQUEST,
            reason="deterministic test result",
        ),
        skill_runs=[
            SkillRunResult(
                run_id=f"fatal-{index}",
                recipe_name=recipe,
                recipe_version="1.0.0",
                allowed_tools=(),
                status="failed",
            )
            for index, recipe in enumerate(recipes)
        ],
        rendered_output="P1 execution failed.",
    )

    evaluated = eval_runner._evaluate_p1_case(case, fatal_result)

    assert evaluated.actual_guard_result == "fail"
    assert evaluated.passed is False


def test_disallowed_domain_case_crosses_real_policy_boundary_without_persistence() -> None:
    """A fabricated empty bundle would not prove the production URL policy was consulted."""
    cases = load_p1_eval_cases(Path("src/financial_evidence_agent/evals/dataset.jsonl"))
    case = next(item for item in cases if item.id == "p1-disallowed-domain-fallback")

    evaluated = evaluate_p1_cases([case])[0]

    assert evaluated.web_policy_rejections == 3
    assert evaluated.persisted_source_ids == []
    assert evaluated.citation_source_ids == []
    assert evaluated.resolved_source_ids == []
    assert "example.com" not in evaluated.model_dump_json()


def test_disallowed_domain_eval_fails_if_source_policy_is_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval must expose rather than bless a forbidden candidate accepted by mutation."""
    cases = load_p1_eval_cases(Path("src/financial_evidence_agent/evals/dataset.jsonl"))
    case = next(item for item in cases if item.id == "p1-disallowed-domain-fallback")

    def bypass_policy(self, *, ticker, url):
        del self, ticker, url
        return SourceKind.AUTHORITATIVE_WEB, SourceTier.AUTHORITATIVE_SECONDARY

    monkeypatch.setattr(SourcePolicy, "classify", bypass_policy)

    evaluated = evaluate_p1_cases([case])[0]

    assert evaluated.web_policy_rejections == 0
    assert evaluated.persisted_source_ids
    assert evaluated.source_policy_violations > 0
    assert evaluated.passed is False


def test_p1_eval_uses_real_collector_and_authoritative_source_resolution() -> None:
    """Synthetic coverage flags or unpersisted evidence must not satisfy the eval."""
    cases = load_p1_eval_cases(Path("src/financial_evidence_agent/evals/dataset.jsonl"))
    case = next(item for item in cases if item.id == "p1-earnings-review")
    dependencies, audit = eval_runner._offline_p1_dependencies(case)

    assert isinstance(dependencies.skill_collector, EvidenceCollector)
    graph_result = eval_runner.run_research(case.ticker, case.request, dependencies)
    evaluated = eval_runner._evaluate_p1_case(case, graph_result, audit=audit)
    assert evaluated.passed is True
    assert set(evaluated.citation_source_ids) <= set(audit.authoritative_source_ids)

    audit.delete_authoritative_source(evaluated.citation_source_ids[0])
    mutated = eval_runner._evaluate_p1_case(case, graph_result, audit=audit)

    assert mutated.valid_citation_count < mutated.verified_claim_count
    assert mutated.passed is False


def test_eval_case_is_strict_and_jsonl_errors_include_line_number(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 2"):
        load_eval_cases(
            _write_dataset(
                tmp_path / "invalid.jsonl",
                [
                    _research_case(),
                    {**_research_case(), "case_id": "bad", "unexpected": True},
                ],
            )
        )

    duplicate_dataset = tmp_path / "duplicates.jsonl"
    duplicate_line = json.dumps(_research_case())
    duplicate_dataset.write_text(
        f"{duplicate_line}\n\n\n{duplicate_line}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=("duplicate case_id 'partial-grounding' at line 4; first defined at line 1"),
    ):
        load_eval_cases(duplicate_dataset)

    empty_dataset = tmp_path / "empty.jsonl"
    empty_dataset.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_eval_cases(empty_dataset)

    assert EvalCase.model_config["extra"] == "forbid"


def test_runner_reports_honest_draft_denominator_metrics_and_eval_trace(tmp_path: Path) -> None:
    recorder = Recorder()
    dataset = _write_dataset(
        tmp_path / "one.jsonl",
        [{**_research_case(), "expected_status": "insufficient_evidence"}],
    )
    dependencies = Dependencies(
        mcp_client=FakeMCPClient(recorder),
        fast_model=FakeFastModel(),
        analyst_model=PartiallyGroundedAnalyst(recorder),
        trace_sink=RecordingTraceSink(recorder),
    )

    summary = run_eval(dataset, dependencies)

    assert summary.case_count == 1
    assert summary.passed_count == 0
    assert summary.pass_rate == 0.0
    assert summary.citation_validity == 0.5
    assert summary.claim_supported == 0.5
    assert summary.claim_supported_label == "deterministic_extractive_proxy"
    assert summary.counterevidence_present == 0.0
    assert summary.coverage == pytest.approx(2 / 3)
    assert summary.cost_usd == 0.0
    assert summary.cost_source == "deterministic_offline"
    assert len(summary.results) == 1
    result = summary.results[0]
    assert result.citation_validity == 0.5
    assert result.claim_supported == 0.5
    assert result.counterevidence_present is False
    assert result.coverage == pytest.approx(2 / 3)
    assert result.passed is False
    assert recorder.mcp_calls == 2
    assert recorder.analyst_calls == 1
    assert recorder.trace_events[-2:] == ["evaluation_case", "evaluation_summary"]


def test_advice_case_has_null_factual_metrics_and_never_calls_tools(tmp_path: Path) -> None:
    recorder = Recorder()
    dataset = _write_dataset(
        tmp_path / "advice.jsonl",
        [
            {
                "case_id": "advice-en",
                "ticker": "NVDA",
                "thesis": "Should I buy NVDA now?",
                "expected_intent": "prohibited_advice",
                "expected_status": "refused",
                "expected_facets": ["refusal"],
            }
        ],
    )

    summary = run_eval(
        dataset,
        Dependencies(
            mcp_client=FakeMCPClient(recorder),
            fast_model=FakeFastModel(),
            analyst_model=PartiallyGroundedAnalyst(recorder),
            trace_sink=RecordingTraceSink(recorder),
        ),
    )

    assert summary.passed_count == 1
    assert summary.citation_validity is None
    assert summary.claim_supported is None
    assert summary.counterevidence_present is None
    assert summary.results[0].citation_validity is None
    assert summary.results[0].claim_supported is None
    assert recorder.mcp_calls == 0
    assert recorder.analyst_calls == 0


def test_factual_claim_in_wrong_draft_section_remains_in_metric_denominator(
    tmp_path: Path,
) -> None:
    """A guard-dropped fact must not disappear from the reported draft-quality metric."""
    recorder = Recorder()
    dataset = _write_dataset(
        tmp_path / "misfiled.jsonl",
        [{**_research_case(), "expected_status": "insufficient_evidence"}],
    )

    summary = run_eval(
        dataset,
        Dependencies(
            mcp_client=FakeMCPClient(recorder),
            fast_model=FakeFastModel(),
            analyst_model=MisfiledFactAnalyst(),
        ),
    )

    assert summary.citation_validity == 0.0
    assert summary.claim_supported == 0.0
    assert summary.results[0].passed is False


def test_summary_uses_micro_denominators_for_claims_and_coverage(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path / "unequal.jsonl",
        [
            {
                "case_id": "small",
                "ticker": "NVDA",
                "thesis": "small revenue growth evidence risk",
                "expected_intent": "research_request",
                "expected_status": "insufficient_evidence",
                "expected_facets": ["supporting_evidence"],
            },
            {
                "case_id": "large",
                "ticker": "NVDA",
                "thesis": "large revenue growth evidence risk",
                "expected_intent": "research_request",
                "expected_status": "completed",
                "expected_facets": [
                    "supporting_evidence",
                    "counter_evidence",
                    "refusal",
                ],
            },
        ],
    )
    recorder = Recorder()

    summary = run_eval(
        dataset,
        Dependencies(
            mcp_client=FakeMCPClient(recorder),
            fast_model=FakeFastModel(),
            analyst_model=UnequalCaseSizeAnalyst(),
        ),
    )

    assert [result.citation_validity for result in summary.results] == [1.0, 1 / 3]
    assert [result.claim_supported for result in summary.results] == [1.0, 1 / 3]
    assert [result.coverage for result in summary.results] == [1.0, 0.25]
    assert summary.citation_validity == 0.5  # (1 + 1) / (1 + 3)
    assert summary.claim_supported == 0.5  # (1 + 1) / (1 + 3)
    assert summary.coverage == 0.5  # (2 + 1) / (2 + 4)


def test_eval_cli_runs_bundled_dataset_through_fixture_mcp_and_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance command must emit parseable local JSON without provider calls."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-p0-key")

    result = cli_runner.invoke(
        app,
        ["eval", "--dataset", "src/financial_evidence_agent/evals/dataset.jsonl"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["case_count"] == 7
    assert payload["passed_count"] == 7
    assert payload["pass_rate"] == 1.0
    assert payload["citation_validity"] == 1.0
    assert payload["claim_supported"] == 1.0
    assert payload["claim_supported_label"] == "deterministic_extractive_proxy"
    assert payload["counterevidence_present"] == 1.0
    assert payload["coverage"] == 1.0
    assert payload["cost_usd"] == 0.0
    assert payload["cost_source"] == "deterministic_offline"
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
    assert {item["actual_status"] for item in payload["results"]} >= {
        "completed",
        "insufficient_evidence",
        "refused",
    }
    irrelevant = next(
        item for item in payload["results"] if item["case_id"] == "same-ticker-irrelevant-thesis"
    )
    assert irrelevant["actual_status"] == "insufficient_evidence"
    assert irrelevant["passed"] is True

    repeated = cli_runner.invoke(
        app,
        ["eval", "--dataset", "src/financial_evidence_agent/evals/dataset.jsonl"],
    )
    assert repeated.exit_code == 0, repeated.output
    assert repeated.output == result.output
    assert Decimal(payload["p1_metrics"]["citation_validity"]) == Decimal("1")
    for p1_result in payload["p1_results"]:
        if p1_result["verified_claim_count"] == 0:
            continue
        assert p1_result["citation_source_ids"]
        assert set(p1_result["citation_source_ids"]) <= set(
            p1_result["resolved_source_ids"]
        )


def test_eval_cli_returns_one_after_printing_a_red_legacy_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    red_dataset = _write_dataset(
        tmp_path / "red.jsonl",
        [{**_research_case(), "expected_intent": "prohibited_advice"}],
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-p0-key")

    result = cli_runner.invoke(app, ["eval", "--dataset", str(red_dataset)])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed_count"] < payload["case_count"]


def test_red_all_suite_prints_json_before_settings_or_langfuse_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import financial_evidence_agent.cli as cli_module
    import financial_evidence_agent.evals.runner as runner_module
    import financial_evidence_agent.observability as observability_module

    red_dataset = _write_dataset(
        tmp_path / "red-all.jsonl",
        [{**_research_case(), "expected_intent": "prohibited_advice"}],
    )
    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: pytest.fail("red all-suite eval constructed Settings"),
    )
    monkeypatch.setattr(
        observability_module,
        "build_langfuse_operation_client",
        lambda settings: pytest.fail(f"red all-suite eval constructed client: {settings}"),
    )
    monkeypatch.setattr(
        observability_module,
        "sync_langfuse_dataset",
        lambda *args: pytest.fail(f"red all-suite eval synchronized dataset: {args}"),
    )
    monkeypatch.setattr(
        runner_module,
        "run_comparable_application_experiment",
        lambda *args, **kwargs: pytest.fail(
            f"red all-suite eval ran experiment: {args!r} {kwargs!r}"
        ),
    )

    result = cli_runner.invoke(
        app,
        [
            "eval",
            "--dataset",
            str(red_dataset),
            "--langfuse-dataset",
            "financial-evidence-red",
            "--sync-langfuse-dataset",
            "--langfuse-experiment",
            "must-not-run",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed_count"] < payload["case_count"]
    assert payload["p2_passed_count"] == payload["p2_case_count"]


def test_eval_cli_langfuse_experiment_is_opt_in_and_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "financial_evidence_agent.observability.import_module",
        lambda name: pytest.fail(f"Langfuse client construction should not run: {name}"),
    )

    result = cli_runner.invoke(
        app,
        [
            "eval",
            "--suite",
            "p2",
            "--langfuse-dataset",
            "financial-evidence-p2",
            "--langfuse-experiment",
            "p2-local",
        ],
    )

    assert result.exit_code == 2
    assert "complete Langfuse credentials" in result.output


def test_eval_cli_routes_all_suites_to_the_comparable_application_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import financial_evidence_agent.evals.runner as runner_module
    import financial_evidence_agent.observability as observability_module

    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-p0-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "langfuse-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "langfuse-secret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.invalid")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(observability_module, "build_langfuse_operation_client", lambda _: object())
    monkeypatch.setattr(observability_module, "sync_langfuse_dataset", lambda *_: None)
    monkeypatch.setattr(
        runner_module,
        "run_comparable_application_experiment",
        lambda _, *, dataset_name, experiment_name, cases: calls.append(
            (dataset_name, experiment_name)
        ),
    )

    result = cli_runner.invoke(
        app,
        [
            "eval",
            "--langfuse-dataset",
            "financial-evidence-canonical",
            "--sync-langfuse-dataset",
            "--langfuse-experiment",
            "canonical-local",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("financial-evidence-canonical", "canonical-local")]


def test_eval_cli_reports_safe_error_when_langfuse_client_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'eval.sqlite3'}")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "langfuse-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "langfuse-secret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.invalid")
    monkeypatch.setattr(
        "financial_evidence_agent.observability.import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"broken import: {name}")),
    )

    result = cli_runner.invoke(
        app,
        [
            "eval",
            "--suite",
            "p2",
            "--langfuse-dataset",
            "financial-evidence-p2",
            "--sync-langfuse-dataset",
        ],
    )

    assert result.exit_code == 2
    assert "LANGFUSE_EXPORT_FAILED: failed to construct Langfuse client" in result.output
    assert "langfuse-secret" not in result.output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["eval", "--suite", "p2", "--sync-langfuse-dataset"],
            "--sync-langfuse-dataset requires --langfuse-dataset",
        ),
        (
            ["eval", "--suite", "p2", "--langfuse-experiment", "p2-local"],
            "--langfuse-experiment requires --langfuse-dataset",
        ),
        (
            ["eval", "--suite", "p2", "--langfuse-dataset", "financial-evidence-p2"],
            "--langfuse-dataset requires a sync and/or experiment flag",
        ),
    ],
)
def test_eval_cli_rejects_invalid_langfuse_flag_combinations_before_settings(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import financial_evidence_agent.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: pytest.fail("Settings() should not be constructed for invalid Langfuse flags"),
    )

    result = cli_runner.invoke(app, argv)

    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            [
                "eval",
                "--suite",
                "p2",
                "--langfuse-dataset",
                "   ",
                "--sync-langfuse-dataset",
            ],
            "--langfuse-dataset must not be blank",
        ),
        (
            [
                "eval",
                "--suite",
                "p2",
                "--langfuse-dataset",
                "financial-evidence-p2",
                "--langfuse-experiment",
                "   ",
            ],
            "--langfuse-experiment must not be blank",
        ),
    ],
)
def test_eval_cli_rejects_blank_langfuse_names_before_settings(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import financial_evidence_agent.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: pytest.fail("Settings() should not be constructed for blank Langfuse names"),
    )

    result = cli_runner.invoke(app, argv)

    assert result.exit_code == 2
    assert message in result.output
