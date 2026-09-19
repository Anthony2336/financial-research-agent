"""Reproducible P0 evaluation over the controlled research workflow."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, HttpUrl, ValidationError, field_validator

from fra.context import MemoryHint
from fra.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    StrictModel,
    WebEvidence,
)
from fra.evals.p2_runner import P2EvalMetrics, P2EvalResult
from fra.graph.models import (
    Dependencies,
    ResearchResult,
    SkillAnalysisInput,
    SkillPlanningInput,
)
from fra.graph.workflow import run_research
from fra.observability import LangfuseOperationError
from fra.retrieval.collector import (
    EvidenceBundle,
    EvidenceCollector,
    LocalEvidenceHit,
    LocalSearchResponse,
    WebEvidenceHit,
    WebSearchRequest,
    WebSearchResponse,
)
from fra.retrieval.coverage import (
    EvidenceSide,
    source_allowed_for_recipe,
)
from fra.skills.models import ResearchFacet, SkillName
from fra.skills.recipes import RECIPES
from fra.skills.schemas import (
    FinancialDataPoint,
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)
from fra.web_evidence.gateway import (
    AllowlistedWebGateway,
    WebGatewayError,
)
from fra.web_evidence.providers import RawSearchHit
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    PolicyValidatedWebEvidence,
    SourcePolicy,
)

EvalStatus = Literal["completed", "refused", "declined", "insufficient_evidence"]
ExpectedFacet = Literal[
    "supporting_evidence",
    "counter_evidence",
    "insufficient_evidence",
    "refusal",
]
_PROXY_LABEL = "deterministic_extractive_proxy"
_COST_SOURCE = "deterministic_offline"


class EvalCase(StrictModel):
    """One expected P0 behavior loaded from a JSONL dataset."""

    case_id: str = Field(min_length=1, max_length=100)
    ticker: str = Field(min_length=1, max_length=10)
    thesis: str = Field(min_length=1, max_length=2_000)
    expected_intent: Intent
    expected_status: EvalStatus
    expected_facets: list[ExpectedFacet] = Field(min_length=1)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("case_id must not contain whitespace")
        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.upper()

    @field_validator("expected_facets")
    @classmethod
    def unique_facets(cls, values: list[ExpectedFacet]) -> list[ExpectedFacet]:
        if len(values) != len(set(values)):
            raise ValueError("expected_facets must be unique")
        return values


class EvalCaseResult(StrictModel):
    """Measured behavior for one evaluation case."""

    case_id: str
    expected_intent: Intent
    actual_intent: Intent
    expected_status: EvalStatus
    actual_status: EvalStatus
    citation_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_supported: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_supported_label: Literal["deterministic_extractive_proxy"] = _PROXY_LABEL
    counterevidence_present: bool
    coverage: float = Field(ge=0.0, le=1.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_source: Literal["deterministic_offline"] = _COST_SOURCE
    passed: bool


P1GuardResult = Literal["pass", "partial", "refuse"]
P1ObservedGuardResult = Literal["pass", "partial", "refuse", "fail"]


class P1EvalCase(StrictModel):
    """One local P1 routing, recipe, grounding, and policy expectation."""

    id: str = Field(min_length=1, max_length=100)
    ticker: str = Field(min_length=1, max_length=10)
    request: str = Field(min_length=1, max_length=2_000)
    expected_intent: Intent
    expected_recipes: list[SkillName]
    required_facets: list[ResearchFacet]
    expected_guard_result: P1GuardResult

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("id must not contain whitespace")
        return value

    @field_validator("ticker")
    @classmethod
    def normalize_p1_ticker(cls, value: str) -> str:
        return value.upper()

    @field_validator("expected_recipes", "required_facets")
    @classmethod
    def unique_p1_values(cls, values: list[Any]) -> list[Any]:
        if len(values) != len(set(values)):
            raise ValueError("P1 expected lists must contain unique values")
        return values


class P1EvalResult(StrictModel):
    """One deterministic local P1 contract-probe result."""

    id: str
    actual_intent: Intent
    actual_recipes: list[SkillName]
    covered_facets: list[ResearchFacet]
    actual_guard_result: P1ObservedGuardResult
    verified_claim_count: int = Field(ge=0)
    valid_citation_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    citation_source_ids: list[str]
    resolved_source_ids: list[str]
    web_policy_rejections: int = Field(ge=0)
    persisted_source_ids: list[str]
    source_policy_violations: int = Field(ge=0)
    budget_violations: int = Field(ge=0)
    passed: bool


class P1EvalMetrics(StrictModel):
    """Aggregate P1 metrics computed from local, provider-free observations."""

    intent_accuracy: Decimal = Field(ge=0, le=1)
    recipe_accuracy: Decimal = Field(ge=0, le=1)
    facet_coverage: Decimal = Field(ge=0, le=1)
    citation_validity: Decimal = Field(ge=0, le=1)
    unsupported_claim_rate: Decimal = Field(ge=0, le=1)
    source_policy_violations: int = Field(ge=0)
    budget_violations: int = Field(ge=0)


class EvalSummary(StrictModel):
    """Aggregate metrics plus complete per-case local results."""

    dataset_path: str
    case_count: int = Field(ge=1)
    passed_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    citation_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_supported: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_supported_label: Literal["deterministic_extractive_proxy"] = _PROXY_LABEL
    counterevidence_present: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_source: Literal["deterministic_offline"] = _COST_SOURCE
    results: list[EvalCaseResult]
    p1_case_count: int = Field(default=0, ge=0)
    p1_passed_count: int = Field(default=0, ge=0)
    p1_metrics: P1EvalMetrics | None = None
    p1_results: list[P1EvalResult] = Field(default_factory=list)
    p2_case_count: int = Field(default=0, ge=0)
    p2_passed_count: int = Field(default=0, ge=0)
    p2_metrics: P2EvalMetrics | None = None
    p2_results: list[P2EvalResult] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ComparableApplicationExperimentCase:
    """One canonical offline case executed through its real application boundary."""

    id: str
    suite: Literal["p0", "p1", "p2"]
    input: dict[str, object]
    expected_output: dict[str, object]
    metadata: dict[str, object]
    execute: Callable[[], dict[str, object]]

    def input_payload(self) -> dict[str, object]:
        return dict(self.input)

    def expected_payload(self) -> dict[str, object]:
        return dict(self.expected_output)

    def metadata_payload(self) -> dict[str, object]:
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class _RatioCounts:
    numerator: int
    denominator: int

    @property
    def ratio(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True, slots=True)
class _EvaluatedCase:
    result: EvalCaseResult
    citation_validity: _RatioCounts
    claim_supported: _RatioCounts
    coverage: _RatioCounts


def load_comparable_application_experiment_cases(
    dataset_path: Path,
    p2_dataset_path: Path,
) -> list[ComparableApplicationExperimentCase]:
    """Load the fixed cross-suite cases required for one comparable offline experiment."""

    from fra.evals.p2_runner import load_p2_eval_cases

    p0_by_id = {case.case_id: case for case in load_eval_cases(dataset_path)}
    p1_by_id = {case.id: case for case in load_p1_eval_cases(dataset_path)}
    p2_by_id = {case.id: case for case in load_p2_eval_cases(p2_dataset_path)}
    try:
        p0_cases = [
            p0_by_id["normal-research-en"],
            p0_by_id["counter-evidence-priority"],
            p0_by_id["insufficient-evidence"],
            p0_by_id["prohibited-advice-en"],
        ]
        p1_case = p1_by_id["p1-company-profile"]
        p2_cases = [p2_by_id["market-complete"], p2_by_id["cross-ticker-attack"]]
    except KeyError as error:
        raise ValueError(
            f"canonical comparable experiment case is missing: {error.args[0]}"
        ) from error

    cases = [_comparable_p0_case(case) for case in p0_cases]
    cases.append(_comparable_p1_case(p1_case))
    cases.extend(_comparable_p2_case(case) for case in p2_cases)
    return cases


def run_comparable_application_experiment(
    client: object,
    *,
    dataset_name: str,
    experiment_name: str,
    cases: Sequence[ComparableApplicationExperimentCase],
) -> object:
    """Run canonical P0/P1/P2 cases through applications with deterministic scores."""

    case_by_id = {case.id: case for case in cases}
    if len(case_by_id) < 6 or len(case_by_id) != len(cases):
        raise LangfuseOperationError("comparable experiments require at least six unique cases")
    try:
        dataset = getattr(client, "get_dataset")(dataset_name)
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to fetch Langfuse dataset '{dataset_name}'"
        ) from error
    _validate_comparable_dataset_items(getattr(dataset, "items", None), case_by_id)
    try:
        result = getattr(dataset, "run_experiment")(
            name=experiment_name,
            task=lambda *, item, **_: _run_comparable_application_case(item, case_by_id),
            evaluators=[_comparable_overall_pass_evaluator],
            max_concurrency=1,
            metadata={
                "suite": "p0-p1-p2",
                "source": "canonical-jsonl",
                "score_source": "deterministic_offline",
            },
        )
        _validate_comparable_experiment_result(result, case_by_id)
        return result
    except LangfuseOperationError:
        raise
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to run Langfuse experiment '{experiment_name}' for dataset '{dataset_name}'"
        ) from error


def _comparable_p0_case(case: EvalCase) -> ComparableApplicationExperimentCase:
    from fra.contracts import ResearchCommand, ResearchMode

    command = ResearchCommand(ticker=case.ticker, request=case.thesis, mode=ResearchMode.THESIS)
    return ComparableApplicationExperimentCase(
        id=f"p0:{case.case_id}",
        suite="p0",
        input=command.model_dump(mode="json"),
        expected_output={
            "intent": case.expected_intent.value,
            "status": case.expected_status,
            "facets": list(case.expected_facets),
        },
        metadata={
            "suite": "p0",
            "case_kind": case.case_id,
            "score_source": "deterministic_offline",
            "claim_supported_label": _PROXY_LABEL,
        },
        execute=lambda: _run_comparable_p0_case(case),
    )


def _comparable_p1_case(case: P1EvalCase) -> ComparableApplicationExperimentCase:
    from fra.contracts import ResearchCommand, ResearchMode

    command = ResearchCommand(ticker=case.ticker, request=case.request, mode=ResearchMode.AUTO)
    return ComparableApplicationExperimentCase(
        id=f"p1:{case.id}",
        suite="p1",
        input=command.model_dump(mode="json"),
        expected_output={
            "intent": case.expected_intent.value,
            "recipes": [recipe.value for recipe in case.expected_recipes],
            "guard_result": case.expected_guard_result,
        },
        metadata={
            "suite": "p1",
            "case_kind": case.id,
            "score_source": "deterministic_offline",
        },
        execute=lambda: _run_comparable_p1_case(case),
    )


def _comparable_p2_case(case: object) -> ComparableApplicationExperimentCase:
    from fra.evals.p2_runner import (
        DeterministicP2ApplicationFactory,
        P2EvalCase,
        evaluate_p2_case,
    )

    assert isinstance(case, P2EvalCase)
    return ComparableApplicationExperimentCase(
        id=f"p2:{case.id}",
        suite="p2",
        input=case.input_payload(),
        expected_output=case.expected_payload(),
        metadata={**case.metadata_payload(), "score_source": "deterministic_offline"},
        execute=lambda: {
            **evaluate_p2_case(
                case,
                DeterministicP2ApplicationFactory(),
            ).model_dump(mode="json"),
            "score_source": "deterministic_offline",
        },
    )


def _run_comparable_p0_case(case: EvalCase) -> dict[str, object]:
    from fra.application import ResearchApplication
    from fra.bootstrap import build_eval_runtime
    from fra.config import Settings
    from fra.contracts import ResearchCommand, ResearchMode

    application = ResearchApplication(
        _ComparableNeverRouter(),
        _ComparableRuntimeFactory(build_eval_runtime(Settings())),
        company_resolver=_ComparableCompanyResolver(case.ticker),
    )
    result = application.run(
        ResearchCommand(ticker=case.ticker, request=case.thesis, mode=ResearchMode.THESIS)
    )
    measured = _evaluate_case(case, result)
    return {**measured.result.model_dump(mode="json"), "score_source": "deterministic_offline"}


def _run_comparable_p1_case(case: P1EvalCase) -> dict[str, object]:
    from fra.application import ResearchApplication
    from fra.bootstrap import ResearchRuntime
    from fra.contracts import ResearchCommand, ResearchMode

    dependencies, audit = _offline_p1_dependencies(case)
    application = ResearchApplication(
        _ComparableNeverRouter(),
        _ComparableRuntimeFactory(ResearchRuntime(dependencies)),
        company_resolver=_ComparableCompanyResolver(case.ticker),
    )
    result = application.run(
        ResearchCommand(ticker=case.ticker, request=case.request, mode=ResearchMode.AUTO)
    )
    measured = _evaluate_p1_case(case, result, audit=audit)
    return {**measured.model_dump(mode="json"), "score_source": "deterministic_offline"}


class _ComparableNeverRouter:
    def route(self, request: str) -> RouterDecision:
        raise AssertionError(f"comparable evaluation should not use the fallback router: {request}")


class _ComparableRuntimeFactory:
    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    def build(self, command: object, intent: object) -> object:
        del command, intent
        return self._runtime


@dataclass(frozen=True)
class _ComparableCompanyResolver:
    ticker: str

    def resolve(self, ticker: str) -> str | None:
        canonical = self.ticker.strip().upper()
        return canonical if ticker.strip().upper() == canonical else None


def _run_comparable_application_case(
    item: object,
    case_by_id: Mapping[str, ComparableApplicationExperimentCase],
) -> dict[str, object]:
    item_id = getattr(item, "id", None)
    if not isinstance(item_id, str) or item_id not in case_by_id:
        raise LangfuseOperationError("dataset item is not a canonical comparable case")
    case = case_by_id[item_id]
    _validate_comparable_dataset_item(item, case)
    return case.execute()


def _comparable_overall_pass_evaluator(
    *, output: Mapping[str, object], **_: object
) -> dict[str, object]:
    if output.get("score_source") != "deterministic_offline":
        raise LangfuseOperationError("comparable experiment output lost its deterministic label")
    passed = output.get("passed")
    if not isinstance(passed, bool):
        raise LangfuseOperationError("comparable experiment output must include boolean passed")
    return {"name": "overall_pass", "value": float(passed)}


def _validate_comparable_dataset_items(
    items: object,
    case_by_id: Mapping[str, ComparableApplicationExperimentCase],
) -> None:
    if not isinstance(items, Sequence) or len(items) != len(case_by_id):
        raise LangfuseOperationError("dataset items do not match canonical comparable cases")
    seen: set[str] = set()
    for item in items:
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str) or item_id in seen or item_id not in case_by_id:
            raise LangfuseOperationError("dataset items do not match canonical comparable cases")
        seen.add(item_id)
        _validate_comparable_dataset_item(item, case_by_id[item_id])


def _validate_comparable_dataset_item(
    item: object,
    case: ComparableApplicationExperimentCase,
) -> None:
    for field_name, expected in (
        ("input", case.input_payload()),
        ("expected_output", case.expected_payload()),
        ("metadata", case.metadata_payload()),
    ):
        actual = getattr(item, field_name, None)
        if not isinstance(actual, Mapping) or dict(actual) != expected:
            raise LangfuseOperationError(
                f"dataset item '{case.id}' {field_name} does not match the canonical case"
            )


def _validate_comparable_experiment_result(
    result: object,
    case_by_id: Mapping[str, ComparableApplicationExperimentCase],
) -> None:
    item_results = getattr(result, "item_results", None)
    if not isinstance(item_results, Sequence) or len(item_results) != len(case_by_id):
        raise LangfuseOperationError("comparable experiment result does not cover every case")
    seen: set[str] = set()
    for item_result in item_results:
        item = getattr(item_result, "item", None)
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str) or item_id in seen or item_id not in case_by_id:
            raise LangfuseOperationError("comparable experiment result does not cover every case")
        seen.add(item_id)
        _validate_comparable_dataset_item(item, case_by_id[item_id])
        output = getattr(item_result, "output", None)
        if (
            not isinstance(output, Mapping)
            or output.get("score_source") != "deterministic_offline"
        ):
            raise LangfuseOperationError(
                "comparable experiment output lost its deterministic label"
            )
        passed = output.get("passed")
        if not isinstance(passed, bool):
            raise LangfuseOperationError("comparable experiment output must include boolean passed")
        evaluations = getattr(item_result, "evaluations", None)
        if not isinstance(evaluations, Sequence) or len(evaluations) != 1:
            raise LangfuseOperationError("comparable experiment result is missing overall_pass")
        evaluation = evaluations[0]
        name = getattr(evaluation, "name", None)
        value = getattr(evaluation, "value", None)
        if isinstance(evaluation, Mapping):
            name = evaluation.get("name")
            value = evaluation.get("value")
        if (
            name != "overall_pass"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value != float(passed)
        ):
            raise LangfuseOperationError("comparable experiment result is missing overall_pass")


def load_eval_cases(dataset_path: Path) -> list[EvalCase]:
    """Parse strict P0 JSONL cases while validating adjacent P1 rows."""
    cases, _ = _load_dataset(dataset_path)
    if not cases:
        raise ValueError("evaluation dataset must not be empty")
    return cases


def load_p1_eval_cases(dataset_path: Path) -> list[P1EvalCase]:
    """Parse the additive P1 rows from a mixed P0/P1 JSONL dataset."""
    _, cases = _load_dataset(dataset_path)
    return cases


def _load_dataset(dataset_path: Path) -> tuple[list[EvalCase], list[P1EvalCase]]:
    p0_cases: list[EvalCase] = []
    p1_cases: list[P1EvalCase] = []
    first_line_by_p0_id: dict[str, int] = {}
    first_line_by_p1_id: dict[str, int] = {}
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read evaluation dataset: {dataset_path}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("evaluation row must be a JSON object")
            if "case_id" in payload and "id" not in payload:
                p0_case = EvalCase.model_validate(payload)
                first_line = first_line_by_p0_id.get(p0_case.case_id)
                if first_line is not None:
                    raise ValueError(
                        f"duplicate case_id '{p0_case.case_id}' at line {line_number}; "
                        f"first defined at line {first_line}"
                    )
                first_line_by_p0_id[p0_case.case_id] = line_number
                p0_cases.append(p0_case)
            elif "id" in payload and "case_id" not in payload:
                p1_case = P1EvalCase.model_validate(payload)
                first_line = first_line_by_p1_id.get(p1_case.id)
                if first_line is not None:
                    raise ValueError(
                        f"duplicate id '{p1_case.id}' at line {line_number}; "
                        f"first defined at line {first_line}"
                    )
                first_line_by_p1_id[p1_case.id] = line_number
                p1_cases.append(p1_case)
            else:
                raise ValueError("evaluation row must contain exactly one of case_id or id")
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            message = str(error)
            if message.startswith(("duplicate case_id", "duplicate id")):
                raise ValueError(message) from error
            raise ValueError(
                f"invalid evaluation dataset at line {line_number}: {error}"
            ) from error

    if not p0_cases and not p1_cases:
        raise ValueError("evaluation dataset must not be empty")
    return p0_cases, p1_cases


def run_eval(dataset_path: Path, dependencies: Dependencies) -> EvalSummary:
    """Run P0 cases and additive provider-free P1 contract probes."""
    cases = load_eval_cases(dataset_path)
    p1_cases = load_p1_eval_cases(dataset_path)
    p1_results = evaluate_p1_cases(p1_cases)
    p1_metrics = score_p1_eval(p1_cases, p1_results) if p1_cases else None
    evaluated_cases: list[_EvaluatedCase] = []
    for case in cases:
        research_result = run_research(case.ticker, case.thesis, dependencies)
        evaluated = _evaluate_case(case, research_result)
        result = evaluated.result
        evaluated_cases.append(evaluated)
        _record_trace(
            dependencies,
            "evaluation_case",
            {
                "case_id": case.case_id,
                "passed": result.passed,
                "citation_validity": result.citation_validity,
                "claim_supported": result.claim_supported,
                "cost_usd": 0.0,
                "cost_source": _COST_SOURCE,
            },
        )

    results = [evaluated.result for evaluated in evaluated_cases]
    counter_expected = [
        evaluated.result
        for case, evaluated in zip(cases, evaluated_cases, strict=True)
        if "counter_evidence" in case.expected_facets
    ]
    summary = EvalSummary(
        dataset_path=str(dataset_path),
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        pass_rate=sum(result.passed for result in results) / len(results),
        citation_validity=_combined_ratio(
            [evaluated.citation_validity for evaluated in evaluated_cases]
        ),
        claim_supported=_combined_ratio(
            [evaluated.claim_supported for evaluated in evaluated_cases]
        ),
        counterevidence_present=(
            sum(result.counterevidence_present for result in counter_expected)
            / len(counter_expected)
            if counter_expected
            else None
        ),
        coverage=_combined_ratio([evaluated.coverage for evaluated in evaluated_cases]) or 0.0,
        results=results,
        p1_case_count=len(p1_cases),
        p1_passed_count=sum(result.passed for result in p1_results),
        p1_metrics=p1_metrics,
        p1_results=p1_results,
    )
    _record_trace(
        dependencies,
        "evaluation_summary",
        {
            "case_count": summary.case_count,
            "passed_count": summary.passed_count,
            "pass_rate": summary.pass_rate,
            "cost_usd": 0.0,
            "cost_source": _COST_SOURCE,
        },
    )
    return summary


def evaluate_p1_cases(cases: Sequence[P1EvalCase]) -> list[P1EvalResult]:
    """Run P1 cases through the real graph with deterministic local adapters."""

    results: list[P1EvalResult] = []
    for case in cases:
        dependencies, audit = _offline_p1_dependencies(case)
        graph_result = run_research(
            case.ticker,
            case.request,
            dependencies,
        )
        results.append(_evaluate_p1_case(case, graph_result, audit=audit))
    return results


def _evaluate_p1_case(
    case: P1EvalCase,
    result: ResearchResult,
    *,
    audit: _OfflineP1Audit | None = None,
) -> P1EvalResult:
    actual_recipes = [run.recipe_name for run in result.skill_runs]
    covered_facets: list[ResearchFacet] = []
    citation_source_ids: list[str] = []
    resolved_source_ids: list[str] = []
    verified_claim_count = 0
    valid_citation_count = 0
    unsupported_claim_count = 0
    source_policy_violations = 0
    budget_violations = 0
    recipes = {recipe.name: recipe for recipe in RECIPES}

    for run in result.skill_runs:
        recipe = recipes[run.recipe_name]
        evidence = run.evidence
        if evidence is not None:
            budget_violations += int(
                evidence.retrieval_rounds > recipe.budget.max_retrieval_rounds
                or evidence.web_calls > recipe.budget.max_web_calls
                or len(evidence.web_evidence) > recipe.budget.max_web_results
            )
        guarded = run.guarded_memo
        if guarded is None:
            continue
        source_index = {
            source.id: source for source in (*guarded.filing_sources, *guarded.web_sources)
        }
        resolved_source_ids.extend(source_index)
        for source in source_index.values():
            if not source_allowed_for_recipe(source, recipe):
                source_policy_violations += 1
        for section in guarded.memo.sections:
            if section.claims and section.facet not in covered_facets:
                covered_facets.append(section.facet)
            for claim in section.claims:
                if claim.kind is not ClaimKind.VERIFIED_FACT:
                    continue
                verified_claim_count += 1
                source_ids = [*claim.evidence_chunk_ids, *claim.web_evidence_ids]
                citation_source_ids.extend(source_ids)
                if source_ids and _p1_eval_citations_are_authoritative(
                    source_ids=source_ids,
                    source_index=source_index,
                    evidence=evidence,
                    facet=section.facet,
                    audit=audit,
                ):
                    valid_citation_count += 1
                else:
                    unsupported_claim_count += 1
        for point in guarded.memo.data_points:
            verified_claim_count += 1
            citation_source_ids.extend(point.source_ids)
            if point.source_ids and _p1_eval_citations_are_authoritative(
                source_ids=point.source_ids,
                source_index=source_index,
                evidence=evidence,
                facet=ResearchFacet.DATA_VERIFICATION,
                audit=audit,
            ):
                valid_citation_count += 1
            else:
                unsupported_claim_count += 1

    if audit is not None:
        source_policy_violations += len(audit.forbidden_persisted_source_ids)

    guard_result = _guard_result(result)
    passed = (
        result.decision.intent is case.expected_intent
        and actual_recipes == case.expected_recipes
        and (
            case.expected_guard_result != "pass"
            or set(case.required_facets).issubset(covered_facets)
        )
        and guard_result == case.expected_guard_result
        and valid_citation_count == verified_claim_count
        and unsupported_claim_count == 0
        and source_policy_violations == 0
        and budget_violations == 0
    )
    return P1EvalResult(
        id=case.id,
        actual_intent=result.decision.intent,
        actual_recipes=actual_recipes,
        covered_facets=covered_facets,
        actual_guard_result=guard_result,
        verified_claim_count=verified_claim_count,
        valid_citation_count=valid_citation_count,
        unsupported_claim_count=unsupported_claim_count,
        citation_source_ids=list(dict.fromkeys(citation_source_ids)),
        resolved_source_ids=list(dict.fromkeys(resolved_source_ids)),
        web_policy_rejections=audit.web_policy_rejections if audit is not None else 0,
        persisted_source_ids=(audit.persisted_source_ids if audit is not None else []),
        source_policy_violations=source_policy_violations,
        budget_violations=budget_violations,
        passed=passed,
    )


def _p1_eval_citations_are_authoritative(
    *,
    source_ids: Sequence[str],
    source_index: dict[str, EvidenceChunk | WebEvidence],
    evidence: EvidenceBundle | None,
    facet: ResearchFacet,
    audit: _OfflineP1Audit | None,
) -> bool:
    if evidence is None:
        return False
    assignment_keys = {
        (assignment.question_index, assignment.side, assignment.source_id)
        for assignment in evidence.assignments
    }
    required_sides = (
        frozenset({EvidenceSide.SUPPORT})
        if facet is ResearchFacet.BULL_CASE
        else frozenset({EvidenceSide.CHALLENGE})
        if facet is ResearchFacet.BEAR_CASE
        else frozenset(EvidenceSide)
    )
    facet_keys = {
        (assignment.facet, assignment.side, assignment.source_id)
        for assignment in evidence.facet_assignments
        if (assignment.question_index, assignment.side, assignment.source_id)
        in assignment_keys
    }
    for source_id in source_ids:
        source = source_index.get(source_id)
        if source is None:
            return False
        if audit is not None and not _same_authoritative_source(
            audit.authoritative_sources.get(source_id),
            source,
        ):
            return False
        if not any((facet, side, source_id) in facet_keys for side in required_sides):
            return False
    return True


def _same_authoritative_source(
    persisted: EvidenceChunk | WebEvidence | None,
    guarded: EvidenceChunk | WebEvidence,
) -> bool:
    if isinstance(guarded, PolicyValidatedWebEvidence):
        return isinstance(persisted, WebEvidence) and guarded.model_dump(
            exclude={"policy_version", "canonical_url"}
        ) == persisted.model_dump()
    return persisted == guarded


def score_p1_eval(
    cases: Sequence[P1EvalCase],
    results: Sequence[P1EvalResult],
) -> P1EvalMetrics:
    """Compute deterministic micro-averaged P1 metrics from local observations."""

    if not cases:
        raise ValueError("P1 evaluation cases must not be empty")
    result_by_id = {result.id: result for result in results}
    if len(result_by_id) != len(results) or set(result_by_id) != {case.id for case in cases}:
        raise ValueError("P1 results must match case IDs exactly")

    paired = [(case, result_by_id[case.id]) for case in cases]
    facet_denominator = sum(len(case.required_facets) for case in cases)
    matched_facets = sum(
        len(set(case.required_facets).intersection(result.covered_facets))
        for case, result in paired
    )
    verified_claims = sum(result.verified_claim_count for result in results)
    return P1EvalMetrics(
        intent_accuracy=_decimal_ratio(
            sum(result.actual_intent is case.expected_intent for case, result in paired),
            len(cases),
        ),
        recipe_accuracy=_decimal_ratio(
            sum(result.actual_recipes == case.expected_recipes for case, result in paired),
            len(cases),
        ),
        facet_coverage=_decimal_ratio(matched_facets, facet_denominator, empty=Decimal(1)),
        citation_validity=_decimal_ratio(
            sum(result.valid_citation_count for result in results),
            verified_claims,
            empty=Decimal(1),
        ),
        unsupported_claim_rate=_decimal_ratio(
            sum(result.unsupported_claim_count for result in results),
            verified_claims,
            empty=Decimal(0),
        ),
        source_policy_violations=sum(result.source_policy_violations for result in results),
        budget_violations=sum(result.budget_violations for result in results),
    )


def _guard_result(result: ResearchResult) -> P1ObservedGuardResult:
    if result.status == "refused":
        return "refuse"
    if result.status == "completed":
        return "pass"
    if result.status in {"partial", "insufficient_evidence"}:
        return "partial"
    return "fail"


class _OfflineP0Client:
    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"P1 eval entered P0 tool path: {name} {arguments!r}")


class _OfflineP0Model:
    def route(self, thesis: str) -> RouterDecision:
        return RouterDecision(intent=Intent.AMBIGUOUS, reason=f"offline P1 eval: {thesis[:20]}")

    def plan(
        self,
        ticker: str,
        thesis: str,
        *,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> list[ResearchQuestion]:
        del memory_hints
        raise AssertionError(f"P1 eval entered P0 planner: {ticker} {thesis}")


class _OfflineP0Analyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        raise AssertionError(f"P1 eval entered P0 analyst: {questions!r} {evidence!r}")


@dataclass(slots=True)
class _OfflineP1Audit:
    web_policy_rejections: int = 0
    gateway_persisted_source_ids: list[str] = field(default_factory=list)
    forbidden_persisted_source_ids: list[str] = field(default_factory=list)
    run_persisted_source_ids: list[str] = field(default_factory=list)
    authoritative_sources: dict[str, EvidenceChunk | WebEvidence] = field(default_factory=dict)

    @property
    def authoritative_source_ids(self) -> list[str]:
        return list(self.authoritative_sources)

    def persist_authoritative_source(self, source: EvidenceChunk | WebEvidence) -> None:
        self.authoritative_sources[source.id] = source

    def delete_authoritative_source(self, source_id: str) -> None:
        self.authoritative_sources.pop(source_id, None)

    @property
    def persisted_source_ids(self) -> list[str]:
        return self.authoritative_source_ids


class _OfflineP1Planner:
    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        return [
            ResearchQuestion(
                question=f"What evidence covers {request.recipe.name.value}?",
                support_query="supporting disclosure",
                challenge_query="challenging disclosure",
            )
        ]


class _OfflineP1Analyst:
    def __init__(self, case: P1EvalCase) -> None:
        self._case = case

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        filing_ids = [source.id for source in evidence.filing_evidence]
        web_ids = [source.id for source in evidence.web_evidence]
        filing_id_set = set(filing_ids)
        web_id_set = set(web_ids)

        def ids_for_facet(facet: ResearchFacet) -> tuple[list[str], list[str]]:
            source_ids = list(
                dict.fromkeys(
                    assignment.source_id
                    for assignment in evidence.facet_assignments
                    if assignment.facet is facet
                )
            )
            return (
                [source_id for source_id in source_ids if source_id in filing_id_set],
                [source_id for source_id in source_ids if source_id in web_id_set],
            )

        sections = [
            SkillResearchSection(
                facet=facet,
                claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text=f"Disclosed evidence supports the {facet.value} research section.",
                        confidence=Confidence.HIGH,
                        evidence_chunk_ids=ids_for_facet(facet)[0],
                        web_evidence_ids=ids_for_facet(facet)[1],
                    )
                ],
            )
            for facet in request.recipe.required_facets
        ]
        data_points: list[FinancialDataPoint] = []
        if request.recipe.name is SkillName.FINANCIAL_DATA_VERIFICATION:
            data_filing_ids, data_web_ids = ids_for_facet(ResearchFacet.DATA_VERIFICATION)
            source_ids = [*data_filing_ids, *data_web_ids]
            data_points = [_offline_financial_point(source_ids, Decimal("100"))]
            if "差异" in self._case.request:
                data_points.append(_offline_financial_point(source_ids, Decimal("101")))
        return SkillResearchMemo(
            recipe_name=request.recipe.name,
            recipe_version=request.recipe.version,
            research_question=f"Review {request.recipe.name.value} evidence.",
            sections=sections,
            data_points=data_points,
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            information_gaps=["No additional evidence gap was identified."],
            confidence=Decimal("0.9"),
        )


class _OfflineSkillRunRepository:
    def __init__(self, audit: _OfflineP1Audit) -> None:
        self._next_id = 0
        self._audit = audit

    def start(
        self,
        *,
        application_run_id: str,
        ticker: str,
        recipe_name: str,
        recipe_version: str,
        recipe_snapshot: dict[str, object],
    ) -> str:
        del application_run_id, ticker, recipe_name, recipe_version, recipe_snapshot
        run_id = f"offline-eval-{self._next_id}"
        self._next_id += 1
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: Sequence[str],
        errors: Sequence[str],
    ) -> None:
        del run_id, status, errors
        self._audit.run_persisted_source_ids.extend(source_ids)


def _offline_p1_dependencies(
    case: P1EvalCase,
) -> tuple[Dependencies, _OfflineP1Audit]:
    audit = _OfflineP1Audit()

    async def local_search(queries):
        if case.id in {
            "p1-insufficient-local-evidence",
            "p1-disallowed-domain-fallback",
        }:
            return LocalSearchResponse()
        selected_queries = queries
        if case.id == "p1-successful-web-fallback":
            if any("retry=missing_only" in query.query for query in queries):
                return LocalSearchResponse()
            available_facets = {query.facet for query in queries}
            target_facet = (
                ResearchFacet.BULL_CASE
                if ResearchFacet.BULL_CASE in available_facets
                else queries[0].facet
            )
            selected_queries = tuple(
                query
                for query in queries
                if query.side is EvidenceSide.CHALLENGE and query.facet is not target_facet
            )
        hits: list[LocalEvidenceHit] = []
        sources: dict[str, EvidenceChunk] = {}
        for query in selected_queries:
            source = sources.get(query.facet.value)
            if source is None:
                source = _offline_filing_source_for_facet(query.facet, query.ticker)
                sources[query.facet.value] = source
                audit.persist_authoritative_source(source)
            hits.append(
                LocalEvidenceHit(
                    evidence=source,
                    question_index=query.question_index,
                    side=query.side,
                    facet=query.facet,
                )
            )
        return LocalSearchResponse(evidence=tuple(hits))

    async def web_search(request: WebSearchRequest) -> WebSearchResponse:
        if case.id == "p1-disallowed-domain-fallback":
            sources = await _offline_web_evidence(request.ticker, audit, forbidden=True)
        elif case.id == "p1-successful-web-fallback":
            sources = await _offline_web_evidence(request.ticker, audit, forbidden=False)
        else:
            return WebSearchResponse()
        if not sources:
            return WebSearchResponse()
        question_index, side = request.missing_pairs[0]
        return WebSearchResponse(
            evidence=tuple(
                WebEvidenceHit(
                    evidence=source,
                    question_index=question_index,
                    side=side,
                    facet=request.missing_facets[0],
                )
                for source in sources
            )
        )

    collector = EvidenceCollector(local_search=local_search, web_search=web_search)
    source_policy = SourcePolicy(
        issuer_domains={case.ticker: frozenset({"investor.nvidia.com"})}
    )
    return (
        Dependencies(
            mcp_client=_OfflineP0Client(),
            fast_model=_OfflineP0Model(),
            analyst_model=_OfflineP0Analyst(),
            skill_planner=_OfflineP1Planner(),
            skill_collector=collector,
            skill_analyst=_OfflineP1Analyst(case),
            skill_run_repository=_OfflineSkillRunRepository(audit),
            web_evidence_validator=PersistedWebEvidenceValidator(
                source_policy,
                _OfflineGatewayRepository(audit, forbidden=False),
            ),
        ),
        audit,
    )


class _OfflineForbiddenProvider:
    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        return [
            RawSearchHit(
                title="Forbidden provider candidate",
                url=HttpUrl("https://example.com/forbidden-disclosure"),
                excerpt="This candidate must be rejected before persistence.",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]


class _OfflineAllowedProvider:
    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        return [
            RawSearchHit(
                title="Allowlisted issuer disclosure",
                url=HttpUrl("https://investor.nvidia.com/offline-disclosure"),
                excerpt="This issuer disclosure is persisted before it can be cited.",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]


class _OfflineIdentityResolver:
    async def resolve(self, url: HttpUrl) -> HttpUrl:
        return url

    async def resolve_with_policy(
        self,
        url: HttpUrl,
        *,
        ticker: str,
        source_policy: SourcePolicy,
    ) -> HttpUrl:
        source_policy.classify(ticker=ticker, url=url)
        return url


class _OfflineGatewayRepository:
    def __init__(self, audit: _OfflineP1Audit, *, forbidden: bool) -> None:
        self._audit = audit
        self._forbidden = forbidden

    def upsert(self, evidence: WebEvidence) -> WebEvidence:
        source_id = sha256(
            f"{evidence.ticker.upper()}\n{evidence.source_url}\n{evidence.content_hash}".encode()
        ).hexdigest()
        existing = self._audit.authoritative_sources.get(source_id)
        if isinstance(existing, WebEvidence):
            return existing
        stored = evidence.model_copy(update={"id": source_id, "ticker": evidence.ticker.upper()})
        self._audit.gateway_persisted_source_ids.append(source_id)
        if self._forbidden:
            self._audit.forbidden_persisted_source_ids.append(source_id)
        self._audit.persist_authoritative_source(stored)
        return stored

    def get_many(self, evidence_ids: Sequence[str]) -> list[WebEvidence]:
        return [
            source
            for source_id in evidence_ids
            if isinstance(
                (source := self._audit.authoritative_sources.get(source_id)),
                WebEvidence,
            )
        ]


async def _offline_web_evidence(
    ticker: str,
    audit: _OfflineP1Audit,
    *,
    forbidden: bool,
) -> list[WebEvidence]:
    gateway = AllowlistedWebGateway(
        provider=(
            _OfflineForbiddenProvider()
            if forbidden
            else _OfflineAllowedProvider()
        ),
        redirect_resolver=_OfflineIdentityResolver(),
        source_policy=SourcePolicy(
            issuer_domains={ticker: frozenset({"investor.nvidia.com"})}
        ),
        repository=_OfflineGatewayRepository(audit, forbidden=forbidden),  # type: ignore[arg-type]
    )
    try:
        evidence = await gateway.search(
            ticker=ticker,
            query=(
                "forbidden-domain policy probe"
                if forbidden
                else "allowlisted issuer evidence probe"
            ),
            max_results=1,
        )
        if forbidden and not evidence:
            audit.web_policy_rejections += 1
        return evidence
    except WebGatewayError as error:
        if error.code != "SOURCE_NOT_ALLOWED":
            raise
        audit.web_policy_rejections += 1
        return []


def _offline_filing_source_for_facet(
    facet: ResearchFacet,
    ticker: str,
) -> EvidenceChunk:
    source_id = f"sec-{facet.value}"
    return EvidenceChunk(
        id=source_id,
        ticker=ticker,
        corpus_version=f"{ticker}-offline-p1-v1",
        content=f"Source evidence for {facet.value}.",
        source_url=f"https://www.sec.gov/Archives/{source_id}.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=50,
    )


def _offline_financial_point(
    source_ids: list[str],
    value: Decimal,
) -> FinancialDataPoint:
    return FinancialDataPoint(
        name="Revenue",
        value=value,
        currency="USD",
        unit="millions",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        definition="GAAP revenue",
        source_ids=source_ids,
    )


def _decimal_ratio(numerator: int, denominator: int, *, empty: Decimal = Decimal(0)) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else empty


def _evaluate_case(case: EvalCase, result: ResearchResult) -> _EvaluatedCase:
    citation_validity, claim_supported = _factual_metrics(result)
    counterevidence_present = bool(result.guarded_memo and result.guarded_memo.counter_claims)
    matched_facets = sum(_facet_matches(facet, result) for facet in case.expected_facets)
    status_matches = result.status == case.expected_status
    coverage = _RatioCounts(
        numerator=matched_facets + status_matches,
        denominator=len(case.expected_facets) + 1,
    )
    proxies_valid = all(
        counts.ratio is None or counts.ratio == 1.0
        for counts in (citation_validity, claim_supported)
    )
    passed = (
        result.decision.intent is case.expected_intent
        and status_matches
        and matched_facets == len(case.expected_facets)
        and proxies_valid
    )
    return _EvaluatedCase(
        result=EvalCaseResult(
            case_id=case.case_id,
            expected_intent=case.expected_intent,
            actual_intent=result.decision.intent,
            expected_status=case.expected_status,
            actual_status=result.status,
            citation_validity=citation_validity.ratio,
            claim_supported=claim_supported.ratio,
            counterevidence_present=counterevidence_present,
            coverage=coverage.ratio or 0.0,
            passed=passed,
        ),
        citation_validity=citation_validity,
        claim_supported=claim_supported,
        coverage=coverage,
    )


def _factual_metrics(result: ResearchResult) -> tuple[_RatioCounts, _RatioCounts]:
    if result.memo is None:
        return _RatioCounts(0, 0), _RatioCounts(0, 0)
    draft_facts = [
        (section, claim)
        for section, claims in (
            ("supporting", result.memo.supporting_claims),
            ("counter", result.memo.counter_claims),
            ("inferences", result.memo.inferences),
            ("open_questions", result.memo.open_questions),
        )
        for claim in claims
        if claim.kind is ClaimKind.VERIFIED_FACT
    ]
    if not draft_facts:
        return _RatioCounts(0, 0), _RatioCounts(0, 0)
    guarded_sections = {
        "supporting": result.guarded_memo.supporting_claims if result.guarded_memo else [],
        "counter": result.guarded_memo.counter_claims if result.guarded_memo else [],
        "inferences": result.guarded_memo.inferences if result.guarded_memo else [],
        "open_questions": result.guarded_memo.open_questions if result.guarded_memo else [],
    }
    valid_count = 0
    supported_count = 0
    for section, claim in draft_facts:
        if not _survives(claim, guarded_sections[section]):
            continue
        cited_chunks = [result.evidence.get(chunk_id) for chunk_id in claim.evidence_chunk_ids]
        citations_valid = bool(cited_chunks) and all(
            chunk is not None
            and chunk.ticker.upper() == result.ticker
            and chunk.corpus_version == result.corpus_version
            for chunk in cited_chunks
        )
        if not citations_valid:
            continue
        valid_count += 1
        normalized_claim = _normalize_text(claim.text)
        if any(
            normalized_claim in _normalize_text(chunk.content)
            for chunk in cited_chunks
            if chunk is not None
        ):
            supported_count += 1
    denominator = len(draft_facts)
    return _RatioCounts(valid_count, denominator), _RatioCounts(supported_count, denominator)


def _survives(claim: Claim, guarded_facts: list[Claim]) -> bool:
    return any(
        candidate.kind is claim.kind
        and candidate.text == claim.text
        and candidate.evidence_chunk_ids == list(dict.fromkeys(claim.evidence_chunk_ids))
        for candidate in guarded_facts
    )


def _facet_matches(facet: ExpectedFacet, result: ResearchResult) -> bool:
    guarded = result.guarded_memo
    if facet == "supporting_evidence":
        return bool(guarded and guarded.supporting_claims)
    if facet == "counter_evidence":
        return bool(guarded and guarded.counter_claims)
    if facet == "insufficient_evidence":
        return result.status == "insufficient_evidence"
    return result.status == "refused"


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _combined_ratio(counts: list[_RatioCounts]) -> float | None:
    numerator = sum(item.numerator for item in counts)
    denominator = sum(item.denominator for item in counts)
    return numerator / denominator if denominator else None


def _record_trace(
    dependencies: Dependencies,
    name: str,
    attributes: dict[str, object],
) -> None:
    try:
        dependencies.trace_sink.record(name, attributes)
    except Exception:
        pass
