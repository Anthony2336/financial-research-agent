"""Deterministic P2 evaluation contracts and runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal, Protocol

from fastmcp.client.client import CallToolResult
from pydantic import Field, ValidationError, field_validator, model_validator

from financial_evidence_agent.application import (
    ResearchApplication,
    ResearchCommand,
    ResearchMode,
)
from financial_evidence_agent.bootstrap import ResearchRuntime
from financial_evidence_agent.context import MemoryHint
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchQuestion,
    RouterDecision,
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    StrictModel,
    WebEvidence,
    content_addressed_web_evidence_id,
)
from financial_evidence_agent.graph.market_workflow import run_market_workflow
from financial_evidence_agent.graph.models import (
    Dependencies,
    MarketDependencies,
    MarketResearchResult,
    ResearchResult,
    SkillAnalysisInput,
    SkillPlanningInput,
)
from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketSnapshot,
    MarketStatus,
    market_bar_observation_id,
    market_snapshot_observation_id,
)
from financial_evidence_agent.observability import LangfuseOperationError
from financial_evidence_agent.research_packages.models import (
    ComparabilityStatus,
    GuardedResearchPackage,
    PackageClaim,
    ResearchQualityDecision,
)
from financial_evidence_agent.research_packages.orchestrator import PeerResearchResult
from financial_evidence_agent.research_packages.quality import (
    QualityResearchResult,
    QualityResearchRuntime,
)
from financial_evidence_agent.skills.models import ResearchFacet, SkillName
from financial_evidence_agent.skills.recipes import INDUSTRY_RESEARCH, RECIPES
from financial_evidence_agent.skills.schemas import (
    FinancialDataPoint,
    InformationSufficiency,
    RecipeProvenance,
    ReportProvenance,
    SkillResearchMemo,
    SkillResearchSection,
)
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.run_repositories import RunFinish, RunStart, SourceFetchWrite
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
)

P2CaseKind = Literal[
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
]
P2Status = Literal["completed", "partial", "failed", "refused", "declined"]
_MARKET_KINDS = frozenset(
    {
        "market_complete",
        "market_stale",
        "market_unavailable",
        "market_context_unknown",
    }
)
_SOURCE_REF_REQUIRED_KINDS = frozenset(
    {
        "market_complete",
        "market_context_unknown",
        "industry",
        "peer_partial",
        "peer_not_comparable",
    }
)
_ALLOWED_WEB_SOURCE_KINDS = frozenset({SourceKind.ISSUER_IR, SourceKind.AUTHORITATIVE_WEB})
_RECIPE_BY_NAME = {recipe.name.value: recipe for recipe in RECIPES}
_FIXED_QUALITY_DATE = date(2026, 8, 31)
_FIXED_MARKET_GUARD_TIME = datetime(2026, 8, 31, 14, 0, tzinfo=UTC)
_P2_DATASET_SCHEMA_VERSION = "p2-dataset-v1"
_P2_FIXTURE_VERSION = "p2-fixture-v1"


class P2EvalCase(StrictModel):
    """One strict P2 evaluation case loaded from JSONL."""

    id: str = Field(min_length=1, max_length=100)
    kind: P2CaseKind
    ticker: str = Field(min_length=1, max_length=10)
    request: str = Field(min_length=1, max_length=2_000)
    mode: ResearchMode
    peer_tickers: tuple[str, ...] = ()
    peer_scope: str | None = None
    with_context: bool = False
    network_access: bool = False
    expected_intent: Intent
    expected_status: P2Status
    expected_recipe_names: list[str] = Field(default_factory=list)
    expected_source_refs: list[SourceRef] = Field(default_factory=list)
    expected_market_price: Decimal | None = None
    expected_market_errors: list[str] = Field(default_factory=list)
    expected_freshness_label: str | None = None
    expected_quality_decision: ResearchQualityDecision | None = None
    expected_comparison_statuses: list[ComparabilityStatus] = Field(default_factory=list)
    expected_cause_assessment: str | None = None
    expected_cross_ticker_rejection_count: int = Field(default=0, ge=0)
    expected_cross_ticker_leakage_count: int = Field(default=0, ge=0)
    expected_source_policy_violations: int = Field(default=0, ge=0)
    expected_budget_violations: int = Field(default=0, ge=0)

    def input_payload(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "request": self.request,
            "mode": self.mode.value,
            "peer_tickers": list(self.peer_tickers),
            "peer_scope": self.peer_scope,
            "with_context": self.with_context,
        }

    def expected_payload(self) -> dict[str, object]:
        return {
            "intent": self.expected_intent.value,
            "status": self.expected_status,
            "recipe_names": list(self.expected_recipe_names),
            "source_refs": [reference.encode() for reference in self.expected_source_refs],
            "market_price": (
                None if self.expected_market_price is None else str(self.expected_market_price)
            ),
            "market_errors": list(self.expected_market_errors),
            "freshness_label": self.expected_freshness_label,
            "quality_decision": (
                None
                if self.expected_quality_decision is None
                else self.expected_quality_decision.value
            ),
            "comparison_statuses": [
                status.value for status in self.expected_comparison_statuses
            ],
            "cause_assessment": self.expected_cause_assessment,
            "cross_ticker_rejection_count": self.expected_cross_ticker_rejection_count,
            "cross_ticker_leakage_count": self.expected_cross_ticker_leakage_count,
            "source_policy_violations": self.expected_source_policy_violations,
            "budget_violations": self.expected_budget_violations,
        }

    def metadata_payload(self) -> dict[str, object]:
        return {
            "suite": "p2",
            "case_kind": self.kind,
            "dataset_schema_version": _P2_DATASET_SCHEMA_VERSION,
            "fixture_version": _P2_FIXTURE_VERSION,
            "recipe_versions": [
                _RECIPE_BY_NAME[recipe_name].version
                for recipe_name in self.expected_recipe_names
            ],
            "source_policy_version": _industry_policy().version,
        }

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("id must not contain whitespace")
        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.upper()

    @field_validator("peer_tickers", mode="before")
    @classmethod
    def normalize_peer_tickers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().upper() for item in value if isinstance(item, str))
        return value

    @field_validator("expected_source_refs", mode="before")
    @classmethod
    def normalize_source_refs(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        normalized: list[SourceRef | object] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(SourceRef.decode(item))
            else:
                normalized.append(item)
        return normalized

    @field_validator("expected_market_price", mode="before")
    @classmethod
    def require_finite_market_decimal(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        try:
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as error:
            raise ValueError("invalid Decimal for expected_market_price") from error
        if not decimal_value.is_finite():
            raise ValueError("invalid Decimal for expected_market_price")
        return decimal_value

    @field_validator("network_access")
    @classmethod
    def reject_network_access(cls, value: bool) -> bool:
        if value:
            raise ValueError("network access is not allowed in deterministic P2 cases")
        return value

    @model_validator(mode="after")
    def validate_case_shape(self) -> P2EvalCase:
        if self.kind in _SOURCE_REF_REQUIRED_KINDS and not self.expected_source_refs:
            raise ValueError("expected_source_refs must not be empty for this P2 case")
        if self.peer_tickers and self.mode is not ResearchMode.INDUSTRY_RESEARCH:
            raise ValueError("peer_tickers are only valid in industry-research mode")
        if self.peer_scope is not None and self.mode is not ResearchMode.INDUSTRY_RESEARCH:
            raise ValueError("peer_scope is only valid in industry-research mode")
        if self.peer_tickers and (self.peer_scope is None or not self.peer_scope.strip()):
            raise ValueError("peer_scope is required when peer_tickers are supplied")
        if self.kind in _MARKET_KINDS and self.mode is not ResearchMode.MARKET_SNAPSHOT:
            raise ValueError("market P2 cases must use market-snapshot mode")
        if self.kind.startswith("peer_") and not self.peer_tickers:
            raise ValueError("peer P2 cases require explicit peer_tickers")
        if self.kind == "quality_out_of_scope" and self.mode is not ResearchMode.QUALITY_SCREEN:
            raise ValueError("quality_out_of_scope must use quality-screen mode")
        if self.kind == "cross_ticker_attack" and self.mode is not ResearchMode.QUALITY_SCREEN:
            raise ValueError("cross_ticker_attack must use quality-screen mode")
        return self


class P2EvalResult(StrictModel):
    """One measured deterministic P2 evaluation result."""

    id: str
    kind: P2CaseKind
    actual_intent: Intent
    actual_status: str
    actual_recipe_names: list[str] = Field(default_factory=list)
    actual_source_refs: list[str] = Field(default_factory=list)
    actual_errors: list[str] = Field(default_factory=list)
    actual_market_price: Decimal | None = None
    actual_quality_decision: ResearchQualityDecision | None = None
    actual_comparison_statuses: list[str] = Field(default_factory=list)
    intent_matched: bool = True
    recipe_matched: bool = True
    market_metadata_valid: bool | None = None
    freshness_handled_correctly: bool | None = None
    source_ref_valid: bool
    cross_ticker_rejection_count: int = Field(default=0, ge=0)
    cross_ticker_leakage_count: int = Field(default=0, ge=0)
    comparability_correct: bool | None = None
    quality_decision_correct: bool | None = None
    neutral_causality_safe: bool | None = None
    source_policy_violations: int = Field(default=0, ge=0)
    budget_violations: int = Field(default=0, ge=0)
    passed: bool

    @field_validator("actual_market_price")
    @classmethod
    def require_finite_price(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("actual_market_price must be finite")
        return value


class P2EvalMetrics(StrictModel):
    """Aggregate deterministic P2 metrics."""

    intent_accuracy: Decimal = Field(ge=0, le=1)
    recipe_accuracy: Decimal = Field(ge=0, le=1)
    market_metadata_validity: Decimal = Field(ge=0, le=1)
    freshness_handling_accuracy: Decimal = Field(ge=0, le=1)
    source_ref_validity: Decimal = Field(ge=0, le=1)
    cross_ticker_leakage_count: int = Field(ge=0)
    comparability_accuracy: Decimal = Field(ge=0, le=1)
    quality_decision_accuracy: Decimal = Field(ge=0, le=1)
    neutral_causality_accuracy: Decimal = Field(ge=0, le=1)
    source_policy_violations: int = Field(ge=0)
    budget_violations: int = Field(ge=0)
    pass_rate: Decimal = Field(ge=0, le=1)


class P2EvalSummary(StrictModel):
    """Complete deterministic P2 evaluation output."""

    dataset_path: str
    case_count: int = Field(ge=1)
    passed_count: int = Field(ge=0)
    pass_rate: Decimal = Field(ge=0, le=1)
    metrics: P2EvalMetrics
    results: list[P2EvalResult]


class P2EvalApplicationFactory(Protocol):
    """Build one deterministic application per P2 case."""

    def for_case(self, case: P2EvalCase) -> ResearchApplication:
        """Return one case-scoped deterministic application."""


def load_p2_eval_cases(dataset_path: Path) -> list[P2EvalCase]:
    """Parse a strict standalone P2 JSONL dataset."""

    cases: list[P2EvalCase] = []
    first_line_by_id: dict[str, int] = {}
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read P2 evaluation dataset: {dataset_path}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("P2 evaluation row must be a JSON object")
            case = P2EvalCase.model_validate(payload)
            first_line = first_line_by_id.get(case.id)
            if first_line is not None:
                raise ValueError(
                    f"duplicate id '{case.id}' at line {line_number}; "
                    f"first defined at line {first_line}"
                )
            first_line_by_id[case.id] = line_number
            cases.append(case)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            message = str(error)
            if message.startswith("duplicate id"):
                raise ValueError(message) from error
            raise ValueError(
                f"invalid P2 evaluation dataset at line {line_number}: {error}"
            ) from error

    if not cases:
        raise ValueError("P2 evaluation dataset must not be empty")
    return cases


def run_p2_eval(
    dataset_path: Path,
    application_factory: P2EvalApplicationFactory | None = None,
) -> P2EvalSummary:
    """Run deterministic P2 cases through the real application boundary."""

    cases = load_p2_eval_cases(dataset_path)
    factory = application_factory or DeterministicP2ApplicationFactory()
    results = [evaluate_p2_case(case, factory) for case in cases]
    metrics = score_p2_eval(results)
    return P2EvalSummary(
        dataset_path=str(dataset_path),
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        pass_rate=metrics.pass_rate,
        metrics=metrics,
        results=results,
    )


def evaluate_p2_case(
    case: P2EvalCase,
    application_factory: P2EvalApplicationFactory,
) -> P2EvalResult:
    """Execute one strict P2 case through a deterministic application."""

    application = application_factory.for_case(case)
    result = application.run(_command(case))
    audit_getter = getattr(application_factory, "audit_for_case", None)
    audit = audit_getter(case.id) if callable(audit_getter) else None
    expected_source_refs = [reference.encode() for reference in case.expected_source_refs]
    actual_source_refs = _actual_source_refs(result)
    actual_intent = _actual_intent(result)
    actual_recipe_names = _actual_recipe_names(result)
    actual_errors = list(getattr(result, "errors", []))
    actual_market_price = _actual_market_price(result)
    actual_quality_decision = _actual_quality_decision(result)
    actual_comparison_statuses = _actual_comparison_statuses(result)
    intent_matched = actual_intent is case.expected_intent
    recipe_matched = actual_recipe_names == case.expected_recipe_names
    status_matched = actual_status == case.expected_status if (
        actual_status := str(getattr(result, "status"))
    ) else False
    market_metadata_valid = _market_metadata_valid(case, result)
    freshness_handled_correctly = _freshness_handled_correctly(case, result)
    comparability_correct = _comparability_correct(case, actual_comparison_statuses)
    quality_decision_correct = (
        None
        if case.expected_quality_decision is None
        else actual_quality_decision == case.expected_quality_decision
    )
    neutral_causality_safe = _neutral_causality_safe(case, result)
    cross_ticker_rejection_count = _cross_ticker_rejection_count(result)
    cross_ticker_leakage_count = _cross_ticker_leakage_count(result)
    source_policy_violations = _source_policy_violation_count(result)
    budget_violations = _budget_violation_count(audit)
    source_ref_valid = _source_refs_are_valid(
        expected_source_refs,
        actual_source_refs,
        expected_status=case.expected_status,
    )
    lifecycle_ok = _lifecycle_ok(audit, expected_status=case.expected_status)
    passed = all(
        (
            intent_matched,
            status_matched,
            recipe_matched,
            source_ref_valid,
            market_metadata_valid is not False,
            freshness_handled_correctly is not False,
            comparability_correct is not False,
            quality_decision_correct is not False,
            neutral_causality_safe is not False,
            cross_ticker_rejection_count == case.expected_cross_ticker_rejection_count,
            cross_ticker_leakage_count == case.expected_cross_ticker_leakage_count,
            cross_ticker_leakage_count == 0,
            source_policy_violations == case.expected_source_policy_violations,
            budget_violations == case.expected_budget_violations,
            lifecycle_ok,
        )
    )
    return P2EvalResult(
        id=case.id,
        kind=case.kind,
        actual_intent=actual_intent,
        actual_status=actual_status,
        actual_recipe_names=actual_recipe_names,
        actual_source_refs=actual_source_refs,
        actual_errors=actual_errors,
        actual_market_price=actual_market_price,
        actual_quality_decision=actual_quality_decision,
        actual_comparison_statuses=actual_comparison_statuses,
        intent_matched=intent_matched,
        recipe_matched=recipe_matched,
        market_metadata_valid=market_metadata_valid,
        freshness_handled_correctly=freshness_handled_correctly,
        source_ref_valid=source_ref_valid,
        cross_ticker_rejection_count=cross_ticker_rejection_count,
        cross_ticker_leakage_count=cross_ticker_leakage_count,
        comparability_correct=comparability_correct,
        quality_decision_correct=quality_decision_correct,
        neutral_causality_safe=neutral_causality_safe,
        source_policy_violations=source_policy_violations,
        budget_violations=budget_violations,
        passed=passed,
    )


def score_p2_eval(results: Sequence[P2EvalResult]) -> P2EvalMetrics:
    """Compute deterministic aggregate P2 metrics from case results."""

    if not results:
        raise ValueError("P2 evaluation results must not be empty")

    def applicable_ratio(values: list[bool | None]) -> Decimal:
        applicable = [value for value in values if value is not None]
        return _decimal_ratio(
            sum(value is True for value in applicable),
            len(applicable),
            empty=Decimal("1"),
        )

    return P2EvalMetrics(
        intent_accuracy=_decimal_ratio(
            sum(result.intent_matched for result in results),
            len(results),
        ),
        recipe_accuracy=_decimal_ratio(
            sum(result.recipe_matched for result in results),
            len(results),
        ),
        market_metadata_validity=applicable_ratio(
            [result.market_metadata_valid for result in results]
        ),
        freshness_handling_accuracy=applicable_ratio(
            [result.freshness_handled_correctly for result in results]
        ),
        source_ref_validity=_decimal_ratio(
            sum(result.source_ref_valid for result in results),
            len(results),
        ),
        cross_ticker_leakage_count=sum(
            result.cross_ticker_leakage_count for result in results
        ),
        comparability_accuracy=applicable_ratio(
            [result.comparability_correct for result in results]
        ),
        quality_decision_accuracy=applicable_ratio(
            [result.quality_decision_correct for result in results]
        ),
        neutral_causality_accuracy=applicable_ratio(
            [result.neutral_causality_safe for result in results]
        ),
        source_policy_violations=sum(result.source_policy_violations for result in results),
        budget_violations=sum(result.budget_violations for result in results),
        pass_rate=_decimal_ratio(sum(result.passed for result in results), len(results)),
    )


def run_application_experiment(
    client: object,
    *,
    dataset_name: str,
    experiment_name: str,
    application_factory: P2EvalApplicationFactory | None = None,
    cases: Sequence[P2EvalCase] | None = None,
) -> object:
    """Run the real deterministic P2 application against a Langfuse dataset."""

    canonical_cases = list(cases) if cases is not None else _load_bundled_p2_cases()
    case_by_id = {case.id: case for case in canonical_cases}
    if not case_by_id:
        raise LangfuseOperationError("P2 experiments require at least one canonical case")
    if len(case_by_id) != len(canonical_cases):
        raise ValueError("P2 experiment cases must have unique IDs")
    factory = application_factory or DeterministicP2ApplicationFactory()
    try:
        dataset = getattr(client, "get_dataset")(dataset_name)
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to fetch Langfuse dataset '{dataset_name}'"
        ) from error
    _validate_dataset_items(
        getattr(dataset, "items", None),
        case_by_id=case_by_id,
        collection_name="dataset items",
    )
    try:
        result = getattr(dataset, "run_experiment")(
            name=experiment_name,
            task=lambda *, item, **_: _run_experiment_task(
                item=item,
                case_by_id=case_by_id,
                application_factory=factory,
            ),
            evaluators=list(P2_EXPERIMENT_EVALUATORS),
            max_concurrency=1,
            metadata={"suite": "p2", "source": "canonical-jsonl"},
        )
        _validate_experiment_result(result, case_by_id=case_by_id)
        return result
    except LangfuseOperationError:
        raise
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to run Langfuse experiment '{experiment_name}' for dataset '{dataset_name}'"
        ) from error


def _load_bundled_p2_cases() -> list[P2EvalCase]:
    resource = files("financial_evidence_agent.evals").joinpath("p2_dataset.jsonl")
    with as_file(resource) as path:
        return load_p2_eval_cases(path)


def _run_experiment_task(
    *,
    item: object,
    case_by_id: Mapping[str, P2EvalCase],
    application_factory: P2EvalApplicationFactory,
) -> dict[str, object]:
    case_id = _dataset_item_id(item)
    try:
        case = case_by_id[case_id]
    except KeyError as error:
        raise LangfuseOperationError(
            f"dataset item '{case_id}' is not present in the canonical P2 dataset"
        ) from error
    _validate_dataset_item(item, case)
    return evaluate_p2_case(case, application_factory).model_dump(mode="json")


def _dataset_item_id(item: object) -> str:
    case_id = getattr(item, "id", None)
    if not isinstance(case_id, str) or not case_id.strip():
        raise LangfuseOperationError("dataset item id must be a non-empty string")
    return case_id


def _validate_dataset_item(item: object, case: P2EvalCase) -> None:
    actual_input = _dataset_item_mapping(item, "input")
    if actual_input != case.input_payload():
        raise LangfuseOperationError(
            f"dataset item '{case.id}' input does not match the canonical P2 case"
        )
    actual_expected_output = _dataset_item_mapping(item, "expected_output")
    if actual_expected_output != case.expected_payload():
        raise LangfuseOperationError(
            f"dataset item '{case.id}' expected output does not match the canonical P2 case"
        )
    actual_metadata = _dataset_item_mapping(item, "metadata")
    if actual_metadata != case.metadata_payload():
        raise LangfuseOperationError(
            f"dataset item '{case.id}' metadata does not match the canonical P2 case"
        )


def _dataset_item_mapping(item: object, field_name: str) -> dict[str, object]:
    value = getattr(item, field_name, None)
    if not isinstance(value, Mapping):
        raise LangfuseOperationError(
            f"dataset item '{_dataset_item_id(item)}' {field_name} must be a JSON object"
        )
    return dict(value)


def _accuracy_evaluation(name: str, value: bool) -> dict[str, object]:
    return {"name": name, "value": float(value)}


def _output_bool(output: Mapping[str, object], field_name: str, *, metric_name: str) -> bool:
    value = output[field_name]
    if not isinstance(value, bool):
        raise LangfuseOperationError(f"experiment output field for '{metric_name}' must be boolean")
    return value


def _output_optional_bool(
    output: Mapping[str, object],
    field_name: str,
    *,
    metric_name: str,
) -> bool | None:
    value = output[field_name]
    if value is None:
        return None
    if not isinstance(value, bool):
        raise LangfuseOperationError(
            f"experiment output field for '{metric_name}' must be boolean"
        )
    return value


def _output_non_negative_int(
    output: Mapping[str, object],
    field_name: str,
    *,
    metric_name: str,
) -> int:
    value = output[field_name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LangfuseOperationError(
            f"experiment output field for '{metric_name}' must be a non-negative integer"
        )
    return value


def _applicable_accuracy_evaluation(name: str, value: bool | None) -> list[dict[str, object]]:
    if value is None:
        return []
    return [_accuracy_evaluation(name, value)]


def _experiment_intent_evaluator(*, output: Mapping[str, object], **_: object) -> dict[str, object]:
    return _accuracy_evaluation(
        "intent_accuracy",
        _output_bool(output, "intent_matched", metric_name="intent_accuracy"),
    )


def _experiment_recipe_evaluator(*, output: Mapping[str, object], **_: object) -> dict[str, object]:
    return _accuracy_evaluation(
        "recipe_accuracy",
        _output_bool(output, "recipe_matched", metric_name="recipe_accuracy"),
    )


def _experiment_market_metadata_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return _applicable_accuracy_evaluation(
        "market_metadata_validity",
        _output_optional_bool(
            output,
            "market_metadata_valid",
            metric_name="market_metadata_validity",
        ),
    )


def _experiment_freshness_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return _applicable_accuracy_evaluation(
        "freshness_handling_accuracy",
        _output_optional_bool(
            output,
            "freshness_handled_correctly",
            metric_name="freshness_handling_accuracy",
        ),
    )


def _experiment_source_ref_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> dict[str, object]:
    return _accuracy_evaluation(
        "source_ref_validity",
        _output_bool(output, "source_ref_valid", metric_name="source_ref_validity"),
    )


def _experiment_cross_ticker_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> dict[str, object]:
    return {
        "name": "cross_ticker_leakage_count",
        "value": _output_non_negative_int(
            output,
            "cross_ticker_leakage_count",
            metric_name="cross_ticker_leakage_count",
        ),
    }


def _experiment_comparability_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return _applicable_accuracy_evaluation(
        "comparability_accuracy",
        _output_optional_bool(
            output,
            "comparability_correct",
            metric_name="comparability_accuracy",
        ),
    )


def _experiment_quality_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return _applicable_accuracy_evaluation(
        "quality_decision_accuracy",
        _output_optional_bool(
            output,
            "quality_decision_correct",
            metric_name="quality_decision_accuracy",
        ),
    )


def _experiment_neutral_causality_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return _applicable_accuracy_evaluation(
        "neutral_causality_accuracy",
        _output_optional_bool(
            output,
            "neutral_causality_safe",
            metric_name="neutral_causality_accuracy",
        ),
    )


def _experiment_source_policy_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> dict[str, object]:
    return {
        "name": "source_policy_violations",
        "value": _output_non_negative_int(
            output,
            "source_policy_violations",
            metric_name="source_policy_violations",
        ),
    }


def _experiment_budget_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> dict[str, object]:
    return {
        "name": "budget_violations",
        "value": _output_non_negative_int(
            output,
            "budget_violations",
            metric_name="budget_violations",
        ),
    }


def _experiment_overall_pass_evaluator(
    *,
    output: Mapping[str, object],
    **_: object,
) -> dict[str, object]:
    return _accuracy_evaluation(
        "overall_pass",
        _output_bool(output, "passed", metric_name="overall_pass"),
    )


def _validate_dataset_items(
    items: object,
    *,
    case_by_id: Mapping[str, P2EvalCase],
    collection_name: str,
) -> None:
    if not isinstance(items, Sequence):
        raise LangfuseOperationError(f"{collection_name} must be a sequence")
    item_list = list(items)
    _validate_case_id_collection(
        (_dataset_item_id(item) for item in item_list),
        expected_ids=case_by_id.keys(),
        collection_name=collection_name,
    )
    for item in item_list:
        _validate_dataset_item(item, case_by_id[_dataset_item_id(item)])


def _validate_case_id_collection(
    actual_ids: Iterable[str],
    *,
    expected_ids: Iterable[str],
    collection_name: str,
) -> None:
    actual_list = list(actual_ids)
    expected_list = list(expected_ids)
    if len(actual_list) != len(set(actual_list)) or set(actual_list) != set(expected_list):
        raise LangfuseOperationError(
            f"{collection_name} do not match canonical P2 case IDs"
        )


def _validate_experiment_result(
    result: object,
    *,
    case_by_id: Mapping[str, P2EvalCase],
) -> None:
    item_results = getattr(result, "item_results", None)
    if not isinstance(item_results, Sequence):
        raise LangfuseOperationError("experiment result must expose item_results")
    item_result_list = list(item_results)
    result_ids = [_experiment_result_item_id(item_result) for item_result in item_result_list]
    _validate_case_id_collection(
        result_ids,
        expected_ids=case_by_id.keys(),
        collection_name="experiment results",
    )
    for item_result in item_result_list:
        case_id = _experiment_result_item_id(item_result)
        case = case_by_id[case_id]
        item = getattr(item_result, "item", None)
        _validate_dataset_item(item, case)
        output = _experiment_result_output(item_result, case_id=case_id)
        expected_names = _expected_evaluator_names(output)
        actual_names = _evaluation_names(item_result, case_id=case_id)
        if actual_names != expected_names:
            raise LangfuseOperationError(
                f"dataset item '{case_id}' is missing evaluator results"
            )


def _experiment_result_item_id(item_result: object) -> str:
    return _dataset_item_id(getattr(item_result, "item", None))


def _experiment_result_output(
    item_result: object,
    *,
    case_id: str,
) -> Mapping[str, object]:
    output = getattr(item_result, "output", None)
    if not isinstance(output, Mapping):
        raise LangfuseOperationError(
            f"dataset item '{case_id}' experiment output must be a JSON object"
        )
    return output


def _evaluation_names(item_result: object, *, case_id: str) -> set[str]:
    evaluations = getattr(item_result, "evaluations", None)
    if not isinstance(evaluations, Sequence):
        raise LangfuseOperationError(
            f"dataset item '{case_id}' evaluations must be a sequence"
        )
    names: list[str] = []
    for evaluation in evaluations:
        if isinstance(evaluation, Mapping):
            name = evaluation.get("name")
        else:
            name = getattr(evaluation, "name", None)
        if not isinstance(name, str) or not name:
            raise LangfuseOperationError(
                f"dataset item '{case_id}' contains an evaluator without a name"
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise LangfuseOperationError(
            f"dataset item '{case_id}' contains duplicate evaluator results"
        )
    return set(names)


def _expected_evaluator_names(output: Mapping[str, object]) -> set[str]:
    names = {
        "intent_accuracy",
        "recipe_accuracy",
        "source_ref_validity",
        "cross_ticker_leakage_count",
        "source_policy_violations",
        "budget_violations",
        "overall_pass",
    }
    _output_bool(output, "intent_matched", metric_name="intent_accuracy")
    _output_bool(output, "recipe_matched", metric_name="recipe_accuracy")
    _output_bool(output, "source_ref_valid", metric_name="source_ref_validity")
    _output_non_negative_int(
        output,
        "cross_ticker_leakage_count",
        metric_name="cross_ticker_leakage_count",
    )
    _output_non_negative_int(
        output,
        "source_policy_violations",
        metric_name="source_policy_violations",
    )
    _output_non_negative_int(
        output,
        "budget_violations",
        metric_name="budget_violations",
    )
    _output_bool(output, "passed", metric_name="overall_pass")
    if _output_optional_bool(
        output,
        "market_metadata_valid",
        metric_name="market_metadata_validity",
    ) is not None:
        names.add("market_metadata_validity")
    if _output_optional_bool(
        output,
        "freshness_handled_correctly",
        metric_name="freshness_handling_accuracy",
    ) is not None:
        names.add("freshness_handling_accuracy")
    if _output_optional_bool(
        output,
        "comparability_correct",
        metric_name="comparability_accuracy",
    ) is not None:
        names.add("comparability_accuracy")
    if _output_optional_bool(
        output,
        "quality_decision_correct",
        metric_name="quality_decision_accuracy",
    ) is not None:
        names.add("quality_decision_accuracy")
    if _output_optional_bool(
        output,
        "neutral_causality_safe",
        metric_name="neutral_causality_accuracy",
    ) is not None:
        names.add("neutral_causality_accuracy")
    return names


P2_EXPERIMENT_EVALUATORS: tuple[Callable[..., object], ...] = (
    _experiment_intent_evaluator,
    _experiment_recipe_evaluator,
    _experiment_market_metadata_evaluator,
    _experiment_freshness_evaluator,
    _experiment_source_ref_evaluator,
    _experiment_cross_ticker_evaluator,
    _experiment_comparability_evaluator,
    _experiment_quality_evaluator,
    _experiment_neutral_causality_evaluator,
    _experiment_source_policy_evaluator,
    _experiment_budget_evaluator,
    _experiment_overall_pass_evaluator,
)


class DeterministicP2ApplicationFactory:
    """Build one deterministic application and audit surface per P2 case."""

    def __init__(self) -> None:
        self._audits: dict[str, _P2Audit] = {}

    def for_case(self, case: P2EvalCase) -> ResearchApplication:
        audit = _P2Audit()
        self._audits[case.id] = audit
        return ResearchApplication(
            _NeverRouter(),
            _DeterministicRuntimeFactory(case, audit),
            run_repository=audit.run_repository,
            trace_sink_factory=audit.new_trace_sink,
            id_generator=_FixedIds(case.id),
            quality_date_factory=lambda: _FIXED_QUALITY_DATE,
            company_resolver=_P2CompanyResolver(
                frozenset((case.ticker, *case.peer_tickers))
            ),
        )

    def audit_for_case(self, case_id: str) -> _P2Audit | None:
        return self._audits.get(case_id)


@dataclass(frozen=True)
class _P2CompanyResolver:
    supported: frozenset[str]

    def resolve(self, ticker: str) -> str | None:
        normalized = ticker.strip().upper()
        return normalized if normalized in self.supported else None


@dataclass(slots=True)
class _P2Audit:
    run_repository: _MemoryRunRepository = field(default_factory=lambda: _MemoryRunRepository())
    trace_roots: list[_Observation] = field(default_factory=list)
    trace_flush_calls: list[int] = field(default_factory=list)
    p1_events: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def new_trace_sink(self):
        return _TraceSink(self.trace_roots, self.trace_flush_calls)


class _Observation:
    def __init__(
        self,
        *,
        name: str,
        kind: str,
        metadata: dict[str, object],
        trace_id: str | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.metadata = metadata
        self.trace_id = trace_id
        self.output: dict[str, object] | None = None
        self.children: list[_Observation] = []

    def update(
        self,
        *,
        output: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata.update(metadata)

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: str,
        input: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Iterator[_Observation]:
        del input
        child = _Observation(name=name, kind=kind, metadata=dict(metadata or {}))
        self.children.append(child)
        yield child


class _TraceSink:
    def __init__(self, roots: list[_Observation], flush_calls: list[int]) -> None:
        self._roots = roots
        self._flush_calls = flush_calls

    @contextmanager
    def run(
        self,
        *,
        run_id: str,
        input: object,
        metadata: dict[str, object],
    ) -> Iterator[_Observation]:
        del input
        root = _Observation(
            name="financial-evidence-agent.run",
            kind="agent",
            metadata=dict(metadata),
            trace_id=f"trace-{run_id}",
        )
        self._roots.append(root)
        yield root

    def flush(self) -> None:
        self._flush_calls.append(1)


class _MemoryRunRepository:
    def __init__(self) -> None:
        self.started: list[RunStart] = []
        self.finished: list[RunFinish] = []
        self.fetches: list[object] = []

    def start(self, value: RunStart) -> None:
        self.started.append(value)

    def finish(self, value: RunFinish) -> None:
        self.finished.append(value)

    def record_fetch(self, value: SourceFetchWrite) -> None:
        self.fetches.append(value)


class _P1TraceSink:
    def __init__(self, audit: _P2Audit) -> None:
        self._audit = audit

    def record(self, node: str, attributes: dict[str, object]) -> None:
        self._audit.p1_events.append((node, dict(attributes)))


class _NeverRouter:
    def route(self, request: str) -> RouterDecision:
        raise AssertionError(f"explicit P2 eval path called the semantic router: {request}")


class _FixedIds:
    def __init__(self, case_id: str) -> None:
        self._case_id = case_id
        self._next_id = 0

    def new_run_id(self) -> str:
        run_id = f"{self._case_id}-run-{self._next_id}"
        self._next_id += 1
        return run_id


@dataclass(frozen=True, slots=True)
class _DirectMarketRuntime:
    dependencies: MarketDependencies

    def execute(
        self,
        command: ResearchCommand,
        decision: RouterDecision,
    ) -> MarketResearchResult:
        assert decision.intent is Intent.MARKET_SNAPSHOT_REQUEST
        return run_market_workflow(command, self.dependencies)


@dataclass(frozen=True, slots=True)
class _RaisingIndustryRuntime:
    error: Exception

    def execute(self, command: ResearchCommand, decision: RouterDecision) -> object:
        del command, decision
        raise self.error


@dataclass(frozen=True, slots=True)
class _CrossTickerAttackQualityRuntime:
    runtime: QualityResearchRuntime

    def execute(self, command: ResearchCommand, decision: RouterDecision) -> QualityResearchResult:
        result = self.runtime.execute(command, decision)
        assert result.package is not None
        return result.model_copy(
            update={"package": _inject_cross_ticker_attack(result.package)}
        )


class _DeterministicRuntimeFactory:
    def __init__(self, case: P2EvalCase, audit: _P2Audit) -> None:
        self._case = case
        self._audit = audit

    def build(self, command: ResearchCommand, intent: Intent) -> object:
        if intent is Intent.MARKET_SNAPSHOT_REQUEST:
            return _DirectMarketRuntime(_market_dependencies(self._case))
        if intent is Intent.INDUSTRY_RESEARCH_REQUEST:
            return ResearchRuntime(
                dependencies=_p1_dependencies(self._case, self._audit, ticker=command.ticker),
                source_policy_version=_industry_policy().version,
            )
        if intent is Intent.RESEARCH_QUALITY_SCREEN_REQUEST:
            return _quality_runtime(self._case, self._audit, ticker=command.ticker)
        raise AssertionError(f"unsupported deterministic P2 intent: {intent.value}")


def _command(case: P2EvalCase) -> ResearchCommand:
    return ResearchCommand(
        ticker=case.ticker,
        request=case.request,
        mode=case.mode,
        peer_tickers=case.peer_tickers,
        peer_scope=case.peer_scope,
        with_context=case.with_context,
    )


def _actual_intent(result: object) -> Intent:
    if isinstance(result, MarketResearchResult):
        return Intent.MARKET_SNAPSHOT_REQUEST
    if isinstance(result, PeerResearchResult):
        return Intent.INDUSTRY_RESEARCH_REQUEST
    if isinstance(result, QualityResearchResult):
        return Intent.RESEARCH_QUALITY_SCREEN_REQUEST
    if isinstance(result, ResearchResult):
        return result.decision.intent
    decision = getattr(result, "decision", None)
    if decision is not None:
        return decision.intent
    raise TypeError("unsupported P2 evaluation result type")


def _actual_recipe_names(result: object) -> list[str]:
    metadata = getattr(result, "root_metadata", None)
    if callable(metadata):
        recipe_names = metadata().get("recipe_names")
        if isinstance(recipe_names, list):
            return [str(value) for value in recipe_names]
    return []


def _actual_source_refs(result: object) -> list[str]:
    if isinstance(result, MarketResearchResult):
        if result.guarded_report is None:
            return []
        return [reference.encode() for reference in result.guarded_report.source_refs]
    guarded_report = getattr(result, "guarded_report", None)
    if guarded_report is not None:
        return [source.ref.encode() for source in guarded_report.retained_sources]
    if isinstance(result, QualityResearchResult):
        return [reference.encode() for reference in result.quality.source_refs]
    return []


def _actual_market_price(result: object) -> Decimal | None:
    if not isinstance(result, MarketResearchResult):
        return None
    if result.guarded_report is None or result.guarded_report.snapshot is None:
        return None
    return result.guarded_report.snapshot.price


def _actual_quality_decision(result: object) -> ResearchQualityDecision | None:
    if isinstance(result, QualityResearchResult):
        return result.quality.decision
    guarded_report = getattr(result, "guarded_report", None)
    quality = getattr(guarded_report, "quality", None)
    if quality is not None:
        return quality.decision
    return None


def _actual_comparison_statuses(result: object) -> list[str]:
    guarded_report = getattr(result, "guarded_report", None)
    comparisons = getattr(guarded_report, "comparisons", None)
    if comparisons is None:
        return []
    return [comparison.status.value for comparison in comparisons]


def _market_metadata_valid(case: P2EvalCase, result: object) -> bool | None:
    if case.expected_market_price is None:
        return None
    if not isinstance(result, MarketResearchResult):
        return False
    report = result.guarded_report
    if report is None or report.snapshot is None:
        return False
    snapshot = report.snapshot
    return (
        snapshot.provider == "alpaca"
        and snapshot.feed == "iex"
        and snapshot.coverage == "IEX-only"
        and snapshot.symbol == case.ticker
        and snapshot.exchange == "IEX"
        and snapshot.currency == "USD"
        and snapshot.price == case.expected_market_price
    )


def _freshness_handled_correctly(case: P2EvalCase, result: object) -> bool | None:
    if case.kind not in _MARKET_KINDS:
        return None
    if not isinstance(result, MarketResearchResult):
        return False
    report = result.guarded_report
    if list(result.errors) != case.expected_market_errors:
        return False
    if report is None:
        return False
    if case.expected_freshness_label is None:
        return report.freshness_label is None
    return report.freshness_label == case.expected_freshness_label


def _comparability_correct(
    case: P2EvalCase,
    actual_statuses: list[str],
) -> bool | None:
    if not case.expected_comparison_statuses:
        return None
    expected = [status.value for status in case.expected_comparison_statuses]
    return actual_statuses == expected


def _neutral_causality_safe(case: P2EvalCase, result: object) -> bool | None:
    if case.expected_cause_assessment is None:
        return None
    if not isinstance(result, MarketResearchResult):
        return False
    report = result.guarded_report
    if report is None or report.market_context is None:
        return False
    return report.market_context.cause_assessment == case.expected_cause_assessment


def _cross_ticker_leakage_count(result: object) -> int:
    guarded_report = getattr(result, "guarded_report", None)
    return int(getattr(guarded_report, "cross_ticker_leakage_count", 0) or 0)


def _cross_ticker_rejection_count(result: object) -> int:
    guarded_report = getattr(result, "guarded_report", None)
    return int(getattr(guarded_report, "cross_ticker_rejection_count", 0) or 0)


def _source_policy_violation_count(result: object) -> int:
    count = 0
    if isinstance(result, MarketResearchResult):
        report = result.guarded_report
        if report is not None and report.market_context is not None:
            count += sum(
                event.source_kind not in _ALLOWED_WEB_SOURCE_KINDS
                for event in report.market_context.events
            )
        return count

    guarded_report = getattr(result, "guarded_report", None)
    if guarded_report is None:
        return 0
    for package in guarded_report.packages:
        count += sum(
            source.source_kind not in _ALLOWED_WEB_SOURCE_KINDS for source in package.web_sources
        )
    return count


def _budget_violation_count(audit: _P2Audit | None) -> int:
    if audit is None:
        return 0
    violations = 0
    for observation in _iter_trace_observations(audit):
        if observation.name != "graph.p1_recipe":
            continue
        attributes = observation.metadata
        recipe_name = str(attributes.get("recipe_name", ""))
        recipe = _RECIPE_BY_NAME.get(recipe_name)
        if recipe is None:
            violations += 1
            continue
        retrieval_rounds = int(attributes.get("retrieval_rounds", 0))
        web_calls = int(attributes.get("web_calls", 0))
        if (
            retrieval_rounds > recipe.budget.max_retrieval_rounds
            or web_calls > recipe.budget.max_web_calls
        ):
            violations += 1
    return violations


def _source_refs_are_valid(
    expected: list[str],
    actual: list[str],
    *,
    expected_status: P2Status,
) -> bool:
    try:
        decoded = [SourceRef.decode(value) for value in actual]
    except ValueError:
        return False
    if expected_status in {"failed", "refused"} and not expected:
        return actual == []
    actual_set = {reference.encode() for reference in decoded}
    return set(expected).issubset(actual_set)


def _lifecycle_ok(
    audit: _P2Audit | None,
    *,
    expected_status: P2Status,
) -> bool:
    if audit is None:
        return True
    if not audit.run_repository.started or not audit.run_repository.finished:
        return False
    if expected_status == "refused":
        return True
    return bool(audit.trace_roots)


def _iter_trace_observations(audit: _P2Audit) -> Iterator[_Observation]:
    pending = list(audit.trace_roots)
    while pending:
        observation = pending.pop(0)
        yield observation
        pending.extend(observation.children)


def _decimal_ratio(
    numerator: int,
    denominator: int,
    *,
    empty: Decimal | None = None,
) -> Decimal:
    if denominator == 0:
        if empty is None:
            raise ValueError("ratio denominator must be positive")
        return empty
    return Decimal(numerator) / Decimal(denominator)


def _market_dependencies(case: P2EvalCase) -> MarketDependencies:
    if case.kind == "market_complete":
        return MarketDependencies(
            _RecordingMCP(
                [
                    _tool_result(
                        {
                            "snapshot": _snapshot("market-complete").model_dump(mode="json"),
                            "error": None,
                        }
                    ),
                    _tool_result(
                        {
                            "bars": [_bar("market-complete").model_dump(mode="json")],
                            "error": None,
                        }
                    ),
                ]
            ),
            max_bars=1,
            clock=SequenceClock(_FIXED_MARKET_GUARD_TIME),
        )
    if case.kind == "market_stale":
        return MarketDependencies(
            _RecordingMCP(
                [
                    _tool_result(
                        {
                            "snapshot": _snapshot(
                                "market-stale",
                                market_status=MarketStatus.OPEN,
                                as_of=_FIXED_MARKET_GUARD_TIME - timedelta(minutes=5),
                                fetched_at=_FIXED_MARKET_GUARD_TIME
                                - timedelta(minutes=4, seconds=30),
                            ).model_dump(mode="json"),
                            "error": None,
                        }
                    ),
                    _tool_result(
                        {
                            "bars": [_bar("market-stale").model_dump(mode="json")],
                            "error": None,
                        }
                    ),
                ]
            ),
            max_bars=1,
            clock=SequenceClock(_FIXED_MARKET_GUARD_TIME),
        )
    if case.kind == "market_unavailable":
        return MarketDependencies(
            _RecordingMCP(
                [
                    _tool_result(
                        {
                            "snapshot": None,
                            "error": {
                                "code": "MARKET_DATA_UNAVAILABLE",
                                "message": "offline deterministic unavailability",
                            },
                        }
                    )
                ]
            ),
            max_bars=1,
            clock=SequenceClock(_FIXED_MARKET_GUARD_TIME),
        )
    if case.kind == "market_context_unknown":
        return MarketDependencies(
            _RecordingMCP(
                [
                    _tool_result(
                        {
                            "snapshot": _snapshot("market-context-unknown").model_dump(mode="json"),
                            "error": None,
                        }
                    ),
                    _tool_result(
                        {
                            "bars": [_bar("market-context-unknown").model_dump(mode="json")],
                            "error": None,
                        }
                    ),
                    _tool_result({"evidence": [], "error": None}),
                ]
            ),
            max_bars=1,
            context_window_days=3,
            clock=SequenceClock(_FIXED_MARKET_GUARD_TIME, _FIXED_MARKET_GUARD_TIME),
        )
    raise AssertionError(f"unsupported market case: {case.kind}")


def _p1_dependencies(
    case: P2EvalCase,
    audit: _P2Audit,
    *,
    ticker: str,
) -> Dependencies:
    validator = _web_validator(ticker)
    return Dependencies(
        mcp_client=_UnusedP0MCPClient(),
        fast_model=_OfflineP0Model(),
        analyst_model=_OfflineP0Analyst(),
        trace_sink=_P1TraceSink(audit),
        skill_planner=_IndustryPlanner(),
        skill_collector=_IndustryCollector(case, ticker=ticker),
        skill_analyst=_IndustryAnalyst(case, ticker=ticker),
        skill_run_repository=_DeterministicSkillRunRepository(),
        web_evidence_validator=validator,
    )


def _quality_runtime(
    case: P2EvalCase,
    audit: _P2Audit,
    *,
    ticker: str,
) -> object:
    quality_kwargs: dict[str, object] = {
        "current_date_factory": lambda: _FIXED_QUALITY_DATE,
        "max_source_age_days": 365,
        "source_policy_version": _industry_policy().version,
    }
    if case.kind == "cross_ticker_attack":
        quality_kwargs["guarded_package"] = _cross_ticker_attack_package(ticker)
        return _CrossTickerAttackQualityRuntime(QualityResearchRuntime(**quality_kwargs))
    else:
        quality_kwargs["dependencies"] = _p1_dependencies(case, audit, ticker=ticker)
    return QualityResearchRuntime(**quality_kwargs)


class _UnusedP0MCPClient:
    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"P2 evaluation entered the P0 MCP path: {name} {arguments!r}")


class _OfflineP0Model:
    def route(self, thesis: str) -> RouterDecision:
        return RouterDecision(intent=Intent.AMBIGUOUS, reason=f"offline P2 eval: {thesis[:20]}")

    def plan(
        self,
        ticker: str,
        thesis: str,
        *,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> list[ResearchQuestion]:
        del memory_hints
        raise AssertionError(f"P2 evaluation entered the P0 planner: {ticker} {thesis}")


class _OfflineP0Analyst:
    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> object:
        raise AssertionError(f"P2 evaluation entered the P0 analyst: {questions!r} {evidence!r}")


class _IndustryPlanner:
    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        return [
            ResearchQuestion(
                question=f"What retained evidence covers {request.recipe.name.value}?",
                support_query="supporting disclosure",
                challenge_query="challenging disclosure",
            )
        ]


class _IndustryCollector:
    def __init__(self, case: P2EvalCase, *, ticker: str) -> None:
        self._case = case
        self._ticker = ticker.strip().upper()

    async def collect(
        self,
        *,
        ticker: str,
        recipe,
        questions: Sequence[ResearchQuestion],
    ):
        del questions
        assert ticker.strip().upper() == self._ticker
        if self._case.kind == "peer_partial" and self._ticker == "INTC":
            raise RuntimeError("deterministic peer failure")
        filing = _filing(recipe.name.value, ticker=self._ticker)
        web = _web(recipe.name.value, ticker=self._ticker)
        from financial_evidence_agent.retrieval.collector import EvidenceBundle
        from financial_evidence_agent.retrieval.coverage import (
            CoverageReport,
            EvidenceAssignment,
            EvidenceSide,
            FacetAssignment,
        )

        assignments = [
            EvidenceAssignment(
                question_index=0,
                side=side,
                source_id=source.id,
                source_kind=(
                    SourceKind.FILING if isinstance(source, EvidenceChunk) else source.source_kind
                ),
            )
            for source in (filing, web)
            for side in EvidenceSide
        ]
        facet_assignments = [
            FacetAssignment(
                question_index=0,
                side=side,
                facet=facet,
                source_id=source.id,
            )
            for source in (filing, web)
            for side in EvidenceSide
            for facet in recipe.required_facets
        ]
        if self._case.kind in {"peer_partial", "peer_not_comparable"}:
            facet_assignments.extend(
                [
                    FacetAssignment(
                        question_index=0,
                        side=side,
                        facet=ResearchFacet.DATA_VERIFICATION,
                        source_id=filing.id,
                    )
                    for side in EvidenceSide
                ]
            )
        return EvidenceBundle(
            filing_evidence=[filing],
            web_evidence=[web],
            assignments=assignments,
            facet_assignments=facet_assignments,
            coverage=CoverageReport(
                complete=True,
                missing_facets=(),
                missing_pairs=(),
                invalid_source_ids=(),
                ticker_mismatches=(),
                date_mismatches=(),
                new_valid_source_count=2,
                reason_codes=(),
            ),
            retrieval_rounds=1,
            web_calls=0,
        )


class _IndustryAnalyst:
    def __init__(self, case: P2EvalCase, *, ticker: str) -> None:
        self._case = case
        self._ticker = ticker.strip().upper()

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence,
    ) -> SkillResearchMemo:
        filing_id = evidence.filing_evidence[0].id
        web_id = evidence.web_evidence[0].id
        sections = [
            SkillResearchSection(
                facet=facet,
                claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text=f"{self._ticker} retained {facet.value} evidence.",
                        confidence=Confidence.HIGH,
                        evidence_chunk_ids=[filing_id],
                        web_evidence_ids=[web_id],
                    )
                ],
            )
            for facet in request.recipe.required_facets
        ]
        return SkillResearchMemo(
            recipe_name=request.recipe.name,
            recipe_version=request.recipe.version,
            research_question=f"Review {request.recipe.name.value}.",
            sections=sections,
            data_points=_data_points(self._case, self._ticker, filing_id),
            information_sufficiency=InformationSufficiency.SUFFICIENT,
            information_gaps=[],
            confidence=Decimal("0.9"),
        )


class _DeterministicSkillRunRepository:
    def __init__(self) -> None:
        self._next_id = 0

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
        run_id = f"p2-skill-run-{self._next_id}"
        self._next_id += 1
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: Sequence[SourceRef],
        errors: Sequence[str],
    ) -> None:
        del run_id, status, source_ids, errors


class _RecordingMCP:
    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        del name, arguments
        if not self._responses:
            raise AssertionError("deterministic market MCP responses exhausted")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        assert self._values, "deterministic market clock exhausted"
        return self._values.pop(0)


def _snapshot(
    case_id: str,
    *,
    market_status: MarketStatus = MarketStatus.CLOSED,
    as_of: datetime | None = None,
    fetched_at: datetime | None = None,
) -> MarketSnapshot:
    observed_fetched_at = fetched_at or (_FIXED_MARKET_GUARD_TIME - timedelta(minutes=1))
    observed_as_of = as_of or (observed_fetched_at - timedelta(seconds=60))
    raw_payload_hash = sha256(f"snapshot:{case_id}".encode()).hexdigest()
    return MarketSnapshot(
        id=market_snapshot_observation_id(
            provider="alpaca",
            feed="iex",
            symbol="NVDA",
            as_of=observed_as_of,
            fetched_at=observed_fetched_at,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        price=Decimal("123.450000000000000001"),
        open=Decimal("122.00"),
        day_high=Decimal("124.00"),
        day_low=Decimal("121.50"),
        previous_close=Decimal("121.00"),
        as_of=observed_as_of,
        fetched_at=observed_fetched_at,
        market_status=market_status,
        delayed_by_seconds=max(0, int((observed_fetched_at - observed_as_of).total_seconds())),
        raw_payload_hash=raw_payload_hash,
    )


def _bar(case_id: str) -> MarketBar:
    timestamp = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)
    fetched_at = _FIXED_MARKET_GUARD_TIME - timedelta(minutes=1)
    raw_payload_hash = sha256(f"bar:{case_id}".encode()).hexdigest()
    return MarketBar(
        id=market_bar_observation_id(
            provider="alpaca",
            feed="iex",
            symbol="NVDA",
            timestamp=timestamp,
            fetched_at=fetched_at,
            raw_payload_hash=raw_payload_hash,
        ),
        provider="alpaca",
        feed="iex",
        coverage="IEX-only",
        symbol="NVDA",
        exchange="IEX",
        currency="USD",
        interval="1Day",
        timestamp=timestamp,
        open=Decimal("120.00"),
        high=Decimal("124.00"),
        low=Decimal("119.00"),
        close=Decimal("123.45"),
        volume=1000,
        fetched_at=fetched_at,
        raw_payload_hash=raw_payload_hash,
    )


def _tool_result(payload: Mapping[str, object]) -> CallToolResult:
    return CallToolResult(
        content=[],
        structured_content=dict(payload),  # type: ignore[arg-type]
        meta=None,
        is_error=False,
    )


def _filing(recipe_name: str, *, ticker: str) -> EvidenceChunk:
    return EvidenceChunk(
        id=f"sec-{ticker.lower()}-{recipe_name.replace('_', '-')}",
        ticker=ticker,
        corpus_version=f"{ticker}-p2-v1",
        content=f"{ticker} filing evidence for {recipe_name}.",
        source_url=f"https://www.sec.gov/Archives/{ticker.lower()}-{recipe_name}.htm",
        form="10-Q",
        filed_at=date(2026, 5, 20),
        accession_no="0001045810-26-000001",
        section="MD&A",
        raw_start=0,
        raw_end=40,
    )


def _web(recipe_name: str, *, ticker: str) -> WebEvidence:
    source_url = f"https://investor.{ticker.lower()}.example.com/{recipe_name}"
    content = f"{ticker} issuer web evidence for {recipe_name}."
    content_hash = sha256(content.encode("utf-8")).hexdigest()
    return WebEvidence(
        id=content_addressed_web_evidence_id(ticker, source_url, content_hash),
        ticker=ticker,
        title=f"{ticker} issuer evidence for {recipe_name}",
        content=content,
        source_url=source_url,
        source_kind=SourceKind.ISSUER_IR,
        source_tier=SourceTier.PRIMARY,
        published_at=datetime(2026, 5, 18, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, tzinfo=UTC),
        content_hash=content_hash,
    )


def _data_points(case: P2EvalCase, ticker: str, filing_id: str) -> list[FinancialDataPoint]:
    if case.kind not in {"peer_partial", "peer_not_comparable"}:
        return []
    unit = "millions" if case.kind == "peer_not_comparable" and ticker == "AMD" else "billions"
    value = Decimal("27.100") if ticker == "AMD" else Decimal("44.062")
    return [
        FinancialDataPoint(
            name="Revenue",
            value=value,
            currency="USD",
            unit=unit,
            period_start=date(2026, 2, 1),
            period_end=date(2026, 4, 30),
            definition="GAAP revenue",
            source_ids=[filing_id],
        )
    ]


def _industry_policy() -> SourcePolicy:
    return SourcePolicy(issuer_domains=_issuer_domains())


def _issuer_domains() -> dict[str, frozenset[str]]:
    return {
        "NVDA": frozenset({"investor.nvda.example.com"}),
        "AMD": frozenset({"investor.amd.example.com"}),
        "INTC": frozenset({"investor.intc.example.com"}),
    }


def _web_validator(ticker: str) -> PersistedWebEvidenceValidator:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = WebEvidenceRepository(engine)
    normalized_ticker = ticker.strip().upper()
    repository.upsert(_web(INDUSTRY_RESEARCH.name.value, ticker=normalized_ticker))
    repository.upsert(_web(SkillName.COMPANY_DEEP_RESEARCH.value, ticker=normalized_ticker))
    return PersistedWebEvidenceValidator(_industry_policy(), repository)


def _cross_ticker_attack_package(ticker: str) -> GuardedResearchPackage:
    normalized_ticker = ticker.strip().upper()
    filing = _filing(SkillName.COMPANY_DEEP_RESEARCH.value, ticker=normalized_ticker)
    valid_package = GuardedResearchPackage(
        ticker=normalized_ticker,
        claims=[
            PackageClaim(
                facet=ResearchFacet.COMPANY_OVERVIEW,
                kind=ClaimKind.VERIFIED_FACT,
                text=f"{normalized_ticker} retained company evidence.",
                confidence=Confidence.HIGH,
                source_refs=[
                    SourceRef(
                        ticker=normalized_ticker,
                        kind=SourceRefKind.FILING,
                        source_id=filing.id,
                    )
                ],
            )
        ],
        financial_metrics=[],
        filing_sources=[filing],
        web_sources=[],
        provenance=ReportProvenance(
            recipes=(
                RecipeProvenance(
                    name=SkillName.COMPANY_DEEP_RESEARCH,
                    version="1.0.0",
                ),
            ),
            corpus_versions=(filing.corpus_version,),
            evidence_cutoff_dates=(filing.filed_at,),
            information_sufficiency=InformationSufficiency.PARTIAL,
            source_refs=(
                SourceRef(
                    ticker=normalized_ticker,
                    kind=SourceRefKind.FILING,
                    source_id=filing.id,
                ),
            ),
        ),
        evidence_dates=[filing.filed_at],
        coverage="partial",
        information_gaps=["Counterevidence remains limited."],
        guard_notes=[],
    )
    return valid_package


def _inject_cross_ticker_attack(package: GuardedResearchPackage) -> GuardedResearchPackage:
    rejected_claim = PackageClaim.model_construct(
        facet=ResearchFacet.COMPANY_OVERVIEW,
        kind=ClaimKind.VERIFIED_FACT,
        text="Injected AMD source must be rejected.",
        confidence=Confidence.HIGH,
        source_refs=[
            SourceRef(
                ticker="AMD",
                kind=SourceRefKind.FILING,
                source_id="sec-amd-cross-ticker-attack",
            )
        ],
    )
    return package.model_copy(update={"claims": [*package.claims, rejected_claim]})


__all__ = [
    "DeterministicP2ApplicationFactory",
    "P2EvalApplicationFactory",
    "P2EvalCase",
    "P2EvalMetrics",
    "P2EvalResult",
    "P2EvalSummary",
    "evaluate_p2_case",
    "load_p2_eval_cases",
    "run_application_experiment",
    "run_p2_eval",
    "score_p2_eval",
]
