"""Contract tests for the explicit P2 Langfuse dataset experiment path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult

from fra.contracts import ResearchCommand
from fra.evals.p2_runner import (
    DeterministicP2ApplicationFactory,
    load_p2_eval_cases,
    run_application_experiment,
)
from fra.observability import LangfuseOperationError


@dataclass(slots=True)
class FakeDatasetItem:
    id: str
    input: dict[str, object]
    expected_output: dict[str, object]
    metadata: dict[str, object]
    dataset_id: str = "dataset-1"


class FakeDataset:
    def __init__(
        self,
        items: list[FakeDatasetItem],
        *,
        dropped_item_ids: set[str] | None = None,
        missing_evaluation_names_by_item: dict[str, set[str]] | None = None,
        output_mutations: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.items = items
        self.calls: list[dict[str, object]] = []
        self.dropped_item_ids = dropped_item_ids or set()
        self.missing_evaluation_names_by_item = missing_evaluation_names_by_item or {}
        self.output_mutations = output_mutations or {}

    def run_experiment(self, **kwargs: object) -> ExperimentResult:
        self.calls.append(kwargs)
        item_results: list[ExperimentItemResult] = []
        for item in self.items:
            if item.id in self.dropped_item_ids:
                continue
            output = kwargs["task"](item=item)
            output = {
                **output,
                **self.output_mutations.get(item.id, {}),
            }
            evaluations: list[Evaluation] = []
            for evaluator in kwargs["evaluators"]:
                for evaluation in _normalize_evaluations(
                    evaluator(
                        input=item.input,
                        output=output,
                        expected_output=item.expected_output,
                        metadata=item.metadata,
                    )
                ):
                    if evaluation["name"] in self.missing_evaluation_names_by_item.get(
                        item.id,
                        set(),
                    ):
                        continue
                    evaluations.append(Evaluation(**evaluation))
            item_results.append(
                ExperimentItemResult(
                    item=item,
                    output=output,
                    evaluations=evaluations,
                    trace_id=f"trace-{item.id}",
                    dataset_run_id="dataset-run-1",
                )
            )
        return ExperimentResult(
            name=str(kwargs["name"]),
            run_name="dataset-run-1",
            description=None,
            item_results=item_results,
            run_evaluations=[],
            experiment_id="dataset-run-1",
            dataset_run_id="dataset-run-1",
        )


class FakeLangfuseClient:
    def __init__(self, dataset: FakeDataset) -> None:
        self.dataset = dataset
        self.dataset_names: list[str] = []

    def get_dataset(self, name: str) -> FakeDataset:
        self.dataset_names.append(name)
        return self.dataset


class RecordingApplicationFactory(DeterministicP2ApplicationFactory):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[ResearchCommand] = []

    def for_case(self, case):
        application = super().for_case(case)
        run = application.run

        def record(command: ResearchCommand):
            self.commands.append(command)
            return run(command)

        application.run = record  # type: ignore[method-assign]
        return application


def _dataset_item(case_id: str) -> FakeDatasetItem:
    case = next(
        candidate
        for candidate in load_p2_eval_cases(
            Path("src/fra/evals/p2_dataset.jsonl")
        )
        if candidate.id == case_id
    )
    return FakeDatasetItem(
        id=case.id,
        input=case.input_payload(),
        expected_output=case.expected_payload(),
        metadata=case.metadata_payload(),
    )


def _cases(*case_ids: str):
    allowed = set(case_ids)
    return [
        candidate
        for candidate in load_p2_eval_cases(
            Path("src/fra/evals/p2_dataset.jsonl")
        )
        if candidate.id in allowed
    ]


def _normalize_evaluations(result: object) -> list[dict[str, object]]:
    if isinstance(result, list):
        return [value for value in result if isinstance(value, dict)]
    if isinstance(result, dict):
        return [result]
    return []


def test_application_experiment_uses_dataset_client_and_real_task() -> None:
    dataset = FakeDataset([_dataset_item("market-complete")])
    client = FakeLangfuseClient(dataset)
    applications = RecordingApplicationFactory()
    expected_command = ResearchCommand(
        ticker="NVDA",
        request="What is the current price?",
        mode="market-snapshot",
        peer_tickers=(),
        peer_scope=None,
        with_context=False,
    )

    result = run_application_experiment(
        client,
        dataset_name="financial-evidence-p2",
        experiment_name="p2-local",
        application_factory=applications,
        cases=_cases("market-complete"),
    )

    assert [item.item.id for item in result.item_results] == ["market-complete"]
    assert applications.commands == [expected_command]
    assert client.dataset_names == ["financial-evidence-p2"]
    call = dataset.calls[0]
    assert call["name"] == "p2-local"
    assert call["max_concurrency"] == 1
    assert call["metadata"] == {"suite": "p2", "source": "canonical-jsonl"}
    task = call["task"]
    output = task(item=_dataset_item("market-complete"))
    assert applications.commands == [expected_command, expected_command]
    assert output["id"] == "market-complete"
    assert output["actual_status"] == "completed"


def test_application_experiment_registers_p2_metric_evaluators() -> None:
    dataset = FakeDataset([_dataset_item("market-context-unknown")])
    client = FakeLangfuseClient(dataset)

    run_application_experiment(
        client,
        dataset_name="financial-evidence-p2",
        experiment_name="p2-local",
        cases=_cases("market-context-unknown"),
    )

    call = dataset.calls[0]
    task = call["task"]
    item = _dataset_item("market-context-unknown")
    output = task(item=item)
    names = {
        evaluation["name"]
        for evaluator in call["evaluators"]
        for evaluation in _normalize_evaluations(
            evaluator(
                input=item.input,
                output=output,
                expected_output=item.expected_output,
                metadata=item.metadata,
            )
        )
    }
    assert {
        "intent_accuracy",
        "recipe_accuracy",
        "market_metadata_validity",
        "freshness_handling_accuracy",
        "source_ref_validity",
        "cross_ticker_leakage_count",
        "neutral_causality_accuracy",
        "source_policy_violations",
        "budget_violations",
        "overall_pass",
    } <= names


@pytest.mark.parametrize(
    ("items", "cases", "message"),
    [
        (
            [_dataset_item("market-complete"), _dataset_item("market-context-unknown")],
            _cases("market-complete"),
            "dataset items do not match canonical P2 case IDs",
        ),
        (
            [_dataset_item("market-complete"), _dataset_item("market-complete")],
            _cases("market-complete"),
            "dataset items do not match canonical P2 case IDs",
        ),
    ],
)
def test_application_experiment_rejects_dataset_item_id_mismatches_before_running(
    items: list[FakeDatasetItem],
    cases,
    message: str,
) -> None:
    dataset = FakeDataset(items)
    client = FakeLangfuseClient(dataset)

    with pytest.raises(LangfuseOperationError, match=message):
        run_application_experiment(
            client,
            dataset_name="financial-evidence-p2",
            experiment_name="p2-local",
            cases=cases,
        )

    assert dataset.calls == []


def test_application_experiment_fails_when_langfuse_drops_a_task_result() -> None:
    cases = _cases("market-complete", "market-context-unknown")
    dataset = FakeDataset(
        [_dataset_item("market-complete"), _dataset_item("market-context-unknown")],
        dropped_item_ids={"market-context-unknown"},
    )
    client = FakeLangfuseClient(dataset)

    with pytest.raises(
        LangfuseOperationError,
        match="experiment results do not match canonical P2 case IDs",
    ):
        run_application_experiment(
            client,
            dataset_name="financial-evidence-p2",
            experiment_name="p2-local",
            cases=cases,
        )

    assert len(dataset.calls) == 1


def test_application_experiment_fails_when_langfuse_drops_an_applicable_evaluator() -> None:
    dataset = FakeDataset(
        [_dataset_item("market-context-unknown")],
        missing_evaluation_names_by_item={
            "market-context-unknown": {"neutral_causality_accuracy"}
        },
    )
    client = FakeLangfuseClient(dataset)

    with pytest.raises(
        LangfuseOperationError,
        match="dataset item 'market-context-unknown' is missing evaluator results",
    ):
        run_application_experiment(
            client,
            dataset_name="financial-evidence-p2",
            experiment_name="p2-local",
            cases=_cases("market-context-unknown"),
        )


def test_application_experiment_rejects_non_boolean_metric_output() -> None:
    dataset = FakeDataset(
        [_dataset_item("market-context-unknown")],
        output_mutations={
            "market-context-unknown": {"neutral_causality_safe": "false"}
        },
    )
    client = FakeLangfuseClient(dataset)

    with pytest.raises(
        LangfuseOperationError,
        match="experiment output field for 'neutral_causality_accuracy' must be boolean",
    ):
        run_application_experiment(
            client,
            dataset_name="financial-evidence-p2",
            experiment_name="p2-local",
            cases=_cases("market-context-unknown"),
        )
