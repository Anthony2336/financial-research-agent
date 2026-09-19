"""Integration coverage for the dataset-backed Langfuse application experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult

from fra.contracts import ResearchCommand
from fra.evals.p2_runner import (
    DeterministicP2ApplicationFactory,
    load_p2_eval_cases,
    run_application_experiment,
)
from fra.evals.runner import (
    load_comparable_application_experiment_cases,
    run_comparable_application_experiment,
)
from fra.observability import sync_langfuse_dataset


@dataclass(slots=True)
class FakeDatasetItem:
    id: str
    input: dict[str, object]
    expected_output: dict[str, object]
    metadata: dict[str, object]


@dataclass(slots=True)
class FakeApiError(Exception):
    status_code: int


class FakeDataset:
    def __init__(self) -> None:
        self.items: list[FakeDatasetItem] = []
        self.calls: list[dict[str, object]] = []

    def run_experiment(self, **kwargs: object) -> ExperimentResult:
        self.calls.append(kwargs)
        item_results: list[ExperimentItemResult] = []
        for item in self.items:
            output = kwargs["task"](item=item)
            evaluations: list[Evaluation] = []
            for evaluator in kwargs["evaluators"]:
                result = evaluator(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
                if isinstance(result, list):
                    evaluations.extend(Evaluation(**value) for value in result)
                elif isinstance(result, dict):
                    evaluations.append(Evaluation(**result))
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
    def __init__(self) -> None:
        self.datasets: dict[str, FakeDataset] = {}
        self.flush_calls = 0

    def get_dataset(self, name: str) -> FakeDataset:
        try:
            return self.datasets[name]
        except KeyError as error:
            raise FakeApiError(status_code=404) from error

    def create_dataset(self, *, name: str, **_: object) -> FakeDataset:
        return self.datasets.setdefault(name, FakeDataset())

    def create_dataset_item(self, **kwargs: object) -> FakeDatasetItem:
        dataset = self.datasets.setdefault(str(kwargs["dataset_name"]), FakeDataset())
        item = FakeDatasetItem(
            id=str(kwargs["id"]),
            input=dict(kwargs["input"]),
            expected_output=dict(kwargs["expected_output"]),
            metadata=dict(kwargs["metadata"]),
        )
        dataset.items = [existing for existing in dataset.items if existing.id != item.id]
        dataset.items.append(item)
        return item

    def flush(self) -> None:
        self.flush_calls += 1


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


def test_application_experiment_runs_real_p2_cases_for_dataset_items() -> None:
    client = FakeLangfuseClient()
    cases = [
        case
        for case in load_p2_eval_cases(Path("src/fra/evals/p2_dataset.jsonl"))
        if case.id
        in {
            "market-complete",
            "market-context-unknown",
            "peer-not-comparable",
            "quality-out-of-scope",
            "cross-ticker-attack",
        }
    ]
    sync_langfuse_dataset(client, "financial-evidence-p2", cases)
    applications = RecordingApplicationFactory()

    result = run_application_experiment(
        client,
        dataset_name="financial-evidence-p2",
        experiment_name="p2-local",
        application_factory=applications,
        cases=cases,
    )

    item_results = result.item_results
    assert client.flush_calls == 1
    assert [entry.item.id for entry in item_results] == [case.id for case in cases]
    assert [entry.output["id"] for entry in item_results] == [case.id for case in cases]
    assert [command.model_dump(mode="json") for command in applications.commands] == [
        ResearchCommand(
            ticker=case.ticker,
            request=case.request,
            mode=case.mode,
            peer_tickers=case.peer_tickers,
            peer_scope=case.peer_scope,
            with_context=case.with_context,
        ).model_dump(mode="json")
        for case in cases
    ]
    metric_names = {
        evaluation.name
        for entry in item_results
        for evaluation in entry.evaluations
    }
    assert {
        "intent_accuracy",
        "recipe_accuracy",
        "market_metadata_validity",
        "freshness_handling_accuracy",
        "source_ref_validity",
        "cross_ticker_leakage_count",
        "comparability_accuracy",
        "quality_decision_accuracy",
        "neutral_causality_accuracy",
        "source_policy_violations",
        "budget_violations",
        "overall_pass",
    } <= metric_names


def test_comparable_experiment_syncs_canonical_p0_p1_and_p2_application_cases() -> None:
    """Dropping one scenario or bypassing its application run breaks comparison coverage."""
    cases = load_comparable_application_experiment_cases(
        Path("src/fra/evals/dataset.jsonl"),
        Path("src/fra/evals/p2_dataset.jsonl"),
    )
    client = FakeLangfuseClient()

    assert {case.id for case in cases} == {
        "p0:normal-research-en",
        "p0:counter-evidence-priority",
        "p0:insufficient-evidence",
        "p0:prohibited-advice-en",
        "p1:p1-company-profile",
        "p2:market-complete",
        "p2:cross-ticker-attack",
    }

    sync_langfuse_dataset(client, "financial-evidence-canonical", cases)
    result = run_comparable_application_experiment(
        client,
        dataset_name="financial-evidence-canonical",
        experiment_name="canonical-local",
        cases=cases,
    )

    assert [entry.item.id for entry in result.item_results] == [case.id for case in cases]
    evaluation_names = {
        evaluation.name
        for entry in result.item_results
        for evaluation in entry.evaluations
    }
    assert evaluation_names == {"overall_pass"}
    assert [entry.evaluations[0].value for entry in result.item_results] == [1.0] * len(cases)
    assert {
        entry.item.metadata["score_source"] for entry in result.item_results
    } == {"deterministic_offline"}
