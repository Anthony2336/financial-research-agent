"""Opt-in live Langfuse smoke tests for trace flush and P2 experiments."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from financial_evidence_agent.config import Settings
from financial_evidence_agent.evals.p2_runner import (
    load_p2_eval_cases,
    run_application_experiment,
)
from financial_evidence_agent.observability import (
    build_langfuse_operation_client,
    build_trace_sink,
    sync_langfuse_dataset,
)

_DATASET_NAME = "financial-evidence-p2-live-smoke"
_CASE_IDS = ("market-complete", "quality-out-of-scope")


class _FakeCase:
    def __init__(self, case_id: str) -> None:
        self.id = case_id

    def input_payload(self) -> dict[str, object]:
        return {"id": self.id, "mode": "quality-screen"}

    def expected_payload(self) -> dict[str, object]:
        return {"id": self.id, "passed": True}

    def metadata_payload(self) -> dict[str, object]:
        return {"suite": "p2", "id": self.id}


class _FakeDataset:
    def __init__(self) -> None:
        self.items: list[SimpleNamespace] = []


class _FakeOperationClient:
    def __init__(self) -> None:
        self.datasets: dict[str, _FakeDataset] = {}
        self.flush_calls = 0

    def get_dataset(self, name: str) -> _FakeDataset:
        return self.datasets.setdefault(name, _FakeDataset())

    def create_dataset(self, *, name: str, **_: object) -> _FakeDataset:
        return self.datasets.setdefault(name, _FakeDataset())

    def create_dataset_item(self, **kwargs: object) -> SimpleNamespace:
        dataset = self.datasets.setdefault(str(kwargs["dataset_name"]), _FakeDataset())
        item = SimpleNamespace(
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


def _require_live_values(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail(
            "live Langfuse smoke requires environment variables: "
            + ", ".join(sorted(missing))
        )


def _live_settings() -> Settings:
    _require_live_values(
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    )
    return Settings(_env_file=None)


def _live_cases():
    cases = load_p2_eval_cases(Path("src/financial_evidence_agent/evals/p2_dataset.jsonl"))
    selected = [case for case in cases if case.id in _CASE_IDS]
    assert [case.id for case in selected] == list(_CASE_IDS)
    return selected


def _wait_for_dataset_items(client: object, *, name: str, expected_ids: tuple[str, ...]) -> None:
    deadline = monotonic() + 30
    while True:
        dataset = getattr(client, "get_dataset")(name)
        items = getattr(dataset, "items", None)
        if items is not None:
            item_ids = [getattr(item, "id", None) for item in items]
            if all(expected_id in item_ids for expected_id in expected_ids):
                return
        if monotonic() >= deadline:
            pytest.fail(
                f"Langfuse dataset '{name}' did not expose {expected_ids} within 30 seconds"
            )
        sleep(1)


@pytest.mark.live_provider
def test_live_langfuse_trace_sink_flushes_one_root_run() -> None:
    settings = _live_settings()
    sink = build_trace_sink(settings)

    with sink.run(
        run_id="task20-live-langfuse-smoke",
        input={"ticker": "NVDA", "mode": "quality-screen"},
        metadata={"suite": "task20", "source": "live-smoke"},
    ) as run:
        with run.observation(
            name="task20.live-smoke",
            kind="chain",
            input={"step": "trace"},
            metadata={"mode": "root-flush"},
        ) as observation:
            observation.update(output={"status": "ok"})
        run.update(output={"status": "completed"})
        trace_id = run.trace_id

    sink.flush()

    assert isinstance(trace_id, str)
    assert trace_id


@pytest.mark.live_provider
def test_live_langfuse_dataset_sync_and_experiment() -> None:
    settings = _live_settings()
    client = build_langfuse_operation_client(settings)
    cases = _live_cases()

    sync_langfuse_dataset(client, _DATASET_NAME, cases)
    _wait_for_dataset_items(client, name=_DATASET_NAME, expected_ids=_CASE_IDS)
    result = run_application_experiment(
        client,
        dataset_name=_DATASET_NAME,
        experiment_name=(
            "p2-live-smoke-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        ),
        cases=cases,
    )

    item_results = list(getattr(result, "item_results"))
    assert [entry.item.id for entry in item_results] == list(_CASE_IDS)


def test_live_langfuse_dataset_smoke_does_not_force_a_second_client_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeOperationClient()
    cases = [_FakeCase(case_id) for case_id in _CASE_IDS]
    module = sys.modules[__name__]

    monkeypatch.setattr(module, "_live_settings", lambda: object())
    monkeypatch.setattr(
        module,
        "build_langfuse_operation_client",
        lambda settings: client,
    )
    monkeypatch.setattr(module, "_live_cases", lambda: cases)
    monkeypatch.setattr(
        module,
        "_wait_for_dataset_items",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "run_application_experiment",
        lambda *args, **kwargs: SimpleNamespace(
            item_results=[SimpleNamespace(item=SimpleNamespace(id=case.id)) for case in cases]
        ),
    )

    test_live_langfuse_dataset_sync_and_experiment()

    assert client.flush_calls == 1
