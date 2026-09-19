"""Contract tests for explicit Langfuse dataset synchronization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fra.evals.p2_runner import load_p2_eval_cases
from fra.observability import LangfuseOperationError, sync_langfuse_dataset


@dataclass(slots=True)
class FakeApiError(Exception):
    status_code: int


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.created_datasets: list[str] = []
        self.dataset_items: dict[str, dict[str, dict[str, object]]] = {}
        self.create_dataset_item_calls: list[dict[str, object]] = []
        self.flush_calls = 0
        self.get_dataset_error: Exception | None = None

    @property
    def item_ids(self) -> list[str]:
        return sorted(
            item_id
            for items in self.dataset_items.values()
            for item_id in items
        )

    def get_dataset(self, name: str) -> object:
        if self.get_dataset_error is not None:
            raise self.get_dataset_error
        if name not in self.dataset_items:
            raise FakeApiError(status_code=404)
        return object()

    def create_dataset(self, *, name: str, **_: object) -> object:
        self.created_datasets.append(name)
        self.dataset_items.setdefault(name, {})
        return object()

    def create_dataset_item(self, **kwargs: object) -> object:
        dataset_name = str(kwargs["dataset_name"])
        item_id = str(kwargs["id"])
        self.dataset_items.setdefault(dataset_name, {})[item_id] = dict(kwargs)
        self.create_dataset_item_calls.append(dict(kwargs))
        return object()

    def flush(self) -> None:
        self.flush_calls += 1


def test_dataset_sync_upserts_by_case_id_and_flushes_once_per_sync() -> None:
    client = FakeLangfuseClient()
    cases = load_p2_eval_cases(Path("src/fra/evals/p2_dataset.jsonl"))

    sync_langfuse_dataset(client, "financial-evidence-p2", cases)
    sync_langfuse_dataset(client, "financial-evidence-p2", cases)

    assert client.created_datasets == ["financial-evidence-p2"]
    assert client.item_ids == sorted(case.id for case in cases)
    assert client.flush_calls == 2
    peer_partial = next(
        call for call in client.create_dataset_item_calls if call["id"] == "peer-partial"
    )
    assert set(peer_partial["metadata"]) == {
        "case_kind",
        "dataset_schema_version",
        "fixture_version",
        "recipe_versions",
        "source_policy_version",
        "suite",
    }
    assert peer_partial["metadata"]["recipe_versions"] == ["1.0.0", "1.0.0"]
    json.dumps(peer_partial["input"])
    json.dumps(peer_partial["expected_output"])
    json.dumps(peer_partial["metadata"])


def test_dataset_sync_propagates_non_not_found_lookup_failures() -> None:
    client = FakeLangfuseClient()
    client.get_dataset_error = FakeApiError(status_code=500)
    cases = load_p2_eval_cases(Path("src/fra/evals/p2_dataset.jsonl"))[:1]

    with pytest.raises(LangfuseOperationError, match="failed to fetch Langfuse dataset"):
        sync_langfuse_dataset(client, "financial-evidence-p2", cases)

    assert client.created_datasets == []
    assert client.flush_calls == 0
