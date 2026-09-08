"""Deterministic bundled evaluation contracts and runners."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from financial_evidence_agent.evals.p2_runner import (
    P2EvalCase,
    P2EvalMetrics,
    P2EvalResult,
    P2EvalSummary,
    load_p2_eval_cases,
    run_p2_eval,
)
from financial_evidence_agent.evals.runner import (
    EvalCase,
    EvalCaseResult,
    EvalSummary,
    load_eval_cases,
    run_eval,
)


@contextmanager
def bundled_dataset_path() -> Iterator[Path]:
    """Materialize the packaged default JSONL dataset for one evaluation run."""
    resource = files("financial_evidence_agent.evals").joinpath("dataset.jsonl")
    with as_file(resource) as path:
        yield path


@contextmanager
def bundled_p2_dataset_path() -> Iterator[Path]:
    """Materialize the packaged strict P2 JSONL dataset for one evaluation run."""
    resource = files("financial_evidence_agent.evals").joinpath("p2_dataset.jsonl")
    with as_file(resource) as path:
        yield path


__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalSummary",
    "P2EvalCase",
    "P2EvalMetrics",
    "P2EvalResult",
    "P2EvalSummary",
    "bundled_dataset_path",
    "bundled_p2_dataset_path",
    "load_eval_cases",
    "load_p2_eval_cases",
    "run_eval",
    "run_p2_eval",
]
