"""Distribution smoke tests for the bundled deterministic demo resources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "OPENAI_API_KEY",
        "FAST_MODEL",
        "ANALYST_MODEL",
    ):
        environment.pop(name, None)
    return environment


def test_default_eval_runs_from_outside_checkout(tmp_path: Path) -> None:
    eval_script = Path(sys.executable).parent / "eval"
    assert eval_script.is_file()

    completed = subprocess.run(
        [str(eval_script)],
        cwd=tmp_path,
        env=_offline_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["case_count"] == 7
    assert payload["passed_count"] == 7
    assert payload["p2_case_count"] == 10
    assert len(payload["p2_results"]) == 10


def test_wheel_and_sdist_contain_bundled_dataset_and_fixture(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the distribution smoke test")
    repository_root = Path(__file__).resolve().parents[2]
    output_directory = tmp_path / "dist"
    environment = _offline_environment()
    environment["UV_CACHE_DIR"] = str(tmp_path / "empty-uv-cache")

    completed = subprocess.run(
        [uv, "build", "--offline", "--no-build-isolation", "--out-dir", str(output_directory)],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    wheel = next(output_directory.glob("*.whl"))
    source_distribution = next(output_directory.glob("*.tar.gz"))
    expected_suffixes = {
        "fra/evals/dataset.jsonl",
        "fra/evals/p2_dataset.jsonl",
        "fra/resources/nvda_10q.html",
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(source_distribution, mode="r:gz") as archive:
        sdist_names = set(archive.getnames())

    assert expected_suffixes <= wheel_names
    assert all(any(name.endswith(suffix) for name in sdist_names) for suffix in expected_suffixes)

    wheel_environment = _offline_environment()
    wheel_environment["PYTHONPATH"] = str(wheel)
    wheel_eval = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fra.cli import evaluate; evaluate()",
        ],
        cwd=tmp_path,
        env=wheel_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert wheel_eval.returncode == 0, wheel_eval.stderr
    assert json.loads(wheel_eval.stdout)["passed_count"] == 7

    wheel_p2_eval = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fra.cli import EvalSuite, evaluate; "
                "evaluate(suite=EvalSuite.P2)"
            ),
        ],
        cwd=tmp_path,
        env=wheel_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert wheel_p2_eval.returncode == 0, wheel_p2_eval.stderr
    p2_payload = json.loads(wheel_p2_eval.stdout)
    assert p2_payload["case_count"] == 10
    assert "metrics" in p2_payload
