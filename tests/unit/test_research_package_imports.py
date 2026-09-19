"""Regression coverage for package imports that must not form a cycle."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import fra.application",
        "import fra.cli",
        (
            "from fra.research_packages import "
            "PeerResearchOrchestrator, QualityResearchRuntime"
        ),
    ],
)
def test_imports_succeed_in_fresh_python_process(statement: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
