"""Offline CLI acceptance coverage for the P1 earnings-review entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fra import cli as cli_module
from fra.bootstrap import ResearchRuntime
from fra.cli import app
from fra.skills.models import SkillName

from . import seed_supported_companies
from .test_p1_workflow import P1Recorder, _dependencies

runner = CliRunner()


def test_earnings_review_cli_runs_two_recipes_with_citations_and_disclaimer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'earnings-cli.sqlite3'}"
    seed_supported_companies(database_url, "NVDA")
    monkeypatch.setenv("DATABASE_URL", database_url)
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)
    runtime = ResearchRuntime(dependencies=dependencies)
    monkeypatch.setattr(
        cli_module,
        "build_p1_runtime",
        lambda settings, *, ticker: runtime,
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "分析最近一期财报",
            "--mode",
            "earnings-review",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorder.planner_recipes == [
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    assert r"Recipe: earnings\_review" in result.output
    assert r"Recipe: financial\_data\_verification" in result.output
    assert r"SEC: sec-earnings\_review" in result.output
    assert "https://www.sec.gov/Archives/sec-earnings_review.htm" in result.output
    assert "Research assistance only; not investment advice." in result.output
