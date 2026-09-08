"""Offline CLI acceptance coverage for the industry-research entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from financial_evidence_agent import cli as cli_module
from financial_evidence_agent.bootstrap import ResearchRuntime
from financial_evidence_agent.cli import app
from financial_evidence_agent.safety.router import REFUSAL_TEXT
from financial_evidence_agent.skills.models import SkillName

from . import seed_supported_companies
from .test_p1_workflow import P1Recorder, _dependencies

runner = CliRunner()


def _inject_industry_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    execution_note: str | None = None,
) -> P1Recorder:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'industry-cli.sqlite3'}"
    seed_supported_companies(database_url, "NVDA")
    monkeypatch.setenv("DATABASE_URL", database_url)
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)
    runtime = ResearchRuntime(
        dependencies=dependencies,
        execution_note=execution_note,
    )
    monkeypatch.setattr(cli_module, "build_industry_runtime", lambda settings, *, ticker: runtime)
    return recorder


def test_industry_cli_runs_one_recipe_with_guarded_sources(monkeypatch, tmp_path: Path) -> None:
    recorder = _inject_industry_runtime(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Describe the accelerator industry",
            "--mode",
            "industry-research",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorder.planner_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert "## Scope" in result.output
    assert "## Per-company evidence" in result.output
    assert "industry\\_research" in result.output
    assert "NVDA:filing:sec-industry\\_research" in result.output
    assert "https://www.sec.gov/Archives/sec-industry_research.htm" in result.output


def test_industry_cli_reports_local_only_information_gap_when_web_fallback_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _inject_industry_runtime(
        monkeypatch,
        tmp_path,
        execution_note="allowlisted web fallback unavailable; local evidence only",
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Describe the accelerator industry",
            "--mode",
            "industry-research",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "> Information gap: allowlisted web fallback unavailable; local evidence only." in (
        result.output
    )


def test_industry_cli_still_refuses_investment_action_phrasing(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_industry_runtime",
        lambda *args, **kwargs: pytest.fail("runtime constructed before safety refusal"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Should I buy NVDA after reviewing the industry?",
            "--mode",
            "industry-research",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == REFUSAL_TEXT


def test_industry_cli_requires_peer_scope_when_peers_are_supplied() -> None:
    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Compare direct peers using exact reported metrics.",
            "--mode",
            "industry-research",
            "--peer-ticker",
            "AMD",
        ],
    )

    assert result.exit_code != 0
    assert "--peer-scope is required when --peer-ticker is supplied" in result.output


def test_industry_cli_passes_explicit_peer_inputs_to_application(monkeypatch) -> None:
    class FakeResult:
        rendered_output = "peer comparison prepared"

    class RecordingApplication:
        def __init__(self) -> None:
            self.commands = []

        def run(self, command):
            self.commands.append(command)
            return FakeResult()

    application = RecordingApplication()
    monkeypatch.setattr(
        cli_module,
        "build_research_application",
        lambda *args, **kwargs: application,
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Compare direct peers using exact reported metrics.",
            "--mode",
            "industry-research",
            "--peer-ticker",
            "AMD",
            "--peer-ticker",
            "INTC",
            "--peer-scope",
            "US semiconductors",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "peer comparison prepared"
    assert len(application.commands) == 1
    command = application.commands[0]
    assert command.peer_tickers == ("AMD", "INTC")
    assert command.peer_scope == "US semiconductors"


def test_industry_cli_rejects_blank_peer_before_application_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_research_application",
        lambda *args, **kwargs: pytest.fail("application built before peer validation"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Compare direct peers using exact reported metrics.",
            "--mode",
            "industry-research",
            "--peer-ticker",
            "   ",
            "--peer-scope",
            "US semiconductors",
        ],
    )

    assert result.exit_code != 0
    assert "--peer-ticker must not be blank" in result.output


def test_industry_cli_rejects_peer_scope_without_peer_tickers_before_application_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_research_application",
        lambda *args, **kwargs: pytest.fail("application built before peer validation"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--question",
            "Describe the accelerator industry",
            "--mode",
            "industry-research",
            "--peer-scope",
            "US semiconductors",
        ],
    )

    assert result.exit_code != 0
    assert "--peer-scope requires at least one --peer-ticker" in result.output


@pytest.mark.parametrize(
    "args",
    [
        [
            "research",
            "NVDA",
            "--thesis",
            "Validate this thesis against the retained evidence.",
            "--mode",
            "thesis",
            "--peer-scope",
            "US semiconductors",
        ],
        [
            "research",
            "NVDA",
            "--question",
            "Describe the company.",
            "--mode",
            "company-profile",
            "--peer-scope",
            "US semiconductors",
        ],
        [
            "research",
            "NVDA",
            "--question",
            "Review the latest earnings evidence.",
            "--mode",
            "earnings-review",
            "--peer-scope",
            "US semiconductors",
        ],
        [
            "research",
            "NVDA",
            "--question",
            "Summarize the current market snapshot.",
            "--mode",
            "market-snapshot",
            "--peer-scope",
            "US semiconductors",
        ],
        [
            "research",
            "NVDA",
            "--question",
            "Assess whether the retained evidence is balanced enough for more research.",
            "--mode",
            "quality-screen",
            "--peer-scope",
            "US semiconductors",
        ],
        [
            "research",
            "NVDA",
            "--question",
            "Assess whether the retained evidence is balanced enough for more research.",
            "--mode",
            "auto",
            "--peer-scope",
            "US semiconductors",
        ],
    ],
)
def test_non_industry_cli_rejects_peer_scope_before_application_construction(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_research_application",
        lambda *args, **kwargs: pytest.fail("application built before peer validation"),
    )

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "--peer-scope is only valid in industry-research mode" in result.output
