"""CLI contracts for capability-gated IEX-only market mode."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from financial_evidence_agent import cli as cli_module
from financial_evidence_agent.application import ResearchMode
from financial_evidence_agent.cli import app
from financial_evidence_agent.config import Settings
from financial_evidence_agent.domain import Intent, RouterDecision
from financial_evidence_agent.graph.models import MarketDependencies, ResearchResult
from financial_evidence_agent.safety.router import REFUSAL_TEXT

from . import seed_supported_companies
from .test_market_workflow import (
    DirectMarketRuntime,
    RecordingMCP,
    _success_responses,
)

runner = CliRunner()


def _settings(path: Path) -> Settings:
    database_url = f"sqlite+pysqlite:///{path}"
    seed_supported_companies(database_url, "NVDA")
    return Settings(database_url=database_url, _env_file=None)


def test_market_cli_missing_keys_fails_before_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential gating happens before server, gateway, or tool construction."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(cli_module, "Settings", lambda: _settings(tmp_path / "missing.sqlite3"))

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "market-snapshot",
            "--question",
            "current price",
        ],
    )

    assert result.exit_code == 2
    assert "MARKET_DATA_CONFIGURATION_MISSING" in result.stderr
    assert "tool_calls=0" in result.stderr


def test_market_cli_renders_fixed_iex_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Typer boundary delegates to the guarded application result unchanged."""
    client = RecordingMCP(_success_responses())
    runtime = DirectMarketRuntime(MarketDependencies(client, max_bars=5))
    monkeypatch.setattr(cli_module, "Settings", lambda: _settings(tmp_path / "market.sqlite3"))
    monkeypatch.setattr(
        cli_module,
        "build_market_runtime",
        lambda settings, *, ticker: runtime,
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "market-snapshot",
            "--question",
            "current price",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "# IEX-only market snapshot — NVDA" in result.output
    assert "Provider: Alpaca" in result.output
    assert "Feed: IEX" in result.output
    assert "Research assistance only; not investment advice." in result.output
    assert len(client.calls) == 2


def test_market_cli_advice_refuses_before_market_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit market selection never bypasses the deterministic safety boundary."""
    monkeypatch.setattr(cli_module, "Settings", lambda: _settings(tmp_path / "refusal.sqlite3"))
    monkeypatch.setattr(
        cli_module,
        "build_market_runtime",
        lambda *args, **kwargs: pytest.fail("refusal constructed market runtime"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "market-snapshot",
            "--question",
            "Should I buy NVDA at the current price?",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == REFUSAL_TEXT


def test_non_market_cli_mode_does_not_require_alpaca_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent market credentials cannot become a global P0/P1 startup requirement."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(cli_module, "Settings", lambda: _settings(tmp_path / "p0.sqlite3"))

    class P0Runtime:
        def execute(self, command, decision) -> ResearchResult:
            assert command.mode is ResearchMode.THESIS
            assert decision.intent is Intent.RESEARCH_REQUEST
            return ResearchResult(
                status="completed",
                ticker="NVDA",
                thesis=command.request,
                decision=RouterDecision(
                    intent=Intent.RESEARCH_REQUEST,
                    reason="explicit thesis mode",
                ),
                rendered_output="# P0 remains available\n",
            )

    monkeypatch.setattr(
        cli_module,
        "build_p0_runtime",
        lambda settings, *, ticker: P0Runtime(),
    )
    monkeypatch.setattr(
        cli_module,
        "build_market_runtime",
        lambda *args, **kwargs: pytest.fail("P0 constructed market runtime"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "thesis",
            "--thesis",
            "Verify whether disclosed demand supports continued revenue growth.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "# P0 remains available"


def test_market_cli_requires_question_instead_of_thesis() -> None:
    """Market mode has one explicit research-question input shape."""
    missing = runner.invoke(app, ["research", "NVDA", "--mode", "market-snapshot"])
    wrong_option = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "market-snapshot",
            "--thesis",
            "current price",
        ],
    )

    assert missing.exit_code == 2
    assert "--question is required in market-snapshot mode" in missing.output
    assert wrong_option.exit_code == 2
    assert "--thesis is not valid in market-snapshot mode" in wrong_option.output
