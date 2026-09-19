"""Production-path CLI coverage for structured AUTO intent selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from typer.testing import CliRunner

from fra import cli as cli_module
from fra.bootstrap import BootstrapErrorCode, ResearchRuntime
from fra.cli import app
from fra.config import Settings
from fra.domain import Intent, RouterDecision
from fra.safety.router import REFUSAL_TEXT
from fra.skills.models import SkillName

from . import seed_supported_companies
from .test_p1_workflow import P1Recorder, _dependencies

runner = CliRunner()


class RecordingStructuredChat:
    def __init__(self, output: object) -> None:
        self.output = output
        self.schemas: list[type[Any]] = []
        self.route_calls = 0
        self.token_counter = lambda value: len(value.split())

    def with_structured_output(
        self,
        schema: type[Any],
        *,
        strict: bool,
        include_raw: bool,
    ) -> RecordingStructuredChat:
        assert strict is True
        assert include_raw is True
        self.schemas.append(schema)
        return self

    def invoke(self, prompt: list[BaseMessage]) -> object:
        assert len(prompt) == 2
        assert isinstance(prompt[0], SystemMessage)
        assert isinstance(prompt[1], HumanMessage)
        assert "must not analyze the security" in str(prompt[1].content).lower()
        self.route_calls += 1
        return {
            "raw": AIMessage(content=""),
            "parsed": self.output,
            "parsing_error": None,
        }


def _configure_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'auto-cli.sqlite3'}"
    seed_supported_companies(database_url, "NVDA")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("FAST_MODEL", "fast-route-model")
    monkeypatch.setenv("OPENAI_API_KEY", "route-key")
    monkeypatch.setenv("ANALYST_MODEL", "unused-analyst-model")


def test_auto_earnings_cli_builds_p1_after_one_model_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_auto(monkeypatch, tmp_path)
    chat = RecordingStructuredChat(
        RouterDecision(
            intent=Intent.EARNINGS_REVIEW_REQUEST,
            reason="latest operating update",
        )
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: chat)
    recorder = P1Recorder()
    dependencies, _ = _dependencies(recorder)
    runtime = ResearchRuntime(dependencies=dependencies)
    monkeypatch.setattr(
        cli_module,
        "build_p1_runtime",
        lambda settings, *, ticker: runtime,
    )
    monkeypatch.setattr(
        cli_module,
        "build_p0_runtime",
        lambda *args, **kwargs: pytest.fail("earnings AUTO route selected P0"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "auto",
            "--question",
            "Review the latest operating update",
        ],
    )

    assert result.exit_code == 0, result.output
    assert chat.route_calls == 1
    assert recorder.planner_recipes == [
        SkillName.EARNINGS_REVIEW,
        SkillName.FINANCIAL_DATA_VERIFICATION,
    ]
    assert r"Recipe: earnings\_review" in result.output


def test_auto_rule_refusal_constructs_neither_router_nor_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli_module,
        "build_p0_runtime",
        lambda *args, **kwargs: pytest.fail("rule refusal constructed P0"),
    )
    monkeypatch.setattr(
        cli_module,
        "build_p1_runtime",
        lambda *args, **kwargs: pytest.fail("rule refusal constructed P1"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "auto",
            "--question",
            "Should I buy NVDA?",
            "--session-id",
            "future-memory-scope",
            "--forms",
            "10-Q,8-K",
            "--as-of-date",
            "2099-01-01",
            "--market",
            "US",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == REFUSAL_TEXT


def test_unmatched_auto_without_fast_model_configuration_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    database_url = f"sqlite+pysqlite:///{tmp_path / 'auto-config.sqlite3'}"
    seed_supported_companies(database_url, "NVDA")
    monkeypatch.setattr(
        cli_module,
        "Settings",
        lambda: Settings(database_url=database_url, _env_file=None),
    )
    monkeypatch.setattr(
        cli_module,
        "build_p0_runtime",
        lambda *args, **kwargs: pytest.fail("missing route config silently selected P0"),
    )
    monkeypatch.setattr(
        cli_module,
        "build_p1_runtime",
        lambda *args, **kwargs: pytest.fail("missing route config constructed P1"),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--mode",
            "auto",
            "--question",
            "Give me a business overview",
        ],
    )

    assert result.exit_code == 2
    assert BootstrapErrorCode.FAST_MODEL_CONFIGURATION_MISSING.value in result.output
    assert "FAST_MODEL" in result.output
    assert "OPENAI_API_KEY" in result.output
