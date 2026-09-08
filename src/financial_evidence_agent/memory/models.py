"""Strict, source-free values retained for short-lived conversation continuity."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, field_validator

from financial_evidence_agent.domain import StrictModel


class ConversationTurn(StrictModel):
    """One normalized question and deterministic guarded-result summary."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    answer_summary: str = Field(min_length=1, max_length=1_200)
    run_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Z][A-Z0-9.-]*$")
    open_questions: tuple[Annotated[str, Field(min_length=1, max_length=500)], ...] = Field(
        default=(), max_length=5
    )

    @field_validator("question", "answer_summary", "open_questions", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        if isinstance(value, (list, tuple)):
            return tuple(
                " ".join(item.split()) if isinstance(item, str) else item for item in value
            )
        return value

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class SessionMemory(StrictModel):
    """Rolling, at-most-five-turn session value stored as canonical JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    summary: str = Field(default="", max_length=8_000)
    turns: tuple[ConversationTurn, ...] = Field(default=(), max_length=5)
