"""Run-scoped context shared by orchestration and workflows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from fra.context import BudgetAuthority
from fra.memory.research import ResearchMemoryStore
from fra.memory.session import SessionMemoryRepository
from fra.observability import TraceRun
from fra.prompts import PromptUsage
from fra.storage.run_repositories import RunFinish, RunStart, SourceFetchWrite


class RunIdGenerator(Protocol):
    """Application-owned source of public correlation identifiers."""

    def new_run_id(self) -> str:
        """Return one new identifier before safety routing begins."""


class ResearchRunWriter(Protocol):
    """Persistence boundary used by application runs and P1 fetch adapters."""

    def start(self, value: RunStart) -> None:
        """Persist the immutable input boundary."""

    def finish(self, value: RunFinish) -> None:
        """Persist only final guarded output."""

    def record_fetch(self, value: SourceFetchWrite) -> None:
        """Persist one safe source-fetch attempt."""


class UuidRunIdGenerator:
    """Default application run-ID source."""

    def new_run_id(self) -> str:
        return str(uuid4())


class NoopResearchRunWriter:
    """Explicit lower-level context for callers without local persistence."""

    def start(self, value: RunStart) -> None:
        del value

    def finish(self, value: RunFinish) -> None:
        del value

    def record_fetch(self, value: SourceFetchWrite) -> None:
        del value


@dataclass(frozen=True, slots=True)
class ResearchExecutionContext:
    """The one application correlation context visible to a selected runtime."""

    run_id: str
    trace: TraceRun
    run_repository: ResearchRunWriter
    budget: BudgetAuthority
    prompt_usage: PromptUsage
    session_memory_store: SessionMemoryRepository | None = None
    research_memory_store: ResearchMemoryStore | None = None


_CURRENT_CONTEXT: ContextVar[ResearchExecutionContext | None] = ContextVar(
    "fra_research_context",
    default=None,
)


def current_research_context() -> ResearchExecutionContext | None:
    """Return application context, or None for explicit lower-level execution."""
    return _CURRENT_CONTEXT.get()


@contextmanager
def bind_research_context(context: ResearchExecutionContext) -> Iterator[None]:
    token: Token[ResearchExecutionContext | None] = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)
