"""Synchronous short-term session-memory boundary over JSON-only cache storage."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from pydantic import ValidationError

from financial_evidence_agent.context import MemoryHint
from financial_evidence_agent.memory.models import ConversationTurn, SessionMemory
from financial_evidence_agent.memory.privacy import contains_private_financial_or_secret
from financial_evidence_agent.storage.cache import SyncJsonCache

SESSION_TTL_SECONDS = 86_400
_MAX_TURNS = 5
_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
logger = logging.getLogger(__name__)


class SessionMemoryRepository(Protocol):
    """Application/graph boundary for isolated short-term conversation state."""

    def load(self, session_id: str) -> SessionMemory:
        """Return one session's current bounded value."""

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Best-effort append after guarded local run persistence succeeds."""

    def clear(self, session_id: str) -> None:
        """Delete one explicitly addressed session."""


class SessionMemoryStore:
    """Load and update one isolated TTL session without surfacing cache failures."""

    def __init__(self, cache: SyncJsonCache) -> None:
        self._cache = cache

    def load(self, session_id: str) -> SessionMemory:
        """Return validated memory, or an empty value for invalid/cache-miss data."""
        key = _session_key(session_id)
        if key is None:
            return SessionMemory()
        try:
            value = self._cache.get_json_sync(key)
            if value is None:
                return SessionMemory()
            return SessionMemory.model_validate(value)
        except (TypeError, ValueError, ValidationError):
            logger.warning("session memory payload rejected")
            return SessionMemory()
        except Exception:
            logger.warning("session memory load failed")
            return SessionMemory()

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Append one typed turn, trim oldest values, and refresh the exact TTL."""
        key = _session_key(session_id)
        if key is None:
            return
        memory = self.load(session_id)
        turns = (*memory.turns, turn)[-_MAX_TURNS:]
        updated = SessionMemory(summary=build_rolling_summary(turns), turns=turns)
        try:
            self._cache.set_json_sync(
                key,
                updated.model_dump(mode="json"),
                ttl_seconds=SESSION_TTL_SECONDS,
            )
        except Exception:
            logger.warning("session memory write failed")

    def clear(self, session_id: str) -> None:
        """Delete only the explicitly addressed session value."""
        key = _session_key(session_id)
        if key is None:
            return
        try:
            self._cache.delete_json_sync(key)
        except Exception:
            logger.warning("session memory clear failed")


def _session_key(session_id: str) -> str | None:
    normalized = session_id.strip()
    if _SESSION_ID.fullmatch(normalized) is None:
        logger.warning("invalid session memory identifier rejected")
        return None
    return f"session:{normalized}"


def build_rolling_summary(turns: tuple[ConversationTurn, ...]) -> str:
    """Build the single bounded deterministic summary used by every memory view."""
    summary = "\n".join(
        f"[{turn.ticker}] {turn.question} — {turn.answer_summary}" for turn in turns
    )
    return summary[-8_000:]


def load_session_memory_for_ticker(
    repository: SessionMemoryRepository | None,
    session_id: str | None,
    ticker: str,
) -> SessionMemory:
    """Fail open to an empty, same-ticker bounded memory view."""
    if repository is None or not session_id:
        return SessionMemory()
    try:
        loaded = repository.load(session_id)
    except Exception:
        logger.warning("session memory load failed")
        return SessionMemory()
    normalized_ticker = ticker.strip().upper()
    turns = tuple(turn for turn in loaded.turns if turn.ticker == normalized_ticker)[-5:]
    return SessionMemory(summary=build_rolling_summary(turns), turns=turns)


def build_session_memory_hints(memory: SessionMemory) -> tuple[MemoryHint, ...]:
    """Build at most three typed planner-only hints from one scoped memory value."""
    return tuple(
        MemoryHint(
            text=(
                f"Prior guarded turn for {turn.ticker} (run {turn.run_id}): "
                f"question={turn.question}; answer summary={turn.answer_summary}"
                + (
                    "; open questions=" + " | ".join(turn.open_questions)
                    if turn.open_questions
                    else ""
                )
            ),
            score=float(index + 1),
            identity="session",
            pointer_id=turn.run_id,
        )
        for index, turn in enumerate(memory.turns[-3:])
    )


def is_session_memory_eligible_request(
    request: str,
    *,
    current_ticker: str | None = None,
) -> bool:
    """Reject personal-finance and secret-bearing text without logging its contents."""
    return not contains_private_financial_or_secret(
        request,
        current_ticker=current_ticker,
    )
