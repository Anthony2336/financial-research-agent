"""Project final guarded reports into eligible session and research memory."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property

from fra.contracts import ApplicationResult, ResearchCommand
from fra.domain import Claim, SourceRef, SourceRefKind
from fra.graph.models import MarketResearchResult, ResearchResult, persisted_claim
from fra.memory.models import ConversationTurn
from fra.memory.research import (
    ResearchMemoryKind,
    ResearchMemoryStore,
    is_research_memory_summary_eligible,
)
from fra.memory.session import SessionMemoryRepository, is_session_memory_eligible_request
from fra.storage.run_repositories import PersistedClaim, RunFinish

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MemoryProjection:
    """Reuse the exact persisted result and lazily build its safety index once."""

    result: ApplicationResult
    finish: RunFinish

    @cached_property
    def safety(self) -> _SummarySafety:
        return _summary_safety(self.result, self.finish)


def _persist_session_memory(
    repository: SessionMemoryRepository | None,
    command: ResearchCommand,
    projection: MemoryProjection,
) -> None:
    """Best-effort write of source-free guarded continuity after durable run finish."""
    if repository is None or command.session_id is None:
        return
    try:
        turn = _conversation_turn(command, projection)
        if turn is None:
            return
        repository.append(command.session_id, turn)
    except Exception:
        logger.warning("session memory persistence failed")


@dataclass(frozen=True, slots=True)
class _ResearchMemoryCandidate:
    memory_kind: ResearchMemoryKind
    summary: str
    evidence_source_refs: tuple[SourceRef, ...]
    corpus_version: str
    importance: float


def _persist_research_memories(
    repository: ResearchMemoryStore | None,
    command: ResearchCommand,
    projection: MemoryProjection,
) -> None:
    """Best-effort post-persistence writes of bounded, cited research hints."""
    result = projection.result
    if repository is None or result.status not in {"completed", "partial"}:
        return
    if isinstance(result, MarketResearchResult):
        return
    if not is_session_memory_eligible_request(
        command.request,
        current_ticker=command.ticker,
    ):
        return
    finish = projection.finish
    safety = projection.safety
    candidates = (
        _research_memory_candidates(result, safety)
        if isinstance(result, ResearchResult)
        else _alternate_research_memory_candidates(
            finish,
            ticker=command.ticker,
            safety=safety,
        )
    )
    for candidate in candidates:
        try:
            repository.store_guarded(
                ticker=result.ticker,
                memory_kind=candidate.memory_kind,
                summary=candidate.summary,
                source_run_id=result.run_id,
                evidence_source_refs=candidate.evidence_source_refs,
                corpus_version=candidate.corpus_version,
                importance=candidate.importance,
            )
        except Exception:
            logger.warning("research memory persistence failed")


def _alternate_research_memory_candidates(
    finish: RunFinish,
    *,
    ticker: str,
    safety: _SummarySafety,
) -> tuple[_ResearchMemoryCandidate, ...]:
    """Map other final guarded research shapes through their persisted contract."""
    normalized_ticker = ticker.strip().upper()
    ticker_versions = [
        version
        for version in finish.corpus_scope
        if version.upper().startswith(f"{normalized_ticker}-")
    ]
    if not ticker_versions and len(finish.corpus_scope) == 1:
        ticker_versions = list(finish.corpus_scope)
    if len(set(ticker_versions)) != 1:
        return ()
    corpus_version = ticker_versions[0]
    candidates: list[_ResearchMemoryCandidate] = []
    for claim in finish.claims:
        if claim.guard_status != "retained" or claim.kind.startswith(
            ("market_", "research_quality_", "p2_guard_")
        ):
            continue
        references = tuple(
            reference
            for reference in claim.source_refs
            if reference.ticker == normalized_ticker
            and reference.kind in {SourceRefKind.FILING, SourceRefKind.WEB}
        )
        summary = " ".join(claim.text.split())
        if (
            not references
            or len(summary) > 1_200
            or not _is_research_memory_summary_eligible(
                summary,
                ticker=normalized_ticker,
                safety=safety,
            )
        ):
            continue
        folded_kind = claim.kind.casefold()
        if "open_question" in folded_kind:
            memory_kind = ResearchMemoryKind.OPEN_QUESTION
            importance = 0.6
        elif any(
            marker in folded_kind
            for marker in ("counter", "challenge", "bear", "risk")
        ):
            memory_kind = ResearchMemoryKind.COUNTEREVIDENCE
            importance = 0.9
        elif "source_pointer" in folded_kind:
            memory_kind = ResearchMemoryKind.SOURCE_POINTER
            importance = 0.5
        else:
            memory_kind = ResearchMemoryKind.RESEARCH_SUMMARY
            importance = 0.8
        candidates.append(
            _ResearchMemoryCandidate(
                memory_kind=memory_kind,
                summary=summary,
                evidence_source_refs=references,
                corpus_version=corpus_version,
                importance=importance,
            )
        )
    return tuple(candidates)


def _research_memory_candidates(
    result: ResearchResult,
    safety: _SummarySafety,
) -> tuple[_ResearchMemoryCandidate, ...]:
    candidates: list[_ResearchMemoryCandidate] = []
    guarded = result.guarded_memo
    if guarded is not None:
        for memory_kind, claims, importance in (
            (ResearchMemoryKind.RESEARCH_SUMMARY, guarded.supporting_claims, 0.8),
            (ResearchMemoryKind.COUNTEREVIDENCE, guarded.counter_claims, 0.9),
            (ResearchMemoryKind.RESEARCH_SUMMARY, guarded.inferences, 0.6),
            (ResearchMemoryKind.OPEN_QUESTION, guarded.open_questions, 0.6),
        ):
            for claim in claims:
                _append_research_candidate(
                    candidates,
                    claim,
                    ticker=result.ticker,
                    memory_kind=memory_kind,
                    corpus_version=guarded.corpus_version,
                    importance=importance,
                    safety=safety,
                )

    from fra.skills.models import ResearchFacet

    counter_facets = {
        ResearchFacet.BEAR_CASE,
        ResearchFacet.RISKS,
        ResearchFacet.GUIDANCE_AND_RISKS,
    }
    for skill_run in result.skill_runs:
        guarded_skill = skill_run.guarded_memo
        if guarded_skill is None or skill_run.status not in {"completed", "partial"}:
            continue
        source_versions = {
            source.id: source.corpus_version for source in guarded_skill.filing_sources
        }
        fallback_versions = set(source_versions.values())
        for section in guarded_skill.memo.sections:
            for claim in section.claims:
                stored = persisted_claim(claim, ticker=result.ticker)
                cited_versions = {
                    source_versions[reference.source_id]
                    for reference in stored.source_refs
                    if reference.kind is SourceRefKind.FILING
                    and reference.source_id in source_versions
                }
                versions = cited_versions or fallback_versions
                if len(versions) != 1:
                    continue
                if claim.kind.value == "open_question" or (
                    section.facet is ResearchFacet.INFORMATION_GAPS
                ):
                    memory_kind = ResearchMemoryKind.OPEN_QUESTION
                    importance = 0.6
                elif section.facet in counter_facets:
                    memory_kind = ResearchMemoryKind.COUNTEREVIDENCE
                    importance = 0.9
                else:
                    memory_kind = ResearchMemoryKind.RESEARCH_SUMMARY
                    importance = 0.8
                _append_research_candidate(
                    candidates,
                    claim,
                    ticker=result.ticker,
                    memory_kind=memory_kind,
                    corpus_version=next(iter(versions)),
                    importance=importance,
                    safety=safety,
                )

    deduplicated: list[_ResearchMemoryCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = (
            candidate.memory_kind,
            candidate.summary,
            candidate.corpus_version,
            *(reference.encode() for reference in candidate.evidence_source_refs),
        )
        if key not in seen:
            deduplicated.append(candidate)
            seen.add(key)
    return tuple(deduplicated)


def _append_research_candidate(
    candidates: list[_ResearchMemoryCandidate],
    claim: Claim,
    *,
    ticker: str,
    memory_kind: ResearchMemoryKind,
    corpus_version: str,
    importance: float,
    safety: _SummarySafety,
) -> None:
    stored = persisted_claim(claim, ticker=ticker)
    references = tuple(
        reference
        for reference in stored.source_refs
        if reference.ticker == ticker
        and reference.kind in {SourceRefKind.FILING, SourceRefKind.WEB}
    )
    summary = " ".join(stored.text.split())
    if (
        not references
        or not corpus_version.strip()
        or len(summary) > 1_200
        or not _is_research_memory_summary_eligible(
            summary,
            ticker=ticker,
            safety=safety,
        )
    ):
        return
    candidates.append(
        _ResearchMemoryCandidate(
            memory_kind=memory_kind,
            summary=summary,
            evidence_source_refs=references,
            corpus_version=corpus_version,
            importance=importance,
        )
    )


def _is_research_memory_summary_eligible(
    summary: str,
    *,
    ticker: str,
    safety: _SummarySafety,
) -> bool:
    return safety.allows(summary) and is_research_memory_summary_eligible(
        summary,
        ticker,
    )


def _conversation_turn(
    command: ResearchCommand,
    projection: MemoryProjection,
) -> ConversationTurn | None:
    result = projection.result
    if result.status not in {"completed", "partial"}:
        return None
    normalized_question = " ".join(command.request.split())
    if not is_session_memory_eligible_request(
        normalized_question,
        current_ticker=command.ticker,
    ):
        return None
    question = normalized_question[:500].rstrip()
    from fra.research_packages.quality import QualityResearchResult

    if isinstance(result, QualityResearchResult) and result.guarded_report is None:
        return None
    finish = projection.finish
    safety = projection.safety
    retained = [
        claim
        for claim in finish.claims
        if claim.guard_status == "retained"
        and all(reference.ticker == command.ticker for reference in claim.source_refs)
    ]
    answers = [
        text for text in _guarded_answer_parts(result, retained) if safety.allows(text)
    ][:3]
    if not answers:
        return None
    open_question_values = [
        claim.text for claim in retained if claim.kind == "open_question"
    ]
    if isinstance(result, MarketResearchResult):
        report = result.guarded_report
        if report is not None and report.market_context is not None:
            open_question_values.extend(report.market_context.open_questions)
    open_questions = tuple(
        text
        for text in dict.fromkeys(open_question_values)
        if safety.allows(text) and len(" ".join(text.split())) <= 500
    )[:5]
    answer_summary = " ".join("; ".join(answers).split())[:1_200].rstrip()
    return ConversationTurn(
        question=question,
        answer_summary=answer_summary,
        run_id=result.run_id,
        ticker=command.ticker,
        open_questions=open_questions,
    )


def _guarded_answer_parts(
    result: ApplicationResult,
    retained_claims: list[PersistedClaim],
) -> list[str]:
    if isinstance(result, MarketResearchResult):
        report = result.guarded_report
        snapshot = None if report is None else report.snapshot
        if snapshot is None:
            return []
        return [
            (
                f"{result.ticker} was {snapshot.price} {snapshot.currency} as of "
                f"{snapshot.as_of.isoformat()}, with an IEX-only day range of "
                f"{snapshot.day_low} to {snapshot.day_high} and previous close "
                f"{snapshot.previous_close}."
            )
        ]
    return [
        claim.text for claim in retained_claims if claim.kind != "open_question"
    ]


_URL_OR_DOMAIN = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)
_COMMON_WORD_TICKERS = frozenset({"A", "AN", "AT", "BY", "FOR", "IN", "IT", "ON", "OR"})


@dataclass(frozen=True, slots=True)
class _SummarySafety:
    forbidden_values: tuple[str, ...]
    other_tickers: tuple[str, ...]

    def allows(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if not normalized or _URL_OR_DOMAIN.search(normalized):
            return False
        folded = normalized.casefold()
        for value in self.forbidden_values:
            folded_value = value.casefold()
            if len(folded_value) >= 3 and folded_value in folded:
                return False
            if len(folded_value) < 3 and re.search(
                rf"(?<!\w){re.escape(value)}(?!\w)", normalized, re.I
            ):
                return False
        return not any(_contains_other_ticker(normalized, ticker) for ticker in self.other_tickers)


def _contains_other_ticker(text: str, ticker: str) -> bool:
    normalized_ticker = ticker.strip().upper()
    if len(normalized_ticker) <= 1:
        return False
    boundary_pattern = (
        rf"(?<![A-Z0-9]){re.escape(normalized_ticker)}(?![A-Z0-9])"
    )
    if re.search(boundary_pattern, text):
        return True
    return (
        len(normalized_ticker) >= 3
        and normalized_ticker not in _COMMON_WORD_TICKERS
        and re.search(boundary_pattern, text, re.IGNORECASE) is not None
    )


def _summary_safety(result: ApplicationResult, finish: RunFinish) -> _SummarySafety:
    source_values: set[str] = set()
    tickers: set[str] = set()
    for claim in finish.claims:
        for reference in claim.source_refs:
            source_values.update((reference.source_id, reference.encode()))
            tickers.add(reference.ticker)

    errors = getattr(result, "errors", ())
    source_values.update(
        str(error) for error in errors if isinstance(error, str) and len(error.strip()) >= 4
    )
    evidence = getattr(result, "evidence", {})
    if isinstance(evidence, dict):
        for source_id, source in evidence.items():
            source_values.add(str(source_id))
            source_ticker = getattr(source, "ticker", None)
            if isinstance(source_ticker, str):
                tickers.add(source_ticker)

    for run in getattr(result, "skill_runs", ()):
        source_values.update(
            str(error)
            for error in getattr(run, "errors", ())
            if isinstance(error, str) and len(error.strip()) >= 4
        )
        guarded = getattr(run, "guarded_memo", None)
        if guarded is not None:
            for source in (*guarded.filing_sources, *guarded.web_sources):
                source_values.add(source.id)
                tickers.add(source.ticker)

    guarded_report = getattr(result, "guarded_report", None)
    if guarded_report is not None:
        source_values.update(
            str(error)
            for error in getattr(guarded_report, "guard_errors", ())
            if isinstance(error, str) and len(error.strip()) >= 4
        )
        for package in getattr(guarded_report, "packages", ()):
            tickers.add(package.ticker)
            source_values.update(package.guard_notes)
            for source in (*package.filing_sources, *package.web_sources):
                source_values.add(source.id)
                tickers.add(source.ticker)

    current_ticker = str(getattr(result, "ticker", "")).strip().upper()
    scope = getattr(result, "scope", None)
    if scope is not None:
        tickers.update(getattr(scope, "peer_tickers", ()))
        current_ticker = getattr(scope, "primary_ticker", current_ticker)
    return _SummarySafety(
        forbidden_values=tuple(
            sorted(value for value in source_values if value and value.strip())
        ),
        other_tickers=tuple(sorted(ticker for ticker in tickers if ticker != current_ticker)),
    )
