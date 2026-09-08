"""Small deterministic model substitutes for the bundled offline NVDA demo."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from financial_evidence_agent.context import MemoryHint
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
)

DEMO_TICKER = "NVDA"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = frozenset(
    """
    a an and are as at by challenge challenges did disclose disclosed despite does
    evidence filing for from has have how in is its material nvda of on or show
    support supports that the their thesis this to was were what whether which with
    """.split()
)
_TOKEN_ALIASES = {
    "constraints": "constraint",
    "customers": "customer",
    "declined": "decline",
    "decreased": "decline",
    "decreases": "decline",
    "dropped": "decline",
    "grew": "growth",
    "growing": "growth",
    "increased": "growth",
    "increases": "growth",
    "reduce": "decline",
    "reduced": "decline",
    "reduces": "decline",
    "revenues": "revenue",
    "risks": "risk",
}
# Deliberately narrow: the offline adapter must fail closed outside bundled-fixture topics.
_CONCEPT_PATTERNS = {
    "concentration": re.compile(r"\bconcentration\b|集中", re.IGNORECASE),
    "customer": re.compile(r"\bcustomers?\b|客户", re.IGNORECASE),
    "data_center": re.compile(r"\bdata centers?\b|数据中心", re.IGNORECASE),
    "decline": re.compile(
        r"\b(?:declin(?:e|ed)|decreas(?:e|ed)|drop(?:ped)?|fell|reduc(?:e[sd]?|ing))\b|"
        r"下降|下滑|减少",
        re.IGNORECASE,
    ),
    "demand": re.compile(r"\bdemand\b|需求", re.IGNORECASE),
    "growth": re.compile(r"\b(?:grow(?:th|ing)?|grew|increas(?:e|ed))\b|增长", re.IGNORECASE),
    "revenue": re.compile(r"\brevenues?\b|收入", re.IGNORECASE),
    "risk": re.compile(r"\brisks?\b|风险", re.IGNORECASE),
}
_NEGATIVE_REVENUE_DIRECTION = re.compile(
    r"(?:\brevenue\b.{0,24}\b(?:declin(?:e|ed)|decreas(?:e|ed)|fell|drop(?:ped)?)\b|"
    r"\b(?:declin(?:e|ed)|decreas(?:e|ed)|drop(?:ped)?)\b.{0,16}\brevenue\b|"
    r"收入.{0,12}(?:下降|下滑|减少))",
    flags=re.IGNORECASE,
)


class DeterministicDemoFastModel:
    """Plan thesis-derived queries for the deterministic filing demonstration."""

    def route(self, thesis: str) -> RouterDecision:
        del thesis
        return RouterDecision(
            intent=Intent.RESEARCH_REQUEST,
            reason="deterministic offline demo research route",
        )

    def plan(
        self,
        ticker: str,
        thesis: str,
        *,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> list[ResearchQuestion]:
        del memory_hints
        normalized_ticker = ticker.strip().upper()
        normalized_thesis = " ".join(thesis.split())
        return [
            ResearchQuestion(
                question=normalized_thesis[:500],
                support_query=_bounded_query(normalized_ticker, normalized_thesis),
                challenge_query=_bounded_query(
                    normalized_ticker,
                    f"risks constraints counter evidence {normalized_thesis}",
                ),
                forms=["10-Q"],
            )
        ]


class DeterministicDemoAnalystModel:
    """Extract only fixture sentences that align with the requested thesis."""

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        thesis = questions[0].question
        supporting_claims: list[Claim] = []
        counter_claims: list[Claim] = []
        if _has_relevant_evidence(thesis, evidence):
            concepts = _concepts(thesis)
            growth_fact = _fact_with_concepts(
                evidence,
                section="MD&A",
                required={"data_center", "growth", "revenue"},
            )
            if growth_fact is not None and concepts & {
                "data_center",
                "decline",
                "demand",
                "growth",
            }:
                if _NEGATIVE_REVENUE_DIRECTION.search(thesis):
                    counter_claims.append(growth_fact)
                else:
                    supporting_claims.append(growth_fact)

            if concepts & {"concentration", "decline", "demand", "growth", "revenue", "risk"}:
                concentration_fact = _fact_with_concepts(
                    evidence,
                    section="Risk Factors",
                    required={"concentration", "customer", "revenue"},
                )
                if concentration_fact is not None:
                    counter_claims.append(concentration_fact)

        complete = bool(supporting_claims and counter_claims)
        open_question = (
            "How did these disclosed factors change after the filing cutoff?"
            if complete
            else "The local demo corpus does not contain enough thesis-relevant evidence."
        )
        return ResearchMemo(
            research_question=thesis,
            supporting_claims=supporting_claims,
            counter_claims=counter_claims,
            open_questions=[
                Claim(
                    kind=ClaimKind.OPEN_QUESTION,
                    text=open_question,
                    confidence=Confidence.LOW,
                )
            ],
            information_sufficiency="A" if complete else "C",
            confidence=Confidence.HIGH if complete else Confidence.LOW,
        )


@contextmanager
def bundled_nvda_fixture_path() -> Iterator[Path]:
    """Materialize the packaged filing fixture for one deterministic demo run."""
    resource = files("financial_evidence_agent.resources").joinpath("nvda_10q.html")
    with as_file(resource) as path:
        yield path


def _bounded_query(ticker: str, terms: str) -> str:
    return f"{ticker} {terms}".strip()[:500]


def _has_relevant_evidence(thesis: str, evidence: list[EvidenceChunk]) -> bool:
    """Require two filing concepts and reject unknown substantive English terms."""
    evidence_text = " ".join(chunk.content for chunk in evidence)
    thesis_concepts = _concepts(thesis)
    if len(thesis_concepts & _concepts(evidence_text)) < 2:
        return False

    thesis_tokens = _meaningful_tokens(thesis)
    if not thesis_tokens:
        return True
    evidence_tokens = _meaningful_tokens(evidence_text)
    matched_tokens = thesis_tokens & evidence_tokens
    return len(matched_tokens) >= 2 and matched_tokens == thesis_tokens


def _fact_with_concepts(
    evidence: list[EvidenceChunk],
    *,
    section: str,
    required: set[str],
) -> Claim | None:
    """Return the first verbatim sentence containing every required concept."""
    for chunk, sentence in _sentences(evidence, section=section):
        if required <= _concepts(sentence):
            return _fact_from_sentence(chunk, sentence)
    return None


def _sentences(
    evidence: list[EvidenceChunk],
    *,
    section: str,
) -> Iterator[tuple[EvidenceChunk, str]]:
    for chunk in evidence:
        if chunk.section != section:
            continue
        for sentence in _SENTENCE_END.split(chunk.content.strip()):
            if sentence:
                yield chunk, sentence


def _concepts(text: str) -> set[str]:
    return {
        concept
        for concept, pattern in _CONCEPT_PATTERNS.items()
        if pattern.search(text) is not None
    }


def _meaningful_tokens(text: str) -> set[str]:
    return {
        _TOKEN_ALIASES.get(token, token)
        for token in _WORD.findall(text.casefold())
        if token not in _QUERY_STOP_WORDS
    }


def _fact_from_sentence(chunk: EvidenceChunk, sentence: str) -> Claim:
    return Claim(
        kind=ClaimKind.VERIFIED_FACT,
        text=sentence,
        confidence=Confidence.HIGH,
        evidence_chunk_ids=[chunk.id],
    )
