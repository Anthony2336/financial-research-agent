"""Citation-bound long-term research-memory contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from itertools import product
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import financial_evidence_agent.memory.research as research_memory_module
from financial_evidence_agent.config import Settings
from financial_evidence_agent.domain import (
    SourceKind,
    SourceRef,
    SourceRefKind,
    SourceTier,
    WebEvidence,
)
from financial_evidence_agent.memory.privacy import contains_private_financial_or_secret
from financial_evidence_agent.memory.privacy_relations import (
    contains_private_named_finance,
)
from financial_evidence_agent.memory.research import (
    ResearchMemory,
    ResearchMemoryKind,
    ResearchMemoryService,
    is_research_memory_summary_eligible,
)
from financial_evidence_agent.memory.session import is_session_memory_eligible_request
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.memory_repositories import ResearchMemoryRepository
from financial_evidence_agent.storage.models import (
    ClaimRecord,
    ResearchMemoryRecord,
    WebEvidenceRecord,
)
from financial_evidence_agent.storage.repositories import ChunkToStore, FilingRepository
from financial_evidence_agent.storage.run_repositories import (
    PersistedClaim,
    ResearchRunRepository,
    RunFinish,
    RunStart,
)
from financial_evidence_agent.storage.web_repositories import WebEvidenceRepository
from financial_evidence_agent.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
)

NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)
SUMMARIES = (
    "Revenue growth remained supported by current-corpus demand disclosures.",
    "Supply constraints remained a material counterevidence item.",
    "Whether demand remains durable is still an open question.",
    "The filing span remains a source pointer for follow-up retrieval.",
    "Gross-margin sensitivity warrants current-corpus rechecking.",
)


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("alice has a high risk tolerance", True),
        ("ALICE has a 10% stake in msft", True),
        ("alice owns msft", True),
        ("NVIDIA holds a stake in CoreWeave", False),
        ("ALICE holds a stake in CoreWeave", True),
        ("TAYLOR owns a stake in CoreWeave", True),
        ("NVIDIA holds a stake in CoreWeave; Alice owns 10 NVDA shares", True),
        ("institutional investors' holdings reported on Form 13F", False),
        ("client account balances disclosed by the company", False),
        ("client account balances disclosed by the company for Alice", True),
        ("client account balances disclosed by the issuer for account ABC-123", True),
        ("client account balances disclosed by the issuer for client Alice", True),
        (
            "client account balances disclosed by the issuer for account holder Alice",
            True,
        ),
        ("client account balances disclosed by the issuer for alice", True),
        ("client account balances disclosed by the issuer for account id abc-123", True),
        (
            "Client account balances disclosed by the issuer. "
            "Compare public revenue for Alice.",
            False,
        ),
    ],
)
def test_relation_scanner_distinguishes_people_aggregates_and_companies(
    text: str,
    private: bool,
) -> None:
    """Broad aggregate/company wording must not exempt an identified financial relation."""
    assert contains_private_named_finance(text, current_ticker="NVDA") is private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("NVIDIA holds a stake in CoreWeave for strategic purposes", False),
        ("NVIDIA's holdings increased", False),
        ("NVIDIA's holdings of common stock", False),
        ("NVIDIA's holdings of strategic importance", False),
        ("NVIDIA's holdings held by institutional investors", False),
        ("NVIDIA's holdings belonging to public shareholders", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors", False),
        ("NVIDIA's holdings of common stock increased", False),
        ("NVIDIA's holdings held by institutional investors worldwide", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors increased", False),
        ("NVIDIA's holdings belonging to public shareholders of record", False),
        ("NVIDIA's holdings of Class A common stock", False),
        ("NVIDIA's holdings of Class A common stock and preferred stock", False),
        (
            "NVIDIA's holdings held by institutional investors or public shareholders worldwide",
            False,
        ),
        ("NVIDIA's holdings of common stock increased and remained outstanding", False),
        ("NVIDIA holds a stake in CoreWeave for strategic growth", False),
        ("NVIDIA holds a stake in CoreWeave for long-term investment", False),
        ("NVIDIA holds a stake in CoreWeave for disclosure purposes", False),
        ("NVIDIA's holdings held by Rose Morgan", True),
        ("NVDA's holdings belonging to Rose Morgan", True),
        ("NVIDIA holds a stake in CoreWeave of Rose Morgan", True),
        ("NVIDIA's holdings held by institutional investor Alice", True),
        ("NVDA's holdings belonging to institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by institutional investor Alice", True),
        ("NVIDIA's holdings for institutional investor Alice", True),
        ("NVDA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA's holdings held by institutional investors, Alice", True),
        ("NVIDIA's holdings held by institutional investors or Alice", True),
        ("NVIDIA's holdings held by institutional investors as well as Alice", True),
        ("NVIDIA's holdings held by institutional investors including Alice", True),
        ("NVIDIA's holdings held by the client Alice and institutional investors", True),
        ("NVIDIA's holdings held by institutional investors and Alice", True),
        ("NVIDIA's holdings of common stock held by Alice", True),
        ("NVIDIA's holdings in Alice's account", True),
        ("NVIDIA holds a stake in CoreWeave for Alice", True),
        ("NVIDIA's holdings held by Alice", True),
        ("NVIDIA's holdings belonging to Alice", True),
        ("NVIDIA's holdings of Alice", True),
        ("NVDA's holdings held by Alice", True),
        ("NVDA's holdings belonging to Alice", True),
        ("NVDA's holdings of Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by Alice", True),
        ("NVIDIA holds a stake in CoreWeave belonging to Alice", True),
        ("NVIDIA holds a stake in CoreWeave of Alice", True),
        ("Institutional investors' holdings in Alice's account", True),
        ("NVDA holds a stake in CoreWeave for strategic purposes", False),
        ("NVDA holds a stake in CoreWeave for Alice", True),
    ],
)
def test_issuer_and_aggregate_exceptions_inspect_the_complete_relation(
    text: str, private: bool
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("NVIDIA holds a stake in CoreWeave for strategic purposes", False),
        ("NVIDIA's holdings increased", False),
        ("NVIDIA's holdings of common stock", False),
        ("NVIDIA's holdings of strategic importance", False),
        ("NVIDIA's holdings held by institutional investors", False),
        ("NVIDIA's holdings belonging to public shareholders", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors", False),
        ("NVIDIA's holdings of common stock increased", False),
        ("NVIDIA's holdings held by institutional investors worldwide", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors increased", False),
        ("NVIDIA's holdings belonging to public shareholders of record", False),
        ("NVIDIA's holdings of Class A common stock", False),
        ("NVIDIA's holdings of Class A common stock and preferred stock", False),
        (
            "NVIDIA's holdings held by institutional investors or public shareholders worldwide",
            False,
        ),
        ("NVIDIA's holdings of common stock increased and remained outstanding", False),
        ("NVIDIA holds a stake in CoreWeave for strategic growth", False),
        ("NVIDIA holds a stake in CoreWeave for long-term investment", False),
        ("NVIDIA holds a stake in CoreWeave for disclosure purposes", False),
        ("NVIDIA's holdings held by Rose Morgan", True),
        ("NVDA's holdings belonging to Rose Morgan", True),
        ("NVIDIA holds a stake in CoreWeave of Rose Morgan", True),
        ("NVIDIA's holdings held by institutional investor Alice", True),
        ("NVDA's holdings belonging to institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by institutional investor Alice", True),
        ("NVIDIA's holdings for institutional investor Alice", True),
        ("NVDA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA's holdings held by institutional investors, Alice", True),
        ("NVIDIA's holdings held by institutional investors or Alice", True),
        ("NVIDIA's holdings held by institutional investors as well as Alice", True),
        ("NVIDIA's holdings held by institutional investors including Alice", True),
        ("NVIDIA's holdings held by the client Alice and institutional investors", True),
        ("NVIDIA's holdings held by institutional investors and Alice", True),
        ("NVIDIA's holdings of common stock held by Alice", True),
        ("NVIDIA's holdings in Alice's account", True),
        ("NVIDIA holds a stake in CoreWeave for Alice", True),
        ("NVIDIA's holdings held by Alice", True),
        ("NVIDIA's holdings belonging to Alice", True),
        ("NVIDIA's holdings of Alice", True),
        ("NVDA's holdings held by Alice", True),
        ("NVDA's holdings belonging to Alice", True),
        ("NVDA's holdings of Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by Alice", True),
        ("NVIDIA holds a stake in CoreWeave belonging to Alice", True),
        ("NVIDIA holds a stake in CoreWeave of Alice", True),
        ("Institutional investors' holdings in Alice's account", True),
        ("NVDA holds a stake in CoreWeave for strategic purposes", False),
        ("NVDA holds a stake in CoreWeave for Alice", True),
    ],
)
def test_complete_relation_privacy_is_enforced_at_research_memory_boundary(
    text: str, private: bool
) -> None:
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


def _direction_matrix_cases() -> list[tuple[str, bool]]:
    cases: list[tuple[str, bool]] = []
    languages = (
        (
            ("Alice", "Bob", "Carol"),
            lambda actor: {
                "before": f"According to Form 4, {actor} owns 10 NVDA shares",
                "after": f"{actor} owns 10 NVDA shares according to Form 4",
                "none": f"{actor} owns 10 NVDA shares",
            },
        ),
        (
            ("李明", "王芳", "赵强"),
            lambda actor: {
                "before": f"根据Form 4披露，{actor}持有10股NVDA",
                "after": f"{actor}持有10股NVDA，根据Form 4披露",
                "none": f"{actor}持有10股NVDA",
            },
        ),
    )
    source_directions = ("before", "after", "none")
    for actors, clauses_for in languages:
        for relation_count in (2, 3):
            for directions in product(
                source_directions,
                repeat=relation_count,
            ):
                clauses = [
                    clauses_for(actor)[direction]
                    for actor, direction in zip(
                        actors[:relation_count],
                        directions,
                        strict=True,
                    )
                ]
                cases.append(("; ".join(clauses) + ".", "none" in directions))
    return cases


@pytest.mark.parametrize(("text", "private"), _direction_matrix_cases())
def test_relation_scanner_keeps_public_attribution_directional_and_relation_local(
    text: str,
    private: bool,
) -> None:
    """Making source attribution document-wide would exempt an unsourced relation."""
    assert contains_private_named_finance(text, current_ticker="NVDA") is private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("Account no. ABC-12345", True),
        ("Alice owns 100 NVDA shares.", True),
        ("José García owns NVDA.", True),
        ("李明持有100股NVDA。", True),
        ("Alice's risk tolerance is aggressive.", True),
        (
            "Analyze insider ownership, then note Taylor Morgan's portfolio is "
            "concentrated in NVDA.",
            True,
        ),
        ("Analyze Taylor Morgan's beneficial ownership disclosed on Form 4.", False),
        ("Analyze NVDA's bearer bonds and debt maturity profile.", False),
        ("Analyze NVDA's risk tolerance and risk-management disclosures.", False),
        ("Acme Holdings LLC owns NVDA shares in its corporate account.", False),
    ],
)
def test_shared_classifier_and_memory_boundaries_agree_on_reviewed_cases(
    text: str,
    private: bool,
) -> None:
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        (
            "Taylor Morgan holds NVDA shares according to Form 4 and Alice owns NVDA.",
            True,
        ),
        (
            "Form 4 shows Taylor Morgan's portfolio is concentrated in NVDA.",
            True,
        ),
        (
            "Taylor Morgan holds NVDA shares per Form 4以及李明的风险偏好较高。",
            True,
        ),
        ("Her risk tolerance: low.", True),
        ("Taylor Morgan's risk profile: aggressive.", True),
        ("李明的风险偏好：较高。", True),
        ("According to Form 4, Taylor Morgan holds NVDA shares.", False),
        ("NVDA risk tolerance: conservative.", False),
        ("Issuer risk appetite: conservative.", False),
        ("发行人风险偏好：保守。", False),
        ("Risk profile: conservative.", False),
    ],
)
def test_round_2_classifier_and_memory_boundaries_have_identical_privacy_results(
    text: str,
    private: bool,
) -> None:
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        (
            "Taylor Morgan owns 100 NVDA shares. Form 4 discusses a public disclosure.",
            True,
        ),
        (
            "Form 4 discusses a public disclosure. Taylor Morgan owns 100 NVDA shares.",
            True,
        ),
        (
            "Taylor Morgan owns NVDA. Schedule 13G discusses filing mechanics.",
            True,
        ),
        ("Taylor Morgan owns NVDA. A public disclosure discusses governance.", True),
        ("Taylor Morgan owns NVDA while Alice owns NVDA according to Form 4.", True),
        ("李明持有NVDA股份，王芳持有NVDA股份 according to Form 4.", True),
        ("Taylor Morgan owns NVDA as disclosed in Schedule 13D.", False),
        ("Form 4 shows Taylor Morgan owns 100 NVDA shares.", False),
        ("Public disclosure reports Taylor Morgan owns NVDA.", False),
        ("根据Form 4披露，李明持有100股NVDA。", False),
        ("公开披露显示，李明持有100股NVDA。", False),
    ],
)
def test_round_3_source_binding_has_classifier_and_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        (
            "Taylor Morgan owns NVDA and Alice owns 100 NVDA shares according to Form 4.",
            True,
        ),
        ("李明持有NVDA，王芳持有NVDA股票，根据Form 4披露。", True),
        (
            "Taylor Morgan owns 100 NVDA shares according to Form 4 and Alice owns NVDA.",
            True,
        ),
        ("根据Form 4披露，李明持有100股NVDA，王芳持有NVDA股票。", True),
        (
            "Taylor Morgan owns NVDA and Alice owns NVDA and Robert Chen owns 100 "
            "NVDA shares according to Form 4.",
            True,
        ),
        ("李明持有NVDA，王芳持有NVDA，赵强持有NVDA股票，根据Form 4披露。", True),
        ("Taylor Morgan owns 100 shares of NVDA according to Form 4.", False),
        ("李明持有100股NVDA，根据Form 4披露。", False),
        ("李明持有NVDA股票100股，根据Form 4披露。", False),
        (
            "According to Form 4, Taylor Morgan owns NVDA, and according to Schedule "
            "13D, Alice owns 100 shares of NVDA.",
            False,
        ),
    ],
)
def test_round_4_relationship_local_source_binding_has_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("根据Form 4披露，李明持有100股NVDA和王芳持有50股NVDA。", True),
        ("根据Form 4披露，李明持有100股NVDA或王芳持有50股NVDA。", True),
        ("根据Form 4披露，李明持有100股NVDA、王芳持有50股NVDA。", True),
        ("根据Form 4披露，李明持有100股NVDA。王芳持有50股NVDA。", True),
        ("根据Form 4披露，李明持有100股NVDA王芳持有50股NVDA。", True),
        (
            "Taylor owns 100 NVDA shares according to Form 4, Alice owns 50 NVDA shares.",
            True,
        ),
        ("李明持有100股NVDA，根据Form 4披露，王芳持有50股NVDA。", True),
        (
            "Taylor owns 100 NVDA shares, Form 4 shows Alice owns 50 NVDA shares.",
            True,
        ),
        (
            "According to Form 4, Taylor owns 100 NVDA shares, Alice owns 50 NVDA "
            "shares as disclosed in Schedule 13D.",
            False,
        ),
        (
            "根据Form 4披露，李明持有100股NVDA，王芳持有50股NVDA，"
            "根据Schedule 13D披露。",
            False,
        ),
        (
            "Taylor owns 100 NVDA shares according to Form 4 and compare NVDA revenue.",
            False,
        ),
        ("李明持有100股NVDA，根据Form 4披露，并比较NVDA收入。", False),
    ],
)
def test_round_5_directional_relation_scanner_has_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("Acme Company employee Alice owns 50 NVDA shares.", True),
        ("Acme Company employee José García owns 50 NVDA shares.", True),
        ("Acme Company employee 王小明 owns 50 NVDA shares.", True),
        ("Acme公司员工王小明持有100股NVDA。", True),
        ("Acme Company owns 50 NVDA shares.", False),
        ("Société Générale Bank owns 50 NVDA shares.", False),
        ("腾讯公司持有100股NVDA。", False),
    ],
)
def test_exact_ownership_subject_has_classifier_and_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("王小明持有100股NVDA。", True),
        ("欧阳娜娜持有100股NVDA。", True),
        ("根据Form 4披露，王小明持有100股NVDA。", False),
        ("根据Form 4披露，欧阳娜娜持有100股NVDA。", False),
    ],
)
def test_full_chinese_ownership_subject_has_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("Acme Company employee Alice's holdings include 50 NVDA shares.", True),
        ("Acme Company employee Alice's portfolio is concentrated in NVDA.", True),
        ("Acme Company employee Alice's risk tolerance is high.", True),
        ("Acme Company employee José García's holdings include 50 NVDA shares.", True),
        (
            "Acme Company employee José García's portfolio is concentrated in NVDA.",
            True,
        ),
        ("Acme Company employee José García's risk tolerance is high.", True),
        ("Acme Company employee 王小明's holdings include 50 NVDA shares.", True),
        ("Acme Company employee 王小明's portfolio is concentrated in NVDA.", True),
        ("Acme Company employee 王小明's risk tolerance is high.", True),
        ("Acme公司员工王小明的持仓包括50股NVDA。", True),
        ("Acme公司员工王小明的投资组合集中于NVDA。", True),
        ("Acme公司员工王小明的风险偏好较高。", True),
        ("Acme Company's holdings include 50 NVDA shares.", False),
        ("Société Générale Bank's portfolio includes NVDA.", False),
        ("腾讯公司的风险偏好反映企业政策。", False),
        ("According to Form 4, Alice's holdings include 50 NVDA shares.", False),
        (
            "According to Form 4, José García's holdings include 50 NVDA shares.",
            False,
        ),
        ("根据Form 4披露，王小明的持仓包括100股NVDA。", False),
    ],
)
def test_exact_profile_subject_has_classifier_and_memory_parity(
    text: str,
    private: bool,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is private
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private
    assert is_research_memory_summary_eligible(text, "NVDA") is not private


@dataclass
class _MemoryFixture:
    engine: object
    repository: ResearchMemoryRepository
    source_ref: SourceRef
    corpus_version: str
    now: list[datetime]


@pytest.fixture
def memory_fixture() -> _MemoryFixture:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    filings = FilingRepository(engine)
    corpus_version = filings.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000001",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/nvda.htm",
        raw_text="Current-corpus evidence.",
        content_hash="a" * 64,
        chunks=[ChunkToStore("MD&A", 0, "Current-corpus evidence.", 3, 0, 24)],
    )
    chunk = filings.list_chunks("NVDA", corpus_version)[0]
    source_ref = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.FILING,
        source_id=chunk.id,
    )
    runs = ResearchRunRepository(engine)
    runs.start(RunStart(run_id="run-completed", ticker="NVDA", request="Research request"))
    runs.finish(
        RunFinish(
            run_id="run-completed",
            effective_intent="research_request",
            status="completed",
            corpus_scope=[corpus_version],
            prompt_version="research-v1",
            trace_id="trace-1",
            report_markdown="# Guarded report",
            claims=[
                PersistedClaim(
                    kind="verified_fact",
                    text=summary,
                    confidence="high",
                    source_refs=[source_ref],
                    guard_status="retained",
                )
                for summary in SUMMARIES
            ],
        )
    )
    now = [NOW]
    return _MemoryFixture(
        engine=engine,
        repository=ResearchMemoryRepository(engine, clock=lambda: now[0]),
        source_ref=source_ref,
        corpus_version=corpus_version,
        now=now,
    )


def _vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * 1022)]


def _store(
    fixture: _MemoryFixture,
    *,
    summary: str = SUMMARIES[0],
    memory_kind: ResearchMemoryKind = ResearchMemoryKind.RESEARCH_SUMMARY,
    vector: list[float] | None = None,
    source_run_id: str = "run-completed",
    evidence_source_refs: tuple[SourceRef, ...] | None = None,
    corpus_version: str | None = None,
    created_at: datetime = NOW,
    expires_at: datetime | None = None,
    embedding_model: str = "hash-1024-v1",
) -> ResearchMemory:
    return fixture.repository.store_guarded(
        ticker="NVDA",
        memory_kind=memory_kind,
        summary=summary,
        source_run_id=source_run_id,
        evidence_source_refs=evidence_source_refs or (fixture.source_ref,),
        corpus_version=corpus_version or fixture.corpus_version,
        embedding=vector or _vector(1.0),
        embedding_model=embedding_model,
        importance=0.8,
        created_at=created_at,
        expires_at=expires_at or created_at + timedelta(days=90),
    )


@pytest.mark.parametrize(
    "kind",
    [
        ResearchMemoryKind.RESEARCH_SUMMARY,
        ResearchMemoryKind.COUNTEREVIDENCE,
        ResearchMemoryKind.OPEN_QUESTION,
        ResearchMemoryKind.SOURCE_POINTER,
    ],
)
def test_research_memory_accepts_exactly_four_frozen_kinds(
    memory_fixture: _MemoryFixture,
    kind: ResearchMemoryKind,
) -> None:
    memory = _store(memory_fixture, memory_kind=kind)

    assert memory.memory_kind is kind
    with pytest.raises(ValidationError, match="memory_kind"):
        ResearchMemory.model_validate(
            memory.model_dump(mode="python")
            | {
                "embedding": memory.embedding,
                "memory_kind": "portfolio_preference",
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evidence_source_refs": ()}, "evidence_source_refs"),
        ({"corpus_version": ""}, "corpus_version"),
        ({"embedding": (1.0, 0.0)}, "embedding"),
        ({"expires_at": NOW}, "expires_at"),
    ],
)
def test_research_memory_rejects_incomplete_or_expired_values(
    memory_fixture: _MemoryFixture,
    changes: dict[str, object],
    message: str,
) -> None:
    valid = _store(memory_fixture)

    with pytest.raises(ValidationError, match=message):
        ResearchMemory.model_validate(
            valid.model_dump(mode="python")
            | {"embedding": valid.embedding}
            | changes
        )


def test_research_memory_value_rejects_sensitive_summary() -> None:
    with pytest.raises(ValidationError, match="privacy"):
        ResearchMemory(
            id="a" * 64,
            scope_key="ticker:NVDA",
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary="My brokerage account is 12345.",
            source_run_id="run-1",
            evidence_source_refs=(
                SourceRef(
                    ticker="NVDA",
                    kind=SourceRefKind.FILING,
                    source_id="chunk-1",
                ),
            ),
            corpus_version="NVDA-v1",
            embedding=tuple(_vector(1.0)),
            embedding_model="hash-1024-v1",
            importance=0.8,
            created_at=NOW,
            expires_at=NOW + timedelta(days=90),
        )


@pytest.mark.parametrize(
    "summary",
    [
        "I hold 100 NVDA shares.",
        "My portfolio contains NVDA.",
        "We have a position in NVDA.",
        "Our brokerage account number is 12345.",
        "Our NVDA shares are held in custody.",
        "You own NVDA shares.",
        "Your risk tolerance is aggressive.",
        "She owns 100 NVDA shares.",
        "His holdings include NVDA.",
        "They plan to sell NVDA.",
        "Their portfolio contains an NVDA position.",
        "The NVDA shares belong to her.",
        "The portfolio belongs to him.",
        "The user holds 20 NVDA shares.",
        "The client has a position in NVDA.",
        "An investor would consider buying NVDA.",
        "The investor's NVDA shares increased.",
        "The account holder plans to sell NVDA.",
        "Taylor Morgan owns 100 NVDA shares.",
        "User risk appetite is high.",
        "Client brokerage account is ABC-123.",
        "password is top-secret-value",
        "api_key=top-secret-value",
        "Bearer private-access-token",
        "access token is private-access-token",
        "token=private-access-token",
        "-----BEGIN PRIVATE KEY-----",
        "Account number: 12345",
        "我持有100股NVDA。",
        "她持有100股NVDA。",
        "他們的投資組合包含NVDA。",
        "本人風險偏好較高。",
        "我的投资组合包含NVDA。",
        "我們的風險偏好很激進。",
        "你打算買入NVDA。",
        "您的券商账户号码是12345。",
        "您的NVDA股票由券商托管。",
        "该用户持有NVDA股票。",
        "客戶計劃賣出NVDA。",
        "投資者的風險承受能力較高。",
        "帳戶持有人持有NVDA倉位。",
        "密码是 top-secret-value",
        "访问令牌是 private-access-token",
        "账户号码是 12345",
    ],
)
def test_generated_summary_privacy_classifier_rejects_personal_finance(
    summary: str,
) -> None:
    assert not research_memory_module.is_research_memory_summary_eligible(
        summary,
        "NVDA",
    )


@pytest.mark.parametrize(
    "summary",
    [
        "The NVDA trade belongs to her.",
        "The NVDA trade is his.",
        "The NVDA position belongs to the user.",
        "The NVDA position is hers.",
        "The holdings belong to them.",
        "The portfolio is the account holder's.",
        "The brokerage account belongs to the client.",
        "The brokerage account is theirs.",
        "The security interest belongs to an investor.",
        "The security interests are ours.",
        "The trade is exclusively his.",
        "The NVDA trade is entirely hers.",
        "The security interest was personally mine.",
        "The security interests were solely theirs.",
        "The stock is wholly the client's.",
        "NVDA 股票交易属于她。",
        "NVDA持仓归他。",
        "NVDA投资组合是她的。",
        "NVDA券商账户属于该客户。",
        "NVDA证券权益是投资者的。",
        "NVDA的持股属于她。",
        "NVDA的持股屬於她。",
        "NVDA持股是她的。",
        "NVDA持股為該客戶的。",
        "NVDA持股为她所有。",
        "NVDA持股属于她。",
    ],
)
def test_generated_summary_privacy_classifier_rejects_reverse_financial_ownership(
    summary: str,
) -> None:
    assert not research_memory_module.is_research_memory_summary_eligible(
        summary,
        "NVDA",
    )


@pytest.mark.parametrize(
    "summary",
    [
        "NVDA's corporate risk tolerance is conservative.",
        "NVDA's portfolio includes strategic subsidiary stakes.",
        "The company holds a minority stake in a supplier.",
        "Customer buying behavior supported demand.",
        "NVDA plans to buy components from suppliers.",
        "Trade restrictions affected the company's sales.",
        "Clients share usage feedback with the company.",
        "The client shares usage feedback with the company.",
        "The issuer holds shares in a consolidated subsidiary.",
        "NVDA owns a subsidiary stake.",
        "It owns a subsidiary stake.",
        "Its corporate risk tolerance is conservative.",
        "The company owns a treasury stake.",
        "Company treasury holdings include subsidiary shares.",
        "The company shares information with her.",
        "The trade belongs to NVDA's operating strategy.",
        "NVDA trades are part of its hedging disclosure.",
        "The treasury position belongs to the company.",
        "The subsidiary's trading account belongs to NVDA.",
        "The security interest belongs to NVDA's consolidated subsidiary.",
        "The company shares business information with the client.",
        "Public Form 4 filings disclose Jensen Huang's insider ownership.",
        "The trade is discussed in the filing while his comments address execution.",
        "NVDA的企业风险承受能力较为保守。",
        "公司持有一家子公司的少数股权。",
        "客户购买行为支持了需求。",
        "供应商采购和贸易限制影响了交付。",
        "该交易属于NVDA的经营策略。",
        "NVDA交易是其对冲披露的一部分。",
        "公司的证券权益归其全资子公司。",
        "公司与客户分享业务信息。",
        "NVDA的持股属于其全资子公司。",
        "NVDA持股是其资本配置的一部分。",
        "NVDA持股是她审阅的披露主题。",
        "NVDA持股是该客户研究的发行人披露。",
    ],
)
def test_generated_summary_privacy_classifier_allows_issuer_disclosures(
    summary: str,
) -> None:
    assert research_memory_module.is_research_memory_summary_eligible(
        summary,
        "NVDA",
    )


@pytest.mark.parametrize(
    "eligible_summary",
    [
        "NVDA持股是她审阅的披露主题。",
        "NVDA持股是该客户研究的发行人披露。",
    ],
)
def test_search_retains_persisted_issuer_summary_with_personal_reviewer_subject(
    memory_fixture: _MemoryFixture,
    eligible_summary: str,
) -> None:
    memory = _store(memory_fixture)
    with Session(memory_fixture.engine) as session, session.begin():
        record = session.get(ResearchMemoryRecord, memory.id)
        claim = session.scalar(
            select(ClaimRecord).where(ClaimRecord.text == SUMMARIES[0])
        )
        assert record is not None
        assert claim is not None
        record.summary = eligible_summary
        claim.text = eligible_summary

    found = memory_fixture.repository.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    )

    assert [item.summary for item in found] == [eligible_summary]


@pytest.mark.parametrize(
    "hostile_summary",
    [
        "The client has a position in NVDA.",
        "She owns 100 NVDA shares.",
        "They plan to sell NVDA.",
        "她持有100股NVDA。",
        "The NVDA trade belongs to her.",
        "The NVDA trade is his.",
        "The NVDA position is hers.",
        "The brokerage account belongs to the client.",
        "The trade is exclusively his.",
        "The NVDA trade is entirely hers.",
        "The security interest was personally mine.",
        "The security interests were solely theirs.",
        "The stock is wholly the client's.",
        "NVDA 股票交易属于她。",
        "NVDA持仓归他。",
        "NVDA账户是她的。",
        "NVDA的持股属于她。",
        "NVDA的持股屬於她。",
        "NVDA持股是她的。",
        "NVDA持股為該客戶的。",
        "NVDA持股为她所有。",
        "NVDA持股属于她。",
    ],
)
def test_search_excludes_hostile_persisted_personal_summary(
    memory_fixture: _MemoryFixture,
    hostile_summary: str,
) -> None:
    memory = _store(memory_fixture)
    with Session(memory_fixture.engine) as session, session.begin():
        record = session.get(ResearchMemoryRecord, memory.id)
        claim = session.scalar(
            select(ClaimRecord).where(ClaimRecord.text == SUMMARIES[0])
        )
        assert record is not None
        assert claim is not None
        record.summary = hostile_summary
        claim.text = record.summary

    assert memory_fixture.repository.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    ) == []


def test_store_requires_completed_same_ticker_run_and_retained_current_corpus_citations(
    memory_fixture: _MemoryFixture,
) -> None:
    with pytest.raises(ValueError, match="source run"):
        _store(memory_fixture, source_run_id="missing-run")

    cross_ticker = SourceRef(
        ticker="AMD",
        kind=SourceRefKind.FILING,
        source_id=memory_fixture.source_ref.source_id,
    )
    with pytest.raises((ValidationError, ValueError), match="ticker"):
        _store(memory_fixture, evidence_source_refs=(cross_ticker,))

    uncited = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.FILING,
        source_id="not-retained-by-run",
    )
    with pytest.raises(ValueError, match="retained citation"):
        _store(memory_fixture, evidence_source_refs=(uncited,))

    with pytest.raises(ValueError, match="corpus"):
        _store(memory_fixture, corpus_version="NVDA-v999")


def test_store_is_content_addressed_and_idempotent(memory_fixture: _MemoryFixture) -> None:
    first = _store(memory_fixture)
    repeated = _store(
        memory_fixture,
        vector=_vector(0.5),
        expires_at=NOW + timedelta(days=30),
    )

    with Session(memory_fixture.engine) as session:
        count = session.scalar(select(func.count()).select_from(ResearchMemoryRecord))

    assert repeated == first
    assert count == 1
    assert len(first.id) == 64
    assert first.scope_key == "ticker:NVDA"
    assert first.evidence_source_refs == (memory_fixture.source_ref,)


def test_concurrent_retries_create_one_memory_row(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'memory-race.sqlite3'}")
    create_schema(engine)
    filings = FilingRepository(engine)
    corpus_version = filings.store_filing(
        ticker="NVDA",
        form="10-Q",
        accession_no="0001045810-26-000010",
        filed_at=date(2026, 5, 20),
        source_url="https://www.sec.gov/Archives/race.htm",
        raw_text="Current-corpus evidence.",
        content_hash="c" * 64,
        chunks=[ChunkToStore("MD&A", 0, "Current-corpus evidence.", 3, 0, 24)],
    )
    source_ref = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.FILING,
        source_id=filings.list_chunks("NVDA", corpus_version)[0].id,
    )
    runs = ResearchRunRepository(engine)
    runs.start(RunStart(run_id="run-race", ticker="NVDA", request="Research request"))
    runs.finish(
        RunFinish(
            run_id="run-race",
            effective_intent="research_request",
            status="completed",
            corpus_scope=[corpus_version],
            prompt_version="research-v1",
            trace_id="trace-race",
            report_markdown="# Guarded report",
            claims=[
                PersistedClaim(
                    kind="verified_fact",
                    text=SUMMARIES[0],
                    confidence="high",
                    source_refs=[source_ref],
                    guard_status="retained",
                )
            ],
        )
    )
    repository = ResearchMemoryRepository(engine, clock=lambda: NOW)
    barrier = Barrier(4)

    def write() -> str:
        barrier.wait()
        return repository.store_guarded(
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary=SUMMARIES[0],
            source_run_id="run-race",
            evidence_source_refs=(source_ref,),
            corpus_version=corpus_version,
            embedding=_vector(1.0),
            embedding_model="hash-1024-v1",
            importance=0.8,
            created_at=NOW,
            expires_at=NOW + timedelta(days=90),
        ).id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: write(), range(4)))
    with Session(engine) as session:
        count = session.scalar(select(func.count()).select_from(ResearchMemoryRecord))

    assert len(set(ids)) == 1
    assert count == 1


def test_sqlite_search_is_semantic_ticker_scoped_unexpired_top_three_and_stale_aware(
    memory_fixture: _MemoryFixture,
) -> None:
    vectors = (
        _vector(1.0),
        _vector(0.9, 0.1),
        _vector(0.8, 0.2),
        _vector(-1.0),
    )
    kinds = (
        ResearchMemoryKind.RESEARCH_SUMMARY,
        ResearchMemoryKind.COUNTEREVIDENCE,
        ResearchMemoryKind.OPEN_QUESTION,
        ResearchMemoryKind.SOURCE_POINTER,
    )
    for summary, vector, kind in zip(SUMMARIES[:4], vectors, kinds, strict=True):
        _store(memory_fixture, summary=summary, memory_kind=kind, vector=vector)
    _store(
        memory_fixture,
        summary=SUMMARIES[4],
        vector=_vector(2.0),
        expires_at=NOW + timedelta(hours=1),
    )
    memory_fixture.now[0] = NOW + timedelta(hours=2)

    hits = memory_fixture.repository.search(
        "nvda",
        _vector(1.0),
        limit=99,
        embedding_model="hash-1024-v1",
        current_corpus_version="NVDA-v2",
    )

    assert [memory.summary for memory in hits] == list(SUMMARIES[:3])
    assert len(hits) == 3
    assert all(memory.ticker == "NVDA" and memory.stale for memory in hits)
    assert memory_fixture.repository.search(
        "AMD",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    ) == []


def test_validated_top_three_skips_more_than_one_page_of_invalid_leaders(
    memory_fixture: _MemoryFixture,
) -> None:
    valid_vectors = (
        _vector(0.8, 0.2),
        _vector(0.7, 0.3),
        _vector(0.6, 0.4),
    )
    for summary, vector in zip(SUMMARIES[:3], valid_vectors, strict=True):
        _store(memory_fixture, summary=summary, vector=vector)
    with Session(memory_fixture.engine) as session, session.begin():
        session.add_all(
            ResearchMemoryRecord(
                id=f"{index + 100:064x}",
                scope_key="ticker:NVDA",
                ticker="NVDA",
                memory_kind="research_summary",
                summary=f"Invalid leading memory {index}.",
                source_run_id="run-completed",
                evidence_source_refs=[f"NVDA:filing:missing-{index}"],
                corpus_version=memory_fixture.corpus_version,
                embedding=_vector(1.0),
                embedding_model="hash-1024-v1",
                importance=1.0,
                created_at=NOW,
                expires_at=NOW + timedelta(days=90),
            )
            for index in range(20)
        )

    hits = memory_fixture.repository.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
        limit=99,
    )

    assert [memory.summary for memory in hits] == list(SUMMARIES[:3])
    assert len(hits) == 3


def test_search_drops_rows_whose_stored_citation_binding_is_no_longer_valid(
    memory_fixture: _MemoryFixture,
) -> None:
    valid = _store(memory_fixture)
    with Session(memory_fixture.engine) as session, session.begin():
        session.add(
            ResearchMemoryRecord(
                id="f" * 64,
                scope_key="ticker:NVDA",
                ticker="NVDA",
                memory_kind="research_summary",
                summary="Tempting but citation-invalid remembered fact.",
                source_run_id="run-completed",
                evidence_source_refs=["NVDA:filing:missing-source"],
                corpus_version=memory_fixture.corpus_version,
                embedding=_vector(2.0),
                embedding_model="hash-1024-v1",
                importance=1.0,
                created_at=NOW,
                expires_at=NOW + timedelta(days=90),
            )
        )

    hits = memory_fixture.repository.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    )

    assert [memory.id for memory in hits] == [valid.id]


def _store_web_backed_memory(
    fixture: _MemoryFixture,
) -> tuple[ResearchMemory, str]:
    content = "Issuer IR evidence retained under the original policy."
    content_hash = sha256(content.encode()).hexdigest()
    web_repository = WebEvidenceRepository(fixture.engine)
    evidence = web_repository.upsert(
        WebEvidence(
            id="incoming-id-is-normalized-by-repository",
            ticker="NVDA",
            title="Issuer update",
            content=content,
            source_url="https://old.example.com/results",
            source_kind=SourceKind.ISSUER_IR,
            source_tier=SourceTier.PRIMARY,
            published_at=NOW - timedelta(days=2),
            fetched_at=NOW - timedelta(days=1),
            content_hash=content_hash,
        )
    )
    source_ref = SourceRef(
        ticker="NVDA",
        kind=SourceRefKind.WEB,
        source_id=evidence.id,
    )
    summary = "Issuer IR evidence remained relevant to the guarded research question."
    runs = ResearchRunRepository(fixture.engine)
    runs.start(RunStart(run_id="run-web", ticker="NVDA", request="Research request"))
    runs.finish(
        RunFinish(
            run_id="run-web",
            effective_intent="research_request",
            status="completed",
            corpus_scope=[fixture.corpus_version],
            prompt_version="research-v1",
            trace_id="trace-web",
            report_markdown="# Guarded web report",
            claims=[
                PersistedClaim(
                    kind="verified_fact",
                    text=summary,
                    confidence="high",
                    source_refs=[source_ref],
                    guard_status="retained",
                )
            ],
        )
    )
    old_policy = SourcePolicy(
        issuer_domains={"NVDA": frozenset({"old.example.com"})}
    )
    memory = ResearchMemoryRepository(
        fixture.engine,
        clock=lambda: NOW,
        web_evidence_validator=PersistedWebEvidenceValidator(
            old_policy,
            web_repository,
        ),
    ).store_guarded(
        ticker="NVDA",
        memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
        summary=summary,
        source_run_id="run-web",
        evidence_source_refs=(source_ref,),
        corpus_version=fixture.corpus_version,
        embedding=_vector(1.0),
        embedding_model="hash-1024-v1",
        importance=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    return memory, evidence.id


def test_web_memory_is_revalidated_under_current_policy_on_every_search(
    memory_fixture: _MemoryFixture,
) -> None:
    memory, _ = _store_web_backed_memory(memory_fixture)
    web_repository = WebEvidenceRepository(memory_fixture.engine)
    allowed = ResearchMemoryRepository(
        memory_fixture.engine,
        clock=lambda: NOW,
        web_evidence_validator=PersistedWebEvidenceValidator(
            SourcePolicy(
                issuer_domains={"NVDA": frozenset({"old.example.com"})}
            ),
            web_repository,
        ),
    )
    denied = ResearchMemoryRepository(
        memory_fixture.engine,
        clock=lambda: NOW,
        web_evidence_validator=PersistedWebEvidenceValidator(
            SourcePolicy(issuer_domains={}),
            web_repository,
        ),
    )
    no_web_policy = ResearchMemoryRepository(
        memory_fixture.engine,
        clock=lambda: NOW,
    )

    assert [item.id for item in allowed.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    )] == [memory.id]
    assert denied.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    ) == []
    assert no_web_policy.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    ) == []
    with pytest.raises(ValueError, match="WEB_SOURCE_POLICY_REJECTED"):
        denied.store_guarded(
            ticker="NVDA",
            memory_kind=memory.memory_kind,
            summary=memory.summary,
            source_run_id=memory.source_run_id,
            evidence_source_refs=memory.evidence_source_refs,
            corpus_version=memory.corpus_version,
            embedding=list(memory.embedding),
            embedding_model=memory.embedding_model,
            importance=memory.importance,
            created_at=NOW,
            expires_at=NOW + timedelta(days=90),
        )


@pytest.mark.parametrize("corruption", ["content", "url", "ticker", "removed"])
def test_web_memory_read_rejects_missing_or_corrupted_canonical_source(
    memory_fixture: _MemoryFixture,
    corruption: str,
) -> None:
    _store_web_backed_memory(memory_fixture)
    with Session(memory_fixture.engine) as session, session.begin():
        record = session.scalar(select(WebEvidenceRecord))
        assert record is not None
        if corruption == "content":
            record.content = "tampered content"
        elif corruption == "url":
            record.source_url = "https://old.example.com/tampered"
        elif corruption == "ticker":
            record.ticker = "AMD"
        else:
            session.delete(record)
    web_repository = WebEvidenceRepository(memory_fixture.engine)
    repository = ResearchMemoryRepository(
        memory_fixture.engine,
        clock=lambda: NOW,
        web_evidence_validator=PersistedWebEvidenceValidator(
            SourcePolicy(
                issuer_domains={"NVDA": frozenset({"old.example.com"})}
            ),
            web_repository,
        ),
    )

    assert repository.search(
        "NVDA",
        _vector(1.0),
        embedding_model="hash-1024-v1",
    ) == []


def test_service_search_never_compares_vectors_from_another_embedding_model(
    memory_fixture: _MemoryFixture,
) -> None:
    _store(
        memory_fixture,
        summary=SUMMARIES[0],
        vector=_vector(0.8, 0.2),
        embedding_model="hash-1024-v1",
    )
    _store(
        memory_fixture,
        summary=SUMMARIES[1],
        memory_kind=ResearchMemoryKind.COUNTEREVIDENCE,
        vector=_vector(1.0),
        embedding_model="legacy-other-space",
    )
    service = ResearchMemoryService(
        memory_fixture.repository,
        _EmbeddingProvider(),
        clock=lambda: NOW,
    )

    hits = service.search("NVDA", "revenue demand")

    assert [memory.embedding_model for memory in hits] == ["hash-1024-v1"]
    assert [memory.summary for memory in hits] == [SUMMARIES[0]]


def test_direct_repository_search_requires_nonblank_embedding_model(
    memory_fixture: _MemoryFixture,
) -> None:
    with pytest.raises(TypeError):
        memory_fixture.repository.search("NVDA", _vector(1.0))
    with pytest.raises(ValueError, match="embedding model"):
        memory_fixture.repository.search(
            "NVDA",
            _vector(1.0),
            embedding_model="   ",
        )


@dataclass
class _EmbeddingProvider:
    dimensions: int = 1024
    version: str = "hash-1024-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_vector(1.0) for _ in texts]


@dataclass
class _RejectUnexpectedStoreRepository:
    calls: list[dict[str, object]] = field(default_factory=list)

    def store_guarded(self, **values: object) -> ResearchMemory:
        self.calls.append(values)
        raise AssertionError("sensitive summary reached repository")

    def search(self, *args: object, **kwargs: object) -> list[ResearchMemory]:
        del args, kwargs
        return []


def test_service_embeds_only_bounded_summaries_and_uses_configured_ttl(
    memory_fixture: _MemoryFixture,
) -> None:
    provider = _EmbeddingProvider()
    service = ResearchMemoryService(
        memory_fixture.repository,
        provider,
        ttl_days=30,
        clock=lambda: NOW,
    )

    stored = service.store_guarded(
        ticker="NVDA",
        memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
        summary=SUMMARIES[0],
        source_run_id="run-completed",
        evidence_source_refs=(memory_fixture.source_ref,),
        corpus_version=memory_fixture.corpus_version,
    )
    hits = service.search(
        "NVDA",
        "revenue demand",
        current_corpus_version=memory_fixture.corpus_version,
    )

    assert provider.calls == [[SUMMARIES[0]], ["revenue demand"]]
    assert stored.expires_at == NOW + timedelta(days=30)
    assert hits[0].stale is False


@pytest.mark.parametrize("summary", ["", "x" * 1_201])
def test_service_rejects_unbounded_summary_before_embedding(
    memory_fixture: _MemoryFixture,
    summary: str,
) -> None:
    provider = _EmbeddingProvider()
    service = ResearchMemoryService(
        memory_fixture.repository,
        provider,
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="summary"):
        service.store_guarded(
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary=summary,
            source_run_id="run-completed",
            evidence_source_refs=(memory_fixture.source_ref,),
            corpus_version=memory_fixture.corpus_version,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "summary",
    [
        "My brokerage account is 12345.",
        "password is top-secret-value",
        "Bearer private-access-token",
        "My holdings include 100 NVDA shares.",
        "My risk tolerance is aggressive.",
        "I would consider buying NVDA.",
    ],
)
def test_service_rejects_sensitive_summary_before_embedding_or_store(
    memory_fixture: _MemoryFixture,
    summary: str,
) -> None:
    provider = _EmbeddingProvider()
    repository = _RejectUnexpectedStoreRepository()
    service = ResearchMemoryService(repository, provider, clock=lambda: NOW)

    with pytest.raises(ValueError, match="privacy"):
        service.store_guarded(
            ticker="NVDA",
            memory_kind=ResearchMemoryKind.RESEARCH_SUMMARY,
            summary=summary,
            source_run_id="run-completed",
            evidence_source_refs=(memory_fixture.source_ref,),
            corpus_version=memory_fixture.corpus_version,
        )

    assert provider.calls == []
    assert repository.calls == []


def test_settings_use_bounded_research_memory_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_MEMORY_TTL_DAYS", raising=False)
    assert Settings(_env_file=None).research_memory_ttl_days == 90

    monkeypatch.setenv("RESEARCH_MEMORY_TTL_DAYS", "45")
    assert Settings(_env_file=None).research_memory_ttl_days == 45

    for value in (0, 3651):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, research_memory_ttl_days=value)
