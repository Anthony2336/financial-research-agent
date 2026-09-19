"""Dependency protocols and public results for the research graph."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from inspect import isawaitable
from typing import Literal, Protocol

from fastmcp import Client
from pydantic import BaseModel, ConfigDict, Field

from fra.context import BudgetAuthority, BudgetGate, MemoryHint
from fra.domain import (
    Claim,
    EvidenceChunk,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    SourceRef,
    SourceRefKind,
    StrictModel,
)
from fra.market_data.models import MarketDataBundle
from fra.memory.research import ResearchMemorySearch
from fra.memory.session import SessionMemoryRepository
from fra.observability import NoopTraceSink, TraceRun, TraceSink
from fra.reporting.guard import GuardedMemo
from fra.reporting.market_guard import GuardedMarketReport
from fra.retrieval.collector import EvidenceBundle
from fra.skills.models import (
    EvidenceCollectionPolicy,
    ResearchRecipe,
    SkillName,
)
from fra.skills.schemas import GuardedSkillMemo, SkillResearchMemo
from fra.storage.run_repositories import (
    PersistedClaim,
    RunFinish,
    SourceFetchWrite,
)
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
)


class FastModel(Protocol):
    """Fast model operations allowed before evidence synthesis."""

    def route(self, thesis: str) -> RouterDecision:
        """Classify a genuinely ambiguous request once."""

    def plan(
        self,
        ticker: str,
        thesis: str,
        *,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> list[ResearchQuestion]:
        """Return structured research questions."""


class AnalystModel(Protocol):
    """Evidence-only analyst with no tool-binding interface."""

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk] | EvidenceBundle,
    ) -> ResearchMemo:
        """Synthesize a memo from structured questions and evidence values only."""


class ThesisRepairModel(Protocol):
    """One bounded, evidence-only repair attempt after deterministic thesis guarding."""

    def repair(
        self,
        *,
        draft: ResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: list[EvidenceChunk] | EvidenceBundle,
    ) -> ResearchMemo:
        """Return a restrictive draft transformation without tool authority."""


class SkillPlanningInput(BaseModel):
    """Frozen P1 request and recipe snapshot presented to model boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(min_length=1, max_length=10)
    user_request: str = Field(min_length=1, max_length=2_000)
    recipe: ResearchRecipe
    memory_hints: tuple[MemoryHint, ...] = ()


class SkillAnalysisInput(BaseModel):
    """Hint-free frozen P1 request presented only to evidence analysts."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(min_length=1, max_length=10)
    user_request: str = Field(min_length=1, max_length=2_000)
    recipe: ResearchRecipe


class SkillPlannerModel(Protocol):
    """Provider-neutral structured planner for a frozen P1 recipe."""

    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        """Plan bounded questions without choosing or changing the recipe."""


class SkillAnalystModel(Protocol):
    """Provider-neutral analyst limited to a supplied P1 evidence bundle."""

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        """Return a recipe-bound memo using only evidence IDs from the bundle."""


class SkillRepairModel(Protocol):
    """One bounded repair attempt over a guarded P1 draft."""

    async def repair(
        self,
        *,
        request: SkillAnalysisInput,
        draft: SkillResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        """Return a restrictive recipe-bound transformation without tool authority."""


class SkillEvidenceCollector(Protocol):
    """Provider-neutral P1 collector boundary used by graph orchestration."""

    async def collect(
        self,
        *,
        ticker: str,
        recipe: ResearchRecipe,
        questions: Sequence[ResearchQuestion],
    ) -> EvidenceBundle:
        """Collect one recipe's bounded, validated evidence bundle."""

    def with_budget_gate(self, gate: BudgetGate) -> SkillEvidenceCollector:
        """Return a detached collector whose attempts consume the supplied gate."""


class ThesisEvidenceCollector(Protocol):
    """Phased collector boundary used only by the default thesis graph."""

    def with_filing_scope(
        self,
        corpus_version: str | None,
        filing_ids: tuple[str, ...],
        *,
        scope_error: str | None = None,
    ) -> ThesisEvidenceCollector:
        """Bind one immutable application-resolved filing scope."""

    def with_budget_gate(self, gate: BudgetGate) -> ThesisEvidenceCollector:
        """Return a detached collector whose attempts consume the supplied gate."""

    async def retrieve(
        self,
        *,
        ticker: str,
        recipe: EvidenceCollectionPolicy,
        questions: Sequence[ResearchQuestion],
    ) -> EvidenceBundle:
        """Run the first local round."""

    async def retry_missing(
        self,
        *,
        ticker: str,
        recipe: EvidenceCollectionPolicy,
        questions: Sequence[ResearchQuestion],
        evidence: EvidenceBundle,
    ) -> EvidenceBundle:
        """Run the one missing-only local retry."""

    async def web_fallback(
        self,
        *,
        ticker: str,
        recipe: EvidenceCollectionPolicy,
        questions: Sequence[ResearchQuestion],
        evidence: EvidenceBundle,
    ) -> EvidenceBundle:
        """Run the one policy-gated web fallback when eligible."""


class SkillRunWriter(Protocol):
    """Persistence boundary for immutable recipe snapshots and final provenance."""

    def start(
        self,
        *,
        application_run_id: str,
        ticker: str,
        recipe_name: str,
        recipe_version: str,
        recipe_snapshot: dict[str, object],
    ) -> str:
        """Persist a detached recipe snapshot before execution begins."""

    def finish(
        self,
        run_id: str,
        *,
        status: Literal["completed", "partial", "refused", "failed"],
        source_ids: Sequence[SourceRef],
        errors: Sequence[str],
    ) -> None:
        """Persist the final status and stable source identifiers."""


class SkillModelErrorCode(StrEnum):
    """Fail-closed categories for invalid structured model output."""

    INVALID_OUTPUT = "invalid_output"
    RECIPE_IDENTITY_MISMATCH = "recipe_identity_mismatch"
    INVALID_EVIDENCE_ID = "invalid_evidence_id"


class SkillModelError(RuntimeError):
    """Typed model-boundary failure safe for graph-level handling."""

    def __init__(self, code: SkillModelErrorCode, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code.value}: {message}")


class MCPToolClient(Protocol):
    """Small synchronous boundary used by the CLI-compatible workflow."""

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        """Call one named MCP tool and return its structured response envelope."""


class SourceFetchWriter(Protocol):
    """Optional run-scoped persistence boundary used by MCP client adapters."""

    def record_fetch(self, value: SourceFetchWrite) -> None:
        """Persist safe metadata for one source retrieval attempt."""


class MarketBundleWriter(Protocol):
    """Persistence boundary for one final guarded market bundle."""

    def save_bundle(self, bundle: MarketDataBundle) -> None:
        """Persist the final guarded market snapshot and bars."""


class FastMCPToolClient:
    """Open and close a FastMCP client around each synchronous tool invocation.

    ``call_tool`` uses ``asyncio.run`` and therefore must be called outside an active
    event loop. The synchronous graph and CLI satisfy that lifecycle constraint.
    """

    def __init__(self, server: object) -> None:
        self._server = server

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("FastMCPToolClient.call_tool must run outside an active event loop")
        return asyncio.run(self._call_tool(name, arguments))

    async def _call_tool(self, name: str, arguments: dict[str, object]) -> object:
        async with Client(self._server) as client:
            return await client.call_tool(name, arguments, raise_on_error=False)


class FastMCPToolSession:
    """One synchronous FastMCP session and async-resource lifecycle per market run."""

    def __init__(self, server: object, *, resources: tuple[object, ...] = ()) -> None:
        self._server = server
        self._resources = resources
        self._runner: asyncio.Runner | None = None
        self._client: Client | None = None

    def __enter__(self) -> FastMCPToolSession:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("FastMCPToolSession must run outside an active event loop")
        self._runner = asyncio.Runner()
        self._client = Client(self._server)
        try:
            self._runner.run(self._client.__aenter__())
        except BaseException:
            self._runner.close()
            self._runner = None
            self._client = None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        runner = self._runner
        client = self._client
        self._runner = None
        self._client = None
        if runner is None or client is None:
            return False
        try:
            runner.run(self._close(client, exc_type, exc, traceback))
        finally:
            runner.close()
        return False

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        if self._runner is None or self._client is None:
            raise RuntimeError("FastMCPToolSession must be entered before calling tools")
        return self._runner.run(self._client.call_tool(name, arguments, raise_on_error=False))

    async def _close(
        self,
        client: Client,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        try:
            await client.__aexit__(exc_type, exc, traceback)
        finally:
            seen: set[int] = set()
            for resource in self._resources:
                if id(resource) in seen:
                    continue
                seen.add(id(resource))
                close = getattr(resource, "aclose", None)
                if callable(close):
                    result = close()
                    if isawaitable(result):
                        await result


@dataclass(frozen=True, slots=True)
class Dependencies:
    """Explicit graph dependencies; no node reaches repositories or retrievers."""

    mcp_client: MCPToolClient
    fast_model: FastModel
    analyst_model: AnalystModel
    trace_sink: TraceSink = field(default_factory=NoopTraceSink)
    trace_run: TraceRun | None = None
    skill_planner: SkillPlannerModel | None = None
    skill_collector: SkillEvidenceCollector | None = None
    skill_analyst: SkillAnalystModel | None = None
    skill_run_repository: SkillRunWriter | None = None
    web_evidence_validator: PersistedWebEvidenceValidator | None = None
    thesis_collector: ThesisEvidenceCollector | None = None
    thesis_collection_policy: EvidenceCollectionPolicy | None = None
    budget: BudgetAuthority | None = None
    session_memory_store: SessionMemoryRepository | None = None
    research_memory_store: ResearchMemorySearch | None = None
    thesis_repair_model: ThesisRepairModel | None = None
    skill_repair_model: SkillRepairModel | None = None


@dataclass(frozen=True, slots=True)
class MarketDependencies:
    """Only the two dependencies available to the short market workflow."""

    mcp_client: MCPToolClient
    max_bars: int
    max_staleness_seconds: int = 90
    context_window_days: int = 3
    abnormal_move_threshold: Decimal = Decimal("0.05")
    web_evidence_validator: PersistedWebEvidenceValidator | None = None
    bundle_writer: MarketBundleWriter | None = None
    session_factory: Callable[[], AbstractContextManager[MCPToolClient]] | None = None
    clock: Callable[[], datetime] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_bars <= 20:
            raise ValueError("max_bars must be between 1 and 20")
        if self.max_staleness_seconds <= 0:
            raise ValueError("max_staleness_seconds must be positive")
        if not 1 <= self.context_window_days <= 7:
            raise ValueError("context_window_days must be between 1 and 7")
        if not isinstance(self.abnormal_move_threshold, Decimal):
            raise TypeError("abnormal_move_threshold must be Decimal")
        if not Decimal("0") < self.abnormal_move_threshold <= Decimal("1"):
            raise ValueError("abnormal_move_threshold must be between 0 and 1")

    def session(self) -> AbstractContextManager[MCPToolClient]:
        """Open the market tools in one lifecycle; direct test clients need no setup."""
        if self.session_factory is not None:
            return self.session_factory()
        return nullcontext(self.mcp_client)


class SkillRunPlan(BaseModel):
    """One frozen, persisted recipe selected by deterministic dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    recipe: ResearchRecipe


class SkillRunResult(StrictModel):
    """Auditable public result for one P1 recipe execution."""

    run_id: str = Field(min_length=1)
    recipe_name: SkillName
    recipe_version: str = Field(min_length=1)
    allowed_tools: tuple[str, ...]
    status: Literal["completed", "partial", "failed", "refused"]
    questions: list[ResearchQuestion] = Field(default_factory=list)
    evidence: EvidenceBundle | None = None
    memo: SkillResearchMemo | None = None
    guarded_memo: GuardedSkillMemo | None = None
    errors: list[str] = Field(default_factory=list)
    rendered_output: str = ""


class ResearchResult(StrictModel):
    """Strict graph output preserving both analyst draft and guarded report input."""

    run_id: str = Field(default="untracked", min_length=1)
    status: Literal[
        "completed",
        "partial",
        "failed",
        "refused",
        "declined",
        "insufficient_evidence",
    ]
    ticker: str = Field(min_length=1, max_length=10)
    thesis: str = Field(min_length=1, max_length=2_000)
    decision: RouterDecision
    questions: list[ResearchQuestion] = Field(default_factory=list)
    evidence: dict[str, EvidenceChunk] = Field(default_factory=dict)
    corpus_version: str | None = None
    memo: ResearchMemo | None = None
    guarded_memo: GuardedMemo | None = None
    skill_runs: list[SkillRunResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    rendered_output: str
    source_policy_version: str | None = None
    prompt_version: str | None = None
    report_as_of: date | None = None
    node_trace: list[str] = Field(default_factory=list)

    def bind_run_id(self, run_id: str) -> ResearchResult:
        """Return the same immutable-value result correlated to the application run."""
        return self.model_copy(update={"run_id": run_id})

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        """Map guarded P0/P1 state to the common final persistence contract."""
        return RunFinish(
            run_id=self.run_id,
            effective_intent=self.decision.intent.value,
            status=self.status,
            corpus_scope=_result_corpus_scope(self),
            prompt_version=self.prompt_version,
            trace_id=trace_id,
            report_markdown=self.rendered_output,
            claims=_result_persisted_claims(self),
        )

    def root_metadata(self) -> dict[str, object]:
        """Return safe common metadata for the application root observation."""
        metadata: dict[str, object] = {
            "effective_intent": self.decision.intent.value,
            "recipe_names": [run.recipe_name.value for run in self.skill_runs],
            "recipe_versions": [run.recipe_version for run in self.skill_runs],
            "corpus_scope": _result_corpus_scope(self),
        }
        if self.source_policy_version is not None:
            metadata["source_policy_version"] = self.source_policy_version
        if self.prompt_version is not None:
            metadata["prompt_versions"] = [self.prompt_version]
        else:
            metadata["prompt_versions"] = []
        metadata["requested_as_of_dates"] = (
            [] if self.report_as_of is None else [self.report_as_of.isoformat()]
        )
        metadata["evidence_cutoff_dates"] = list(
            dict.fromkeys(
                cutoff.isoformat()
                for run in self.skill_runs
                if run.guarded_memo is not None
                for cutoff in run.guarded_memo.provenance.evidence_cutoff_dates
            )
        )
        return metadata


class MarketResearchResult(StrictModel):
    """Guarded market output implementing the common application lifecycle."""

    run_id: str = Field(default="untracked", min_length=1)
    status: Literal["completed", "partial", "failed", "refused"]
    ticker: str = Field(min_length=1, max_length=10)
    guarded_report: GuardedMarketReport | None
    rendered_output: str
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def from_guarded(
        cls,
        guarded: GuardedMarketReport,
        rendered_output: str,
    ) -> MarketResearchResult:
        return cls(
            status=guarded.status,
            ticker=guarded.ticker,
            guarded_report=guarded,
            rendered_output=rendered_output,
            errors=list(guarded.errors),
        )

    def bind_run_id(self, run_id: str) -> MarketResearchResult:
        return self.model_copy(update={"run_id": run_id})

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        return RunFinish(
            run_id=self.run_id,
            effective_intent="market_snapshot_request",
            status=self.status,
            corpus_scope=[],
            prompt_version=None,
            trace_id=trace_id,
            report_markdown=self.rendered_output,
            claims=_market_persisted_claims(self),
        )

    def root_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "effective_intent": "market_snapshot_request",
            "provider": "alpaca",
            "feed": "iex",
            "coverage": "IEX-only",
            "source_refs": [],
        }
        if self.guarded_report is not None:
            metadata["source_refs"] = [
                reference.encode() for reference in self.guarded_report.source_refs
            ]
            if self.guarded_report.snapshot is not None:
                metadata["exchange"] = self.guarded_report.snapshot.exchange
                metadata["currency"] = self.guarded_report.snapshot.currency
        return metadata


def _market_persisted_claims(result: MarketResearchResult) -> list[PersistedClaim]:
    report = result.guarded_report
    if report is None or report.snapshot is None:
        return []
    snapshot_refs = [
        reference
        for reference in report.source_refs
        if reference.kind is SourceRefKind.MARKET_SNAPSHOT
    ]
    bar_refs = [
        reference for reference in report.source_refs if reference.kind is SourceRefKind.MARKET_BAR
    ]
    claims = [
        PersistedClaim(
            kind="market_snapshot",
            text=f"{result.ticker} guarded Alpaca IEX-only market snapshot.",
            confidence="high",
            source_refs=snapshot_refs,
            guard_status="retained",
        )
    ]
    if bar_refs:
        claims.append(
            PersistedClaim(
                kind="market_bars",
                text=f"{result.ticker} guarded Alpaca IEX-only daily bars.",
                confidence="high",
                source_refs=bar_refs,
                guard_status="retained",
            )
        )
    return claims


def persisted_claim(claim: Claim, *, ticker: str) -> PersistedClaim:
    """Encode one retained claim without flattening filing and web provenance."""
    kind = claim.kind
    confidence = claim.confidence
    return PersistedClaim(
        kind=kind.value,
        text=claim.text,
        confidence=confidence.value,
        source_refs=[
            *(
                SourceRef(
                    ticker=ticker,
                    kind=SourceRefKind.FILING,
                    source_id=source_id,
                )
                for source_id in claim.evidence_chunk_ids
            ),
            *(
                SourceRef(
                    ticker=ticker,
                    kind=SourceRefKind.WEB,
                    source_id=source_id,
                )
                for source_id in claim.web_evidence_ids
            ),
        ],
        guard_status="retained",
    )


def _result_corpus_scope(result: ResearchResult) -> list[str]:
    values: list[str] = []
    if result.guarded_memo is not None:
        values.append(result.guarded_memo.corpus_version)
    for skill_run in result.skill_runs:
        if skill_run.guarded_memo is None:
            continue
        values.extend(source.corpus_version for source in skill_run.guarded_memo.filing_sources)
    return list(dict.fromkeys(values))


def _result_persisted_claims(result: ResearchResult) -> list[PersistedClaim]:
    claims: list[PersistedClaim] = []
    if result.guarded_memo is not None:
        for claim in (
            *result.guarded_memo.supporting_claims,
            *result.guarded_memo.counter_claims,
            *result.guarded_memo.inferences,
            *result.guarded_memo.open_questions,
        ):
            claims.append(persisted_claim(claim, ticker=result.ticker))
    for skill_run in result.skill_runs:
        guarded = skill_run.guarded_memo
        if guarded is None:
            continue
        for section in guarded.memo.sections:
            claims.extend(persisted_claim(claim, ticker=result.ticker) for claim in section.claims)
    return claims
