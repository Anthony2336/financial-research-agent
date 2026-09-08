"""CLI-independent orchestration for one research request."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator, model_validator

from financial_evidence_agent.context import BudgetAuthority
from financial_evidence_agent.domain import (
    DEFAULT_RESEARCH_FORMS,
    FilingForm,
    Intent,
    IntentRoutingError,
    IntentRoutingErrorCode,
    RouterDecision,
    SourceRef,
    SourceRefKind,
    StrictModel,
    is_valid_us_listed_ticker,
)
from financial_evidence_agent.graph.models import (
    MarketResearchResult,
    ResearchResult,
    SkillRunResult,
    SkillRunWriter,
    persisted_claim,
)
from financial_evidence_agent.graph.workflow import build_early_research_result
from financial_evidence_agent.memory.models import ConversationTurn
from financial_evidence_agent.memory.privacy import contains_private_financial_or_secret
from financial_evidence_agent.memory.research import (
    ResearchMemoryKind,
    ResearchMemoryStore,
    is_research_memory_summary_eligible,
    scope_research_memory_store,
)
from financial_evidence_agent.memory.session import (
    SessionMemoryRepository,
    is_session_memory_eligible_request,
)
from financial_evidence_agent.observability import (
    AccountingTraceRun,
    NoopTraceSink,
    RunAccounting,
    TraceRun,
    TraceSink,
    bind_trace_run,
    complete_root_metadata,
)
from financial_evidence_agent.prompts import PromptUsage, bind_prompt_usage
from financial_evidence_agent.research_packages.models import (
    GuardedP2Report,
    GuardedResearchPackage,
    PeerResearchRequest,
    PeerScope,
)
from financial_evidence_agent.research_packages.orchestrator import (
    PeerResearchOrchestrator,
    PeerResearchResult,
    PeerScopeError,
    validate_peer_scope,
)
from financial_evidence_agent.safety.router import route_quality_request, route_request
from financial_evidence_agent.storage.run_repositories import (
    PersistedClaim,
    RunFinish,
    RunStart,
    SourceFetchWrite,
)

logger = logging.getLogger(__name__)
INVALID_TICKER_TEXT = "Invalid ticker input. Provide a valid US-listed ticker symbol."


class ResearchMode(StrEnum):
    """Caller-selected workflow mode."""

    THESIS = "thesis"
    AUTO = "auto"
    COMPANY_PROFILE = "company-profile"
    EARNINGS_REVIEW = "earnings-review"
    INDUSTRY_RESEARCH = "industry-research"
    MARKET_SNAPSHOT = "market-snapshot"
    QUALITY_SCREEN = "quality-screen"


class ResearchCommand(StrictModel):
    """Validated input accepted by the research application boundary."""

    ticker: str
    request: str = Field(min_length=1, max_length=2_000)
    mode: ResearchMode
    session_id: str | None = Field(default=None, min_length=1)
    forms: tuple[FilingForm, ...] = Field(
        default=DEFAULT_RESEARCH_FORMS,
        min_length=1,
    )
    as_of_date: date | None = None
    market: Literal["US"] = "US"
    corpus_version: str | None = Field(default=None, min_length=1, exclude=True)
    filing_ids: tuple[str, ...] = Field(default=(), exclude=True)
    scope_error: str | None = Field(default=None, exclude=True)
    peer_tickers: tuple[str, ...] = ()
    peer_scope: str | None = None
    with_context: bool = False

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("request", mode="before")
    @classmethod
    def normalize_request_text(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("forms", mode="before")
    @classmethod
    def normalize_forms(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().upper() if isinstance(item, str) else item for item in value)
        return value

    @field_validator("market", mode="before")
    @classmethod
    def normalize_market(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("filing_ids", mode="before")
    @classmethod
    def normalize_filing_ids(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip() if isinstance(item, str) else item for item in value)
        return value

    @field_validator("peer_tickers", mode="before")
    @classmethod
    def normalize_peer_tickers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().upper() for item in value if isinstance(item, str))
        return value

    @model_validator(mode="after")
    def validate_mode_specific_request_length(self) -> ResearchCommand:
        if self.mode is ResearchMode.THESIS and not 20 <= len(self.request) <= 500:
            raise ValueError("thesis mode request must contain 20 to 500 characters")
        if any(not filing_id for filing_id in self.filing_ids):
            raise ValueError("filing_ids must not contain empty ids")
        return self


class IntentRouter(Protocol):
    """Synchronous structured intent classifier used only by AUTO mode."""

    def route(self, request: str) -> RouterDecision:
        """Return one validated closed-enum decision."""


class CompanyResolver(Protocol):
    """Local supported-company lookup used before durable execution."""

    def resolve(self, ticker: str) -> str | None:
        """Return the canonical supported ticker from local company metadata."""


@runtime_checkable
class ApplicationResult(Protocol):
    """Common final lifecycle implemented by every accepted application result."""

    run_id: str
    status: str
    rendered_output: str

    def bind_run_id(self, run_id: str) -> ApplicationResult:
        """Return this result correlated to the application-owned run ID."""

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        """Return the guarded persistence payload for this result."""

    def root_metadata(self) -> dict[str, object]:
        """Return safe final metadata for the root observation."""


class ResearchRuntime(Protocol):
    """One selected workflow runtime."""

    def execute(
        self,
        command: ResearchCommand,
        decision: RouterDecision,
    ) -> ApplicationResult:
        """Execute a preselected safe research intent."""


class ResearchRuntimeFactory(Protocol):
    """Select a runtime only after request routing succeeds."""

    def build(self, command: ResearchCommand, intent: Intent) -> ResearchRuntime:
        """Build the runtime for one effective intent."""


class P2IndustryResult(StrictModel):
    """Application-bound single-company industry result backed by the final P2 report."""

    run_id: str = Field(default="untracked", min_length=1)
    status: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    decision: RouterDecision
    skill_runs: list[SkillRunResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    package: GuardedResearchPackage
    guarded_report: GuardedP2Report
    rendered_output: str
    source_policy_version: str | None = None

    def bind_run_id(self, run_id: str) -> P2IndustryResult:
        return self.model_copy(update={"run_id": run_id})

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        from financial_evidence_agent.reporting.p2_guard import p2_persisted_claims

        return RunFinish(
            run_id=self.run_id,
            effective_intent=self.decision.intent.value,
            status=self.status,
            corpus_scope=list(self.guarded_report.provenance.corpus_versions),
            prompt_version=_single_prompt_version(self.guarded_report),
            trace_id=trace_id,
            report_markdown=self.rendered_output,
            claims=p2_persisted_claims(self.guarded_report),
        )

    def root_metadata(self) -> dict[str, object]:
        from financial_evidence_agent.reporting.p2_guard import p2_source_refs

        cross_ticker_leakage_count = self.guarded_report.cross_ticker_leakage_count
        cross_ticker_rejection_count = self.guarded_report.cross_ticker_rejection_count
        metadata: dict[str, object] = {
            "effective_intent": self.decision.intent.value,
            "recipe_names": [
                recipe.name.value for recipe in self.guarded_report.provenance.recipes
            ],
            "recipe_versions": [
                recipe.version for recipe in self.guarded_report.provenance.recipes
            ],
            "corpus_scope": list(self.guarded_report.provenance.corpus_versions),
            "prompt_versions": list(self.guarded_report.provenance.prompt_versions),
            "requested_as_of_dates": [
                value.isoformat()
                for value in self.guarded_report.provenance.requested_as_of_dates
            ],
            "evidence_cutoff_dates": [
                value.isoformat()
                for value in self.guarded_report.provenance.evidence_cutoff_dates
            ],
            "source_policy_versions": list(
                self.guarded_report.provenance.source_policy_versions
            ),
            "source_refs": [
                reference.encode() for reference in p2_source_refs(self.guarded_report)
            ],
            "information_sufficiency": self.guarded_report.information_sufficiency.value,
            "guard_errors": list(self.guarded_report.guard_errors),
            "cross_ticker_leakage_count": cross_ticker_leakage_count,
            "cross_ticker_rejection_count": cross_ticker_rejection_count,
        }
        if self.source_policy_version is not None:
            metadata["source_policy_version"] = self.source_policy_version
        return metadata


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


class _RootAccountingFinalizer:
    """Attempt one root update while the caller still owns the active root context."""

    def __init__(self, trace: TraceRun, accounting: RunAccounting, budget: BudgetAuthority) -> None:
        self._trace = trace
        self._accounting = accounting
        self._budget = budget
        self._metadata: dict[str, object] = {}
        self._finished = False

    @contextmanager
    def active(self) -> Iterator[_RootAccountingFinalizer]:
        try:
            yield self
        except Exception:
            try:
                self.finish({}, status="failed")
            except Exception:
                pass
            raise

    def include_metadata(self, metadata: dict[str, object]) -> None:
        self._metadata.update(metadata)

    def finish(self, metadata: dict[str, object], *, status: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._trace.update(
            output={"status": status},
            metadata={
                **self._metadata,
                **complete_root_metadata(
                    metadata,
                    accounting=self._accounting,
                    budget=self._budget,
                    status=status,
                ),
            },
        )


_CURRENT_CONTEXT: ContextVar[ResearchExecutionContext | None] = ContextVar(
    "financial_evidence_agent_research_context",
    default=None,
)


def current_research_context() -> ResearchExecutionContext | None:
    """Return application context, or None for explicit lower-level execution."""
    return _CURRENT_CONTEXT.get()


@contextmanager
def _bind_research_context(context: ResearchExecutionContext) -> Iterator[None]:
    token: Token[ResearchExecutionContext | None] = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


class ResearchApplication:
    """Route then execute one research command without depending on Typer."""

    def __init__(
        self,
        intent_router: IntentRouter,
        runtime_factory: ResearchRuntimeFactory,
        *,
        run_repository: ResearchRunWriter | None = None,
        trace_sink: TraceSink | None = None,
        trace_sink_factory: Callable[[], TraceSink] | None = None,
        id_generator: RunIdGenerator | None = None,
        quality_date_factory: Callable[[], date] | None = None,
        quality_max_source_age_days: int = 365,
        session_memory_store: SessionMemoryRepository | None = None,
        research_memory_store: ResearchMemoryStore | None = None,
        skill_run_repository: SkillRunWriter | None = None,
        company_resolver: CompanyResolver | None = None,
    ) -> None:
        if trace_sink is not None and trace_sink_factory is not None:
            raise ValueError("provide trace_sink or trace_sink_factory, not both")
        self._intent_router = intent_router
        self._runtime_factory = runtime_factory
        self._runs = run_repository or NoopResearchRunWriter()
        self._trace = trace_sink
        self._trace_factory = trace_sink_factory or NoopTraceSink
        self._ids = id_generator or UuidRunIdGenerator()
        self._quality_date_factory = quality_date_factory
        self._quality_max_source_age_days = quality_max_source_age_days
        self._session_memory = session_memory_store
        self._research_memory = research_memory_store
        self._skill_runs = skill_run_repository
        self._company_resolver = company_resolver

    def run(
        self,
        command: ResearchCommand,
        *,
        trace_sink: TraceSink | None = None,
    ) -> ApplicationResult:
        """Execute with one app-owned ID across trace, fetch, and final persistence."""
        run_id = self._ids.new_run_id()
        sink = trace_sink or self._get_trace_sink()
        result: ApplicationResult | None = None
        decision: RouterDecision | None = None
        runtime: ResearchRuntime | None = None
        run_persistence_started = False
        prompt_usage = PromptUsage()
        budget = BudgetAuthority()
        durable_command = _unknown_ticker_command(command)
        try:
            with sink.run(
                run_id=run_id,
                input=_safe_input(command),
                metadata=_run_metadata(durable_command, run_id, None),
            ) as exported_trace:
                accounting = RunAccounting()
                trace = AccountingTraceRun(exported_trace, accounting)
                root_finalizer = _RootAccountingFinalizer(trace, accounting, budget)
                with root_finalizer.active(), bind_trace_run(trace):
                    try:
                        decision = _configuration_free_safety_route(command)
                        if decision is not None:
                            result = _configuration_free_refusal(
                                durable_command,
                                decision,
                                run_id=run_id,
                            )
                            assert result is not None
                            return self._finish_refusal(
                                durable_command,
                                result,
                                run_id=run_id,
                                trace=trace,
                                root_finalizer=root_finalizer,
                                deterministic=decision,
                            )

                        if command.mode is ResearchMode.QUALITY_SCREEN:
                            early_quality_result = self._out_of_scope_quality_preflight(
                                durable_command,
                                command,
                                run_id,
                            )
                            if early_quality_result is not None:
                                result = early_quality_result
                                return self._finish_without_runtime(
                                    durable_command,
                                    early_quality_result,
                                    run_id=run_id,
                                    trace=trace,
                                    budget=budget,
                                    root_finalizer=root_finalizer,
                                )

                        resolved_command = self._resolve_company_scope(command, trace)
                        if resolved_command is None:
                            result = _invalid_ticker_refusal(
                                durable_command,
                                run_id=run_id,
                            )
                            decision = result.decision
                            return self._finish_refusal(
                                durable_command,
                                result,
                                run_id=run_id,
                                trace=trace,
                                root_finalizer=root_finalizer,
                                deterministic=decision,
                            )

                        command = durable_command = resolved_command
                        deterministic = deterministic_research_route(command)
                        root_finalizer.include_metadata(
                            _run_metadata(command, run_id, deterministic)
                        )
                        refusal = _configuration_free_refusal(
                            command,
                            deterministic,
                            run_id=run_id,
                        )
                        if refusal is not None:
                            assert deterministic is not None
                            result = refusal
                            decision = deterministic
                            return self._finish_refusal(
                                command,
                                refusal,
                                run_id=run_id,
                                trace=trace,
                                root_finalizer=root_finalizer,
                                deterministic=deterministic,
                            )

                        _validate_peer_command(command)
                        context = ResearchExecutionContext(
                            run_id,
                            trace,
                            self._runs,
                            budget,
                            prompt_usage,
                            self._session_memory,
                            self._research_memory,
                        )
                        with (
                            _bind_research_context(context),
                            bind_prompt_usage(prompt_usage),
                        ):
                            with trace.observation(
                                name="safety.route",
                                kind="guardrail",
                                metadata={"rule_match": deterministic is not None},
                            ) as guard:
                                guard.update(
                                    output={
                                        "intent": (
                                            deterministic.intent.value
                                            if deterministic is not None
                                            else Intent.AMBIGUOUS.value
                                        )
                                    }
                                )

                            if _is_rule_refusal(deterministic):
                                decision = deterministic
                                assert decision is not None
                                result = build_early_research_result(
                                    command.ticker,
                                    command.request,
                                    decision,
                                    run_id=run_id,
                                )
                                run_persistence_started = _persist_start(
                                    self._runs, command, run_id, trace
                                )
                            else:
                                run_persistence_started = _persist_start(
                                    self._runs, command, run_id, trace
                                )
                                with trace.observation(
                                    name="router.resolve",
                                    kind="chain",
                                    metadata={"mode": command.mode.value},
                                ) as router_observation:
                                    decision = resolve_effective_intent(
                                        command,
                                        self._intent_router,
                                        deterministic=deterministic,
                                    )
                                    router_observation.update(
                                        output={"intent": decision.intent.value}
                                    )
                                if decision.intent not in {
                                    Intent.RESEARCH_REQUEST,
                                    Intent.COMPANY_PROFILE_REQUEST,
                                    Intent.EARNINGS_REVIEW_REQUEST,
                                    Intent.INDUSTRY_RESEARCH_REQUEST,
                                    Intent.MARKET_SNAPSHOT_REQUEST,
                                    Intent.RESEARCH_QUALITY_SCREEN_REQUEST,
                                }:
                                    result = build_early_research_result(
                                        command.ticker,
                                        command.request,
                                        decision,
                                        run_id=run_id,
                                    )
                                else:
                                    with trace.observation(
                                        name="runtime.execute",
                                        kind="chain",
                                        metadata={"intent": decision.intent.value},
                                    ) as runtime_observation:
                                        runtime = self._runtime_factory.build(
                                            command, decision.intent
                                        )
                                        if _should_run_peers(command, decision):
                                            raw_result = self._run_peer_orchestration(
                                                command,
                                                runtime,
                                            )
                                        else:
                                            raw_result = runtime.execute(command, decision)
                                            raw_result = self._finalize_mode_result(
                                                command,
                                                raw_result,
                                                runtime=runtime,
                                            )
                                        if isinstance(raw_result, ApplicationResult):
                                            result = raw_result.bind_run_id(run_id)
                                        else:
                                            result = _invalid_runtime_result(
                                                command,
                                                decision,
                                                run_id=run_id,
                                            )
                                        runtime_observation.update(
                                            output={"status": result.status}
                                        )

                            assert result is not None
                            persisted = _persist_finish(self._runs, result, trace)
                            if run_persistence_started and persisted:
                                _persist_quality_skill_run(self._skill_runs, result, trace)
                                _persist_session_memory(
                                    self._session_memory,
                                    command,
                                    result,
                                )
                                _persist_research_memories(
                                    scope_research_memory_store(
                                        self._research_memory,
                                        _runtime_web_validator(runtime),
                                    ),
                                    command,
                                    result,
                                )
                            root_finalizer.finish(
                                result.root_metadata(),
                                status=result.status,
                            )
                            return result
                    except Exception:
                        if result is None and run_persistence_started:
                            _persist_failure(
                                self._runs,
                                durable_command,
                                run_id,
                                decision,
                                trace_id=trace.trace_id,
                                prompt_version=prompt_usage.version,
                            )
                        raise
        finally:
            try:
                sink.flush()
            except Exception:
                logger.warning("trace flush failed")

    def _resolve_company_scope(
        self,
        command: ResearchCommand,
        trace: TraceRun,
    ) -> ResearchCommand | None:
        resolver = self._company_resolver
        with trace.observation(
            name="scope.resolve",
            kind="tool",
            input={
                "ticker_sha256": sha256(command.ticker.encode("utf-8")).hexdigest(),
                "peer_ticker_sha256s": [
                    sha256(peer.encode("utf-8")).hexdigest()
                    for peer in command.peer_tickers
                ],
            },
            metadata={"peer_count": len(command.peer_tickers)},
        ) as observation:
            if resolver is None:
                observation.update(output={"status": "unavailable"})
                return None
            resolved_tickers: list[str] = []
            for ticker in (command.ticker, *command.peer_tickers):
                try:
                    resolved = resolver.resolve(ticker)
                except Exception:
                    logger.warning("company resolution failed")
                    observation.update(output={"status": "unavailable"})
                    return None
                if not isinstance(resolved, str) or not is_valid_us_listed_ticker(resolved):
                    observation.update(output={"status": "unavailable"})
                    return None
                resolved_tickers.append(resolved.strip().upper())
            resolved_command = command.model_copy(
                update={
                    "ticker": resolved_tickers[0],
                    "peer_tickers": tuple(resolved_tickers[1:]),
                }
            )
            observation.update(
                output={
                    "status": "completed",
                    "ticker": resolved_command.ticker,
                    "peer_tickers": list(resolved_command.peer_tickers),
                }
            )
            return resolved_command

    def _run_peer_orchestration(
        self,
        command: ResearchCommand,
        runtime: ResearchRuntime,
    ) -> PeerResearchResult:
        from financial_evidence_agent.reporting.p2_guard import guard_p2_report
        from financial_evidence_agent.reporting.p2_render import render_p2_markdown
        from financial_evidence_agent.research_packages.industry import (
            build_industry_package,
        )

        request = PeerResearchRequest(
            primary_ticker=command.ticker,
            peer_tickers=command.peer_tickers,
            peer_scope=command.peer_scope or "",
            question=command.request,
        )

        async def run_child(
            ticker: str,
            question: str,
            *,
            run_id: str,
        ) -> GuardedResearchPackage:
            del run_id
            return await asyncio.to_thread(
                self._run_peer_child_command,
                command,
                ticker,
                question,
                build_industry_package,
            )

        result = asyncio.run(PeerResearchOrchestrator(run_child).run(request))
        guarded_report = guard_p2_report(
            scope=result.scope,
            package=result.package,
            comparisons=result.comparisons,
            quality=None,
            web_validator=_runtime_web_validator(runtime),
        )
        rendered_output = render_p2_markdown(guarded_report).rstrip()
        return result.model_copy(
            update={
                "package": result.package.model_copy(update={"packages": guarded_report.packages}),
                "comparisons": guarded_report.comparisons,
                "guarded_report": guarded_report,
                "rendered_output": _apply_runtime_execution_note(
                    runtime,
                    rendered_output,
                ),
                "source_policy_version": getattr(runtime, "source_policy_version", None),
            }
        )

    def _run_peer_child_command(
        self,
        parent_command: ResearchCommand,
        ticker: str,
        question: str,
        build_industry_package,
    ) -> GuardedResearchPackage:
        child_result = self.run(
            ResearchCommand(
                ticker=ticker,
                request=question,
                mode=ResearchMode.INDUSTRY_RESEARCH,
                session_id=parent_command.session_id,
                forms=parent_command.forms,
                as_of_date=parent_command.as_of_date,
                market=parent_command.market,
            ),
            trace_sink=self._trace_factory(),
        )
        if isinstance(child_result, P2IndustryResult):
            if child_result.status not in {"completed", "partial"}:
                raise RuntimeError("peer child run failed")
            return child_result.package
        if not isinstance(child_result, ResearchResult):
            raise RuntimeError("peer child run returned an invalid result")
        if child_result.status not in {"completed", "partial", "insufficient_evidence"}:
            raise RuntimeError("peer child run failed")
        return build_industry_package(child_result)

    def _finalize_mode_result(
        self,
        command: ResearchCommand,
        raw_result: object,
        *,
        runtime: ResearchRuntime | None,
    ) -> object:
        if command.mode is ResearchMode.INDUSTRY_RESEARCH and not command.peer_tickers:
            return self._finalize_industry_result(raw_result, runtime=runtime)
        if command.mode is ResearchMode.QUALITY_SCREEN:
            return self._finalize_quality_result(command, raw_result, runtime=runtime)
        return raw_result

    def _finalize_industry_result(
        self,
        raw_result: object,
        *,
        runtime: ResearchRuntime | None,
    ) -> object:
        from financial_evidence_agent.reporting.p2_guard import guard_p2_report
        from financial_evidence_agent.reporting.p2_render import render_p2_markdown
        from financial_evidence_agent.research_packages.industry import (
            build_industry_package,
        )

        if not isinstance(raw_result, ResearchResult):
            return raw_result
        package = build_industry_package(raw_result)
        guarded_report = guard_p2_report(
            scope=PeerScope(
                primary_ticker=raw_result.ticker,
                peer_tickers=(),
                description="Single-company industry research",
            ),
            package=package,
            comparisons=[],
            quality=None,
            web_validator=_runtime_web_validator(runtime),
        )
        rendered_output = render_p2_markdown(guarded_report).rstrip()
        if runtime is not None:
            rendered_output = _apply_runtime_execution_note(runtime, rendered_output)
        return P2IndustryResult(
            status=raw_result.status,
            ticker=raw_result.ticker,
            decision=raw_result.decision,
            skill_runs=list(raw_result.skill_runs),
            errors=list(raw_result.errors),
            package=guarded_report.packages[0],
            guarded_report=guarded_report,
            rendered_output=rendered_output,
            source_policy_version=raw_result.source_policy_version,
        )

    def _finalize_quality_result(
        self,
        command: ResearchCommand,
        raw_result: object,
        *,
        runtime: ResearchRuntime | None,
    ) -> object:
        from financial_evidence_agent.reporting.p2_guard import guard_p2_report
        from financial_evidence_agent.reporting.p2_render import render_p2_markdown
        from financial_evidence_agent.research_packages.models import (
            MultiTickerResearchPackage,
            ResearchQualityDecision,
        )
        from financial_evidence_agent.research_packages.quality import (
            QualityResearchResult,
        )

        if not isinstance(raw_result, QualityResearchResult):
            return raw_result
        package: GuardedResearchPackage | MultiTickerResearchPackage | None
        if raw_result.package is None:
            if raw_result.quality.decision is ResearchQualityDecision.OUT_OF_SCOPE:
                package = None
            else:
                package = MultiTickerResearchPackage(
                    primary_ticker=command.ticker,
                    packages=[],
                    missing_tickers=[command.ticker],
                    status="failed",
                    cross_ticker_leakage_count=0,
                    guard_notes=[],
                )
        else:
            package = raw_result.package
        guarded_report = guard_p2_report(
            scope=PeerScope(
                primary_ticker=command.ticker,
                peer_tickers=(),
                description="Single-company research quality screen",
            ),
            package=package,
            comparisons=[],
            quality=raw_result.quality,
            web_validator=_runtime_web_validator(runtime),
            requested_as_of=raw_result.requested_as_of,
        )
        updates: dict[str, object] = {
            "guarded_report": guarded_report,
            "rendered_output": render_p2_markdown(guarded_report).rstrip(),
        }
        if raw_result.package is not None and guarded_report.packages:
            updates["package"] = guarded_report.packages[0]
        return raw_result.model_copy(update=updates)

    def _out_of_scope_quality_preflight(
        self,
        durable_command: ResearchCommand,
        ephemeral_command: ResearchCommand,
        run_id: str,
    ):
        if durable_command.mode is not ResearchMode.QUALITY_SCREEN:
            return None
        from financial_evidence_agent.research_packages.quality import (
            QualityResearchResult,
            classify_quality_scope,
            render_quality_markdown,
        )

        quality = classify_quality_scope(
            ephemeral_command.request,
            ticker=ephemeral_command.ticker,
        )
        if quality is None:
            return None
        effective_date = (
            self._quality_date_factory()
            if self._quality_date_factory is not None
            else datetime.now(UTC).date()
        )
        return QualityResearchResult(
            run_id=run_id,
            status="declined",
            ticker=durable_command.ticker,
            package=None,
            quality=quality,
            rendered_output=render_quality_markdown(
                durable_command.ticker,
                quality,
                package=None,
            ),
            effective_date=effective_date,
            max_source_age_days=self._quality_max_source_age_days,
            requested_as_of=durable_command.as_of_date,
        )

    def _get_trace_sink(self) -> TraceSink:
        if self._trace is None:
            try:
                self._trace = self._trace_factory()
            except Exception:
                logger.warning("trace sink construction failed")
                self._trace = NoopTraceSink()
        return self._trace

    def _finish_without_runtime(
        self,
        command: ResearchCommand,
        result: ApplicationResult,
        *,
        run_id: str,
        trace: TraceRun,
        budget: BudgetAuthority,
        root_finalizer: _RootAccountingFinalizer,
    ) -> ApplicationResult:
        finalized = self._finalize_mode_result(command, result, runtime=None)
        assert isinstance(finalized, ApplicationResult)
        prompt_usage = PromptUsage()
        context = ResearchExecutionContext(
            run_id,
            trace,
            self._runs,
            budget,
            prompt_usage,
        )
        with _bind_research_context(context), bind_prompt_usage(prompt_usage):
            with trace.observation(
                name="quality.scope",
                kind="guardrail",
                metadata={"mode": command.mode.value},
            ) as observation:
                observation.update(
                    output={
                        "decision": getattr(finalized, "quality").decision.value,
                    }
                )
            run_persistence_started = _persist_start(
                self._runs, command, run_id, trace
            )
            persisted = _persist_finish(self._runs, finalized, trace)
            if run_persistence_started and persisted:
                _persist_quality_skill_run(self._skill_runs, finalized, trace)
                _persist_session_memory(self._session_memory, command, finalized)
                _persist_research_memories(
                    self._research_memory,
                    command,
                    finalized,
                )
            root_finalizer.finish(finalized.root_metadata(), status=finalized.status)
            return finalized

    def _finish_refusal(
        self,
        command: ResearchCommand,
        refusal: ApplicationResult,
        *,
        run_id: str,
        trace: TraceRun,
        root_finalizer: _RootAccountingFinalizer,
        deterministic: RouterDecision,
    ) -> ApplicationResult:
        root_finalizer.include_metadata(_run_metadata(command, run_id, deterministic))
        persistence_attempted = False
        try:
            with trace.observation(
                name="safety.route",
                kind="guardrail",
                metadata={"rule_match": True},
            ) as observation:
                observation.update(output={"intent": deterministic.intent.value})
            _persist_refusal(self._runs, command, refusal, trace=trace)
            persistence_attempted = True
            root_finalizer.finish(refusal.root_metadata(), status=refusal.status)
        except Exception:
            logger.warning("research refusal trace failed")
            if not persistence_attempted:
                _persist_refusal(self._runs, command, refusal)
            try:
                root_finalizer.finish({}, status="failed")
            except Exception:
                pass
        return refusal


def deterministic_research_route(command: ResearchCommand) -> RouterDecision | None:
    """Return the configuration-free deterministic route for one validated command."""
    if command.mode is ResearchMode.QUALITY_SCREEN:
        return route_quality_request(command.ticker, command.request)
    return route_request(command.ticker, command.request)


def configuration_free_refusal(command: ResearchCommand) -> ResearchResult | None:
    """Return a fixed safety refusal without constructing configuration or providers."""
    safe_command = _unknown_ticker_command(command)
    deterministic = _configuration_free_safety_route(command)
    if deterministic is not None:
        return _configuration_free_refusal(
            safe_command,
            deterministic,
            run_id="preflight",
        )
    primary_valid = is_valid_us_listed_ticker(command.ticker)
    peers_valid = all(is_valid_us_listed_ticker(peer) for peer in command.peer_tickers)
    if not primary_valid or not peers_valid:
        return _invalid_ticker_refusal(safe_command, run_id="preflight")
    return None


def _configuration_free_safety_route(command: ResearchCommand) -> RouterDecision | None:
    decision = deterministic_research_route(command)
    return decision if _is_rule_refusal(decision) else None


def _configuration_free_refusal(
    command: ResearchCommand,
    deterministic: RouterDecision | None,
    *,
    run_id: str,
) -> ResearchResult | None:
    if not _is_rule_refusal(deterministic):
        return None
    assert deterministic is not None
    return build_early_research_result(
        command.ticker,
        command.request,
        deterministic,
        run_id=run_id,
    )


def _invalid_ticker_refusal(
    command: ResearchCommand,
    *,
    run_id: str,
) -> ResearchResult:
    decision = RouterDecision(intent=Intent.AMBIGUOUS, reason="ticker resolution unavailable")
    return ResearchResult(
        run_id=run_id,
        status="refused",
        ticker="UNKNOWN",
        thesis=command.request,
        decision=decision,
        errors=["INVALID_TICKER"],
        rendered_output=INVALID_TICKER_TEXT,
        node_trace=["safety_router"],
    )


def resolve_effective_intent(
    command: ResearchCommand,
    intent_router: IntentRouter,
    *,
    deterministic: RouterDecision | None | object = ...,
) -> RouterDecision:
    """Apply deterministic rules before explicit selection or one AUTO route."""
    if deterministic is ...:
        deterministic = route_request(command.ticker, command.request)
    assert deterministic is None or isinstance(deterministic, RouterDecision)
    if deterministic is not None and deterministic.intent in {
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
        Intent.UNSAFE_SOURCE_REQUEST,
    }:
        return deterministic
    if command.mode is ResearchMode.COMPANY_PROFILE:
        return RouterDecision(
            intent=Intent.COMPANY_PROFILE_REQUEST,
            reason="explicit company-profile mode",
        )
    if command.mode is ResearchMode.EARNINGS_REVIEW:
        return RouterDecision(
            intent=Intent.EARNINGS_REVIEW_REQUEST,
            reason="explicit earnings-review mode",
        )
    if command.mode is ResearchMode.INDUSTRY_RESEARCH:
        return RouterDecision(
            intent=Intent.INDUSTRY_RESEARCH_REQUEST,
            reason="explicit industry-research mode",
        )
    if command.mode is ResearchMode.QUALITY_SCREEN:
        return RouterDecision(
            intent=Intent.RESEARCH_QUALITY_SCREEN_REQUEST,
            reason="explicit quality-screen mode",
        )
    if command.mode is ResearchMode.THESIS:
        return RouterDecision(
            intent=Intent.RESEARCH_REQUEST,
            reason="explicit thesis mode",
        )
    if command.mode is ResearchMode.MARKET_SNAPSHOT:
        return RouterDecision(
            intent=Intent.MARKET_SNAPSHOT_REQUEST,
            reason="explicit market-snapshot mode",
        )
    if deterministic is not None and deterministic.intent is not Intent.AMBIGUOUS:
        return deterministic
    try:
        return RouterDecision.model_validate(intent_router.route(command.request))
    except (TypeError, ValidationError, ValueError):
        raise IntentRoutingError(
            IntentRoutingErrorCode.FAST_ROUTE_INVALID,
            "structured router returned an invalid decision",
        ) from None


def _is_rule_refusal(decision: RouterDecision | None) -> bool:
    return decision is not None and decision.intent in {
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
        Intent.UNSAFE_SOURCE_REQUEST,
    }


def _should_run_peers(command: ResearchCommand, decision: RouterDecision) -> bool:
    return decision.intent is Intent.INDUSTRY_RESEARCH_REQUEST and bool(command.peer_tickers)


def _validate_peer_command(command: ResearchCommand) -> None:
    if not command.peer_tickers and command.peer_scope is None:
        return
    if command.mode is not ResearchMode.INDUSTRY_RESEARCH:
        raise IntentRoutingError(
            IntentRoutingErrorCode.INVALID_PEER_SCOPE,
            "peer options are only valid in industry-research mode",
        )
    if not command.peer_tickers:
        raise IntentRoutingError(
            IntentRoutingErrorCode.INVALID_PEER_SCOPE,
            "peer_scope requires at least one peer_ticker",
        )
    try:
        validate_peer_scope(command.ticker, command.peer_tickers, command.peer_scope or "")
    except PeerScopeError as error:
        raise IntentRoutingError(IntentRoutingErrorCode(error.code), error.detail) from error


def _apply_runtime_execution_note(
    runtime: ResearchRuntime,
    rendered_output: str,
) -> str:
    apply_note = getattr(runtime, "apply_execution_note", None)
    if callable(apply_note):
        return apply_note(
            rendered_output,
            is_p1=True,
            has_guarded_memo=False,
        )
    return rendered_output


def _runtime_web_validator(runtime: ResearchRuntime | None):
    dependencies = getattr(runtime, "dependencies", None)
    return None if dependencies is None else getattr(dependencies, "web_evidence_validator", None)


def _p2_corpus_scope(packages: list[GuardedResearchPackage]) -> list[str]:
    values = [
        source.corpus_version
        for package in packages
        for source in package.filing_sources
        if source.ticker == package.ticker
    ]
    return list(dict.fromkeys(values))


def _single_prompt_version(report: GuardedP2Report) -> str | None:
    versions = report.provenance.prompt_versions
    return versions[0] if len(versions) == 1 else None


def _safe_input(command: ResearchCommand) -> dict[str, object]:
    return {
        "ticker": "UNKNOWN",
        "ticker_sha256": sha256(command.ticker.encode("utf-8")).hexdigest(),
        "mode": command.mode.value,
        "request_sha256": sha256(command.request.encode("utf-8")).hexdigest(),
        "request_length": len(command.request),
    }


def _unknown_ticker_command(command: ResearchCommand) -> ResearchCommand:
    safe_request = _persisted_request(command)
    if safe_request == command.request:
        safe_request = _redact_unresolved_tickers(
            command.request,
            (command.ticker, *command.peer_tickers),
        )
    return command.model_copy(
        update={
            "ticker": "UNKNOWN",
            "request": safe_request,
            "peer_tickers": (),
            "peer_scope": None,
        }
    )


def _redact_unresolved_tickers(request: str, tickers: tuple[str, ...]) -> str:
    safe_request = request
    candidates = sorted(
        {ticker.strip().upper() for ticker in tickers if ticker.strip()},
        key=len,
        reverse=True,
    )
    for candidate in candidates:
        recognized_count = 0
        if len(candidate) <= 2:
            dollar_pattern = re.compile(
                rf"(?<![A-Za-z0-9])\${re.escape(candidate)}"
                r"(?![A-Za-z0-9]|[.-][A-Za-z0-9])",
                re.IGNORECASE,
            )
            safe_request, recognized_count = dollar_pattern.subn(
                "UNKNOWN",
                safe_request,
            )
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])(?<![A-Za-z0-9][.-]){re.escape(candidate)}"
            r"(?![A-Za-z0-9]|[.-][A-Za-z0-9])",
            0 if len(candidate) <= 2 else re.IGNORECASE,
        )
        safe_request, ordinary_count = pattern.subn("UNKNOWN", safe_request)
        recognized_count += ordinary_count
        if len(candidate) <= 2 and recognized_count == 0:
            ambiguous_pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(candidate.casefold())}"
                r"(?![A-Za-z0-9])"
            )
            if ambiguous_pattern.search(safe_request):
                return _private_request_marker(request)
    return safe_request


def _private_request_marker(request: str) -> str:
    digest = sha256(request.encode("utf-8")).hexdigest()
    return f"[private request redacted; sha256={digest}]"


def _persisted_request(command: ResearchCommand) -> str:
    """Keep ordinary research text while replacing private requests with an opaque marker."""
    if not contains_private_financial_or_secret(
        command.request,
        current_ticker=command.ticker,
    ):
        return command.request
    return _private_request_marker(command.request)


def _run_metadata(
    command: ResearchCommand,
    run_id: str,
    deterministic: RouterDecision | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "ticker": command.ticker.strip().upper(),
        "requested_mode": command.mode.value,
        "safety_rule_match": deterministic is not None,
        "safety_intent": (
            deterministic.intent.value if deterministic is not None else Intent.AMBIGUOUS.value
        ),
    }


def _persist_refusal(
    repository: ResearchRunWriter,
    command: ResearchCommand,
    result: ApplicationResult,
    *,
    trace: TraceRun | None = None,
) -> None:
    start = RunStart(
        run_id=result.run_id,
        ticker=command.ticker.strip().upper(),
        request=_persisted_request(command),
        requested_intent=command.mode.value,
    )
    specialized_start = getattr(repository, "start_refusal", None)
    started = False
    with (
        trace.observation(
            name="persistence.start",
            kind="tool",
            metadata={"operation": "start"},
        )
        if trace is not None
        else nullcontext(None)
    ) as observation:
        try:
            if callable(specialized_start):
                started = bool(specialized_start(start))
            else:
                repository.start(start)
                started = True
        except Exception:
            logger.warning("research refusal start persistence failed")
            if observation is not None:
                observation.update(output={"status": "failed"})
        else:
            if observation is not None:
                observation.update(output={"status": "completed"})
    if not started:
        return

    specialized_finish = getattr(repository, "finish_refusal", None)
    with (
        trace.observation(
            name="persistence.finish",
            kind="tool",
            metadata={"operation": "finish"},
        )
        if trace is not None
        else nullcontext(None)
    ) as observation:
        try:
            finish = result.to_run_finish(None if trace is None else trace.trace_id)
            if callable(specialized_finish):
                specialized_finish(finish)
            else:
                repository.finish(finish)
        except Exception:
            logger.warning("research refusal finish persistence failed")
            if observation is not None:
                observation.update(output={"status": "failed"})
        else:
            if observation is not None:
                observation.update(output={"status": "completed"})


def _persist_start(
    repository: ResearchRunWriter,
    command: ResearchCommand,
    run_id: str,
    trace: TraceRun,
) -> bool:
    with trace.observation(
        name="persistence.start",
        kind="tool",
        metadata={"operation": "start"},
    ) as observation:
        try:
            repository.start(
                RunStart(
                    run_id=run_id,
                    ticker=command.ticker.strip().upper(),
                    request=_persisted_request(command),
                    requested_intent=command.mode.value,
                )
            )
        except Exception:
            logger.warning("research run start persistence failed")
            observation.update(output={"status": "failed"})
            return False
        else:
            observation.update(output={"status": "completed"})
            return True


def _persist_finish(
    repository: ResearchRunWriter,
    result: ApplicationResult,
    trace: TraceRun,
) -> bool:
    with trace.observation(
        name="persistence.finish",
        kind="tool",
        metadata={"operation": "finish"},
    ) as observation:
        try:
            finish = _finish_from_result(result, trace.trace_id)
            context = current_research_context()
            prompt_version = (
                context.prompt_usage.version if context is not None else None
            )
            repository.finish(
                finish.model_copy(
                    update={
                        "prompt_version": (
                            finish.prompt_version
                            if prompt_version is None
                            else prompt_version
                        )
                    }
                )
            )
        except Exception:
            logger.warning("research run finish persistence failed")
            observation.update(output={"status": "failed"})
            return False
        else:
            observation.update(output={"status": "completed"})
            return True


def _persist_quality_skill_run(
    repository: SkillRunWriter | None,
    result: ApplicationResult,
    trace: TraceRun,
) -> None:
    """Persist the deterministic quality recipe under its owning application run."""
    if repository is None:
        return
    from financial_evidence_agent.research_packages.quality import QualityResearchResult
    from financial_evidence_agent.skills.recipes import RESEARCH_QUALITY_SCREEN

    if not isinstance(result, QualityResearchResult):
        return
    source_ids = (
        [source.ref for source in result.guarded_report.retained_sources]
        if result.guarded_report is not None
        else list(result.quality.source_refs)
    )
    errors = (
        list(result.guarded_report.guard_errors)
        if result.guarded_report is not None
        else []
    )
    with trace.observation(
        name="persistence.quality_skill_run",
        kind="tool",
        metadata={"operation": "quality_skill_run"},
    ) as observation:
        try:
            skill_run_id = repository.start(
                application_run_id=result.run_id,
                ticker=result.ticker,
                recipe_name=RESEARCH_QUALITY_SCREEN.name.value,
                recipe_version=RESEARCH_QUALITY_SCREEN.version,
                recipe_snapshot=RESEARCH_QUALITY_SCREEN.model_dump(mode="json"),
            )
            repository.finish(
                skill_run_id,
                status="refused" if result.status == "declined" else "completed",
                source_ids=source_ids,
                errors=errors,
            )
        except Exception:
            logger.warning("quality skill run persistence failed")
            observation.update(output={"status": "failed"})
        else:
            observation.update(output={"status": "completed"})


def _persist_session_memory(
    repository: SessionMemoryRepository | None,
    command: ResearchCommand,
    result: ApplicationResult,
) -> None:
    """Best-effort write of source-free guarded continuity after durable run finish."""
    if repository is None or command.session_id is None:
        return
    try:
        turn = _conversation_turn(command, result)
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
    result: ApplicationResult,
) -> None:
    """Best-effort post-persistence writes of bounded, cited research hints."""
    if repository is None or result.status not in {"completed", "partial"}:
        return
    if isinstance(result, MarketResearchResult):
        return
    if not is_session_memory_eligible_request(
        command.request,
        current_ticker=command.ticker,
    ):
        return
    finish = result.to_run_finish(None)
    safety = _summary_safety(result, finish)
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

    from financial_evidence_agent.skills.models import ResearchFacet

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
    claim: object,
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
    result: ApplicationResult,
) -> ConversationTurn | None:
    if result.status not in {"completed", "partial"}:
        return None
    normalized_question = " ".join(command.request.split())
    if not is_session_memory_eligible_request(
        normalized_question,
        current_ticker=command.ticker,
    ):
        return None
    question = normalized_question[:500].rstrip()
    from financial_evidence_agent.research_packages.quality import QualityResearchResult

    if isinstance(result, QualityResearchResult) and result.guarded_report is None:
        return None
    finish = result.to_run_finish(None)
    safety = _summary_safety(result, finish)
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


def _persist_failure(
    repository: ResearchRunWriter,
    command: ResearchCommand,
    run_id: str,
    decision: RouterDecision | None,
    *,
    trace_id: str | None,
    prompt_version: str | None,
) -> None:
    try:
        repository.finish(
            RunFinish(
                run_id=run_id,
                effective_intent=(decision.intent.value if decision is not None else "unresolved"),
                status="failed",
                corpus_scope=[],
                prompt_version=prompt_version,
                trace_id=trace_id,
                report_markdown="",
                claims=[],
            )
        )
    except Exception:
        logger.warning(
            "research run failure persistence failed for ticker %s",
            command.ticker.strip().upper(),
        )


def _finish_from_result(result: ApplicationResult, trace_id: str | None) -> RunFinish:
    return result.to_run_finish(trace_id)


def _persisted_claim(claim: object, *, ticker: str) -> PersistedClaim:
    return persisted_claim(claim, ticker=ticker)


def _invalid_runtime_result(
    command: ResearchCommand,
    decision: RouterDecision,
    *,
    run_id: str,
) -> ResearchResult:
    return ResearchResult(
        run_id=run_id,
        status="failed",
        ticker=command.ticker.strip().upper(),
        thesis=" ".join(command.request.split()),
        decision=decision,
        errors=["RUNTIME_RESULT_INVALID"],
        rendered_output="Research runtime returned an invalid result.",
    )
