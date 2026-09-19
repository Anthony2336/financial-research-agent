"""CLI-independent orchestration for one research request."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from hashlib import sha256

from pydantic import Field, ValidationError

from fra.context import BudgetAuthority
from fra.contracts import (
    ApplicationResult,
    CompanyResolver,
    IntentRouter,
    ResearchCommand,
    ResearchMode,
    ResearchRuntime,
    ResearchRuntimeFactory,
)
from fra.domain import (
    Intent,
    IntentRoutingError,
    IntentRoutingErrorCode,
    RouterDecision,
    StrictModel,
    is_valid_us_listed_ticker,
)
from fra.execution import (
    NoopResearchRunWriter,
    ResearchExecutionContext,
    ResearchRunWriter,
    RunIdGenerator,
    UuidRunIdGenerator,
    bind_research_context,
    current_research_context,
)
from fra.graph.models import (
    ResearchResult,
    SkillRunResult,
    SkillRunWriter,
)
from fra.graph.workflow import build_early_research_result
from fra.memory.privacy import contains_private_financial_or_secret
from fra.memory.projection import (
    MemoryProjection,
    _persist_research_memories,
    _persist_session_memory,
)
from fra.memory.research import (
    ResearchMemoryStore,
    scope_research_memory_store,
)
from fra.memory.session import (
    SessionMemoryRepository,
)
from fra.observability import (
    AccountingTraceRun,
    NoopTraceRun,
    NoopTraceSink,
    RunAccounting,
    TraceRun,
    TraceSink,
    bind_trace_run,
    complete_root_metadata,
)
from fra.prompts import PromptUsage, bind_prompt_usage
from fra.research_packages.models import (
    GuardedP2Report,
    GuardedResearchPackage,
    PeerResearchRequest,
    PeerScope,
)
from fra.research_packages.orchestrator import (
    PeerResearchOrchestrator,
    PeerResearchResult,
    PeerScopeError,
    validate_peer_scope,
)
from fra.safety.router import route_quality_request, route_request
from fra.storage.run_repositories import (
    RunFinish,
    RunStart,
)

logger = logging.getLogger(__name__)
INVALID_TICKER_TEXT = "Invalid ticker input. Provide a valid US-listed ticker symbol."


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
        from fra.reporting.p2_guard import p2_persisted_claims

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
        from fra.reporting.p2_guard import p2_source_refs

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
                            bind_research_context(context),
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
                            if run_persistence_started and persisted is not None:
                                projection = MemoryProjection(result, persisted)
                                _persist_quality_skill_run(self._skill_runs, result, trace)
                                _persist_session_memory(
                                    self._session_memory,
                                    command,
                                    projection,
                                )
                                _persist_research_memories(
                                    scope_research_memory_store(
                                        self._research_memory,
                                        _runtime_web_validator(runtime),
                                    ),
                                    command,
                                    projection,
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
        from fra.reporting.p2_guard import guard_p2_report
        from fra.reporting.p2_render import render_p2_markdown
        from fra.research_packages.industry import (
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
        from fra.reporting.p2_guard import guard_p2_report
        from fra.reporting.p2_render import render_p2_markdown
        from fra.research_packages.industry import (
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
        from fra.reporting.p2_guard import guard_p2_report
        from fra.reporting.p2_render import render_p2_markdown
        from fra.research_packages.models import (
            MultiTickerResearchPackage,
            ResearchQualityDecision,
        )
        from fra.research_packages.quality import (
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
        from fra.research_packages.quality import (
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
        with bind_research_context(context), bind_prompt_usage(prompt_usage):
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
            if run_persistence_started and persisted is not None:
                projection = MemoryProjection(finalized, persisted)
                _persist_quality_skill_run(self._skill_runs, finalized, trace)
                _persist_session_memory(self._session_memory, command, projection)
                _persist_research_memories(
                    self._research_memory,
                    command,
                    projection,
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
    active_trace = trace if trace is not None else NoopTraceRun()
    if _persist_start(repository, command, result.run_id, active_trace):
        _persist_finish(repository, result, active_trace)


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
) -> RunFinish | None:
    with trace.observation(
        name="persistence.finish",
        kind="tool",
        metadata={"operation": "finish"},
    ) as observation:
        try:
            finish = result.to_run_finish(trace.trace_id)
            context = current_research_context()
            prompt_version = (
                context.prompt_usage.version if context is not None else None
            )
            if prompt_version is not None:
                finish = finish.model_copy(update={"prompt_version": prompt_version})
            repository.finish(finish)
        except Exception:
            logger.warning("research run finish persistence failed")
            observation.update(output={"status": "failed"})
            return None
        else:
            observation.update(output={"status": "completed"})
            return finish


def _persist_quality_skill_run(
    repository: SkillRunWriter | None,
    result: ApplicationResult,
    trace: TraceRun,
) -> None:
    """Persist the deterministic quality recipe under its owning application run."""
    if repository is None:
        return
    from fra.research_packages.quality import QualityResearchResult
    from fra.skills.recipes import RESEARCH_QUALITY_SCREEN

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
