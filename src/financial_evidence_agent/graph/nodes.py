"""Deterministic nodes for the controlled research graph."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Coroutine, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Any, Literal, NotRequired, TypedDict, TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, TypeAdapter, ValidationError

from financial_evidence_agent.context import (
    BudgetExhaustedError,
    BudgetLimits,
    HierarchicalBudgetGate,
    MemoryHint,
)
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    SourceRef,
    SourceRefKind,
)
from financial_evidence_agent.graph.models import (
    Dependencies,
    SkillAnalysisInput,
    SkillModelError,
    SkillPlanningInput,
    SkillRunPlan,
    SkillRunResult,
)
from financial_evidence_agent.mcp_server.tools import HybridSearchFilingsResponse
from financial_evidence_agent.memory.models import ConversationTurn
from financial_evidence_agent.memory.research import (
    ResearchMemory,
    build_research_memory_hints,
)
from financial_evidence_agent.memory.session import (
    build_session_memory_hints,
    load_session_memory_for_ticker,
)
from financial_evidence_agent.reporting import (
    GuardedMemo,
    guard_memo,
    guard_skill_memo,
    render_markdown,
    render_skill_markdown,
)
from financial_evidence_agent.retrieval.collector import (
    CollectionErrorCode,
    EvidenceBundle,
    EvidenceCollectionError,
)
from financial_evidence_agent.safety.router import route_request
from financial_evidence_agent.skills.models import ResearchRecipe
from financial_evidence_agent.skills.recipes import (
    RESEARCH_INPUT_DISPATCH_CONTRACT,
    RESEARCH_INPUT_RECIPES,
)
from financial_evidence_agent.skills.registry import ResearchRecipeRegistry
from financial_evidence_agent.skills.schemas import (
    GuardedSkillMemo,
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)

_ENGLISH_RESEARCH_TERMS = re.compile(
    r"\b(?:revenue|growth|demand|margin|earnings|cash flow|filing|disclosure|"
    r"risk|capacity|support|challenge|headwind|segment|guidance)\b"
)
_CHINESE_RESEARCH_TERMS = (
    "收入",
    "增长",
    "需求",
    "利润",
    "现金流",
    "财报",
    "披露",
    "风险",
    "支持",
    "挑战",
    "数据中心",
)
_RESEARCH_QUESTION_LIST = TypeAdapter(list[ResearchQuestion])
_SKILL_RESEARCH_REGISTRY = ResearchRecipeRegistry(
    RESEARCH_INPUT_RECIPES,
    dispatch_contract=RESEARCH_INPUT_DISPATCH_CONTRACT,
)
_SKILL_RESEARCH_INTENTS = frozenset(RESEARCH_INPUT_DISPATCH_CONTRACT.ordered_recipe_names)
_NONFATAL_SKILL_GUARD_CODES = frozenset(
    {
        "BULL_BEAR_EVIDENCE_INCOMPLETE",
        "FINANCIAL_DATA_DISCREPANCY",
        "FINANCIAL_DATA_NOT_COMPARABLE",
        "MISSING_REQUIRED_FACET",
        "WEB_SOURCE_CANONICAL_MISMATCH",
        "WEB_SOURCE_POLICY_REJECTED",
        "WEB_SOURCE_TICKER_MISMATCH",
    }
)
_TRACE_COVERAGE_REASONS = frozenset(
    {
        "challenge_missing",
        "date_mismatch",
        "insufficient_information",
        "invalid_source_id",
        "missing_facet",
        "support_missing",
        "ticker_mismatch",
        "zero_new_evidence",
    }
)
logger = logging.getLogger(__name__)

_OrderedValue = TypeVar("_OrderedValue")


class SkillDispatchError(ValueError):
    """Raised when a non-P1 intent reaches the static recipe dispatcher."""


def dispatch_skill_recipes(intent: Intent) -> tuple[ResearchRecipe, ...]:
    """Resolve and detach the exact code-owned recipe sequence for one P1 intent."""
    recipes = _SKILL_RESEARCH_REGISTRY.resolve(intent)
    if not recipes:
        raise SkillDispatchError(f"unsupported P1 intent: {intent!s}")
    return tuple(
        ResearchRecipe.model_validate(recipe.model_dump(mode="python")) for recipe in recipes
    )


class ResearchState(TypedDict):
    run_id: str
    ticker: str
    thesis: str
    corpus_version: NotRequired[str | None]
    filing_ids: NotRequired[tuple[str, ...]]
    scope_error: NotRequired[str | None]
    node_trace: list[str]
    requested_intent: NotRequired[Intent]
    decision: NotRequired[RouterDecision]
    questions: NotRequired[list[ResearchQuestion]]
    evidence: NotRequired[dict[str, EvidenceChunk]]
    evidence_bundle: NotRequired[EvidenceBundle | None]
    retrieval_fatal_error: NotRequired[str]
    p0_fatal_error: NotRequired[str]
    memo: NotRequired[ResearchMemo | None]
    guarded_memo: NotRequired[GuardedMemo | None]
    skill_run_plans: NotRequired[list[SkillRunPlan]]
    skill_runs: NotRequired[list[SkillRunResult]]
    p1_fatal_error: NotRequired[str]
    errors: NotRequired[list[str]]
    rendered_output: NotRequired[str]
    p0_budget_exhausted: NotRequired[str]
    session_id: NotRequired[str | None]
    report_as_of: NotRequired[date | None]
    recent_turns: NotRequired[tuple[ConversationTurn, ...]]
    session_summary: NotRequired[str]
    memory_hints: NotRequired[tuple[MemoryHint, ...]]
    session_memory_hints: NotRequired[tuple[MemoryHint, ...]]
    research_memory_hints: NotRequired[tuple[MemoryHint, ...]]
    research_memories: NotRequired[tuple[ResearchMemory, ...]]
    messages: NotRequired[list[HumanMessage]]


def make_safety_router(dependencies: Dependencies):
    def safety_router(state: ResearchState) -> dict[str, object]:
        decision = route_request(state["ticker"], state["thesis"])
        update: dict[str, object] = _trace_update(state, "safety_router", dependencies)
        if decision is not None and decision.intent in {
            Intent.PROHIBITED_ADVICE,
            Intent.PROMPT_INJECTION,
            Intent.UNSAFE_SOURCE_REQUEST,
        }:
            update["decision"] = decision
        elif requested_intent := state.get("requested_intent"):
            update["decision"] = RouterDecision(
                intent=requested_intent,
                reason="caller-selected effective intent",
            )
        elif decision is not None:
            update["decision"] = decision
        return update

    return safety_router


def make_normalize(dependencies: Dependencies):
    def normalize(state: ResearchState) -> dict[str, object]:
        ticker = state["ticker"].strip().upper()
        thesis = " ".join(state["thesis"].split())
        errors = list(state.get("errors", []))
        existing_decision = state.get("decision")
        if existing_decision is not None and existing_decision.intent in {
            Intent.RESEARCH_REQUEST,
            *_SKILL_RESEARCH_INTENTS,
        }:
            decision = existing_decision
        elif _is_clear_research_thesis(thesis):
            decision = RouterDecision(
                intent=Intent.RESEARCH_REQUEST,
                reason="deterministic research-thesis match",
            )
        else:
            try:
                raw_decision = dependencies.fast_model.route(thesis)
            except Exception as error:
                decision = RouterDecision(
                    intent=Intent.AMBIGUOUS,
                    reason="fast router call failed",
                )
                _append_unique(errors, f"FAST_ROUTE_ERROR: {type(error).__name__}")
            else:
                try:
                    decision = RouterDecision.model_validate(raw_decision)
                except (TypeError, ValidationError):
                    decision = RouterDecision(
                        intent=Intent.AMBIGUOUS,
                        reason="invalid fast router output",
                    )
                    _append_unique(errors, "FAST_ROUTE_INVALID: Invalid router decision")
        return {
            **_trace_update(state, "normalize", dependencies),
            "ticker": ticker,
            "thesis": thesis,
            "decision": decision,
            "errors": errors,
        }

    return normalize


def make_load_session_memory(dependencies: Dependencies):
    """Load only same-session, same-ticker continuity hints after safe normalization."""

    def load_session_memory(state: ResearchState) -> dict[str, object]:
        store = dependencies.session_memory_store
        session_id = state.get("session_id")
        memory = load_session_memory_for_ticker(store, session_id, state["ticker"])
        turns = memory.turns
        hints = build_session_memory_hints(memory)
        messages = [
            *(HumanMessage(content=hint.text) for hint in hints),
            HumanMessage(content=state["thesis"]),
        ][-5:]
        return {
            **_trace_update(state, "load_session_memory", dependencies),
            "recent_turns": turns,
            "session_summary": memory.summary,
            "session_memory_hints": hints,
            "memory_hints": hints,
            "messages": messages,
        }

    return load_session_memory


def make_load_research_memory(dependencies: Dependencies):
    """Load at most three ticker-scoped query hints without citation authority."""

    def load_research_memory(state: ResearchState) -> dict[str, object]:
        store = dependencies.research_memory_store
        memories: tuple[ResearchMemory, ...] = ()
        if store is not None:
            try:
                loaded = store.search(
                    state["ticker"],
                    state["thesis"],
                    current_corpus_version=state.get("corpus_version"),
                    limit=3,
                )
                normalized_ticker = state["ticker"].strip().upper()
                current_corpus = state.get("corpus_version")
                now = datetime.now(UTC)
                memories = tuple(
                    memory.model_copy(
                        update={
                            "stale": (
                                current_corpus is not None
                                and memory.corpus_version != current_corpus
                            )
                        }
                    )
                    for memory in loaded
                    if memory.ticker == normalized_ticker
                    and memory.expires_at > now
                    and all(
                        reference.ticker == normalized_ticker
                        for reference in memory.evidence_source_refs
                    )
                )[:3]
            except Exception:
                logger.warning("research memory load failed")
        research_hints = build_research_memory_hints(memories)
        session_hints = state.get("session_memory_hints", ())
        hints = (*session_hints, *research_hints)
        messages = [
            *(HumanMessage(content=hint.text) for hint in hints),
            HumanMessage(content=state["thesis"]),
        ][-5:]
        return {
            **_trace_update(state, "load_research_memory", dependencies),
            "research_memories": memories,
            "research_memory_hints": research_hints,
            "memory_hints": hints,
            "messages": messages,
        }

    return load_research_memory


def make_skill_dispatch(dependencies: Dependencies):
    """Freeze and persist all selected recipes before any model or collector call."""

    def skill_dispatch(state: ResearchState) -> dict[str, object]:
        update = _p1_trace_update(state, "skill_dispatch", dependencies)
        errors = list(state.get("errors", []))
        decision = state.get("decision")
        if (
            dependencies.skill_planner is None
            or dependencies.skill_collector is None
            or dependencies.skill_analyst is None
            or dependencies.skill_run_repository is None
        ):
            _append_unique(errors, "P1_DEPENDENCY_MISSING")
            return {
                **update,
                "skill_run_plans": [],
                "skill_runs": [],
                "p1_fatal_error": "P1_DEPENDENCY_MISSING",
                "errors": errors,
            }
        if decision is None:
            _append_unique(errors, "P1_DISPATCH_FAILURE")
            return {
                **update,
                "skill_run_plans": [],
                "skill_runs": [],
                "p1_fatal_error": "P1_PROTOCOL_FAILURE",
                "errors": errors,
            }

        try:
            recipes = dispatch_skill_recipes(decision.intent)
        except SkillDispatchError:
            _append_unique(errors, "P1_DISPATCH_FAILURE")
            return {
                **update,
                "skill_run_plans": [],
                "skill_runs": [],
                "p1_fatal_error": "P1_PROTOCOL_FAILURE",
                "errors": errors,
            }

        plans: list[SkillRunPlan] = []
        try:
            for recipe in recipes:
                run_id = dependencies.skill_run_repository.start(
                    application_run_id=state["run_id"],
                    ticker=state["ticker"],
                    recipe_name=recipe.name.value,
                    recipe_version=recipe.version,
                    recipe_snapshot=recipe.model_dump(mode="json"),
                )
                plans.append(SkillRunPlan(run_id=run_id, recipe=recipe))
        except Exception:
            for plan in plans:
                _finish_without_raising(
                    dependencies,
                    plan.run_id,
                    status="refused",
                    source_ids=[],
                    errors=["P1_PROTOCOL_FAILURE"],
                )
            _append_unique(errors, "P1_PROTOCOL_FAILURE")
            return {
                **update,
                "skill_run_plans": plans,
                "skill_runs": [
                    _refused_skill_result(plan, "P1_PROTOCOL_FAILURE") for plan in plans
                ],
                "p1_fatal_error": "P1_PROTOCOL_FAILURE",
                "errors": errors,
            }

        _record_trace(
            dependencies,
            "p1_dispatch",
            {
                "recipe_names": [plan.recipe.name.value for plan in plans],
                "recipe_versions": [plan.recipe.version for plan in plans],
            },
        )
        return {
            **update,
            "skill_run_plans": plans,
            "skill_runs": [],
            "errors": errors,
        }

    return skill_dispatch


def make_execute_skill_recipes(dependencies: Dependencies):
    """Execute planner, collector, analyst, and guard once per frozen recipe."""

    def execute_skill_recipes(state: ResearchState) -> dict[str, object]:
        update = _p1_trace_update(state, "execute_skill_recipes", dependencies)
        plans = list(state.get("skill_run_plans", []))
        results = list(state.get("skill_runs", []))
        errors = list(state.get("errors", []))
        if state.get("p1_fatal_error"):
            return {**update, "skill_runs": results, "errors": errors}

        for index, plan in enumerate(plans):
            try:
                result = _execute_skill_recipe(state, dependencies, plan)
            except _FatalP1Error:
                _append_unique(errors, "P1_PROTOCOL_FAILURE")
                aborted = [plan, *plans[index + 1 :]]
                for aborted_plan in aborted:
                    _finish_without_raising(
                        dependencies,
                        aborted_plan.run_id,
                        status="refused",
                        source_ids=[],
                        errors=["P1_PROTOCOL_FAILURE"],
                    )
                    refused = _refused_skill_result(
                        aborted_plan,
                        "P1_PROTOCOL_FAILURE",
                    )
                    results.append(refused)
                    _trace_skill_result(dependencies, refused)
                return {
                    **update,
                    "skill_runs": results,
                    "p1_fatal_error": "P1_PROTOCOL_FAILURE",
                    "errors": errors,
                }
            results.append(result)
            _trace_skill_result(dependencies, result)

        return {**update, "skill_runs": results, "errors": errors}

    return execute_skill_recipes


def make_combine_skill_sections(dependencies: Dependencies):
    """Combine only guarded P1 reports and explicit partial placeholders."""

    def combine_skill_sections(state: ResearchState) -> dict[str, object]:
        update = _p1_trace_update(state, "combine_skill_sections", dependencies)
        with _p1_observation(
            dependencies,
            name="p1.render_final",
            kind="chain",
            metadata={"skill_run_count": len(state.get("skill_runs", []))},
        ) as observation:
            if state.get("p1_fatal_error"):
                rendered = (
                    "Unable to execute the requested research safely. "
                    "No research sections were rendered."
                )
            else:
                sections = [
                    run.rendered_output
                    if run.rendered_output
                    else _render_partial_skill_section(run)
                    for run in state.get("skill_runs", [])
                ]
                rendered = "\n".join(section.rstrip() for section in sections if section).rstrip()
                if rendered:
                    rendered += "\n"
                else:
                    rendered = "Insufficient evidence to produce a P1 research report."
            if observation is not None:
                observation.update(
                    output={
                        "status": "completed",
                        "rendered_length": len(rendered),
                    }
                )
        return {**update, "rendered_output": rendered}

    return combine_skill_sections


def make_plan_questions(dependencies: Dependencies):
    def plan_questions(state: ResearchState) -> dict[str, object]:
        errors = list(state.get("errors", []))
        fatal_error: str | None = None
        budget = dependencies.budget
        if budget is not None:
            try:
                budget.consume(planner_calls=1)
            except BudgetExhaustedError as error:
                detail = f"BUDGET_EXHAUSTED: {error.dimension}"
                _append_unique(errors, detail)
                return {
                    **_trace_update(state, "plan_questions", dependencies),
                    "questions": [],
                    "errors": errors,
                    "p0_budget_exhausted": detail,
                }
        try:
            memory_hints = state.get("memory_hints", ())
            if memory_hints:
                raw_questions = dependencies.fast_model.plan(
                    state["ticker"],
                    state["thesis"],
                    memory_hints=memory_hints,
                )
            else:
                raw_questions = dependencies.fast_model.plan(state["ticker"], state["thesis"])
        except Exception as error:
            questions: list[ResearchQuestion] = []
            detail = f"FAST_PLAN_ERROR: {type(error).__name__}"
            _append_unique(errors, detail)
            if dependencies.thesis_collector is not None:
                fatal_error = detail
        else:
            try:
                questions = _RESEARCH_QUESTION_LIST.validate_python(raw_questions)
            except (TypeError, ValidationError):
                questions = []
                detail = "FAST_PLAN_INVALID: Invalid research question list"
                _append_unique(errors, detail)
                if dependencies.thesis_collector is not None:
                    fatal_error = detail
            else:
                if not questions:
                    detail = "FAST_PLAN_EMPTY: Planner returned no research questions"
                    _append_unique(errors, detail)
                    if dependencies.thesis_collector is not None:
                        fatal_error = detail
        update: dict[str, object] = {
            **_trace_update(state, "plan_questions", dependencies),
            "questions": list(questions[:3]),
            "errors": errors,
        }
        if fatal_error is not None:
            update["p0_fatal_error"] = fatal_error
        return update

    return plan_questions


def make_retrieve_evidence(dependencies: Dependencies):
    def retrieve_evidence(state: ResearchState) -> dict[str, object]:
        evidence: dict[str, EvidenceChunk] = {}
        errors = list(state.get("errors", []))
        if state.get("p0_budget_exhausted"):
            return {
                **_trace_update(state, "retrieve_evidence", dependencies),
                "evidence": {},
                "evidence_bundle": None,
                "errors": errors,
            }
        if state.get("p0_fatal_error") and dependencies.thesis_collector is not None:
            return {
                **_trace_update(state, "retrieve_evidence", dependencies),
                "evidence": {},
                "evidence_bundle": None,
                "errors": errors,
            }
        if (
            dependencies.thesis_collector is not None
            and dependencies.thesis_collection_policy is not None
        ):
            if str(state.get("scope_error", "")).startswith("NO_FILINGS:"):
                _append_unique(errors, str(state["scope_error"]))
            try:
                bundle = asyncio.run(
                    dependencies.thesis_collector.retrieve(
                        ticker=state["ticker"],
                        recipe=dependencies.thesis_collection_policy,
                        questions=state.get("questions", []),
                    )
                )
            except BudgetExhaustedError as error:
                detail = f"BUDGET_EXHAUSTED: {error.dimension}"
                _append_unique(errors, detail)
                return {
                    **_trace_update(state, "retrieve_evidence", dependencies),
                    "evidence": {},
                    "evidence_bundle": None,
                    "p0_budget_exhausted": detail,
                    "errors": errors,
                }
            except EvidenceCollectionError as error:
                detail = _safe_collection_error(error)
                _append_unique(errors, detail)
                return {
                    **_trace_update(state, "retrieve_evidence", dependencies),
                    "evidence": {},
                    "evidence_bundle": None,
                    "retrieval_fatal_error": detail,
                    "errors": errors,
                }
            return {
                **_trace_update(state, "retrieve_evidence", dependencies),
                "evidence_bundle": bundle,
                "evidence": {source.id: source for source in bundle.filing_evidence},
                "corpus_version": state.get("corpus_version"),
                "errors": errors,
            }
        if state.get("scope_error") is not None:
            _append_unique(errors, str(state["scope_error"]))
            return {
                **_trace_update(state, "retrieve_evidence", dependencies),
                "evidence": evidence,
                "corpus_version": None,
                "errors": errors,
            }
        corpus_version: str | None = None
        retrieval_failed = False
        if dependencies.budget is not None:
            try:
                dependencies.budget.consume(retrieval_rounds=1)
            except BudgetExhaustedError as error:
                detail = f"BUDGET_EXHAUSTED: {error.dimension}"
                _append_unique(errors, detail)
                return {
                    **_trace_update(state, "retrieve_evidence", dependencies),
                    "evidence": {},
                    "corpus_version": None,
                    "p0_budget_exhausted": detail,
                    "errors": errors,
                }
        for question in state.get("questions", []):
            for query in (question.support_query, question.challenge_query):
                try:
                    if dependencies.budget is not None:
                        dependencies.budget.consume(tool_calls=1)
                    arguments = {
                        "ticker": state["ticker"],
                        "query": query,
                        "filing_ids": list(state.get("filing_ids", ())),
                        "k": 8,
                    }
                    if state.get("corpus_version") is not None:
                        arguments["corpus_version"] = state["corpus_version"]
                    raw_response = _call_mcp_tool(
                        dependencies,
                        "hybrid_search_filings",
                        arguments,
                    )
                except BudgetExhaustedError as error:
                    detail = f"BUDGET_EXHAUSTED: {error.dimension}"
                    _append_unique(errors, detail)
                    return {
                        **_trace_update(state, "retrieve_evidence", dependencies),
                        "evidence": {},
                        "corpus_version": None,
                        "p0_budget_exhausted": detail,
                        "errors": errors,
                    }
                except Exception as error:
                    _append_unique(errors, f"MCP_CALL_ERROR: {type(error).__name__}")
                    retrieval_failed = True
                    continue
                try:
                    response = _parse_search_response(raw_response)
                except _MCPProtocolError:
                    _append_unique(
                        errors,
                        "MCP_PROTOCOL_ERROR: hybrid_search_filings returned is_error",
                    )
                    retrieval_failed = True
                    continue
                except _MCPResponseError:
                    _append_unique(errors, "MCP_RESPONSE_ERROR: Invalid MCP response envelope")
                    retrieval_failed = True
                    continue

                if response.error is not None:
                    _append_unique(errors, f"{response.error.code}: {response.error.message}")
                    retrieval_failed = True
                    continue
                if retrieval_failed:
                    continue
                for chunk in response.chunks:
                    if chunk.ticker.upper() != state["ticker"]:
                        continue
                    if corpus_version is None:
                        corpus_version = chunk.corpus_version
                    if chunk.corpus_version != corpus_version:
                        continue
                    evidence.setdefault(chunk.id, chunk)

        if retrieval_failed:
            evidence.clear()
            corpus_version = None

        return {
            **_trace_update(state, "retrieve_evidence", dependencies),
            "evidence": evidence,
            "corpus_version": corpus_version,
            "errors": errors,
        }

    return retrieve_evidence


def make_evidence_coverage(dependencies: Dependencies, node_name: str):
    def evidence_coverage(state: ResearchState) -> dict[str, object]:
        return _trace_update(state, node_name, dependencies)

    return evidence_coverage


def make_final_coverage(dependencies: Dependencies):
    def final_coverage(state: ResearchState) -> dict[str, object]:
        bundle = state.get("evidence_bundle")
        update = _trace_update(state, "final_coverage", dependencies)
        if bundle is None or bundle.coverage.complete:
            return update
        reason_codes = bundle.coverage.reason_codes
        if "insufficient_information" not in reason_codes:
            reason_codes = (*reason_codes, "insufficient_information")
        return {
            **update,
            "evidence_bundle": bundle.model_copy(
                update={
                    "coverage": bundle.coverage.model_copy(
                        update={"reason_codes": reason_codes}
                    )
                }
            ),
        }

    return final_coverage


def make_retry_missing_evidence(dependencies: Dependencies):
    def retry_missing_evidence(state: ResearchState) -> dict[str, object]:
        errors = list(state.get("errors", []))
        bundle = state.get("evidence_bundle")
        if (
            bundle is None
            or state.get("p0_budget_exhausted")
            or dependencies.thesis_collector is None
            or dependencies.thesis_collection_policy is None
        ):
            return _trace_update(state, "retry_missing_evidence", dependencies)
        try:
            bundle = asyncio.run(
                dependencies.thesis_collector.retry_missing(
                    ticker=state["ticker"],
                    recipe=dependencies.thesis_collection_policy,
                    questions=state.get("questions", []),
                    evidence=bundle,
                )
            )
        except BudgetExhaustedError as error:
            detail = f"BUDGET_EXHAUSTED: {error.dimension}"
            _append_unique(errors, detail)
            return {
                **_trace_update(state, "retry_missing_evidence", dependencies),
                "evidence": {},
                "evidence_bundle": bundle,
                "p0_budget_exhausted": detail,
                "errors": errors,
            }
        except EvidenceCollectionError as error:
            detail = _safe_collection_error(error)
            _append_unique(errors, detail)
            return {
                **_trace_update(state, "retry_missing_evidence", dependencies),
                "evidence": {},
                "evidence_bundle": None,
                "retrieval_fatal_error": detail,
                "errors": errors,
            }
        return {
            **_trace_update(state, "retry_missing_evidence", dependencies),
            "evidence_bundle": bundle,
            "evidence": {source.id: source for source in bundle.filing_evidence},
            "errors": errors,
        }

    return retry_missing_evidence


def make_web_fallback(dependencies: Dependencies):
    def web_fallback(state: ResearchState) -> dict[str, object]:
        errors = list(state.get("errors", []))
        bundle = state.get("evidence_bundle")
        if (
            bundle is None
            or state.get("p0_budget_exhausted")
            or dependencies.thesis_collector is None
            or dependencies.thesis_collection_policy is None
        ):
            return _trace_update(state, "web_fallback", dependencies)
        try:
            bundle = asyncio.run(
                dependencies.thesis_collector.web_fallback(
                    ticker=state["ticker"],
                    recipe=dependencies.thesis_collection_policy,
                    questions=state.get("questions", []),
                    evidence=bundle,
                )
            )
        except BudgetExhaustedError as error:
            detail = f"BUDGET_EXHAUSTED: {error.dimension}"
            _append_unique(errors, detail)
            return {
                **_trace_update(state, "web_fallback", dependencies),
                "evidence": {},
                "evidence_bundle": bundle,
                "p0_budget_exhausted": detail,
                "errors": errors,
            }
        except EvidenceCollectionError as error:
            detail = _safe_collection_error(error)
            _append_unique(errors, detail)
            return {
                **_trace_update(state, "web_fallback", dependencies),
                "evidence": {},
                "evidence_bundle": None,
                "retrieval_fatal_error": detail,
                "errors": errors,
            }
        return {
            **_trace_update(state, "web_fallback", dependencies),
            "evidence_bundle": bundle,
            "evidence": {source.id: source for source in bundle.filing_evidence},
            "errors": errors,
        }

    return web_fallback


def make_analyze(dependencies: Dependencies):
    def analyze(state: ResearchState) -> dict[str, object]:
        evidence = state.get("evidence", {})
        analysis_questions = _analysis_questions(state)
        errors = list(state.get("errors", []))
        memo = None
        fatal_error: str | None = None
        bundle = state.get("evidence_bundle")
        budget = dependencies.budget
        if state.get("p0_budget_exhausted"):
            return {
                **_trace_update(state, "analyze", dependencies),
                "memo": None,
                "errors": errors,
            }
        if (
            not state.get("retrieval_fatal_error")
            and (evidence or (bundle is not None and bundle.web_evidence))
            and analysis_questions
        ):
            try:
                if budget is not None:
                    budget.consume(analysis_calls=1)
                raw_memo = dependencies.analyst_model.analyze(
                    analysis_questions,
                    bundle if bundle is not None else list(evidence.values()),  # type: ignore[arg-type]
                )
            except BudgetExhaustedError as error:
                detail = f"BUDGET_EXHAUSTED: {error.dimension}"
                _append_unique(errors, detail)
                return {
                    **_trace_update(state, "analyze", dependencies),
                    "memo": None,
                    "errors": errors,
                    "p0_budget_exhausted": detail,
                }
            except Exception as error:
                detail = f"ANALYST_CALL_ERROR: {type(error).__name__}"
                _append_unique(errors, detail)
                if dependencies.thesis_collector is not None:
                    fatal_error = detail
            else:
                try:
                    memo = ResearchMemo.model_validate(raw_memo)
                except (TypeError, ValidationError):
                    detail = "ANALYST_OUTPUT_INVALID: Invalid research memo"
                    _append_unique(errors, detail)
                    if dependencies.thesis_collector is not None:
                        fatal_error = detail
        update: dict[str, object] = {
            **_trace_update(state, "analyze", dependencies),
            "memo": memo,
            "errors": errors,
        }
        if fatal_error is not None:
            update["p0_fatal_error"] = fatal_error
        return update

    return analyze


def _analysis_questions(state: ResearchState) -> list[ResearchQuestion]:
    """Build a wholly hint-free P0 analyst view after memory-influenced planning."""
    questions = list(state.get("questions", []))
    if not questions:
        return []
    hints = state.get("memory_hints", ())
    if not hints:
        return questions
    return [
        ResearchQuestion(
            question=state["thesis"],
            support_query="current-corpus supporting evidence",
            challenge_query="current-corpus challenging evidence",
            period=None,
            forms=[],
        )
    ]


def make_citation_guard(dependencies: Dependencies):
    def citation_guard(state: ResearchState) -> dict[str, object]:
        memo = state.get("memo")
        corpus_version = state.get("corpus_version")
        guarded_memo = None
        errors = list(state.get("errors", []))
        if memo is not None and corpus_version is not None:
            guarded_memo = guard_memo(
                memo,
                state.get("evidence_bundle", state.get("evidence", {})),
                ticker=state["ticker"],
                corpus_version=corpus_version,
                web_validator=dependencies.web_evidence_validator,
            )
            for error in guarded_memo.errors:
                _append_unique(errors, error)
        return {
            **_trace_update(state, "citation_guard", dependencies),
            "guarded_memo": guarded_memo,
            "errors": errors,
        }

    return citation_guard


def make_repair_draft(dependencies: Dependencies):
    def repair_draft(state: ResearchState) -> dict[str, object]:
        original = state.get("guarded_memo")
        draft = state.get("memo")
        errors = list(state.get("errors", []))
        update: dict[str, object] = _trace_update(state, "repair_draft", dependencies)
        if original is None or draft is None or not original.errors:
            return update
        repairer = dependencies.thesis_repair_model
        if repairer is None:
            _append_unique(errors, "REPAIR_UNAVAILABLE")
            return {**update, "errors": errors}
        try:
            if dependencies.budget is not None:
                dependencies.budget.consume(repair_calls=1)
            raw_repair = repairer.repair(
                draft=draft,
                guard_errors=tuple(original.errors),
                evidence=state.get(
                    "evidence_bundle",
                    list(state.get("evidence", {}).values()),
                ),
            )
            repaired = ResearchMemo.model_validate(raw_repair)
            if not _is_restrictive_thesis_repair(draft, repaired):
                raise ValueError("repair expanded the guarded draft")
            reguarded = guard_memo(
                repaired,
                state.get("evidence_bundle", state.get("evidence", {})),
                ticker=state["ticker"],
                corpus_version=original.corpus_version,
                web_validator=dependencies.web_evidence_validator,
            )
            if not _preserves_guarded_thesis(original, reguarded):
                raise ValueError("repair reduced the guarded memo")
        except BudgetExhaustedError:
            _append_unique(errors, "REPAIR_SKIPPED: budget_exhausted")
        except (TypeError, ValidationError, ValueError):
            _append_unique(errors, "REPAIR_OUTPUT_REJECTED")
        except Exception as error:
            _append_unique(errors, f"REPAIR_FAILED: {type(error).__name__}")
        else:
            for error in reguarded.errors:
                _append_unique(errors, error)
            update["guarded_memo"] = reguarded
        update["errors"] = errors
        return update

    return repair_draft


def _is_restrictive_thesis_repair(
    draft: ResearchMemo,
    repaired: ResearchMemo,
) -> bool:
    if draft.research_question != repaired.research_question:
        return False
    if {"A": 2, "B": 1, "C": 0}[repaired.information_sufficiency] > {
        "A": 2,
        "B": 1,
        "C": 0,
    }[draft.information_sufficiency]:
        return False
    if _confidence_rank(repaired.confidence) > _confidence_rank(draft.confidence):
        return False
    originals = _memo_claims(draft)
    for claim in _memo_claims(repaired):
        match = next(
            (
                index
                for index, original in enumerate(originals)
                if _is_restrictive_claim(original, claim)
            ),
            None,
        )
        if match is None:
            return False
        originals.pop(match)
    return True


def _preserves_guarded_thesis(
    original: GuardedMemo,
    candidate: GuardedMemo,
) -> bool:
    return (
        candidate.ticker == original.ticker
        and candidate.corpus_version == original.corpus_version
        and candidate.research_question == original.research_question
        and candidate.source_policy_version == original.source_policy_version
        and _is_ordered_subsequence(
            original.supporting_claims, candidate.supporting_claims
        )
        and _is_ordered_subsequence(original.counter_claims, candidate.counter_claims)
        and _is_ordered_subsequence(original.inferences, candidate.inferences)
        and _is_ordered_subsequence(original.open_questions, candidate.open_questions)
        and _is_ordered_subsequence(original.sources, candidate.sources)
        and _is_ordered_subsequence(original.web_sources, candidate.web_sources)
        and _thesis_sufficiency_rank(candidate.information_sufficiency)
        >= _thesis_sufficiency_rank(original.information_sufficiency)
        and _confidence_rank(candidate.confidence) >= _confidence_rank(original.confidence)
    )


def _is_ordered_subsequence(
    original: Sequence[_OrderedValue],
    candidate: Sequence[_OrderedValue],
) -> bool:
    remaining = iter(candidate)
    return all(any(item == current for current in remaining) for item in original)


def _thesis_sufficiency_rank(value: str) -> int:
    return {"C": 0, "B": 1, "A": 2}[value]


def _memo_claims(memo: ResearchMemo) -> list[Claim]:
    return [
        *memo.supporting_claims,
        *memo.counter_claims,
        *memo.inferences,
        *memo.open_questions,
    ]


def _is_restrictive_claim(original: Claim, repaired: Claim) -> bool:
    kind_rank = {
        ClaimKind.VERIFIED_FACT: 0,
        ClaimKind.INFERENCE: 1,
        ClaimKind.OPEN_QUESTION: 2,
    }
    return (
        repaired.text == original.text
        and kind_rank[repaired.kind] >= kind_rank[original.kind]
        and _confidence_rank(repaired.confidence) <= _confidence_rank(original.confidence)
        and set(repaired.evidence_chunk_ids) <= set(original.evidence_chunk_ids)
        and set(repaired.web_evidence_ids) <= set(original.web_evidence_ids)
    )


def _confidence_rank(confidence: Confidence) -> int:
    return {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}[confidence]


def make_render(dependencies: Dependencies):
    def render(state: ResearchState) -> dict[str, object]:
        memo = state.get("guarded_memo")
        if state.get("p0_budget_exhausted"):
            rendered = "Insufficient evidence: the frozen research budget was exhausted."
        elif state.get("retrieval_fatal_error"):
            rendered = (
                "Unable to retrieve evidence safely. No research memo was produced. "
                f"Error: {state['retrieval_fatal_error']}"
            )
        elif state.get("p0_fatal_error"):
            rendered = "Unable to complete thesis research safely. No memo was produced."
        else:
            rendered = (
                render_markdown(memo)
                if memo is not None
                else "Insufficient evidence to produce a research memo."
            )
        return {
            **_trace_update(state, "render", dependencies),
            "rendered_output": rendered,
        }

    return render


class _FatalP1Error(Exception):
    """Internal marker for failures that invalidate the whole P1 result."""


def _execute_skill_recipe(
    state: ResearchState,
    dependencies: Dependencies,
    plan: SkillRunPlan,
) -> SkillRunResult:
    planner = dependencies.skill_planner
    collector = dependencies.skill_collector
    analyst = dependencies.skill_analyst
    authority = dependencies.budget
    if planner is None or collector is None or analyst is None or authority is None:
        raise _FatalP1Error

    budget = authority.child(BudgetLimits.from_policy(plan.recipe))
    binder = getattr(collector, "with_budget_gate", None)
    run_collector = binder(budget) if callable(binder) else collector
    request = SkillPlanningInput(
        ticker=state["ticker"],
        user_request=state["thesis"],
        recipe=plan.recipe,
        memory_hints=state.get("memory_hints", ()),
    )
    analysis_request = SkillAnalysisInput(
        ticker=request.ticker,
        user_request=request.user_request,
        recipe=request.recipe,
    )
    try:
        budget.consume(planner_calls=1)
        raw_questions = _run_async(planner.plan(request))
    except BudgetExhaustedError as error:
        _record_p1_budget(dependencies, plan, budget, exhausted=error.dimension)
        return _research_budget_exhaustion(dependencies, plan, error.dimension)
    except SkillModelError as error:
        raise _FatalP1Error from error
    except Exception as error:
        _record_p1_budget(dependencies, plan, budget)
        return _research_failure(
            dependencies,
            plan,
            f"RECIPE_EXECUTION_ERROR: {type(error).__name__}",
        )
    try:
        questions = _RESEARCH_QUESTION_LIST.validate_python(raw_questions)
    except (TypeError, ValidationError) as error:
        raise _FatalP1Error from error
    if len(questions) > plan.recipe.budget.max_questions:
        raise _FatalP1Error
    if not questions:
        _record_p1_budget(dependencies, plan, budget)
        return _research_failure(dependencies, plan, "PLANNER_EMPTY")

    try:
        raw_evidence = _run_async(
            run_collector.collect(
                ticker=state["ticker"],
                recipe=plan.recipe,
                questions=questions,
            )
        )
    except BudgetExhaustedError as error:
        _record_p1_budget(dependencies, plan, budget, exhausted=error.dimension)
        return _research_budget_exhaustion(
            dependencies,
            plan,
            error.dimension,
            questions=questions,
        )
    except EvidenceCollectionError as error:
        _record_p1_budget(dependencies, plan, budget)
        if error.code is CollectionErrorCode.RETRIEVAL_ERROR:
            return _research_failure(
                dependencies,
                plan,
                "RECIPE_RETRIEVAL_ERROR",
                questions=questions,
            )
        raise _FatalP1Error from error
    except Exception as error:
        _record_p1_budget(dependencies, plan, budget)
        raise _FatalP1Error from error
    if not isinstance(raw_evidence, EvidenceBundle):
        raise _FatalP1Error
    evidence = raw_evidence
    if not evidence.coverage.complete:
        coverage_errors = [
            "INSUFFICIENT_EVIDENCE",
            *(f"COVERAGE: {reason}" for reason in _safe_coverage_reasons(evidence)),
        ]
        _record_p1_budget(dependencies, plan, budget)
        return _research_failure(
            dependencies,
            plan,
            *coverage_errors,
            questions=questions,
            evidence=evidence,
        )

    try:
        budget.consume(analysis_calls=1)
        raw_memo = _run_async(analyst.analyze(request=analysis_request, evidence=evidence))
    except BudgetExhaustedError as error:
        _record_p1_budget(dependencies, plan, budget, exhausted=error.dimension)
        return _research_budget_exhaustion(
            dependencies,
            plan,
            error.dimension,
            questions=questions,
            evidence=evidence,
        )
    except SkillModelError as error:
        raise _FatalP1Error from error
    except Exception as error:
        _record_p1_budget(dependencies, plan, budget)
        return _research_failure(
            dependencies,
            plan,
            f"RECIPE_EXECUTION_ERROR: {type(error).__name__}",
            questions=questions,
            evidence=evidence,
        )
    try:
        memo = SkillResearchMemo.model_validate(raw_memo)
    except (TypeError, ValidationError) as error:
        raise _FatalP1Error from error
    if memo.recipe_name is not plan.recipe.name or memo.recipe_version != plan.recipe.version:
        raise _FatalP1Error

    try:
        with _p1_observation(
            dependencies,
            name="p1.output_guard",
            kind="guardrail",
            metadata={
                "recipe_name": plan.recipe.name.value,
                "recipe_version": plan.recipe.version,
            },
        ) as observation:
            guarded = guard_skill_memo(
                memo=memo,
                evidence=evidence,
                ticker=state["ticker"],
                recipe=plan.recipe,
                web_validator=dependencies.web_evidence_validator,
                report_as_of=state.get("report_as_of"),
            )
            if observation is not None:
                observation.update(
                    output={
                        "status": "repair_required" if guarded.guard_errors else "completed",
                        "retained_source_count": len(guarded.filing_sources)
                        + len(guarded.web_sources),
                        "guard_error_codes": [
                            _guard_error_code(error) for error in guarded.guard_errors
                        ],
                    }
                )
    except (TypeError, ValidationError, ValueError) as error:
        raise _FatalP1Error from error

    guarded, repair_errors = _repair_skill_draft(
        dependencies=dependencies,
        budget=budget,
        plan=plan,
        request=analysis_request,
        draft=memo,
        evidence=evidence,
        original=guarded,
        report_as_of=state.get("report_as_of"),
    )
    errors = list(dict.fromkeys([*guarded.guard_errors, *repair_errors]))
    fatal_guard_error = any(
        _guard_error_code(error) not in _NONFATAL_SKILL_GUARD_CODES
        for error in guarded.guard_errors
    )
    useful = bool(guarded.filing_sources or guarded.web_sources) and (
        any(section.claims for section in guarded.memo.sections)
        or bool(guarded.memo.data_points)
    )
    if fatal_guard_error:
        raise _FatalP1Error

    try:
        with _p1_observation(
            dependencies,
            name="p1.render_skill",
            kind="chain",
            metadata={
                "recipe_name": plan.recipe.name.value,
                "recipe_version": plan.recipe.version,
            },
        ) as observation:
            rendered_output = render_skill_markdown(guarded) if useful else ""
            if observation is not None:
                observation.update(
                    output={
                        "status": "completed",
                        "rendered": bool(rendered_output),
                        "rendered_length": len(rendered_output),
                    }
                )
    except (TypeError, ValidationError, ValueError) as error:
        raise _FatalP1Error from error

    if not useful:
        _append_unique(errors, "CITATION_GUARD_REJECTED")
        result = _skill_result(
            plan,
            status="failed",
            questions=questions,
            evidence=evidence,
            memo=memo,
            guarded_memo=guarded,
            errors=errors,
        )
        _finish_skill_run(dependencies, result, "failed")
        _record_p1_budget(dependencies, plan, budget)
        return result

    status: Literal["completed", "partial"] = (
        "completed"
        if guarded.memo.information_sufficiency is InformationSufficiency.SUFFICIENT and not errors
        else "partial"
    )
    result = _skill_result(
        plan,
        status=status,
        questions=questions,
        evidence=evidence,
        memo=memo,
        guarded_memo=guarded,
        errors=errors,
        rendered_output=rendered_output,
    )
    _finish_skill_run(dependencies, result, status)
    _record_p1_budget(dependencies, plan, budget)
    return result


def _record_p1_budget(
    dependencies: Dependencies,
    plan: SkillRunPlan,
    budget: HierarchicalBudgetGate,
    *,
    exhausted: str | None = None,
) -> None:
    with _p1_observation(
        dependencies,
        name="p1.budget",
        kind="chain",
        metadata={
            "recipe_name": plan.recipe.name.value,
            "recipe_version": plan.recipe.version,
        },
    ) as observation:
        if observation is not None:
            output = {
                "planner_calls": budget.local.planner_calls,
                "analysis_calls": budget.local.analysis_calls,
                "repair_calls": budget.local.repair_calls,
                "tool_calls": budget.local.tool_calls,
                "retrieval_rounds": budget.local.retrieval_rounds,
                "web_calls": budget.local.web_calls,
                "run_planner_calls": budget.shared.state.planner_calls,
                "run_analysis_calls": budget.shared.state.analysis_calls,
                "run_repair_calls": budget.shared.state.repair_calls,
                "run_tool_calls": budget.shared.state.tool_calls,
                "run_retrieval_rounds": budget.shared.state.retrieval_rounds,
                "run_web_calls": budget.shared.state.web_calls,
            }
            if exhausted is not None:
                output["exhausted"] = exhausted
            observation.update(output=output)


def _repair_skill_draft(
    *,
    dependencies: Dependencies,
    budget: HierarchicalBudgetGate,
    plan: SkillRunPlan,
    request: SkillAnalysisInput,
    draft: SkillResearchMemo,
    evidence: EvidenceBundle,
    original: GuardedSkillMemo,
    report_as_of: date | None,
) -> tuple[GuardedSkillMemo, list[str]]:
    if not original.guard_errors:
        return original, []
    repairer = dependencies.skill_repair_model
    if repairer is None:
        return original, ["REPAIR_UNAVAILABLE"]
    with _p1_observation(
        dependencies,
        name="p1.repair_draft",
        kind="chain",
        metadata={
            "recipe_name": plan.recipe.name.value,
            "recipe_version": plan.recipe.version,
            "max_output_tokens": plan.recipe.budget.max_repair_output_tokens,
        },
    ) as observation:
        try:
            budget.consume(repair_calls=1)
            raw_repair = _run_async(
                repairer.repair(
                    request=request,
                    draft=draft,
                    guard_errors=tuple(original.guard_errors),
                    evidence=evidence,
                )
            )
            repaired = SkillResearchMemo.model_validate(raw_repair)
            if not _is_restrictive_skill_repair(draft, repaired):
                raise ValueError("repair expanded the guarded draft")
            reguarded = guard_skill_memo(
                memo=repaired,
                evidence=evidence,
                ticker=request.ticker,
                recipe=plan.recipe,
                web_validator=dependencies.web_evidence_validator,
                report_as_of=report_as_of,
            )
            if not _preserves_guarded_skill_memo(original, reguarded):
                raise ValueError("repair reduced the guarded skill memo")
        except BudgetExhaustedError:
            if observation is not None:
                observation.update(output={"status": "budget_exhausted"})
            return original, ["REPAIR_SKIPPED: budget_exhausted"]
        except (TypeError, ValidationError, ValueError):
            if observation is not None:
                observation.update(output={"status": "rejected"})
            return original, ["REPAIR_OUTPUT_REJECTED"]
        except Exception as error:
            if observation is not None:
                observation.update(output={"status": "failed"})
            return original, [f"REPAIR_FAILED: {type(error).__name__}"]
        if observation is not None:
            observation.update(
                output={
                    "status": "completed",
                    "guard_error_codes": [
                        _guard_error_code(error) for error in reguarded.guard_errors
                    ],
                }
            )
        return reguarded, ["REPAIR_APPLIED"]


def _is_restrictive_skill_repair(
    draft: SkillResearchMemo,
    repaired: SkillResearchMemo,
) -> bool:
    if (
        repaired.recipe_name is not draft.recipe_name
        or repaired.recipe_version != draft.recipe_version
        or repaired.research_question != draft.research_question
        or _p1_sufficiency_rank(repaired.information_sufficiency)
        > _p1_sufficiency_rank(draft.information_sufficiency)
        or repaired.confidence > draft.confidence
    ):
        return False
    original_sections = [
        (section.facet, list(section.claims)) for section in draft.sections
    ]
    for section in repaired.sections:
        match = next(
            (
                index
                for index, (facet, _) in enumerate(original_sections)
                if facet is section.facet
            ),
            None,
        )
        if match is None:
            return False
        _, original_claims = original_sections.pop(match)
        for claim in section.claims:
            claim_match = next(
                (
                    index
                    for index, original in enumerate(original_claims)
                    if _is_restrictive_claim(original, claim)
                ),
                None,
            )
            if claim_match is None:
                return False
            original_claims.pop(claim_match)
    original_points = [point.model_dump(mode="python") for point in draft.data_points]
    for point in repaired.data_points:
        serialized = point.model_dump(mode="python")
        if serialized not in original_points:
            return False
        original_points.remove(serialized)
    original_gaps = list(draft.information_gaps)
    for gap in repaired.information_gaps:
        if gap not in original_gaps:
            return False
        original_gaps.remove(gap)
    return True


def _preserves_guarded_skill_memo(
    original: GuardedSkillMemo,
    candidate: GuardedSkillMemo,
) -> bool:
    original_memo = original.memo
    candidate_memo = candidate.memo
    return (
        candidate.ticker == original.ticker
        and candidate_memo.recipe_name is original_memo.recipe_name
        and candidate_memo.recipe_version == original_memo.recipe_version
        and candidate_memo.research_question == original_memo.research_question
        and _preserves_guarded_sections(
            original_memo.sections, candidate_memo.sections
        )
        and _is_ordered_subsequence(
            original_memo.data_points, candidate_memo.data_points
        )
        and _is_ordered_subsequence(
            original_memo.information_gaps, candidate_memo.information_gaps
        )
        and _is_ordered_subsequence(original.filing_sources, candidate.filing_sources)
        and _is_ordered_subsequence(original.web_sources, candidate.web_sources)
        and _p1_sufficiency_rank(candidate_memo.information_sufficiency)
        >= _p1_sufficiency_rank(original_memo.information_sufficiency)
        and candidate_memo.confidence >= original_memo.confidence
        and candidate.provenance.recipes == original.provenance.recipes
        and candidate.provenance.prompt_versions
        == original.provenance.prompt_versions
        and candidate.provenance.requested_as_of_dates
        == original.provenance.requested_as_of_dates
        and _is_ordered_subsequence(
            original.provenance.source_policy_versions,
            candidate.provenance.source_policy_versions,
        )
        and _is_ordered_subsequence(
            original.provenance.corpus_versions,
            candidate.provenance.corpus_versions,
        )
        and _is_ordered_subsequence(
            original.provenance.evidence_cutoff_dates,
            candidate.provenance.evidence_cutoff_dates,
        )
        and _is_ordered_subsequence(
            original.provenance.source_refs,
            candidate.provenance.source_refs,
        )
    )


def _preserves_guarded_sections(
    original: Sequence[SkillResearchSection],
    candidate: Sequence[SkillResearchSection],
) -> bool:
    remaining = iter(candidate)
    for original_section in original:
        candidate_section = next(
            (
                section
                for section in remaining
                if section.facet is original_section.facet
            ),
            None,
        )
        if candidate_section is None or not _is_ordered_subsequence(
            original_section.claims,
            candidate_section.claims,
        ):
            return False
    return True


def _p1_sufficiency_rank(value: InformationSufficiency) -> int:
    return {
        InformationSufficiency.INSUFFICIENT: 0,
        InformationSufficiency.PARTIAL: 1,
        InformationSufficiency.SUFFICIENT: 2,
    }[value]


def _research_failure(
    dependencies: Dependencies,
    plan: SkillRunPlan,
    *errors: str,
    questions: list[ResearchQuestion] | None = None,
    evidence: EvidenceBundle | None = None,
) -> SkillRunResult:
    result = _skill_result(
        plan,
        status="failed",
        questions=questions or [],
        evidence=evidence,
        errors=list(errors),
    )
    _finish_skill_run(dependencies, result, "failed")
    return result


def _research_budget_exhaustion(
    dependencies: Dependencies,
    plan: SkillRunPlan,
    dimension: str,
    *,
    questions: list[ResearchQuestion] | None = None,
    evidence: EvidenceBundle | None = None,
) -> SkillRunResult:
    result = _skill_result(
        plan,
        status="partial",
        questions=questions or [],
        evidence=evidence,
        errors=[f"BUDGET_EXHAUSTED: {dimension}"],
    )
    _finish_skill_run(dependencies, result, "partial")
    return result


def _skill_result(
    plan: SkillRunPlan,
    *,
    status: Literal["completed", "partial", "failed", "refused"],
    questions: list[ResearchQuestion] | None = None,
    evidence: EvidenceBundle | None = None,
    memo: SkillResearchMemo | None = None,
    guarded_memo: GuardedSkillMemo | None = None,
    errors: list[str] | None = None,
    rendered_output: str = "",
) -> SkillRunResult:
    return SkillRunResult(
        run_id=plan.run_id,
        recipe_name=plan.recipe.name,
        recipe_version=plan.recipe.version,
        allowed_tools=tuple(sorted(plan.recipe.allowed_tools)),
        status=status,
        questions=questions or [],
        evidence=evidence,
        memo=memo,
        guarded_memo=guarded_memo,
        errors=errors or [],
        rendered_output=rendered_output,
    )


def _refused_skill_result(plan: SkillRunPlan, error: str) -> SkillRunResult:
    return _skill_result(plan, status="refused", errors=[error])


def _finish_skill_run(
    dependencies: Dependencies,
    result: SkillRunResult,
    status: Literal["completed", "partial", "refused", "failed"],
) -> None:
    repository = dependencies.skill_run_repository
    if repository is None:
        raise _FatalP1Error
    try:
        repository.finish(
            result.run_id,
            status=status,
            source_ids=_skill_source_ids(result),
            errors=result.errors,
        )
    except Exception as error:
        raise _FatalP1Error from error


def _finish_without_raising(
    dependencies: Dependencies,
    run_id: str,
    *,
    status: Literal["completed", "refused", "failed"],
    source_ids: list[SourceRef],
    errors: list[str],
) -> None:
    repository = dependencies.skill_run_repository
    if repository is None:
        return
    try:
        repository.finish(
            run_id,
            status=status,
            source_ids=source_ids,
            errors=errors,
        )
    except Exception:
        pass


def _skill_source_ids(result: SkillRunResult) -> list[SourceRef]:
    if result.guarded_memo is not None:
        return [
            *(
                SourceRef(
                    ticker=result.guarded_memo.ticker,
                    kind=SourceRefKind.FILING,
                    source_id=source.id,
                )
                for source in result.guarded_memo.filing_sources
            ),
            *(
                SourceRef(
                    ticker=result.guarded_memo.ticker,
                    kind=SourceRefKind.WEB,
                    source_id=source.id,
                )
                for source in result.guarded_memo.web_sources
            ),
        ]
    return []


def _trace_skill_result(dependencies: Dependencies, result: SkillRunResult) -> None:
    evidence = result.evidence
    _record_trace(
        dependencies,
        "p1_recipe",
        {
            "recipe_name": result.recipe_name.value,
            "recipe_version": result.recipe_version,
            "retrieval_rounds": evidence.retrieval_rounds if evidence is not None else 0,
            "web_calls": evidence.web_calls if evidence is not None else 0,
            "coverage_reasons": _safe_coverage_reasons(evidence),
            "source_ids": [reference.encode() for reference in _skill_source_ids(result)],
            "status": result.status,
        },
    )


def _render_partial_skill_section(result: SkillRunResult) -> str:
    return (
        f"## Partial — {result.recipe_name.value}\n\n"
        "- Status: insufficient evidence for this recipe.\n"
        "- No unguarded research content was retained.\n"
    )


def _guard_error_code(error: str) -> str:
    return error.partition(":")[0].strip()


def _safe_coverage_reasons(evidence: EvidenceBundle | None) -> list[str]:
    if evidence is None:
        return []
    return [
        reason
        for raw_reason in evidence.coverage.reason_codes
        if (reason := str(raw_reason)) in _TRACE_COVERAGE_REASONS
    ]


def _run_async(awaitable: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("P1 synchronous workflow cannot run inside an active event loop")
    return asyncio.run(awaitable)


def is_research_decision(state: ResearchState) -> bool:
    decision = state.get("decision")
    return decision is not None and decision.intent is Intent.RESEARCH_REQUEST


def has_safety_decision(state: ResearchState) -> bool:
    return "decision" in state


def _is_clear_research_thesis(thesis: str) -> bool:
    normalized = thesis.casefold()
    english_matches = set(_ENGLISH_RESEARCH_TERMS.findall(normalized))
    chinese_matches = {term for term in _CHINESE_RESEARCH_TERMS if term in normalized}
    return len(english_matches) >= 2 or len(chinese_matches) >= 2


def _parse_search_response(raw_response: object) -> HybridSearchFilingsResponse:
    if getattr(raw_response, "is_error", False):
        raise _MCPProtocolError
    if isinstance(raw_response, HybridSearchFilingsResponse):
        return raw_response
    structured_content = getattr(raw_response, "structured_content", None)
    if structured_content is not None:
        raw_response = structured_content
    elif isinstance(raw_response, BaseModel):
        raw_response = raw_response.model_dump()
    if not isinstance(raw_response, Mapping):
        raise _MCPResponseError
    try:
        return HybridSearchFilingsResponse.model_validate(raw_response)
    except ValidationError as error:
        raise _MCPResponseError from error


class _MCPProtocolError(Exception):
    pass


class _MCPResponseError(Exception):
    pass


def _safe_collection_error(error: EvidenceCollectionError) -> str:
    """Expose only the typed collection category across graph/output boundaries."""
    return f"THESIS_COLLECTION_{error.code.value.upper()}"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _trace_update(
    state: ResearchState,
    node: str,
    dependencies: Dependencies,
) -> dict[str, object]:
    attributes: dict[str, object] = {"ticker": state["ticker"]}
    _record_trace(dependencies, node, attributes)
    return {"node_trace": [*state.get("node_trace", []), node]}


def _p1_trace_update(
    state: ResearchState,
    node: str,
    dependencies: Dependencies,
) -> dict[str, object]:
    _record_trace(dependencies, node, {})
    return {"node_trace": [*state.get("node_trace", []), node]}


def _record_trace(
    dependencies: Dependencies,
    node: str,
    attributes: dict[str, object],
) -> None:
    try:
        if dependencies.trace_run is None:
            dependencies.trace_sink.record(node, attributes)
            return
        with dependencies.trace_run.observation(
            name=f"graph.{node}",
            kind=_observation_kind(node),
            metadata=attributes,
        ):
            pass
    except Exception:
        pass


def _p1_observation(
    dependencies: Dependencies,
    *,
    name: str,
    kind: Literal["chain", "guardrail"],
    metadata: dict[str, object],
):
    if dependencies.trace_run is None:
        return nullcontext()
    return dependencies.trace_run.observation(
        name=name,
        kind=kind,
        metadata=metadata,
    )


def _call_mcp_tool(
    dependencies: Dependencies,
    name: str,
    arguments: dict[str, object],
) -> object:
    if dependencies.trace_run is None:
        return dependencies.mcp_client.call_tool(name, arguments)
    query = str(arguments.get("query", ""))
    metadata: dict[str, object] = {
        "tool": name,
        "ticker": str(arguments.get("ticker", "")),
        "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
        "limit": int(arguments.get("k", 0)),
    }
    with dependencies.trace_run.observation(
        name=f"mcp.{name}",
        kind="tool",
        metadata=metadata,
    ) as observation:
        try:
            result = dependencies.mcp_client.call_tool(name, arguments)
        except Exception as error:
            observation.update(
                output={"status": "failed"},
                metadata={"error_code": type(error).__name__},
            )
            raise
        observation.update(output={"status": "completed"})
        return result


def _observation_kind(
    node: str,
) -> Literal["chain", "generation", "retriever", "guardrail"]:
    if node in {"safety_router", "citation_guard"}:
        return "guardrail"
    if node in {"plan_questions", "analyze"}:
        return "generation"
    if node in {"retrieve_evidence", "p1_recipe"}:
        return "retriever"
    return "chain"
