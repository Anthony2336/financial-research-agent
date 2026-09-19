"""Compilation and synchronous execution of the controlled LangGraph."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fra.context import BudgetAuthority, BudgetLimits
from fra.domain import Intent, RouterDecision
from fra.graph.models import Dependencies, ResearchResult
from fra.graph.nodes import (
    ResearchState,
    dispatch_skill_recipes,
    is_research_decision,
    make_analyze,
    make_citation_guard,
    make_combine_skill_sections,
    make_evidence_coverage,
    make_execute_skill_recipes,
    make_final_coverage,
    make_load_research_memory,
    make_load_session_memory,
    make_normalize,
    make_plan_questions,
    make_render,
    make_repair_draft,
    make_retrieve_evidence,
    make_retry_missing_evidence,
    make_safety_router,
    make_skill_dispatch,
    make_web_fallback,
)
from fra.prompts import current_prompt_version
from fra.retrieval.collector import THESIS_COLLECTION_POLICY
from fra.safety.router import (
    ARBITRARY_URL_REFUSAL_TEXT,
    PROMPT_INJECTION_TEXT,
    REFUSAL_TEXT,
)
from fra.skills.recipes import RESEARCH_INPUT_RECIPES

_SKILL_RESEARCH_INTENTS = {
    Intent.COMPANY_PROFILE_REQUEST,
    Intent.EARNINGS_REVIEW_REQUEST,
    Intent.INDUSTRY_RESEARCH_REQUEST,
}


def build_research_graph(dependencies: Dependencies) -> CompiledStateGraph:
    """Compile the fixed, non-agentic research graph."""

    builder = StateGraph(ResearchState)
    builder.add_node("safety_router", make_safety_router(dependencies))
    builder.add_node("normalize", make_normalize(dependencies))
    builder.add_node("load_session_memory", make_load_session_memory(dependencies))
    builder.add_node("load_research_memory", make_load_research_memory(dependencies))
    builder.add_node("plan_questions", make_plan_questions(dependencies))
    builder.add_node("retrieve_evidence", make_retrieve_evidence(dependencies))
    builder.add_node(
        "evidence_coverage",
        make_evidence_coverage(dependencies, "evidence_coverage"),
    )
    builder.add_node(
        "retry_missing_evidence",
        make_retry_missing_evidence(dependencies),
    )
    builder.add_node(
        "retry_coverage",
        make_evidence_coverage(dependencies, "retry_coverage"),
    )
    builder.add_node("web_fallback", make_web_fallback(dependencies))
    builder.add_node("final_coverage", make_final_coverage(dependencies))
    builder.add_node("analyze", make_analyze(dependencies))
    builder.add_node("citation_guard", make_citation_guard(dependencies))
    builder.add_node("repair_draft", make_repair_draft(dependencies))
    builder.add_node("render", make_render(dependencies))
    builder.add_node("skill_dispatch", make_skill_dispatch(dependencies))
    builder.add_node("execute_skill_recipes", make_execute_skill_recipes(dependencies))
    builder.add_node("combine_skill_sections", make_combine_skill_sections(dependencies))

    builder.add_edge(START, "safety_router")
    builder.add_conditional_edges(
        "safety_router",
        _after_safety,
        {"stop": END, "continue": "normalize"},
    )
    builder.add_conditional_edges(
        "normalize",
        _after_normalize,
        {"stop": END, "p0": "load_session_memory", "p1": "load_session_memory"},
    )
    builder.add_edge("load_session_memory", "load_research_memory")
    builder.add_conditional_edges(
        "load_research_memory",
        _after_memory,
        {"p0": "plan_questions", "p1": "skill_dispatch"},
    )
    builder.add_edge("plan_questions", "retrieve_evidence")
    builder.add_conditional_edges(
        "retrieve_evidence",
        _after_retrieve,
        {"legacy": "analyze", "production": "evidence_coverage"},
    )
    builder.add_conditional_edges(
        "evidence_coverage",
        _after_initial_coverage,
        {
            "analyze": "analyze",
            "retry": "retry_missing_evidence",
            "web": "web_fallback",
            "final": "final_coverage",
        },
    )
    builder.add_edge("retry_missing_evidence", "retry_coverage")
    builder.add_conditional_edges(
        "retry_coverage",
        _after_retry_coverage,
        {"analyze": "analyze", "web": "web_fallback"},
    )
    builder.add_edge("web_fallback", "final_coverage")
    builder.add_conditional_edges(
        "final_coverage",
        _after_final_coverage,
        {"analyze": "analyze", "render": "render"},
    )
    builder.add_edge("analyze", "citation_guard")
    builder.add_conditional_edges(
        "citation_guard",
        _after_citation_guard,
        {"repair": "repair_draft", "render": "render"},
    )
    builder.add_edge("repair_draft", "render")
    builder.add_edge("render", END)
    builder.add_edge("skill_dispatch", "execute_skill_recipes")
    builder.add_edge("execute_skill_recipes", "combine_skill_sections")
    builder.add_edge("combine_skill_sections", END)
    return builder.compile()


def run_research(
    ticker: str,
    thesis: str,
    dependencies: Dependencies,
    *,
    intent: Intent | None = None,
    run_id: str = "untracked",
    corpus_version: str | None = None,
    filing_ids: tuple[str, ...] = (),
    scope_error: str | None = None,
    session_id: str | None = None,
    report_as_of: date | None = None,
) -> ResearchResult:
    """Run research synchronously for CLI compatibility."""

    if not ticker.strip():
        raise ValueError("ticker must not be empty")
    if not thesis.strip():
        raise ValueError("thesis must not be empty")
    if intent is not None and intent not in {
        Intent.RESEARCH_REQUEST,
        Intent.COMPANY_PROFILE_REQUEST,
        Intent.EARNINGS_REVIEW_REQUEST,
        Intent.INDUSTRY_RESEARCH_REQUEST,
    }:
        raise ValueError("intent must select a supported research workflow")
    dependencies = _dependencies_with_run_budget(dependencies, intent)
    graph = build_research_graph(dependencies)
    graph_input: dict[str, object] = {
        "run_id": run_id,
        "ticker": ticker,
        "thesis": thesis,
        "node_trace": [],
        "corpus_version": corpus_version,
        "filing_ids": filing_ids,
        "scope_error": scope_error,
        "session_id": session_id,
        "report_as_of": report_as_of,
    }
    if intent is not None:
        graph_input["requested_intent"] = intent
    state = cast(
        ResearchState,
        graph.invoke(graph_input),
    )
    return _build_result(state, run_id=run_id)


def _dependencies_with_run_budget(
    dependencies: Dependencies,
    intent: Intent | None,
) -> Dependencies:
    authority = dependencies.budget or BudgetAuthority()
    if not authority.configured:
        if intent in _SKILL_RESEARCH_INTENTS:
            policies = dispatch_skill_recipes(intent)
        elif intent is None and dependencies.skill_planner is not None:
            policies = RESEARCH_INPUT_RECIPES
        else:
            policies = (
                dependencies.thesis_collection_policy or THESIS_COLLECTION_POLICY,
            )
        authority.configure(BudgetLimits.aggregate(policies))
    updates: dict[str, object] = {"budget": authority}
    if dependencies.thesis_collector is not None:
        binder = getattr(dependencies.thesis_collector, "with_budget_gate", None)
        if callable(binder):
            updates["thesis_collector"] = binder(authority)
    return replace(dependencies, **updates)


def build_early_research_result(
    ticker: str,
    request: str,
    decision: RouterDecision,
    *,
    run_id: str = "untracked",
) -> ResearchResult:
    """Build a result for a decision that must not construct or run a runtime."""
    status = (
        "refused"
        if decision.intent
        in {
            Intent.PROHIBITED_ADVICE,
            Intent.PROMPT_INJECTION,
            Intent.UNSAFE_SOURCE_REQUEST,
        }
        else "declined"
    )
    return ResearchResult(
        run_id=run_id,
        status=status,
        ticker=ticker.strip().upper(),
        thesis=" ".join(request.split()),
        decision=decision,
        rendered_output=_early_output(decision),
        node_trace=["safety_router"] if status == "refused" else [],
    )


def _after_safety(state: ResearchState) -> Literal["stop", "continue"]:
    decision = state.get("decision")
    if decision is None or decision.intent in {
        Intent.RESEARCH_REQUEST,
        Intent.COMPANY_PROFILE_REQUEST,
        Intent.EARNINGS_REVIEW_REQUEST,
        Intent.INDUSTRY_RESEARCH_REQUEST,
    }:
        return "continue"
    return "stop"


def _after_normalize(state: ResearchState) -> Literal["stop", "p0", "p1"]:
    if is_research_decision(state):
        return "p0"
    decision = state.get("decision")
    if decision is not None and decision.intent in _SKILL_RESEARCH_INTENTS:
        return "p1"
    return "stop"


def _after_memory(state: ResearchState) -> Literal["p0", "p1"]:
    return "p0" if is_research_decision(state) else "p1"


def _after_retrieve(state: ResearchState) -> Literal["legacy", "production"]:
    is_production = "evidence_bundle" in state or state.get("retrieval_fatal_error")
    return "production" if is_production else "legacy"


def _after_initial_coverage(
    state: ResearchState,
) -> Literal["analyze", "retry", "web", "final"]:
    if state.get("retrieval_fatal_error"):
        return "analyze"
    if state.get("p0_budget_exhausted"):
        return "final"
    if str(state.get("scope_error", "")).startswith("NO_FILINGS:"):
        return "web"
    bundle = state.get("evidence_bundle")
    if bundle is None or bundle.coverage.complete:
        return "analyze"
    return "retry"


def _after_retry_coverage(state: ResearchState) -> Literal["analyze", "web"]:
    if state.get("retrieval_fatal_error") or state.get("p0_budget_exhausted"):
        return "analyze"
    bundle = state.get("evidence_bundle")
    if bundle is None or bundle.coverage.complete:
        return "analyze"
    return "web"


def _after_final_coverage(state: ResearchState) -> Literal["analyze", "render"]:
    if state.get("p0_budget_exhausted"):
        return "render"
    if state.get("retrieval_fatal_error") or state.get("p0_fatal_error"):
        return "analyze"
    bundle = state.get("evidence_bundle")
    if bundle is not None and not bundle.coverage.complete:
        return "render"
    return "analyze"


def _after_citation_guard(state: ResearchState) -> Literal["repair", "render"]:
    guarded = state.get("guarded_memo")
    return "repair" if guarded is not None and guarded.errors else "render"


def _build_result(state: ResearchState, *, run_id: str) -> ResearchResult:
    decision = state.get("decision")
    if decision is None:
        raise RuntimeError("research graph ended without a router decision")
    memo = state.get("memo")
    guarded_memo = state.get("guarded_memo")
    final_bundle = state.get("evidence_bundle")
    final_coverage_incomplete = final_bundle is not None and not final_bundle.coverage.complete
    budget_exhausted = bool(state.get("p0_budget_exhausted"))
    if final_coverage_incomplete or budget_exhausted:
        memo = None
        guarded_memo = None
    evidence = state.get("evidence", {})
    skill_runs = state.get("skill_runs", [])
    status: str
    if decision.intent in {
        Intent.PROHIBITED_ADVICE,
        Intent.PROMPT_INJECTION,
        Intent.UNSAFE_SOURCE_REQUEST,
    }:
        status = "refused"
    elif decision.intent in _SKILL_RESEARCH_INTENTS:
        if state.get("p1_fatal_error"):
            status = "failed"
        elif any(run.status in {"completed", "partial"} for run in skill_runs):
            status = (
                "partial" if any(run.status != "completed" for run in skill_runs) else "completed"
            )
        else:
            status = "insufficient_evidence"
    elif decision.intent is not Intent.RESEARCH_REQUEST:
        status = "declined"
    elif budget_exhausted:
        status = "insufficient_evidence"
    elif state.get("retrieval_fatal_error") or state.get("p0_fatal_error"):
        status = "failed"
    elif final_coverage_incomplete:
        status = "insufficient_evidence"
    elif (
        guarded_memo is None
        or not evidence
        or not guarded_memo.verified_claims
        or guarded_memo.information_sufficiency == "C"
    ):
        status = "insufficient_evidence"
    else:
        status = "completed"

    if budget_exhausted:
        rendered_output = "Insufficient evidence: the frozen research budget was exhausted."
    elif final_coverage_incomplete:
        rendered_output = "Insufficient evidence to produce a research memo."
    else:
        rendered_output = state.get("rendered_output") or _early_output(decision)
    return ResearchResult(
        run_id=run_id,
        status=status,
        ticker=state["ticker"].strip().upper(),
        thesis=" ".join(state["thesis"].split()),
        decision=decision,
        questions=state.get("questions", []),
        evidence=evidence,
        corpus_version=state.get("corpus_version"),
        memo=memo,
        guarded_memo=guarded_memo,
        skill_runs=skill_runs,
        errors=state.get("errors", []),
        rendered_output=rendered_output,
        prompt_version=current_prompt_version(),
        report_as_of=state.get("report_as_of"),
        node_trace=state.get("node_trace", []),
    )


def _early_output(decision: RouterDecision) -> str:
    if decision.intent is Intent.PROHIBITED_ADVICE:
        return REFUSAL_TEXT
    if decision.intent is Intent.PROMPT_INJECTION:
        return PROMPT_INJECTION_TEXT
    if decision.intent is Intent.UNSAFE_SOURCE_REQUEST:
        return ARBITRARY_URL_REFUSAL_TEXT
    return "Please provide a specific, verifiable research thesis."
