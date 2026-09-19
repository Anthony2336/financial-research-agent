"""Lazy OpenAI adapters for P1 Pydantic structured output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fra.context import (
    BoundedContext,
    ContextBuilder,
    ContextCompressor,
    ContextEvidence,
    ContextLimits,
    LazyTiktokenTokenCounter,
    MemoryHint,
    TokenCounter,
)
from fra.domain import (
    Intent,
    IntentRoutingError,
    IntentRoutingErrorCode,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
)
from fra.graph.models import (
    SkillAnalysisInput,
    SkillModelError,
    SkillModelErrorCode,
    SkillPlanningInput,
)
from fra.observability import observe
from fra.prompts import (
    ANALYST_PROMPT,
    REPAIR_PROMPT,
    ROUTER_PROMPT,
    SKILL_PLANNER_PROMPT,
    THESIS_ANALYST_PROMPT,
    THESIS_PLANNER_PROMPT,
    PromptBundle,
    record_prompt_version,
)
from fra.retrieval.collector import EvidenceBundle
from fra.skills.models import ResearchRecipe
from fra.skills.schemas import SkillResearchMemo


class ResearchQuestionPlan(BaseModel):
    """Structured planner response before recipe-budget validation."""

    model_config = ConfigDict(extra="forbid")

    questions: list[ResearchQuestion] = Field(min_length=1, max_length=8)


class ThesisResearchQuestionPlan(BaseModel):
    """The default thesis planner's hard three-question response contract."""

    model_config = ConfigDict(extra="forbid")

    questions: list[ResearchQuestion] = Field(min_length=1, max_length=3)


class _LazyOpenAIAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        chat_model: Any | None = None,
        context_compressor: ContextCompressor | None = None,
        token_counter: TokenCounter | None = None,
        token_cache_dir: str | None = None,
    ) -> None:
        self._model_name = model
        self._api_key = api_key
        self._injected_chat_model = chat_model
        self._chat_models: dict[tuple[str, int], Any] = {}
        self._context_compressor = context_compressor
        self._token_counter = token_counter
        self._token_cache_dir = token_cache_dir

    def _model(self, *, max_completion_tokens: int) -> Any:
        if self._injected_chat_model is not None:
            model = self._injected_chat_model
        else:
            key = (self._model_name, max_completion_tokens)
            model = self._chat_models.get(key)
            if model is None:
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(
                    model=self._model_name,
                    api_key=self._api_key,
                    temperature=0,
                    max_completion_tokens=max_completion_tokens,
                )
                self._chat_models[key] = model
        if self._token_counter is None:
            injected_counter = getattr(model, "token_counter", None)
            self._token_counter = (
                injected_counter
                if callable(injected_counter)
                else LazyTiktokenTokenCounter(
                    model_name=self._model_name,
                    cache_dir=self._token_cache_dir,
                )
            )
        return model

    def _messages(
        self,
        prompt: PromptBundle,
        payload: Mapping[str, object],
        *,
        evidence: EvidenceBundle | None = None,
        max_evidence_tokens: int = 3_000,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> tuple[list[BaseMessage], BoundedContext]:
        builder = ContextBuilder(
            control=prompt.system,
            limits=ContextLimits(max_evidence_tokens=max_evidence_tokens),
            compressor=self._context_compressor,
            token_counter=self._token_counter,
        )
        context = builder.build_dynamic(
            task_factory=lambda retained: _task_text(
                prompt,
                _bounded_source_payload(
                    payload,
                    {item.evidence_id for item in retained},
                ),
            ),
            memory_hints=memory_hints,
            evidence=_context_evidence(evidence),
        )
        return (
            [SystemMessage(content=prompt.system), HumanMessage(content=context.render())],
            context,
        )


class OpenAIIntentRouter(_LazyOpenAIAdapter):
    """Synchronous strict classifier with no tool or analysis authority."""

    def route(self, request: str) -> RouterDecision:
        structured_model = self._model(max_completion_tokens=350).with_structured_output(
            RouterDecision,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.intent_router",
            kind="generation",
            metadata={"model": self._model_name, "prompt_version": ROUTER_PROMPT.version},
        ) as observation:
            try:
                messages, _ = self._messages(
                    ROUTER_PROMPT,
                    {
                        "user_request": request,
                        "allowed_intents": [intent.value for intent in Intent],
                    },
                )
                record_prompt_version(ROUTER_PROMPT.version)
                raw_output = structured_model.invoke(messages)
                decision = RouterDecision.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                observation.update(
                    metadata={
                        "latency_ms": _latency_ms(started_at),
                        "error_code": IntentRoutingErrorCode.FAST_ROUTE_INVALID.value,
                    }
                )
                raise IntentRoutingError(
                    IntentRoutingErrorCode.FAST_ROUTE_INVALID,
                    "structured router returned an invalid decision",
                ) from None
            observation.update(
                output={"intent": decision.intent.value},
                metadata={
                    "latency_ms": _latency_ms(started_at),
                    **_provider_metrics(raw_output),
                },
            )
            return decision


class OpenAIThesisFastModel(OpenAIIntentRouter):
    """Structured thesis router/planner with no tool-binding interface."""

    def plan(
        self,
        ticker: str,
        thesis: str,
        *,
        memory_hints: tuple[MemoryHint, ...] = (),
    ) -> list[ResearchQuestion]:
        structured_model = self._model(max_completion_tokens=350).with_structured_output(
            ThesisResearchQuestionPlan,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.thesis_planner",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": THESIS_PLANNER_PROMPT.version,
                "max_output_tokens": 350,
            },
        ) as observation:
            try:
                messages, _ = self._messages(
                    THESIS_PLANNER_PROMPT,
                    {"ticker": ticker, "thesis": thesis},
                    memory_hints=memory_hints,
                )
                record_prompt_version(THESIS_PLANNER_PROMPT.version)
                raw_output = structured_model.invoke(messages)
                output = ThesisResearchQuestionPlan.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "thesis planner returned invalid structured output",
                ) from None
            observation.update(
                output={"question_count": len(output.questions)},
                metadata={"latency_ms": _latency_ms(started_at), **_provider_metrics(raw_output)},
            )
            return output.questions


class OpenAIThesisAnalystModel(_LazyOpenAIAdapter):
    """Structured thesis analyst restricted to one supplied evidence bundle."""

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: EvidenceBundle,
    ) -> ResearchMemo:
        structured_model = self._model(max_completion_tokens=1_200).with_structured_output(
            ResearchMemo,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.thesis_analyst",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": THESIS_ANALYST_PROMPT.version,
                "max_output_tokens": 1_200,
            },
        ) as observation:
            try:
                messages, bounded_context = self._messages(
                    THESIS_ANALYST_PROMPT,
                    _thesis_analysis_payload(questions, evidence),
                    evidence=evidence,
                )
                record_prompt_version(THESIS_ANALYST_PROMPT.version)
                raw_output = structured_model.invoke(messages)
                memo = ResearchMemo.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "thesis analyst returned invalid structured output",
                ) from None
            observation.update(
                output={"claim_count": _research_memo_claim_count(memo)},
                metadata={"latency_ms": _latency_ms(started_at), **_provider_metrics(raw_output)},
            )
        filing_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "filing"
        }
        web_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "web"
        }
        if not _research_memo_source_ids_are_valid(
            memo,
            filing_ids=filing_ids,
            web_ids=web_ids,
        ):
            raise SkillModelError(
                SkillModelErrorCode.INVALID_EVIDENCE_ID,
                "thesis analyst cited an ID outside the supplied evidence bundle",
            )
        return memo

    def repair(
        self,
        *,
        draft: ResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> ResearchMemo:
        """Attempt one structured, evidence-only repair with a fixed output cap."""
        structured_model = self._model(max_completion_tokens=600).with_structured_output(
            ResearchMemo,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.thesis_repair",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": REPAIR_PROMPT.version,
                "max_output_tokens": 600,
            },
        ) as observation:
            try:
                messages, bounded_context = self._messages(
                    REPAIR_PROMPT,
                    _thesis_repair_payload(draft, guard_errors, evidence),
                    evidence=evidence,
                )
                record_prompt_version(REPAIR_PROMPT.version)
                raw_output = structured_model.invoke(messages)
                repaired = ResearchMemo.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "thesis repair returned invalid structured output",
                ) from None
            observation.update(
                output={"claim_count": _research_memo_claim_count(repaired)},
                metadata={"latency_ms": _latency_ms(started_at), **_provider_metrics(raw_output)},
            )
        filing_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "filing"
        }
        web_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "web"
        }
        if not _research_memo_source_ids_are_valid(
            repaired,
            filing_ids=filing_ids,
            web_ids=web_ids,
        ):
            raise SkillModelError(
                SkillModelErrorCode.INVALID_EVIDENCE_ID,
                "thesis repair cited an ID outside the supplied evidence bundle",
            )
        return repaired


class OpenAISkillPlannerModel(_LazyOpenAIAdapter):
    """OpenAI-backed P1 planner constrained by one frozen recipe snapshot."""

    async def plan(self, request: SkillPlanningInput) -> list[ResearchQuestion]:
        max_output_tokens = request.recipe.budget.max_planner_output_tokens
        structured_model = self._model(
            max_completion_tokens=max_output_tokens
        ).with_structured_output(
            ResearchQuestionPlan,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.skill_planner",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": SKILL_PLANNER_PROMPT.version,
                "max_output_tokens": max_output_tokens,
            },
        ) as observation:
            try:
                messages, _ = self._messages(
                    SKILL_PLANNER_PROMPT,
                    _planning_payload(request),
                    memory_hints=request.memory_hints,
                )
                record_prompt_version(SKILL_PLANNER_PROMPT.version)
                raw_output = await structured_model.ainvoke(messages)
                output = ResearchQuestionPlan.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                observation.update(
                    metadata={
                        "latency_ms": _latency_ms(started_at),
                        "error_code": SkillModelErrorCode.INVALID_OUTPUT.value,
                    }
                )
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "planner returned invalid structured output",
                ) from None
            observation.update(
                output={"question_count": len(output.questions)},
                metadata={
                    "latency_ms": _latency_ms(started_at),
                    **_provider_metrics(raw_output),
                },
            )
        if len(output.questions) > request.recipe.budget.max_questions:
            raise SkillModelError(
                SkillModelErrorCode.INVALID_OUTPUT,
                "planner exceeded the frozen recipe question budget",
            )
        return output.questions


class OpenAISkillAnalystModel(_LazyOpenAIAdapter):
    """OpenAI-backed analyst restricted to source IDs from one evidence bundle."""

    async def analyze(
        self,
        *,
        request: SkillAnalysisInput,
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        request = SkillAnalysisInput(
            ticker=request.ticker,
            user_request=request.user_request,
            recipe=request.recipe,
        )
        max_output_tokens = request.recipe.budget.max_analysis_output_tokens
        structured_model = self._model(
            max_completion_tokens=max_output_tokens
        ).with_structured_output(
            SkillResearchMemo,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.skill_analyst",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": ANALYST_PROMPT.version,
                "max_output_tokens": max_output_tokens,
            },
        ) as observation:
            try:
                messages, bounded_context = self._messages(
                    ANALYST_PROMPT,
                    _analysis_payload(request, evidence),
                    evidence=evidence,
                    max_evidence_tokens=request.recipe.budget.max_evidence_tokens,
                )
                record_prompt_version(ANALYST_PROMPT.version)
                raw_output = await structured_model.ainvoke(messages)
                memo = SkillResearchMemo.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                observation.update(
                    metadata={
                        "latency_ms": _latency_ms(started_at),
                        "error_code": SkillModelErrorCode.INVALID_OUTPUT.value,
                    }
                )
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "analyst returned invalid structured output",
                ) from None
            observation.update(
                output={"section_count": len(memo.sections)},
                metadata={
                    "latency_ms": _latency_ms(started_at),
                    **_provider_metrics(raw_output),
                },
            )

        if (
            memo.recipe_name is not request.recipe.name
            or memo.recipe_version != request.recipe.version
        ):
            raise SkillModelError(
                SkillModelErrorCode.RECIPE_IDENTITY_MISMATCH,
                "analyst output does not match the frozen recipe identity",
            )

        filing_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "filing"
        }
        web_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "web"
        }
        if not _memo_source_ids_are_valid(memo, filing_ids=filing_ids, web_ids=web_ids):
            raise SkillModelError(
                SkillModelErrorCode.INVALID_EVIDENCE_ID,
                "analyst output cites an ID outside the supplied evidence bundle",
            )
        return memo

    async def repair(
        self,
        *,
        request: SkillAnalysisInput,
        draft: SkillResearchMemo,
        guard_errors: tuple[str, ...],
        evidence: EvidenceBundle,
    ) -> SkillResearchMemo:
        """Attempt one recipe-bound repair without expanding evidence authority."""
        max_output_tokens = request.recipe.budget.max_repair_output_tokens
        structured_model = self._model(
            max_completion_tokens=max_output_tokens
        ).with_structured_output(
            SkillResearchMemo,
            strict=True,
            include_raw=True,
        )
        started_at = perf_counter()
        with observe(
            name="model.skill_repair",
            kind="generation",
            metadata={
                "model": self._model_name,
                "prompt_version": REPAIR_PROMPT.version,
                "max_output_tokens": max_output_tokens,
            },
        ) as observation:
            try:
                messages, bounded_context = self._messages(
                    REPAIR_PROMPT,
                    _skill_repair_payload(request, draft, guard_errors, evidence),
                    evidence=evidence,
                    max_evidence_tokens=request.recipe.budget.max_evidence_tokens,
                )
                record_prompt_version(REPAIR_PROMPT.version)
                raw_output = await structured_model.ainvoke(messages)
                repaired = SkillResearchMemo.model_validate(_parsed_output(raw_output))
            except (TypeError, ValidationError, OutputParserException, ValueError):
                raise SkillModelError(
                    SkillModelErrorCode.INVALID_OUTPUT,
                    "skill repair returned invalid structured output",
                ) from None
            observation.update(
                output={"section_count": len(repaired.sections)},
                metadata={
                    "latency_ms": _latency_ms(started_at),
                    **_provider_metrics(raw_output),
                },
            )
        if (
            repaired.recipe_name is not request.recipe.name
            or repaired.recipe_version != request.recipe.version
        ):
            raise SkillModelError(
                SkillModelErrorCode.RECIPE_IDENTITY_MISMATCH,
                "repair output does not match the frozen recipe identity",
            )
        filing_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "filing"
        }
        web_ids = {
            item.evidence_id for item in bounded_context.evidence if item.source_type == "web"
        }
        if not _memo_source_ids_are_valid(
            repaired,
            filing_ids=filing_ids,
            web_ids=web_ids,
        ):
            raise SkillModelError(
                SkillModelErrorCode.INVALID_EVIDENCE_ID,
                "repair output cites an ID outside the supplied evidence bundle",
            )
        return repaired


def _task_text(prompt: PromptBundle, payload: Mapping[str, object]) -> str:
    return f"{prompt.task}\n<task_data>{json.dumps(payload, sort_keys=True)}</task_data>"


def _bounded_source_payload(
    payload: Mapping[str, object],
    retained_ids: set[str],
) -> dict[str, object]:
    bounded = dict(payload)
    for key in ("allowed_filing_source_ids", "allowed_web_source_ids"):
        source_ids = bounded.get(key)
        if isinstance(source_ids, list):
            bounded[key] = [
                source_id
                for source_id in source_ids
                if isinstance(source_id, str) and source_id in retained_ids
            ]
    return bounded


def _thesis_analysis_payload(
    questions: list[ResearchQuestion],
    evidence: EvidenceBundle,
) -> dict[str, object]:
    return {
        "questions": [question.model_dump(mode="json") for question in questions],
        "allowed_filing_source_ids": [item.id for item in evidence.filing_evidence],
        "allowed_web_source_ids": [item.id for item in evidence.web_evidence],
    }


def _thesis_repair_payload(
    draft: ResearchMemo,
    guard_errors: tuple[str, ...],
    evidence: EvidenceBundle,
) -> dict[str, object]:
    return {
        "draft": draft.model_dump(mode="json"),
        "citation_failures": list(guard_errors),
        "allowed_filing_source_ids": [item.id for item in evidence.filing_evidence],
        "allowed_web_source_ids": [item.id for item in evidence.web_evidence],
    }


def _research_memo_source_ids_are_valid(
    memo: ResearchMemo,
    *,
    filing_ids: set[str],
    web_ids: set[str],
) -> bool:
    return all(
        set(claim.evidence_chunk_ids) <= filing_ids and set(claim.web_evidence_ids) <= web_ids
        for claims in (
            memo.supporting_claims,
            memo.counter_claims,
            memo.inferences,
            memo.open_questions,
        )
        for claim in claims
    )


def _planning_payload(request: SkillPlanningInput) -> dict[str, object]:
    return {
        "ticker": request.ticker,
        "user_request": request.user_request,
        "frozen_recipe_constraints": _recipe_constraints(request.recipe),
    }


def _analysis_payload(
    request: SkillAnalysisInput,
    evidence: EvidenceBundle,
) -> dict[str, object]:
    filing_ids = [item.id for item in evidence.filing_evidence]
    web_ids = [item.id for item in evidence.web_evidence]
    return {
        "ticker": request.ticker,
        "user_request": request.user_request,
        "frozen_recipe_constraints": _recipe_constraints(request.recipe),
        "allowed_filing_source_ids": filing_ids,
        "allowed_web_source_ids": web_ids,
    }


def _skill_repair_payload(
    request: SkillAnalysisInput,
    draft: SkillResearchMemo,
    guard_errors: tuple[str, ...],
    evidence: EvidenceBundle,
) -> dict[str, object]:
    payload = _analysis_payload(request, evidence)
    return {
        **payload,
        "draft": draft.model_dump(mode="json"),
        "citation_failures": list(guard_errors),
    }


def _context_evidence(evidence: EvidenceBundle | None) -> tuple[ContextEvidence, ...]:
    if evidence is None:
        return ()
    values: list[ContextEvidence] = []
    all_sources = [*evidence.filing_evidence, *evidence.web_evidence]
    source_count = len(all_sources)
    bindings_by_source: dict[str, list[str]] = {}
    for assignment in evidence.facet_assignments:
        bindings_by_source.setdefault(assignment.source_id, []).append(
            f"question_index={assignment.question_index},"
            f"side={assignment.side.value},facet={assignment.facet.value}"
        )
    for index, item in enumerate(evidence.filing_evidence):
        values.append(
            ContextEvidence(
                evidence_id=item.id,
                source_type="filing",
                ticker=item.ticker,
                body=item.content,
                score=float(source_count - index),
                source_url=item.source_url,
                date=item.filed_at.isoformat(),
                accession_no=item.accession_no,
                section=item.section,
                raw_start=item.raw_start,
                raw_end=item.raw_end,
                form=item.form,
                citation_bindings=tuple(bindings_by_source.get(item.id, ())),
            )
        )
    filing_count = len(evidence.filing_evidence)
    for index, item in enumerate(evidence.web_evidence):
        values.append(
            ContextEvidence(
                evidence_id=item.id,
                source_type="web",
                ticker=item.ticker,
                body=item.content,
                score=float(source_count - filing_count - index),
                source_url=str(item.source_url),
                date=(
                    item.published_at.isoformat()
                    if item.published_at is not None
                    else item.fetched_at.isoformat()
                ),
                title=item.title,
                source_kind=item.source_kind.value,
                source_tier=item.source_tier.value,
                fetched_at=item.fetched_at.isoformat(),
                citation_bindings=tuple(bindings_by_source.get(item.id, ())),
            )
        )
    return tuple(values)


def _research_memo_claim_count(memo: ResearchMemo) -> int:
    return sum(
        len(claims)
        for claims in (
            memo.supporting_claims,
            memo.counter_claims,
            memo.inferences,
            memo.open_questions,
        )
    )


def _recipe_constraints(recipe: ResearchRecipe) -> dict[str, object]:
    return {
        "recipe_name": recipe.name.value,
        "recipe_version": recipe.version,
        "required_facets": [facet.value for facet in recipe.required_facets],
        "source_policy": [source_kind.value for source_kind in recipe.source_policy],
        "max_questions": recipe.budget.max_questions,
        "max_local_results_per_query": recipe.budget.max_local_results_per_query,
        "max_retrieval_rounds": recipe.budget.max_retrieval_rounds,
        "max_web_calls": recipe.budget.max_web_calls,
        "max_web_results": recipe.budget.max_web_results,
        "max_planner_output_tokens": recipe.budget.max_planner_output_tokens,
        "max_analysis_output_tokens": recipe.budget.max_analysis_output_tokens,
        "max_repair_output_tokens": recipe.budget.max_repair_output_tokens,
        "max_evidence_tokens": recipe.budget.max_evidence_tokens,
    }


def _memo_source_ids_are_valid(
    memo: SkillResearchMemo,
    *,
    filing_ids: set[str],
    web_ids: set[str],
) -> bool:
    for section in memo.sections:
        for claim in section.claims:
            if not set(claim.evidence_chunk_ids) <= filing_ids:
                return False
            if not set(claim.web_evidence_ids) <= web_ids:
                return False

    allowed_data_ids = filing_ids | web_ids
    return all(set(point.source_ids) <= allowed_data_ids for point in memo.data_points)


def _parsed_output(output: object) -> object:
    if not isinstance(output, Mapping):
        raise TypeError("structured output did not return a raw envelope")
    if not {"raw", "parsed", "parsing_error"} <= output.keys():
        raise TypeError("structured output envelope is incomplete")
    if not isinstance(output["raw"], AIMessage):
        raise TypeError("structured output raw value is not an AI message")
    if output["parsing_error"] is not None:
        raise ValueError("structured output parser rejected the response")
    if output["parsed"] is None:
        raise ValueError("structured output has no parsed value")
    return output["parsed"]


def _provider_metrics(output: object) -> dict[str, object]:
    raw = output.get("raw") if isinstance(output, Mapping) else None
    if not isinstance(raw, AIMessage):
        return {}
    usage = getattr(raw, "usage_metadata", None)
    response = getattr(raw, "response_metadata", None)
    metrics: dict[str, object] = {}
    if isinstance(usage, Mapping):
        safe_usage = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance((value := usage.get(key)), (int, float))
        }
        if safe_usage:
            metrics["usage"] = safe_usage
    if isinstance(response, Mapping):
        for key in ("cost", "total_cost"):
            value = response.get(key)
            if isinstance(value, (int, float)):
                metrics["cost"] = value
                break
    return metrics


def _latency_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1_000))
