"""Budgeted, deterministic P1 evidence collection over injected callables."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fra.context import BudgetGate
from fra.domain import (
    EvidenceChunk,
    ResearchQuestion,
    SourceKind,
    WebEvidence,
)
from fra.retrieval.coverage import (
    CoverageReport,
    EvidenceAssignment,
    EvidenceSide,
    FacetAssignment,
    assess_coverage,
    valid_assignment_keys,
    valid_source_ids,
)
from fra.skills.models import (
    EvidenceCollectionPolicy,
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    WebUsagePolicy,
)

_LOCAL_TOOL = "hybrid_search_filings"
_WEB_TOOL = "search_allowlisted_web"
_FORM = Literal["10-K", "10-Q", "8-K"]

THESIS_COLLECTION_POLICY = EvidenceCollectionPolicy(
    allowed_tools=frozenset({_LOCAL_TOOL, _WEB_TOOL}),
    source_policy=(
        SourceKind.FILING,
        SourceKind.ISSUER_IR,
        SourceKind.AUTHORITATIVE_WEB,
    ),
    required_facets=(ResearchFacet.INFORMATION_GAPS,),
    budget=RecipeBudget(
        max_questions=3,
        max_local_results_per_query=5,
        max_retrieval_rounds=2,
        max_web_calls=1,
        max_web_results=3,
        max_planner_output_tokens=350,
        max_analysis_output_tokens=1_200,
        max_repair_output_tokens=600,
        max_evidence_tokens=3_000,
    ),
    web_usage_policy=WebUsagePolicy.EVIDENCE,
)

CollectionPolicy = ResearchRecipe | EvidenceCollectionPolicy


class RetrievalErrorCode(StrEnum):
    """Provider-neutral error categories for typed retrieval envelopes."""

    EMPTY_RETRIEVAL = "empty_retrieval"
    INSUFFICIENT_RETRIEVAL = "insufficient_retrieval"
    PROTOCOL_ERROR = "protocol_error"
    OPERATION_ERROR = "operation_error"
    RERANKER_UNAVAILABLE = "reranker_unavailable"


class RetrievalError(BaseModel):
    """A typed error distinct from an empty evidence sequence."""

    model_config = ConfigDict(frozen=True)

    code: RetrievalErrorCode
    message: str = Field(min_length=1, max_length=500)


class EvidenceQuery(BaseModel):
    """One typed local query inside a collector retrieval round."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(min_length=1, max_length=10)
    question_index: int = Field(ge=0)
    facet: ResearchFacet
    side: EvidenceSide
    query: str = Field(min_length=1, max_length=500)
    period: str | None = None
    forms: tuple[_FORM, ...] = ()
    corpus_version: str | None = None
    filing_ids: tuple[str, ...] = ()
    limit: int = Field(ge=1, le=10)


class LocalEvidenceHit(BaseModel):
    """A filing chunk with typed question, side, and facet provenance."""

    model_config = ConfigDict(frozen=True)

    evidence: EvidenceChunk
    question_index: int = Field(ge=0)
    side: EvidenceSide
    facet: ResearchFacet


class LocalSearchResponse(BaseModel):
    """Provider-neutral local-search envelope consumed by the collector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: tuple[LocalEvidenceHit, ...] = ()
    error: RetrievalError | None = None

    @model_validator(mode="after")
    def reject_mixed_success_and_error(self) -> LocalSearchResponse:
        if self.evidence and self.error is not None:
            raise ValueError("retrieval response cannot contain both evidence and error")
        return self


class WebEvidenceHit(BaseModel):
    """A web snapshot with typed question, side, and facet provenance."""

    model_config = ConfigDict(frozen=True)

    evidence: WebEvidence
    question_index: int = Field(ge=0)
    side: EvidenceSide
    facet: ResearchFacet


class WebSearchRequest(BaseModel):
    """The single missing-coverage request permitted for web fallback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    query: str = Field(min_length=1)
    missing_facets: tuple[ResearchFacet, ...]
    missing_pairs: tuple[tuple[int, EvidenceSide], ...]
    max_results: int = Field(ge=1, le=3)
    usage_policy: WebUsagePolicy


class WebSearchResponse(BaseModel):
    """Provider-neutral allowlisted-web envelope consumed by the collector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: tuple[WebEvidenceHit, ...] = ()
    error: RetrievalError | None = None

    @model_validator(mode="after")
    def reject_mixed_success_and_error(self) -> WebSearchResponse:
        if self.evidence and self.error is not None:
            raise ValueError("retrieval response cannot contain both evidence and error")
        return self


class EvidenceBundle(BaseModel):
    """Validated evidence and explicit collection accounting for one recipe."""

    filing_evidence: list[EvidenceChunk]
    web_evidence: list[WebEvidence]
    assignments: list[EvidenceAssignment]
    facet_assignments: list[FacetAssignment] = Field(default_factory=list)
    coverage: CoverageReport
    retrieval_rounds: int = Field(ge=0, le=2)
    web_calls: int = Field(ge=0, le=1)
    tool_calls: int = Field(default=0, ge=0)


class CollectionErrorCode(StrEnum):
    """Fail-closed collection failures that must not become evidence gaps."""

    PROTOCOL_ERROR = "protocol_error"
    RETRIEVAL_ERROR = "retrieval_error"
    POLICY_VIOLATION = "policy_violation"
    DEPENDENCY_ERROR = "dependency_error"


class EvidenceCollectionError(RuntimeError):
    """Typed failure for protocol, operation, or recipe-policy violations."""

    def __init__(self, code: CollectionErrorCode, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code.value}: {message}")


LocalSearch = Callable[[tuple[EvidenceQuery, ...]], Awaitable[LocalSearchResponse]]
WebSearch = Callable[[WebSearchRequest], Awaitable[WebSearchResponse]]


def _with_budget_gate(value: Callable[..., object], gate: BudgetGate):
    binder = getattr(value, "with_budget_gate", None)
    return binder(gate) if callable(binder) else value


class EvidenceCollector:
    """Enforce two local rounds and one policy-authorized web fallback at most."""

    def __init__(
        self,
        *,
        local_search: LocalSearch,
        web_search: WebSearch | None = None,
        corpus_version: str | None = None,
        filing_ids: tuple[str, ...] = (),
        scope_error: str | None = None,
        budget_gate: BudgetGate | None = None,
    ) -> None:
        self._local_search = local_search
        self._web_search = web_search
        self._corpus_version = corpus_version
        self._filing_ids = filing_ids
        self._scope_error = scope_error
        self._budget_gate = budget_gate

    def with_filing_scope(
        self,
        corpus_version: str | None,
        filing_ids: tuple[str, ...],
        *,
        scope_error: str | None = None,
    ) -> EvidenceCollector:
        """Return a detached collector with one application-resolved filing scope."""
        return EvidenceCollector(
            local_search=self._local_search,
            web_search=self._web_search,
            corpus_version=corpus_version,
            filing_ids=filing_ids,
            scope_error=scope_error,
            budget_gate=self._budget_gate,
        )

    def with_budget_gate(self, gate: BudgetGate) -> EvidenceCollector:
        """Return a detached collector and gate-capable adapters sharing one authority."""
        local_search = _with_budget_gate(self._local_search, gate)
        web_search = (
            None if self._web_search is None else _with_budget_gate(self._web_search, gate)
        )
        return EvidenceCollector(
            local_search=local_search,
            web_search=web_search,
            corpus_version=self._corpus_version,
            filing_ids=self._filing_ids,
            scope_error=self._scope_error,
            budget_gate=gate,
        )

    async def collect(
        self,
        *,
        ticker: str,
        recipe: CollectionPolicy,
        questions: Sequence[ResearchQuestion],
    ) -> EvidenceBundle:
        bundle = await self.retrieve(
            ticker=ticker,
            recipe=recipe,
            questions=questions,
        )
        no_filings = self._scope_error is not None and self._scope_error.startswith(
            "NO_FILINGS:"
        )
        if not bundle.coverage.complete and not no_filings:
            bundle = await self.retry_missing(
                ticker=ticker,
                recipe=recipe,
                questions=questions,
                evidence=bundle,
            )
        if not bundle.coverage.complete:
            bundle = await self.web_fallback(
                ticker=ticker,
                recipe=recipe,
                questions=questions,
                evidence=bundle,
            )
        if bundle.coverage.complete:
            return bundle
        return bundle.model_copy(
            update={
                "coverage": bundle.coverage.model_copy(
                    update={
                        "reason_codes": _append_reason(
                            bundle.coverage.reason_codes,
                            "insufficient_information",
                        )
                    }
                )
            }
        )

    async def retrieve(
        self,
        *,
        ticker: str,
        recipe: CollectionPolicy,
        questions: Sequence[ResearchQuestion],
    ) -> EvidenceBundle:
        """Run exactly the first local round for a frozen ticker and filing scope."""
        if self._scope_error is not None:
            if self._scope_error.startswith("NO_FILINGS:"):
                normalized_ticker = ticker.strip().upper()
                self._validate_request(normalized_ticker, recipe, questions)
                coverage = _coverage(
                    normalized_ticker,
                    recipe,
                    questions,
                    (),
                    (),
                    (),
                    (),
                )
                return _bundle(
                    normalized_ticker,
                    recipe,
                    questions,
                    (),
                    (),
                    (),
                    (),
                    coverage,
                    0,
                    0,
                    0,
                )
            raise EvidenceCollectionError(
                CollectionErrorCode.RETRIEVAL_ERROR,
                self._scope_error,
            )
        normalized_ticker = ticker.strip().upper()
        self._validate_request(normalized_ticker, recipe, questions)

        local_hits: list[LocalEvidenceHit] = []
        web_hits: list[WebEvidenceHit] = []
        filing_evidence: list[EvidenceChunk] = []
        web_evidence: list[WebEvidence] = []
        assignments: list[EvidenceAssignment] = []
        facet_assignments: list[FacetAssignment] = []
        retrieval_rounds = 0
        first_queries = _initial_queries(
            normalized_ticker,
            recipe,
            questions,
            corpus_version=self._corpus_version,
            filing_ids=self._filing_ids,
        )
        if self._budget_gate is not None:
            self._budget_gate.consume(retrieval_rounds=1)
        first_response = await self._call_local(first_queries)
        retrieval_rounds += 1
        local_hits.extend(_bounded_local_hits(first_response.evidence, first_queries))
        _materialize_hits(
            local_hits,
            web_hits,
            filing_evidence,
            web_evidence,
            assignments,
            facet_assignments,
        )
        coverage = _coverage(
            normalized_ticker,
            recipe,
            questions,
            filing_evidence,
            web_evidence,
            assignments,
            facet_assignments,
        )

        return _bundle(
            normalized_ticker,
            recipe,
            questions,
            filing_evidence,
            web_evidence,
            assignments,
            facet_assignments,
            coverage,
            retrieval_rounds,
            0,
            len(first_queries),
        )

    async def retry_missing(
        self,
        *,
        ticker: str,
        recipe: CollectionPolicy,
        questions: Sequence[ResearchQuestion],
        evidence: EvidenceBundle,
    ) -> EvidenceBundle:
        """Run the one permitted retry using only missing side/facet bindings."""
        if evidence.coverage.complete or evidence.retrieval_rounds >= 2:
            return evidence
        if recipe.budget.max_retrieval_rounds < 2:
            return evidence
        normalized_ticker = ticker.strip().upper()
        self._validate_request(normalized_ticker, recipe, questions)
        first_queries = _initial_queries(
            normalized_ticker,
            recipe,
            questions,
            corpus_version=self._corpus_version,
            filing_ids=self._filing_ids,
        )
        retry_queries = _retry_queries(first_queries, evidence.coverage)
        if self._budget_gate is not None:
            self._budget_gate.consume(retrieval_rounds=1)
        retry_response = await self._call_local(retry_queries)
        local_hits, web_hits = _bundle_hits(evidence)
        local_hits.extend(_bounded_local_hits(retry_response.evidence, retry_queries))
        return _bundle_from_hits(
            ticker=normalized_ticker,
            recipe=recipe,
            questions=questions,
            local_hits=local_hits,
            web_hits=web_hits,
            previous_valid_source_ids=frozenset(
                source.id
                for source in (*evidence.filing_evidence, *evidence.web_evidence)
            ),
            previous_coverage=evidence.coverage,
            retrieval_rounds=evidence.retrieval_rounds + 1,
            web_calls=evidence.web_calls,
            tool_calls=evidence.tool_calls + len(retry_queries),
            after_second_round=True,
        )

    async def web_fallback(
        self,
        *,
        ticker: str,
        recipe: CollectionPolicy,
        questions: Sequence[ResearchQuestion],
        evidence: EvidenceBundle,
    ) -> EvidenceBundle:
        """Run at most one policy-authorized web request for one missing coverage target."""
        if (
            evidence.coverage.complete
            or (
                evidence.retrieval_rounds != 2
                and not (
                    evidence.retrieval_rounds == 0
                    and self._scope_error is not None
                    and self._scope_error.startswith("NO_FILINGS:")
                )
            )
            or evidence.web_calls >= 1
            or not self._web_authorized(recipe)
            or self._web_search is None
        ):
            return evidence
        normalized_ticker = ticker.strip().upper()
        web_request = _web_request(normalized_ticker, recipe, evidence.coverage)
        if self._budget_gate is not None:
            self._budget_gate.consume(web_calls=1)
        web_response = await self._call_web(web_request)
        local_hits, web_hits = _bundle_hits(evidence)
        web_hits.extend(
            _bounded_web_hits(
                web_response.evidence,
                web_request,
                recipe.required_facets,
            )
        )
        return _bundle_from_hits(
            ticker=normalized_ticker,
            recipe=recipe,
            questions=questions,
            local_hits=local_hits,
            web_hits=web_hits,
            previous_valid_source_ids=frozenset(
                source.id
                for source in (*evidence.filing_evidence, *evidence.web_evidence)
            ),
            previous_coverage=evidence.coverage,
            retrieval_rounds=evidence.retrieval_rounds,
            web_calls=evidence.web_calls + 1,
            tool_calls=evidence.tool_calls + 1,
            after_second_round=True,
        )

    @staticmethod
    def _validate_request(
        ticker: str,
        recipe: CollectionPolicy,
        questions: Sequence[ResearchQuestion],
    ) -> None:
        if not ticker:
            raise EvidenceCollectionError(
                CollectionErrorCode.POLICY_VIOLATION,
                "ticker must not be empty",
            )
        if len(questions) > recipe.budget.max_questions:
            raise EvidenceCollectionError(
                CollectionErrorCode.POLICY_VIOLATION,
                "planned questions exceed recipe budget",
            )
        if _LOCAL_TOOL not in recipe.allowed_tools or SourceKind.FILING not in recipe.source_policy:
            raise EvidenceCollectionError(
                CollectionErrorCode.POLICY_VIOLATION,
                "recipe does not authorize local filing search",
            )

    async def _call_local(
        self,
        queries: tuple[EvidenceQuery, ...],
    ) -> LocalSearchResponse:
        try:
            response = await self._local_search(queries)
        except EvidenceCollectionError:
            raise
        except Exception as error:
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "local search callable failed",
            ) from error
        if not isinstance(response, LocalSearchResponse):
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "local search returned an invalid typed envelope",
            )
        _raise_for_error(response.error, "local search")
        return response

    async def _call_web(self, request: WebSearchRequest) -> WebSearchResponse:
        if self._web_search is None:
            raise AssertionError("web search authorization checked without a callable")
        try:
            response = await self._web_search(request)
        except EvidenceCollectionError:
            raise
        except Exception as error:
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "web search callable failed",
            ) from error
        if not isinstance(response, WebSearchResponse):
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "web search returned an invalid typed envelope",
            )
        _raise_for_error(response.error, "web search")
        return response

    @staticmethod
    def _web_authorized(recipe: CollectionPolicy) -> bool:
        return (
            _WEB_TOOL in recipe.allowed_tools
            and recipe.web_usage_policy is not WebUsagePolicy.NONE
            and recipe.budget.max_web_calls >= 1
            and recipe.budget.max_web_results >= 1
        )


def rewrite_query(query: EvidenceQuery) -> EvidenceQuery:
    """Rewrite one missing query with a fixed template and no model call."""

    suffix = (
        f" | ticker={query.ticker} | facet={query.facet.value} "
        f"| side={query.side.value} | retry=missing_only"
    )
    rewritten = f"{query.query[: 500 - len(suffix)]}{suffix}"
    return query.model_copy(update={"query": rewritten})


def _initial_queries(
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    *,
    corpus_version: str | None = None,
    filing_ids: tuple[str, ...] = (),
) -> tuple[EvidenceQuery, ...]:
    queries: list[EvidenceQuery] = []
    for question_index, question in enumerate(questions):
        for side, query_text in (
            (EvidenceSide.SUPPORT, question.support_query),
            (EvidenceSide.CHALLENGE, question.challenge_query),
        ):
            for facet in recipe.required_facets:
                queries.append(
                    EvidenceQuery(
                        ticker=ticker,
                        question_index=question_index,
                        facet=facet,
                        side=side,
                        query=query_text,
                        period=question.period,
                        forms=tuple(question.forms),
                        corpus_version=corpus_version,
                        filing_ids=filing_ids,
                        limit=recipe.budget.max_local_results_per_query,
                    )
                )
    return tuple(queries)


def _retry_queries(
    initial_queries: tuple[EvidenceQuery, ...],
    coverage: CoverageReport,
) -> tuple[EvidenceQuery, ...]:
    missing_pairs = frozenset(coverage.missing_pairs)
    missing_facets = frozenset(coverage.missing_facets)
    return tuple(
        rewrite_query(query)
        for query in initial_queries
        if (query.question_index, query.side) in missing_pairs or query.facet in missing_facets
    )


def _bounded_local_hits(
    hits: tuple[LocalEvidenceHit, ...],
    queries: tuple[EvidenceQuery, ...],
) -> tuple[LocalEvidenceHit, ...]:
    limits = {(query.question_index, query.side, query.facet): query.limit for query in queries}
    counts: dict[tuple[int, EvidenceSide, ResearchFacet], int] = {}
    bounded: list[LocalEvidenceHit] = []
    for hit in hits:
        key = (hit.question_index, hit.side, hit.facet)
        if key not in limits:
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                "local evidence binding was not requested",
            )
        count = counts.get(key, 0)
        if count >= limits[key]:
            continue
        counts[key] = count + 1
        bounded.append(hit)
    return tuple(bounded)


def _web_request(
    ticker: str,
    recipe: CollectionPolicy,
    coverage: CoverageReport,
) -> WebSearchRequest:
    purpose = (
        "primary disclosure locator"
        if recipe.web_usage_policy is WebUsagePolicy.PRIMARY_SOURCE_LOCATOR
        else "allowlisted evidence"
    )
    if coverage.missing_pairs:
        target_pair = coverage.missing_pairs[0]
        target_facet = (
            coverage.missing_facets[0] if coverage.missing_facets else recipe.required_facets[0]
        )
    else:
        target_pair = (0, EvidenceSide.SUPPORT)
        target_facet = coverage.missing_facets[0]
    question_index, side = target_pair
    query = (
        f"{ticker} {purpose} target question={question_index} "
        f"side={side.value} facet={target_facet.value}"
    )
    return WebSearchRequest(
        ticker=ticker,
        query=query,
        missing_facets=(target_facet,),
        missing_pairs=(target_pair,),
        max_results=recipe.budget.max_web_results,
        usage_policy=recipe.web_usage_policy,
    )


def _bounded_web_hits(
    hits: tuple[WebEvidenceHit, ...],
    request: WebSearchRequest,
    required_facets: tuple[ResearchFacet, ...],
) -> tuple[WebEvidenceHit, ...]:
    selected_ids: list[str] = []
    bounded: list[WebEvidenceHit] = []
    missing_facets = frozenset(request.missing_facets)
    missing_pairs = frozenset(request.missing_pairs)
    allowed_facets = frozenset(required_facets)
    for hit in hits:
        pair = (hit.question_index, hit.side)
        if hit.facet not in allowed_facets:
            continue
        if hit.facet not in missing_facets or pair not in missing_pairs:
            continue
        if hit.evidence.id not in selected_ids:
            if len(selected_ids) >= request.max_results:
                continue
            selected_ids.append(hit.evidence.id)
        bounded.append(hit)
    return tuple(bounded)


def _bundle_hits(
    evidence: EvidenceBundle,
) -> tuple[list[LocalEvidenceHit], list[WebEvidenceHit]]:
    """Rebuild typed hits from a previously validated intermediate bundle."""
    filing_index = {source.id: source for source in evidence.filing_evidence}
    web_index = {source.id: source for source in evidence.web_evidence}
    assignment_keys = {
        (assignment.question_index, assignment.side, assignment.source_id)
        for assignment in evidence.assignments
    }
    local_hits: list[LocalEvidenceHit] = []
    web_hits: list[WebEvidenceHit] = []
    for assignment in evidence.facet_assignments:
        key = (assignment.question_index, assignment.side, assignment.source_id)
        if key not in assignment_keys:
            continue
        if source := filing_index.get(assignment.source_id):
            local_hits.append(
                LocalEvidenceHit(
                    evidence=source,
                    question_index=assignment.question_index,
                    side=assignment.side,
                    facet=assignment.facet,
                )
            )
        elif source := web_index.get(assignment.source_id):
            web_hits.append(
                WebEvidenceHit(
                    evidence=source,
                    question_index=assignment.question_index,
                    side=assignment.side,
                    facet=assignment.facet,
                )
            )
    return local_hits, web_hits


def _bundle_from_hits(
    *,
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    local_hits: Sequence[LocalEvidenceHit],
    web_hits: Sequence[WebEvidenceHit],
    previous_valid_source_ids: frozenset[str],
    previous_coverage: CoverageReport,
    retrieval_rounds: int,
    web_calls: int,
    tool_calls: int,
    after_second_round: bool,
) -> EvidenceBundle:
    filing_evidence: list[EvidenceChunk] = []
    web_evidence: list[WebEvidence] = []
    assignments: list[EvidenceAssignment] = []
    facet_assignments: list[FacetAssignment] = []
    _materialize_hits(
        local_hits,
        web_hits,
        filing_evidence,
        web_evidence,
        assignments,
        facet_assignments,
    )
    coverage = _coverage(
        ticker,
        recipe,
        questions,
        filing_evidence,
        web_evidence,
        assignments,
        facet_assignments,
        previous_valid_source_ids=previous_valid_source_ids,
        after_second_round=after_second_round,
    )
    invalid_source_ids = tuple(
        sorted(
            set(previous_coverage.invalid_source_ids)
            | set(coverage.invalid_source_ids)
        )
    )
    ticker_mismatches = tuple(
        sorted(
            set(previous_coverage.ticker_mismatches)
            | set(coverage.ticker_mismatches)
        )
    )
    date_mismatches = tuple(
        sorted(
            set(previous_coverage.date_mismatches)
            | set(coverage.date_mismatches)
        )
    )
    reason_codes = list(coverage.reason_codes)
    for present, reason in (
        (invalid_source_ids, "invalid_source_id"),
        (ticker_mismatches, "ticker_mismatch"),
        (date_mismatches, "date_mismatch"),
    ):
        if present and reason not in reason_codes:
            reason_codes.append(reason)
    coverage = coverage.model_copy(
        update={
            "invalid_source_ids": invalid_source_ids,
            "ticker_mismatches": ticker_mismatches,
            "date_mismatches": date_mismatches,
            "reason_codes": tuple(reason_codes),
        }
    )
    return _bundle(
        ticker,
        recipe,
        questions,
        filing_evidence,
        web_evidence,
        assignments,
        facet_assignments,
        coverage,
        retrieval_rounds,
        web_calls,
        tool_calls,
    )


def _materialize_hits(
    local_hits: Sequence[LocalEvidenceHit],
    web_hits: Sequence[WebEvidenceHit],
    filing_evidence: list[EvidenceChunk],
    web_evidence: list[WebEvidence],
    assignments: list[EvidenceAssignment],
    facet_assignments: list[FacetAssignment],
) -> None:
    filing_evidence.clear()
    web_evidence.clear()
    assignments.clear()
    facet_assignments.clear()
    source_values: dict[str, EvidenceChunk | WebEvidence] = {}
    assignment_keys: set[tuple[int, EvidenceSide, str, SourceKind]] = set()
    facet_keys: set[tuple[int, EvidenceSide, ResearchFacet, str]] = set()

    for hit in (*local_hits, *web_hits):
        source = hit.evidence
        previous = source_values.get(source.id)
        if previous is not None and previous != source:
            raise EvidenceCollectionError(
                CollectionErrorCode.PROTOCOL_ERROR,
                f"stable source ID {source.id!r} identified conflicting evidence",
            )
        if previous is None:
            source_values[source.id] = source
            if isinstance(source, EvidenceChunk):
                filing_evidence.append(source)
            else:
                web_evidence.append(source)
        source_kind = SourceKind.FILING if isinstance(source, EvidenceChunk) else source.source_kind
        assignment_key = (hit.question_index, hit.side, source.id, source_kind)
        if assignment_key not in assignment_keys:
            assignment_keys.add(assignment_key)
            assignments.append(
                EvidenceAssignment(
                    question_index=hit.question_index,
                    side=hit.side,
                    source_id=source.id,
                    source_kind=source_kind,
                )
            )
        facet_key = (hit.question_index, hit.side, hit.facet, source.id)
        if facet_key not in facet_keys:
            facet_keys.add(facet_key)
            facet_assignments.append(
                FacetAssignment(
                    question_index=hit.question_index,
                    side=hit.side,
                    facet=hit.facet,
                    source_id=source.id,
                )
            )


def _coverage(
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
    *,
    previous_valid_source_ids: frozenset[str] = frozenset(),
    after_second_round: bool = False,
) -> CoverageReport:
    return assess_coverage(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
        previous_valid_source_ids=previous_valid_source_ids,
        after_second_round=after_second_round,
    )


def _bundle(
    ticker: str,
    recipe: CollectionPolicy,
    questions: Sequence[ResearchQuestion],
    filing_evidence: Sequence[EvidenceChunk],
    web_evidence: Sequence[WebEvidence],
    assignments: Sequence[EvidenceAssignment],
    facet_assignments: Sequence[FacetAssignment],
    coverage: CoverageReport,
    retrieval_rounds: int,
    web_calls: int,
    tool_calls: int,
) -> EvidenceBundle:
    valid_ids = valid_source_ids(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
    )
    valid_assignments = valid_assignment_keys(
        ticker=ticker,
        recipe=recipe,
        questions=questions,
        filing_evidence=filing_evidence,
        web_evidence=web_evidence,
        assignments=assignments,
        facet_assignments=facet_assignments,
    )
    return EvidenceBundle(
        filing_evidence=[source for source in filing_evidence if source.id in valid_ids],
        web_evidence=[source for source in web_evidence if source.id in valid_ids],
        assignments=[
            assignment
            for assignment in assignments
            if (assignment.question_index, assignment.side, assignment.source_id)
            in valid_assignments
            and assignment.source_id in valid_ids
        ],
        facet_assignments=[
            assignment
            for assignment in facet_assignments
            if (
                assignment.question_index,
                assignment.side,
                assignment.source_id,
            )
            in valid_assignments
            and assignment.source_id in valid_ids
        ],
        coverage=coverage,
        retrieval_rounds=retrieval_rounds,
        web_calls=web_calls,
        tool_calls=tool_calls,
    )


def _raise_for_error(error: RetrievalError | None, operation: str) -> None:
    if error is None or error.code in {
        RetrievalErrorCode.EMPTY_RETRIEVAL,
        RetrievalErrorCode.INSUFFICIENT_RETRIEVAL,
    }:
        return
    if error.code is RetrievalErrorCode.PROTOCOL_ERROR:
        code = CollectionErrorCode.PROTOCOL_ERROR
    elif error.code is RetrievalErrorCode.RERANKER_UNAVAILABLE:
        code = CollectionErrorCode.DEPENDENCY_ERROR
    else:
        code = CollectionErrorCode.RETRIEVAL_ERROR
    raise EvidenceCollectionError(code, f"{operation}: {error.message}")


def _append_reason(reason_codes: tuple[str, ...], reason: str) -> tuple[str, ...]:
    return reason_codes if reason in reason_codes else (*reason_codes, reason)
