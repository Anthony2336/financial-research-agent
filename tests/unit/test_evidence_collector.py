"""Behavior tests for bounded deterministic P1 evidence collection."""

from collections.abc import Sequence
from datetime import UTC, date, datetime

import pytest

from financial_evidence_agent.context import (
    BudgetAuthority,
    BudgetExhaustedError,
    BudgetLimits,
)
from financial_evidence_agent.domain import (
    EvidenceChunk,
    Intent,
    ResearchQuestion,
    SourceKind,
    SourceTier,
    WebEvidence,
)
from financial_evidence_agent.retrieval.collector import (
    CollectionErrorCode,
    EvidenceCollectionError,
    EvidenceCollector,
    EvidenceQuery,
    LocalEvidenceHit,
    LocalSearchResponse,
    RetrievalError,
    RetrievalErrorCode,
    WebEvidenceHit,
    WebSearchRequest,
    WebSearchResponse,
    rewrite_query,
)
from financial_evidence_agent.retrieval.coverage import EvidenceSide
from financial_evidence_agent.skills.models import (
    RecipeBudget,
    ResearchFacet,
    ResearchRecipe,
    SkillName,
    WebUsagePolicy,
)


def _recipe(
    *facets: ResearchFacet,
    allow_web: bool = True,
    web_usage_policy: WebUsagePolicy = WebUsagePolicy.EVIDENCE,
    max_local_results: int = 5,
    max_web_results: int = 3,
) -> ResearchRecipe:
    allowed_tools = {"hybrid_search_filings"}
    source_policy = [SourceKind.FILING]
    max_web_calls = 0
    if allow_web:
        allowed_tools.add("search_allowlisted_web")
        source_policy.extend((SourceKind.ISSUER_IR, SourceKind.AUTHORITATIVE_WEB))
        max_web_calls = 1
    return ResearchRecipe(
        name=(
            SkillName.FINANCIAL_DATA_VERIFICATION
            if web_usage_policy is WebUsagePolicy.PRIMARY_SOURCE_LOCATOR
            else SkillName.EARNINGS_REVIEW
        ),
        version="test",
        accepted_intents=frozenset({Intent.EARNINGS_REVIEW_REQUEST}),
        allowed_tools=frozenset(allowed_tools),
        source_policy=tuple(source_policy),
        required_facets=facets or (ResearchFacet.EARNINGS_CHANGE,),
        budget=RecipeBudget(
            max_questions=3,
            max_local_results_per_query=max_local_results,
            max_retrieval_rounds=2,
            max_web_calls=max_web_calls,
            max_web_results=max_web_results if allow_web else 0,
            max_planner_output_tokens=350,
            max_analysis_output_tokens=1_200,
            max_repair_output_tokens=600,
            max_evidence_tokens=3_000,
        ),
        web_usage_policy=web_usage_policy if allow_web else WebUsagePolicy.NONE,
        input_schema="ResearchInput",
        output_schema="ResearchMemo",
        guard_profile="strict_citation",
    )


def _question() -> ResearchQuestion:
    return ResearchQuestion(
        question="What changed?",
        support_query="revenue growth",
        challenge_query="revenue headwinds",
        period="2025",
        forms=["10-Q"],
    )


def _chunk(chunk_id: str) -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        ticker="NVDA",
        corpus_version="fixture-v1",
        content=f"Evidence for {chunk_id}",
        source_url=f"https://www.sec.gov/Archives/{chunk_id}",
        form="10-Q",
        filed_at=date(2025, 5, 29),
        accession_no=f"000-{chunk_id}",
        section="MD&A",
        raw_start=0,
        raw_end=20,
    )


def _web(
    evidence_id: str,
    *,
    source_kind: SourceKind = SourceKind.ISSUER_IR,
    source_tier: SourceTier = SourceTier.PRIMARY,
) -> WebEvidence:
    return WebEvidence(
        id=evidence_id,
        ticker="NVDA",
        title="Quarterly disclosure",
        content="Revenue disclosure.",
        source_url=f"https://investor.nvidia.com/{evidence_id}",
        source_kind=source_kind,
        source_tier=source_tier,
        published_at=datetime(2025, 5, 29, tzinfo=UTC),
        fetched_at=datetime(2025, 5, 30, tzinfo=UTC),
        content_hash=f"sha256:{evidence_id}",
    )


def _local_hit(
    source_id: str,
    side: EvidenceSide,
    facet: ResearchFacet = ResearchFacet.EARNINGS_CHANGE,
) -> LocalEvidenceHit:
    return LocalEvidenceHit(
        evidence=_chunk(source_id),
        question_index=0,
        side=side,
        facet=facet,
    )


class SequenceLocalSearch:
    """One result per collector round, recording the real request batches."""

    def __init__(self, responses: Sequence[LocalSearchResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[EvidenceQuery, ...]] = []

    async def __call__(self, queries: tuple[EvidenceQuery, ...]) -> LocalSearchResponse:
        self.calls.append(queries)
        return self.responses.pop(0)


class SequenceWebSearch:
    """One typed web result per collector invocation."""

    def __init__(self, responses: Sequence[WebSearchResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[WebSearchRequest] = []

    async def __call__(self, request: WebSearchRequest) -> WebSearchResponse:
        self.calls.append(request)
        return self.responses.pop(0)


class TypoLocalEnvelopeSearch:
    """Model a provider adapter constructing a misspelled local error envelope."""

    def __init__(self) -> None:
        self.calls: list[tuple[EvidenceQuery, ...]] = []

    async def __call__(self, queries: tuple[EvidenceQuery, ...]) -> LocalSearchResponse:
        self.calls.append(queries)
        return LocalSearchResponse.model_validate(
            {
                "errror": {
                    "code": "protocol_error",
                    "message": "malformed upstream response",
                }
            }
        )


class TypoWebEnvelopeSearch:
    """Model a provider adapter constructing a misspelled web error envelope."""

    def __init__(self) -> None:
        self.calls: list[WebSearchRequest] = []

    async def __call__(self, request: WebSearchRequest) -> WebSearchResponse:
        self.calls.append(request)
        return WebSearchResponse.model_validate(
            {
                "errror": {
                    "code": "protocol_error",
                    "message": "malformed upstream response",
                }
            }
        )


def _web_hit(
    evidence_id: str,
    *,
    side: EvidenceSide,
    facet: ResearchFacet,
    source_kind: SourceKind = SourceKind.ISSUER_IR,
    source_tier: SourceTier = SourceTier.PRIMARY,
) -> WebEvidenceHit:
    return WebEvidenceHit(
        evidence=_web(evidence_id, source_kind=source_kind, source_tier=source_tier),
        question_index=0,
        side=side,
        facet=facet,
    )


def test_query_rewrite_is_deterministic_and_preserves_typed_scope() -> None:
    """Replacing the template with a model or omitting scope would change this exact output."""
    query = EvidenceQuery(
        ticker="NVDA",
        question_index=0,
        facet=ResearchFacet.EARNINGS_CHANGE,
        side=EvidenceSide.CHALLENGE,
        query="revenue headwinds",
        period="2025",
        forms=("10-Q",),
        limit=5,
    )

    first = rewrite_query(query)
    second = rewrite_query(query)

    assert first == second
    assert first.query == (
        "revenue headwinds | ticker=NVDA | facet=earnings_change | side=challenge "
        "| retry=missing_only"
    )


@pytest.mark.asyncio
async def test_collector_stops_after_a_complete_first_round() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit("support", EvidenceSide.SUPPORT),
                    _local_hit("challenge", EvidenceSide.CHALLENGE),
                )
            )
        ]
    )
    web = SequenceWebSearch([WebSearchResponse()])

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="nvda", recipe=_recipe(), questions=[_question()]
    )

    assert bundle.coverage.complete is True
    assert bundle.retrieval_rounds == 1
    assert bundle.web_calls == 0
    assert bundle.tool_calls == 2
    assert len(local.calls) == 1
    assert web.calls == []


@pytest.mark.asyncio
async def test_retrieval_round_is_gated_before_local_search_side_effect() -> None:
    local = SequenceLocalSearch([LocalSearchResponse()])
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=0,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=10,
            max_retrieval_rounds=0,
            max_web_calls=0,
        )
    )

    with pytest.raises(BudgetExhaustedError) as caught:
        await EvidenceCollector(
            local_search=local,
            budget_gate=authority,
        ).collect(ticker="NVDA", recipe=_recipe(), questions=[_question()])

    assert caught.value.dimension == "retrieval_rounds"
    assert local.calls == []
    assert authority.state.retrieval_rounds == 0


@pytest.mark.asyncio
async def test_collector_completes_after_one_missing_only_rewrite() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(_local_hit("support", EvidenceSide.SUPPORT),)
            ),
            LocalSearchResponse(
                evidence=(_local_hit("challenge", EvidenceSide.CHALLENGE),)
            ),
        ]
    )

    bundle = await EvidenceCollector(local_search=local).collect(
        ticker="NVDA", recipe=_recipe(), questions=[_question()]
    )

    assert bundle.coverage.complete is True
    assert bundle.retrieval_rounds == 2
    assert bundle.web_calls == 0
    assert bundle.tool_calls == sum(len(call) for call in local.calls)
    assert len(local.calls[1]) == 1
    assert local.calls[1][0].side is EvidenceSide.CHALLENGE
    assert local.calls[1][0].query.endswith("side=challenge | retry=missing_only")


@pytest.mark.asyncio
async def test_collector_uses_one_web_fallback_for_only_the_missing_facets() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit("support", EvidenceSide.SUPPORT),
                    _local_hit("challenge", EvidenceSide.CHALLENGE),
                )
            ),
            LocalSearchResponse(),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "guidance",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.GUIDANCE_AND_RISKS,
                    ),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=_recipe(
            ResearchFacet.EARNINGS_CHANGE,
            ResearchFacet.GUIDANCE_AND_RISKS,
        ),
        questions=[_question().model_copy(update={"forms": []})],
    )

    assert bundle.coverage.complete is True
    assert bundle.retrieval_rounds == 2
    assert bundle.web_calls == 1
    assert bundle.tool_calls == sum(len(call) for call in local.calls) + len(web.calls)
    assert web.calls[0].missing_facets == (ResearchFacet.GUIDANCE_AND_RISKS,)
    assert "earnings_change" not in web.calls[0].query
    assert "guidance_and_risks" in web.calls[0].query


@pytest.mark.asyncio
async def test_retry_improvement_still_uses_web_when_coverage_remains_incomplete() -> None:
    """One new local source must not suppress fallback for a still-missing facet."""
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit(
                        "earnings-support",
                        EvidenceSide.SUPPORT,
                        ResearchFacet.EARNINGS_CHANGE,
                    ),
                )
            ),
            LocalSearchResponse(
                evidence=(
                    _local_hit(
                        "earnings-challenge",
                        EvidenceSide.CHALLENGE,
                        ResearchFacet.EARNINGS_CHANGE,
                    ),
                )
            ),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "guidance-support",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.GUIDANCE_AND_RISKS,
                    ),
                    _web_hit(
                        "guidance-challenge",
                        side=EvidenceSide.CHALLENGE,
                        facet=ResearchFacet.GUIDANCE_AND_RISKS,
                    ),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=_recipe(
            ResearchFacet.EARNINGS_CHANGE,
            ResearchFacet.GUIDANCE_AND_RISKS,
        ),
        questions=[_question().model_copy(update={"forms": []})],
    )

    assert bundle.coverage.complete is True
    assert bundle.retrieval_rounds == 2
    assert bundle.web_calls == 1
    assert len(web.calls) == 1
    assert web.calls[0].missing_facets == (ResearchFacet.GUIDANCE_AND_RISKS,)


@pytest.mark.asyncio
async def test_web_budget_exhaustion_happens_before_fallback_side_effect() -> None:
    local = SequenceLocalSearch([LocalSearchResponse(), LocalSearchResponse()])
    web = SequenceWebSearch([WebSearchResponse()])
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=0,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=10,
            max_retrieval_rounds=2,
            max_web_calls=0,
        )
    )

    with pytest.raises(BudgetExhaustedError) as caught:
        await EvidenceCollector(
            local_search=local,
            web_search=web,
            budget_gate=authority,
        ).collect(ticker="NVDA", recipe=_recipe(), questions=[_question()])

    assert caught.value.dimension == "web_calls"
    assert web.calls == []


@pytest.mark.asyncio
async def test_collector_does_not_call_unauthorized_web_search() -> None:
    local = SequenceLocalSearch([LocalSearchResponse(), LocalSearchResponse()])
    web = SequenceWebSearch([WebSearchResponse()])

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA", recipe=_recipe(allow_web=False), questions=[_question()]
    )

    assert bundle.coverage.complete is False
    assert bundle.retrieval_rounds == 2
    assert bundle.web_calls == 0
    assert "insufficient_information" in bundle.coverage.reason_codes
    assert web.calls == []


@pytest.mark.asyncio
async def test_collector_fails_closed_on_typed_protocol_error() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                error=RetrievalError(
                    code=RetrievalErrorCode.PROTOCOL_ERROR,
                    message="malformed structured response",
                )
            )
        ]
    )
    web = SequenceWebSearch([WebSearchResponse()])

    with pytest.raises(EvidenceCollectionError) as caught:
        await EvidenceCollector(local_search=local, web_search=web).collect(
            ticker="NVDA", recipe=_recipe(), questions=[_question()]
        )

    assert caught.value.code is CollectionErrorCode.PROTOCOL_ERROR
    assert len(local.calls) == 1
    assert web.calls == []


@pytest.mark.asyncio
async def test_collector_preserves_reranker_dependency_failure() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                error=RetrievalError(
                    code=RetrievalErrorCode.RERANKER_UNAVAILABLE,
                    message=(
                        "RERANKER_MODEL_UNAVAILABLE: configured FlashRank assets "
                        "are unavailable"
                    ),
                )
            )
        ]
    )
    web = SequenceWebSearch([WebSearchResponse()])

    with pytest.raises(EvidenceCollectionError) as caught:
        await EvidenceCollector(local_search=local, web_search=web).collect(
            ticker="NVDA", recipe=_recipe(), questions=[_question()]
        )

    assert caught.value.code is CollectionErrorCode.DEPENDENCY_ERROR
    assert "RERANKER_MODEL_UNAVAILABLE" in caught.value.detail
    assert len(local.calls) == 1
    assert web.calls == []


@pytest.mark.asyncio
async def test_no_filings_application_scope_uses_one_authorized_web_fallback() -> None:
    """An empty filing scope is a quality gap, not a reason to skip bounded web fallback."""
    local = SequenceLocalSearch([LocalSearchResponse()])
    web = SequenceWebSearch([WebSearchResponse()])
    authority = BudgetAuthority()
    authority.configure(
        BudgetLimits(
            max_planner_calls=0,
            max_analysis_calls=0,
            max_repair_calls=0,
            max_tool_calls=1,
            max_retrieval_rounds=0,
            max_web_calls=1,
        )
    )
    collector = EvidenceCollector(local_search=local, web_search=web).with_filing_scope(
        None,
        (),
        scope_error="NO_FILINGS: filing scope could not be resolved",
    ).with_budget_gate(authority)

    bundle = await collector.collect(
        ticker="NVDA",
        recipe=_recipe(),
        questions=[_question()],
    )

    assert bundle.coverage.complete is False
    assert bundle.retrieval_rounds == 0
    assert bundle.web_calls == 1
    assert bundle.tool_calls == 1
    assert "insufficient_information" in bundle.coverage.reason_codes
    assert local.calls == []
    assert len(web.calls) == 1
    assert authority.state.web_calls == 1


@pytest.mark.asyncio
async def test_collector_fails_closed_on_unknown_local_envelope_fields() -> None:
    """A misspelled local error field must not degrade into empty retries or web."""
    local = TypoLocalEnvelopeSearch()
    web = SequenceWebSearch([WebSearchResponse()])

    with pytest.raises(EvidenceCollectionError) as caught:
        await EvidenceCollector(local_search=local, web_search=web).collect(
            ticker="NVDA", recipe=_recipe(), questions=[_question()]
        )

    assert caught.value.code is CollectionErrorCode.PROTOCOL_ERROR
    assert len(local.calls) == 1
    assert web.calls == []


@pytest.mark.asyncio
async def test_collector_fails_closed_on_unknown_web_envelope_fields() -> None:
    """A misspelled web error field must not become an empty successful fallback."""
    local = SequenceLocalSearch([LocalSearchResponse(), LocalSearchResponse()])
    web = TypoWebEnvelopeSearch()

    with pytest.raises(EvidenceCollectionError) as caught:
        await EvidenceCollector(local_search=local, web_search=web).collect(
            ticker="NVDA", recipe=_recipe(), questions=[_question()]
        )

    assert caught.value.code is CollectionErrorCode.PROTOCOL_ERROR
    assert len(local.calls) == 2
    assert len(web.calls) == 1


@pytest.mark.asyncio
async def test_collector_hard_caps_two_empty_local_rounds_and_one_empty_web_call() -> None:
    local = SequenceLocalSearch([LocalSearchResponse(), LocalSearchResponse()])
    web = SequenceWebSearch([WebSearchResponse()])

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA", recipe=_recipe(), questions=[_question()]
    )

    assert len(local.calls) == bundle.retrieval_rounds == 2
    assert len(web.calls) == bundle.web_calls == 1
    assert bundle.coverage.complete is False
    assert "zero_new_evidence" in bundle.coverage.reason_codes
    assert "insufficient_information" in bundle.coverage.reason_codes


@pytest.mark.asyncio
async def test_primary_source_locator_rejects_secondary_web_evidence() -> None:
    local = SequenceLocalSearch([LocalSearchResponse(), LocalSearchResponse()])
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "secondary",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.DATA_VERIFICATION,
                        source_kind=SourceKind.AUTHORITATIVE_WEB,
                        source_tier=SourceTier.AUTHORITATIVE_SECONDARY,
                    ),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=_recipe(
            ResearchFacet.DATA_VERIFICATION,
            allow_web=True,
            web_usage_policy=WebUsagePolicy.PRIMARY_SOURCE_LOCATOR,
        ),
        questions=[_question()],
    )

    assert bundle.coverage.complete is False
    assert bundle.web_evidence == []
    assert bundle.coverage.invalid_source_ids == ("secondary",)
    assert "insufficient_information" in bundle.coverage.reason_codes


@pytest.mark.asyncio
async def test_primary_source_locator_rejects_authoritative_web_even_if_mistyped_primary() -> None:
    """A secondary source category must not become financial evidence via a tier typo."""
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit(
                        "challenge",
                        EvidenceSide.CHALLENGE,
                        ResearchFacet.DATA_VERIFICATION,
                    ),
                )
            ),
            LocalSearchResponse(),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "mistyped-primary",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.DATA_VERIFICATION,
                        source_kind=SourceKind.AUTHORITATIVE_WEB,
                        source_tier=SourceTier.PRIMARY,
                    ),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=_recipe(
            ResearchFacet.DATA_VERIFICATION,
            web_usage_policy=WebUsagePolicy.PRIMARY_SOURCE_LOCATOR,
        ),
        questions=[_question()],
    )

    assert bundle.coverage.complete is False
    assert bundle.coverage.invalid_source_ids == ("mistyped-primary",)
    assert bundle.web_evidence == []


@pytest.mark.asyncio
async def test_primary_source_locator_can_use_web_to_find_an_allowed_filing() -> None:
    """A filing-only source policy still permits the web tool to locate an SEC disclosure."""
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit(
                        "challenge",
                        EvidenceSide.CHALLENGE,
                        ResearchFacet.DATA_VERIFICATION,
                    ),
                )
            ),
            LocalSearchResponse(),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "located-filing",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.DATA_VERIFICATION,
                        source_kind=SourceKind.FILING,
                        source_tier=SourceTier.PRIMARY,
                    ),
                )
            )
        ]
    )
    recipe = _recipe(
        ResearchFacet.DATA_VERIFICATION,
        web_usage_policy=WebUsagePolicy.PRIMARY_SOURCE_LOCATOR,
    ).model_copy(update={"source_policy": (SourceKind.FILING,)})

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=recipe,
        questions=[_question().model_copy(update={"forms": []})],
    )

    assert bundle.coverage.complete is True
    assert bundle.web_calls == 1
    assert [evidence.id for evidence in bundle.web_evidence] == ["located-filing"]


@pytest.mark.asyncio
async def test_web_filing_without_typed_form_cannot_satisfy_a_form_constraint() -> None:
    """WebEvidence has no form field, so a filing label cannot prove a 10-Q match."""
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit(
                        "challenge",
                        EvidenceSide.CHALLENGE,
                        ResearchFacet.DATA_VERIFICATION,
                    ),
                )
            ),
            LocalSearchResponse(),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "formless-filing",
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.DATA_VERIFICATION,
                        source_kind=SourceKind.FILING,
                        source_tier=SourceTier.PRIMARY,
                    ),
                )
            )
        ]
    )
    recipe = _recipe(
        ResearchFacet.DATA_VERIFICATION,
        web_usage_policy=WebUsagePolicy.PRIMARY_SOURCE_LOCATOR,
    ).model_copy(update={"source_policy": (SourceKind.FILING,)})

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=recipe,
        questions=[_question()],
    )

    assert bundle.coverage.complete is False
    assert bundle.coverage.missing_pairs == ((0, EvidenceSide.SUPPORT),)
    assert bundle.coverage.date_mismatches == ("formless-filing",)
    assert bundle.web_evidence == []


@pytest.mark.asyncio
async def test_collector_caps_results_returned_beyond_recipe_budgets() -> None:
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    _local_hit("support-first", EvidenceSide.SUPPORT),
                    _local_hit("support-overflow", EvidenceSide.SUPPORT),
                    _local_hit("challenge-first", EvidenceSide.CHALLENGE),
                    _local_hit("challenge-overflow", EvidenceSide.CHALLENGE),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local).collect(
        ticker="NVDA",
        recipe=_recipe(allow_web=False, max_local_results=1),
        questions=[_question()],
    )

    assert [item.id for item in bundle.filing_evidence] == [
        "support-first",
        "challenge-first",
    ]


@pytest.mark.asyncio
async def test_bundle_excludes_an_invalid_binding_for_an_otherwise_valid_source() -> None:
    """A source valid for one question must not retain a date-invalid second assignment."""
    shared = _chunk("shared")
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(
                    LocalEvidenceHit(
                        evidence=shared,
                        question_index=0,
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.EARNINGS_CHANGE,
                    ),
                    LocalEvidenceHit(
                        evidence=shared,
                        question_index=1,
                        side=EvidenceSide.SUPPORT,
                        facet=ResearchFacet.EARNINGS_CHANGE,
                    ),
                    _local_hit("challenge", EvidenceSide.CHALLENGE),
                )
            ),
            LocalSearchResponse(),
        ]
    )
    older_question = _question().model_copy(update={"period": "2024"})

    bundle = await EvidenceCollector(local_search=local).collect(
        ticker="NVDA",
        recipe=_recipe(allow_web=False),
        questions=[_question(), older_question],
    )

    assert [(item.question_index, item.source_id) for item in bundle.assignments] == [
        (0, "shared"),
        (0, "challenge"),
    ]
    assert bundle.coverage.date_mismatches == ("shared",)


@pytest.mark.asyncio
async def test_web_fallback_rejects_an_unplanned_facet_for_a_missing_pair() -> None:
    """An arbitrary web facet must not mark a missing challenge pair complete."""
    local = SequenceLocalSearch(
        [
            LocalSearchResponse(
                evidence=(_local_hit("support", EvidenceSide.SUPPORT),)
            ),
            LocalSearchResponse(),
        ]
    )
    web = SequenceWebSearch(
        [
            WebSearchResponse(
                evidence=(
                    _web_hit(
                        "wrong-facet",
                        side=EvidenceSide.CHALLENGE,
                        facet=ResearchFacet.GUIDANCE_AND_RISKS,
                    ),
                )
            )
        ]
    )

    bundle = await EvidenceCollector(local_search=local, web_search=web).collect(
        ticker="NVDA",
        recipe=_recipe(ResearchFacet.EARNINGS_CHANGE),
        questions=[_question()],
    )

    assert bundle.coverage.complete is False
    assert bundle.coverage.missing_pairs == ((0, EvidenceSide.CHALLENGE),)
    assert bundle.web_evidence == []
    assert [(item.side, item.source_id) for item in bundle.assignments] == [
        (EvidenceSide.SUPPORT, "support")
    ]
