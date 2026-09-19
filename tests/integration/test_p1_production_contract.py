"""Production-composed P1 run correlation and observation contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import fra.bootstrap as bootstrap_module
import fra.observability as observability_module
from fra.config import Settings
from fra.contracts import ResearchCommand, ResearchMode
from fra.domain import (
    Claim,
    ClaimKind,
    Confidence,
    ResearchMemo,
    ResearchQuestion,
)
from fra.execution import current_research_context
from fra.model_providers.openai import (
    ResearchQuestionPlan,
    ThesisResearchQuestionPlan,
)
from fra.prompts import RESEARCH_PROMPT_VERSION
from fra.retrieval.indexing import HashEmbeddingProvider
from fra.retrieval.ingest import ingest_fixture
from fra.retrieval.rerank import IdentityReranker
from fra.skills.schemas import (
    InformationSufficiency,
    SkillResearchMemo,
    SkillResearchSection,
)
from fra.storage.cache import InMemoryTtlJsonCache
from fra.storage.database import create_schema
from fra.storage.models import ResearchRun, SkillRun
from fra.storage.repositories import FilingRepository
from fra.storage.run_repositories import ResearchRunRepository


class ExportedObservation:
    def __init__(
        self,
        *,
        name: str,
        kind: str,
        metadata: Mapping[str, object] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.metadata = dict(metadata or {})
        self.trace_id = trace_id
        self.output: object | None = None
        self.children: list[ExportedObservation] = []
        self.active = False

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        assert self.active, f"observation updated after close: {self.name}"
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata.update(metadata)

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: str,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        del input
        child = ExportedObservation(name=name, kind=kind, metadata=metadata)
        self.children.append(child)
        child.active = True
        try:
            yield child
        finally:
            child.active = False


class ExportManager(AbstractContextManager[ExportedObservation]):
    def __init__(self, client: FakeLangfuseClient, value: ExportedObservation) -> None:
        self.client = client
        self.value = value

    def __enter__(self) -> ExportedObservation:
        if self.client.stack:
            self.client.stack[-1].children.append(self.value)
        else:
            self.client.roots.append(self.value)
        self.client.stack.append(self.value)
        self.value.active = True
        return self.value

    def __exit__(self, *args: object) -> None:
        del args
        assert self.client.stack.pop() is self.value
        self.value.active = False


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.roots: list[ExportedObservation] = []
        self.stack: list[ExportedObservation] = []
        self.flush_calls = 0

    def start_as_current_observation(self, **values: object) -> ExportManager:
        return ExportManager(
            self,
            ExportedObservation(
                name=str(values["name"]),
                kind=str(values["as_type"]),
                metadata=values.get("metadata"),  # type: ignore[arg-type]
                trace_id="langfuse-trace-p1" if not self.stack else None,
            ),
        )

    def flush(self) -> None:
        self.flush_calls += 1


def _task_payload(messages: list[BaseMessage]) -> dict[str, object]:
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[1].content, str)
    matched = re.search(r"<task_data>(.*)</task_data>", messages[1].content, re.DOTALL)
    assert matched is not None
    return json.loads(matched.group(1))


class RecordingStructuredChat:
    def __init__(self, model: str, max_completion_tokens: int) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.schema: type[object] | None = None
        self.requests: list[dict[str, object]] = []
        self.messages: list[list[BaseMessage]] = []
        self.token_counter = lambda value: len(value.split())

    def with_structured_output(
        self,
        schema: type[object],
        *,
        strict: bool,
        include_raw: bool,
    ) -> RecordingStructuredChat:
        assert strict is True
        assert include_raw is True
        self.schema = schema
        return self

    async def ainvoke(self, prompt: list[BaseMessage]) -> dict[str, object | None]:
        payload = _task_payload(prompt)
        self.requests.append(payload)
        self.messages.append(prompt)
        if self.schema is ResearchQuestionPlan:
            parsed: object = ResearchQuestionPlan(
                questions=[
                    ResearchQuestion(
                        question="Review disclosed performance and risks.",
                        support_query="reported revenue growth and operating strengths",
                        challenge_query="reported risks challenges and operating weaknesses",
                    )
                ]
            )
        else:
            constraints = payload["frozen_recipe_constraints"]
            source_id = payload["allowed_filing_source_ids"][0]
            parsed = SkillResearchMemo(
                recipe_name=constraints["recipe_name"],
                recipe_version=constraints["recipe_version"],
                research_question="Review disclosed performance and risks.",
                sections=[
                    SkillResearchSection(
                        facet=facet,
                        claims=[
                            Claim(
                                kind=ClaimKind.VERIFIED_FACT,
                                text=f"The filing supports the {facet} review.",
                                confidence=Confidence.HIGH,
                                evidence_chunk_ids=[source_id],
                            )
                        ],
                    )
                    for facet in constraints["required_facets"]
                ],
                information_sufficiency=InformationSufficiency.SUFFICIENT,
                information_gaps=[],
                confidence=Decimal("0.9"),
            )
        raw = AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 13,
                "output_tokens": 8,
                "total_tokens": 21,
            },
            response_metadata={
                "total_cost": 0.001,
                "provider_payload": "must-not-export",
            },
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


def _descendants(root: ExportedObservation) -> list[ExportedObservation]:
    values: list[ExportedObservation] = []
    pending = list(root.children)
    while pending:
        value = pending.pop(0)
        values.append(value)
        pending.extend(value.children)
    return values


def _seed_fixture(database_url: str, *, ticker: str = "NVDA") -> None:
    engine = create_engine(database_url)
    create_schema(engine)
    ingest_fixture(
        Path("tests/fixtures/nvda_10q.html"),
        ticker,
        "10-Q",
        FilingRepository(engine),
    )


class ThesisStructuredChat:
    def __init__(self, model: str, max_completion_tokens: int) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.schema: type[object] | None = None
        self.requests: list[dict[str, object]] = []
        self.messages: list[list[BaseMessage]] = []
        self.token_counter = lambda value: len(value.split())

    def with_structured_output(
        self,
        schema: type[object],
        *,
        strict: bool,
        include_raw: bool,
    ) -> ThesisStructuredChat:
        assert strict is True
        assert include_raw is True
        self.schema = schema
        return self

    def invoke(self, prompt: list[BaseMessage]) -> dict[str, object | None]:
        payload = _task_payload(prompt)
        self.requests.append(payload)
        self.messages.append(prompt)
        if self.schema is ThesisResearchQuestionPlan:
            parsed: object = ThesisResearchQuestionPlan(
                questions=[
                    ResearchQuestion(
                        question="Does reported demand support growth despite risks?",
                        support_query="reported demand revenue growth",
                        challenge_query="reported demand revenue concentration risks",
                    )
                ]
            )
        else:
            source_id = payload["allowed_filing_source_ids"][0]
            parsed = ResearchMemo(
                research_question="Does reported demand support growth despite risks?",
                supporting_claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text="The filing contains supporting demand evidence.",
                        confidence=Confidence.HIGH,
                        evidence_chunk_ids=[source_id],
                    )
                ],
                counter_claims=[
                    Claim(
                        kind=ClaimKind.VERIFIED_FACT,
                        text="The filing contains counter-evidence and risks.",
                        confidence=Confidence.MEDIUM,
                        evidence_chunk_ids=[source_id],
                    )
                ],
                information_sufficiency="B",
                confidence=Confidence.MEDIUM,
            )
        return {
            "raw": AIMessage(content=""),
            "parsed": parsed,
            "parsing_error": None,
        }


def test_production_composition_runs_default_thesis_without_demo_substitution(
    monkeypatch,
    tmp_path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'thesis-runs.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        redis_url="redis://cache.example:6379/0",
        offline_demo=False,
        fast_model="fast-test",
        analyst_model="analyst-test",
        openai_api_key=SecretStr("provider-secret"),
        tavily_api_key=None,
        _env_file=None,
    )
    chats: list[ThesisStructuredChat] = []

    def chat_factory(**values: object) -> ThesisStructuredChat:
        chat = ThesisStructuredChat(
            str(values["model"]),
            int(values["max_completion_tokens"]),
        )
        chats.append(chat)
        return chat

    monkeypatch.setattr("langchain_openai.ChatOpenAI", chat_factory)
    monkeypatch.setattr(
        bootstrap_module,
        "BgeM3EmbeddingProvider",
        lambda model_name, *, cache_dir=None: HashEmbeddingProvider(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "LazyFlashRankReranker",
        lambda model_name, *, cache_dir=None, ranker_factory=None, asset_validator=None: (
            IdentityReranker()
        ),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_cache",
        lambda settings: InMemoryTtlJsonCache(),
    )

    _seed_fixture(database_url)
    result = bootstrap_module.build_research_application(settings).run(
        ResearchCommand(
            ticker="NVDA",
            request="Does reported demand support revenue growth despite disclosed risks?",
            mode=ResearchMode.THESIS,
        )
    )
    stored = ResearchRunRepository(create_engine(database_url)).get(result.run_id)

    assert result.status == "completed"
    assert result.skill_runs == []
    assert result.guarded_memo is not None
    assert result.guarded_memo.sources
    assert "deterministic offline demo" not in result.rendered_output
    assert (
        "Execution mode: production thesis "
        "(allowlisted web fallback unavailable; local evidence only)" in result.rendered_output
    )
    assert [chat.model for chat in chats] == ["fast-test", "analyst-test"]
    assert [chat.max_completion_tokens for chat in chats] == [350, 1_200]
    assert all(
        isinstance(chat.messages[0][0], SystemMessage)
        and isinstance(chat.messages[0][1], HumanMessage)
        for chat in chats
    )
    assert stored.status == "completed"
    assert stored.prompt_version == RESEARCH_PROMPT_VERSION
    assert stored.source_fetches
    assert all(fetch.run_id == result.run_id for fetch in stored.source_fetches)
    assert result.node_trace == [
        "safety_router",
        "normalize",
        "load_session_memory",
        "load_research_memory",
        "plan_questions",
        "retrieve_evidence",
        "evidence_coverage",
        "analyze",
        "citation_guard",
        "render",
    ]


def test_production_composition_correlates_p1_persistence_and_observation_tree(
    monkeypatch,
    tmp_path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'p1-runs.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        redis_url=None,
        fast_model="fast-test",
        analyst_model="analyst-test",
        openai_api_key=SecretStr("provider-secret"),
        tavily_api_key=None,
        langfuse_public_key=SecretStr("langfuse-public"),
        langfuse_secret_key=SecretStr("langfuse-secret"),
        langfuse_host="https://langfuse.invalid",
        _env_file=None,
    )
    exporter = FakeLangfuseClient()
    chats: list[RecordingStructuredChat] = []

    def chat_factory(**values: object) -> RecordingStructuredChat:
        chat = RecordingStructuredChat(
            str(values["model"]),
            int(values["max_completion_tokens"]),
        )
        chats.append(chat)
        return chat

    monkeypatch.setattr("langchain_openai.ChatOpenAI", chat_factory)
    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: SimpleNamespace(Langfuse=lambda **values: exporter),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "BgeM3EmbeddingProvider",
        lambda model_name, *, cache_dir=None: HashEmbeddingProvider(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "LazyFlashRankReranker",
        lambda model_name, *, cache_dir=None, ranker_factory=None, asset_validator=None: (
            IdentityReranker()
        ),
    )

    _seed_fixture(database_url)
    application = bootstrap_module.build_research_application(settings)
    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="介绍一下这家公司",
            mode=ResearchMode.COMPANY_PROFILE,
        )
    )
    stored = ResearchRunRepository(create_engine(database_url)).get(result.run_id)

    assert result.status == "completed"
    assert len(exporter.roots) == 1
    root = exporter.roots[0]
    observations = _descendants(root)
    assert root.name == "financial-research-agent.run"
    assert root.output == {"status": result.status}
    assert root.metadata["effective_intent"] == "company_profile_request"
    assert root.metadata["recipe_names"] == [run.recipe_name.value for run in result.skill_runs]
    assert root.metadata["recipe_versions"] == [run.recipe_version for run in result.skill_runs]
    assert root.metadata["corpus_scope"]
    assert root.metadata["source_policy_version"] == result.source_policy_version
    assert stored.run_id == result.run_id
    assert stored.trace_id == root.trace_id == "langfuse-trace-p1"
    assert stored.status == result.status
    assert stored.prompt_version == RESEARCH_PROMPT_VERSION
    assert stored.report_markdown == result.rendered_output
    assert stored.source_fetches
    assert all(fetch.run_id == result.run_id for fetch in stored.source_fetches)
    assert all("sha256:" in fetch.source_ref for fetch in stored.source_fetches)

    runtime = next(item for item in observations if item.name == "runtime.execute")
    assert runtime.output == {"status": result.status}
    guards = [item for item in observations if item.name == "p1.output_guard"]
    skill_renders = [item for item in observations if item.name == "p1.render_skill"]
    final_render = next(item for item in observations if item.name == "p1.render_final")
    assert guards and all(item.kind == "guardrail" for item in guards)
    assert all(item.output["status"] == "completed" for item in guards)  # type: ignore[index]
    assert skill_renders and all(item.kind == "chain" for item in skill_renders)
    assert all(item.output["rendered"] is True for item in skill_renders)  # type: ignore[index]
    assert final_render.kind == "chain"
    assert final_render.output["rendered_length"] > 0  # type: ignore[index]
    tools = [item for item in observations if item.name.startswith("mcp.")]
    assert tools
    assert all(item.output is not None for item in tools)
    assert all("result_count" in item.output for item in tools)  # type: ignore[operator]
    filing_tools = [item for item in tools if item.name == "mcp.hybrid_search_filings"]
    assert filing_tools
    assert all("cache_hit" in item.metadata for item in filing_tools)
    retrievers = [item for item in observations if item.name == "retrieval.hybrid"]
    assert retrievers
    assert all(item.metadata["fused_ids"] for item in retrievers)
    assert all("retained_scores" in item.metadata for item in retrievers)
    generations = [item for item in observations if item.kind == "generation"]
    model_generations = [item for item in generations if item.name.startswith("model.")]
    assert model_generations
    assert all(
        item.metadata["prompt_version"] == RESEARCH_PROMPT_VERSION for item in model_generations
    )
    assert all(item.metadata["usage"]["total_tokens"] == 21 for item in model_generations)  # type: ignore[index]
    assert all(item.metadata["cost"] == 0.001 for item in model_generations)
    assert exporter.flush_calls == 1
    exported = repr(exporter.roots)
    assert "provider-secret" not in exported
    assert "langfuse-secret" not in exported
    assert "must-not-export" not in exported
    assert chats
    assert {chat.max_completion_tokens for chat in chats} == {350, 1_200}
    assert all(
        isinstance(messages[0], SystemMessage) and isinstance(messages[1], HumanMessage)
        for chat in chats
        for messages in chat.messages
    )
    budgets = [item for item in observations if item.name == "p1.budget"]
    assert budgets
    assert all(item.output["planner_calls"] == 1 for item in budgets)  # type: ignore[index]
    assert all(item.output["analysis_calls"] <= 1 for item in budgets)  # type: ignore[index]
    assert all(item.output["retrieval_rounds"] <= 2 for item in budgets)  # type: ignore[index]
    assert all(item.output["web_calls"] <= 1 for item in budgets)  # type: ignore[index]
    assert sum(item.output["tool_calls"] for item in budgets) == len(  # type: ignore[index]
        stored.source_fetches
    )


def test_production_composition_routes_explicit_industry_mode_through_the_industry_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'industry-runs.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        redis_url=None,
        fast_model="fast-test",
        analyst_model="analyst-test",
        openai_api_key=SecretStr("provider-secret"),
        tavily_api_key=None,
        langfuse_public_key=SecretStr("langfuse-public"),
        langfuse_secret_key=SecretStr("langfuse-secret"),
        langfuse_host="https://langfuse.invalid",
        _env_file=None,
    )
    exporter = FakeLangfuseClient()
    chats: list[RecordingStructuredChat] = []

    def chat_factory(**values: object) -> RecordingStructuredChat:
        chat = RecordingStructuredChat(
            str(values["model"]),
            int(values["max_completion_tokens"]),
        )
        chats.append(chat)
        return chat

    monkeypatch.setattr("langchain_openai.ChatOpenAI", chat_factory)
    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: SimpleNamespace(Langfuse=lambda **values: exporter),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "BgeM3EmbeddingProvider",
        lambda model_name, *, cache_dir=None: HashEmbeddingProvider(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "LazyFlashRankReranker",
        lambda model_name, *, cache_dir=None, ranker_factory=None, asset_validator=None: (
            IdentityReranker()
        ),
    )

    _seed_fixture(database_url)
    application = bootstrap_module.build_research_application(settings)
    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="Describe the accelerator industry",
            mode=ResearchMode.INDUSTRY_RESEARCH,
        )
    )
    stored = ResearchRunRepository(create_engine(database_url)).get(result.run_id)

    assert result.status == "completed"
    assert result.decision.intent.value == "industry_research_request"
    assert [run.recipe_name.value for run in result.skill_runs] == ["industry_research"]
    assert result.rendered_output.startswith(
        "> Information gap: allowlisted web fallback unavailable; local evidence only."
    )
    assert stored.status == "completed"
    assert stored.report_markdown == result.rendered_output
    assert len(chats) == 2
    root = exporter.roots[0]
    assert root.metadata["effective_intent"] == "industry_research_request"
    assert root.metadata["recipe_names"] == ["industry_research"]


class PeerTraceSink:
    def __init__(self, roots: list[ExportedObservation], flush_calls: list[int]) -> None:
        self._roots = roots
        self._flush_calls = flush_calls

    def run(self, *, run_id: str, input: object, metadata: Mapping[str, object]):
        del input
        root = ExportedObservation(
            name="financial-research-agent.run",
            kind="agent",
            metadata={**dict(metadata), "run_id": run_id},
            trace_id=f"trace-{run_id}",
        )
        return ExportManager(_PeerTraceClient(self._roots, self._flush_calls), root)

    def flush(self) -> None:
        self._flush_calls.append(1)

    def record(self, node: str, attributes: Mapping[str, object]) -> None:
        del node, attributes


class _PeerTraceClient:
    def __init__(self, roots: list[ExportedObservation], flush_calls: list[int]) -> None:
        self.roots = roots
        self.stack: list[ExportedObservation] = []
        self._flush_calls = flush_calls


def test_production_peer_mode_runs_child_industry_requests_through_full_application_lifecycle(
    monkeypatch,
    tmp_path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'industry-peer-runs.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        redis_url=None,
        fast_model="fast-test",
        analyst_model="analyst-test",
        openai_api_key=SecretStr("provider-secret"),
        tavily_api_key=None,
        langfuse_public_key=None,
        langfuse_secret_key=None,
        langfuse_host=None,
        _env_file=None,
    )
    roots: list[ExportedObservation] = []
    flush_calls: list[int] = []
    chats: list[RecordingStructuredChat] = []

    def chat_factory(**values: object) -> RecordingStructuredChat:
        chat = RecordingStructuredChat(
            str(values["model"]),
            int(values["max_completion_tokens"]),
        )
        chats.append(chat)
        return chat

    monkeypatch.setattr("langchain_openai.ChatOpenAI", chat_factory)
    monkeypatch.setattr(
        bootstrap_module,
        "build_trace_sink",
        lambda settings: PeerTraceSink(roots, flush_calls),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "BgeM3EmbeddingProvider",
        lambda model_name, *, cache_dir=None: HashEmbeddingProvider(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "LazyFlashRankReranker",
        lambda model_name, *, cache_dir=None, ranker_factory=None, asset_validator=None: (
            IdentityReranker()
        ),
    )

    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    fixture = Path("tests/fixtures/nvda_10q.html")
    ingest_fixture(fixture, "NVDA", "10-Q", repository)
    ingest_fixture(fixture, "AMD", "10-Q", repository)

    application = bootstrap_module.build_research_application(settings)
    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="Compare direct peers using exact reported metrics.",
            mode=ResearchMode.INDUSTRY_RESEARCH,
            peer_tickers=("AMD",),
            peer_scope="US semiconductors",
        )
    )

    assert result.status == "completed"
    assert result.run_id
    assert result.rendered_output.startswith(
        "> Information gap: allowlisted web fallback unavailable; local evidence only."
    )
    assert (
        result.rendered_output.count("allowlisted web fallback unavailable; local evidence only")
        == 1
    )
    assert len(roots) == 3
    assert len({root.trace_id for root in roots}) == 3
    roots_by_run_id = {str(root.metadata["run_id"]): root for root in roots}
    assert result.run_id in roots_by_run_id

    run_repository = ResearchRunRepository(create_engine(database_url))
    outer_stored = run_repository.get(result.run_id)
    child_run_ids = [run_id for run_id in roots_by_run_id if run_id != result.run_id]
    assert len(child_run_ids) == 2
    child_runs = [run_repository.get(run_id) for run_id in child_run_ids]

    assert outer_stored.trace_id == roots_by_run_id[result.run_id].trace_id
    assert outer_stored.status == "completed"
    assert outer_stored.source_fetches == []
    assert outer_stored.report_markdown == result.rendered_output
    assert all(run.trace_id == roots_by_run_id[run.run_id].trace_id for run in child_runs)
    assert all(run.status == "completed" for run in child_runs)
    assert all(run.effective_intent == "industry_research_request" for run in child_runs)
    assert all(run.source_fetches for run in child_runs)
    assert all(
        all(fetch.run_id == run.run_id for fetch in run.source_fetches) for run in child_runs
    )
    assert current_research_context() is None
    assert len(chats) == 4
    assert len(flush_calls) == 3

    with Session(engine) as session:
        research_runs = session.scalars(select(ResearchRun)).all()
        skill_runs = session.scalars(select(SkillRun)).all()

    assert len(research_runs) == 3
    assert {run.ticker for run in research_runs} == {"NVDA", "AMD"}
    assert len(skill_runs) == 2
    assert {run.ticker for run in skill_runs} == {"NVDA", "AMD"}
    assert all(run.recipe_name == "industry_research" for run in skill_runs)
    assert all(run.recipe_snapshot["name"] == "industry_research" for run in skill_runs)
