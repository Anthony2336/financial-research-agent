"""Application-level correlation and observation-tree contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from hashlib import sha256
from typing import Any

import pytest

import fra.application as application_module
from fra.application import ResearchApplication
from fra.bootstrap import ResearchRuntime
from fra.context import BudgetLimits
from fra.contracts import ResearchCommand, ResearchMode
from fra.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
)
from fra.execution import current_research_context
from fra.graph.models import Dependencies
from fra.observability import (
    LangfuseTraceSink,
    RunAccounting,
    bind_trace_run,
)
from fra.safety.router import REFUSAL_TEXT
from fra.storage.run_repositories import RunFinish


@dataclass
class FakeObservation:
    name: str
    kind: str
    input: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    output: object | None = None
    children: list[FakeObservation] = field(default_factory=list)
    trace_id: str | None = None
    active: bool = False
    update_calls: int = 0

    def update(
        self,
        *,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        assert self.active, f"observation updated after close: {self.name}"
        self.update_calls += 1
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
        metadata: dict[str, object] | None = None,
    ) -> Iterator[FakeObservation]:
        child = FakeObservation(
            name=name,
            kind=kind,
            input=input,
            metadata=dict(metadata or {}),
        )
        self.children.append(child)
        child.active = True
        try:
            yield child
        finally:
            child.active = False


class FakeTraceSink:
    def __init__(self) -> None:
        self.roots: list[FakeObservation] = []
        self.flush_calls = 0

    @contextmanager
    def run(
        self,
        *,
        run_id: str,
        input: object,
        metadata: dict[str, object],
    ) -> Iterator[FakeObservation]:
        root = FakeObservation(
            name="financial-research-agent.run",
            kind="agent",
            input=input,
            metadata=dict(metadata),
            trace_id=f"trace-{run_id}",
        )
        self.roots.append(root)
        root.active = True
        try:
            yield root
        finally:
            root.active = False

    def flush(self) -> None:
        self.flush_calls += 1

    def single_root(self) -> FakeObservation:
        assert len(self.roots) == 1
        return self.roots[0]


class MemoryRunRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started: list[object] = []
        self.finished: list[object] = []
        self.fetches: list[object] = []
        self.start_calls = 0
        self.finish_calls = 0
        self.attempted_starts: list[object] = []

    def start(self, value: object) -> None:
        self.start_calls += 1
        self.attempted_starts.append(value)
        if self.fail:
            raise RuntimeError("database secret must not escape")
        self.started.append(value)

    def finish(self, value: object) -> None:
        self.finish_calls += 1
        if self.fail:
            raise RuntimeError("database secret must not escape")
        self.finished.append(value)

    def record_fetch(self, value: object) -> None:
        self.fetches.append(value)


class FixedIds:
    def new_run_id(self) -> str:
        return "run-123"


class NeverRouter:
    def route(self, request: str) -> RouterDecision:
        raise AssertionError(f"router called with {request}")


class DeterministicCompanyResolver:
    def __init__(self, *supported: str) -> None:
        self._supported = frozenset(supported)

    def resolve(self, ticker: str) -> str | None:
        normalized = ticker.strip().upper()
        return normalized if normalized in self._supported else None


class RuntimeFactory:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime
        self.calls = 0

    def build(self, command: ResearchCommand, intent: Intent) -> object:
        del command, intent
        self.calls += 1
        return self.runtime


@dataclass
class CallRecorder:
    model_calls: int = 0
    tool_calls: int = 0


class FastModel:
    def __init__(self, recorder: CallRecorder) -> None:
        self.recorder = recorder

    def route(self, thesis: str) -> RouterDecision:
        raise AssertionError(f"unexpected route: {thesis}")

    def plan(self, ticker: str, thesis: str) -> list[ResearchQuestion]:
        del ticker, thesis
        self.recorder.model_calls += 1
        return [
            ResearchQuestion(
                question="What supports and challenges demand?",
                support_query="secret support query",
                challenge_query="secret challenge query",
            )
        ]


class Analyst:
    def __init__(self, recorder: CallRecorder) -> None:
        self.recorder = recorder

    def analyze(
        self,
        questions: list[ResearchQuestion],
        evidence: list[EvidenceChunk],
    ) -> ResearchMemo:
        self.recorder.model_calls += 1
        claim = Claim(
            kind=ClaimKind.VERIFIED_FACT,
            text="Data center demand remained strong.",
            confidence=Confidence.HIGH,
            evidence_chunk_ids=[evidence[0].id],
        )
        return ResearchMemo(
            research_question=questions[0].question,
            supporting_claims=[claim],
            counter_claims=[claim],
            information_sufficiency="B",
            confidence=Confidence.MEDIUM,
        )


class MCPClient:
    def __init__(self, recorder: CallRecorder) -> None:
        self.recorder = recorder

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        del name, arguments
        self.recorder.tool_calls += 1
        return {
            "chunks": [
                EvidenceChunk(
                    id="chunk-1",
                    ticker="NVDA",
                    corpus_version="NVDA-v1",
                    content="Data center demand remained strong.",
                    source_url="https://www.sec.gov/Archives/example.htm",
                    form="10-Q",
                    filed_at=date(2026, 5, 20),
                    accession_no="0001045810-26-000001",
                    section="MD&A",
                    raw_start=0,
                    raw_end=35,
                )
            ],
            "error": None,
        }


def _application(
    *,
    trace_sink: FakeTraceSink,
    repository: MemoryRunRepository,
) -> tuple[ResearchApplication, CallRecorder, RuntimeFactory]:
    recorder = CallRecorder()
    runtime = ResearchRuntime(
        dependencies=Dependencies(
            mcp_client=MCPClient(recorder),
            fast_model=FastModel(recorder),
            analyst_model=Analyst(recorder),
        )
    )
    factory = RuntimeFactory(runtime)
    application = ResearchApplication(
        NeverRouter(),
        factory,
        run_repository=repository,
        trace_sink=trace_sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )
    return application, recorder, factory


def _company_command() -> ResearchCommand:
    return ResearchCommand(
        ticker="NVDA",
        request="Does data center demand support revenue growth?",
        mode=ResearchMode.THESIS,
    )


def _descendants(root: FakeObservation) -> list[FakeObservation]:
    descendants: list[FakeObservation] = []
    pending = list(root.children)
    while pending:
        child = pending.pop(0)
        descendants.append(child)
        pending.extend(child.children)
    return descendants


def test_one_run_has_root_and_typed_children() -> None:
    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application, _, _ = _application(trace_sink=sink, repository=repository)

    result = application.run(_company_command())

    root = sink.single_root()
    assert root.name == "financial-research-agent.run"
    assert root.metadata["run_id"] == result.run_id == "run-123"
    assert {child.kind for child in _descendants(root)} >= {
        "guardrail",
        "chain",
        "tool",
        "retriever",
    }
    assert root.output == {"status": result.status}
    assert root.metadata["effective_intent"] == "research_request"
    assert root.metadata["corpus_scope"] == ["NVDA-v1"]
    assert root.metadata["recipe_names"] == []
    assert root.metadata["recipe_versions"] == []
    assert root.metadata["safety_rule_match"] is False
    assert root.metadata["safety_intent"] == "ambiguous"
    runtime = next(child for child in _descendants(root) if child.name == "runtime.execute")
    assert runtime.output == {"status": result.status}
    assert sink.flush_calls == 1
    assert len(repository.started) == 1
    assert len(repository.finished) == 1


def test_root_metadata_aggregates_safe_child_accounting() -> None:
    """Removing the accounting wrapper would lose actual provider and budget totals."""

    @dataclass(frozen=True)
    class AccountedResult:
        run_id: str = "pending"
        status: str = "completed"
        rendered_output: str = "# Accounted result"

        def bind_run_id(self, run_id: str) -> AccountedResult:
            return AccountedResult(run_id=run_id)

        def to_run_finish(self, trace_id: str | None) -> RunFinish:
            return RunFinish(
                run_id=self.run_id,
                effective_intent="research_request",
                status=self.status,
                corpus_scope=[],
                prompt_version=None,
                trace_id=trace_id,
                report_markdown=self.rendered_output,
                claims=[],
            )

        def root_metadata(self) -> dict[str, object]:
            return {
                "effective_intent": "research_request",
                "guard_errors": ["safe-code"],
                "raw_provider_error": "provider secret must not export",
            }

    class AccountedRuntime:
        def execute(self, command: ResearchCommand, decision: RouterDecision) -> AccountedResult:
            del command, decision
            context = current_research_context()
            assert context is not None
            context.budget.configure(
                BudgetLimits(max_tool_calls=3, max_retrieval_rounds=2, max_web_calls=1)
            )
            context.budget.consume(tool_calls=3, retrieval_rounds=2, web_calls=1)
            with context.trace.observation(
                name="model.planner",
                kind="generation",
                metadata={
                    "model": "gpt-5-mini",
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                    "cost": 0.03,
                },
            ):
                pass
            with context.trace.observation(
                name="model.analyst",
                kind="generation",
                metadata={
                    "model": "gpt-5",
                    "usage": {"input_tokens": 19, "output_tokens": 13},
                    "cost": 0.07,
                },
            ):
                pass
            with context.trace.observation(
                name="mcp.hybrid_search_filings",
                kind="tool",
                metadata={"cache_hit": True, "raw_query": "source text must not export"},
            ):
                pass
            return AccountedResult()

    sink = FakeTraceSink()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(AccountedRuntime()),
        run_repository=MemoryRunRepository(),
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    application.run(_company_command())

    root = sink.single_root()
    assert root.metadata["input_tokens"] == 30
    assert root.metadata["output_tokens"] == 20
    assert root.metadata["provider_cost_usd"] == 0.1
    assert root.metadata["model_ids"] == ["gpt-5-mini", "gpt-5"]
    assert root.metadata["model_calls"] == 2
    assert root.metadata["tool_calls"] == 3
    assert root.metadata["retrieval_rounds"] == 2
    assert root.metadata["web_calls"] == 1
    assert root.metadata["cache_hits"] == 1
    assert root.metadata["final_status"] == "completed"
    assert "raw_provider_error" not in root.metadata
    assert "raw_query" not in root.metadata


def test_early_quality_failure_finalizes_active_root_once() -> None:
    sink = _FailingChildTraceSink()
    repository = MemoryRunRepository()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(object()),
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    with pytest.raises(RuntimeError, match="child exporter failed"):
        application.run(
            ResearchCommand(
                ticker="NVDA",
                request="Rank the best semiconductor stocks to buy",
                mode=ResearchMode.QUALITY_SCREEN,
            )
        )

    root = sink.single_root()
    assert root.output == {"status": "failed"}
    assert root.metadata["final_status"] == "failed"
    assert root.update_calls == 1


def test_refusal_failure_finalizes_active_root_once_and_returns_refusal() -> None:
    sink = _FailingChildTraceSink()
    repository = MemoryRunRepository()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(object()),
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
    )

    result = application.run(
        ResearchCommand(ticker="NVDA", request="Should I buy NVDA now?", mode=ResearchMode.THESIS)
    )

    root = sink.single_root()
    assert result.status == "refused"
    assert root.output == {"status": "failed"}
    assert root.metadata["final_status"] == "failed"
    assert root.update_calls == 1
    assert repository.start_calls == 1


def test_cost_accounting_rejects_nonrepresentable_and_malformed_values_before_mutation() -> None:
    accounting = RunAccounting()
    observation = accounting.begin_observation(kind="generation", metadata=None)

    for cost in (0, 0.25, Decimal("0.5"), Decimal("1e-13")):
        observation.update_metadata({"cost": cost})
    accepted = accounting.provider_cost_usd
    for cost in (
        Decimal("1e-1000000"),
        -1,
        True,
        "0.5",
        Decimal("NaN"),
        Decimal("Infinity"),
        10**1000,
    ):
        observation.update_metadata({"cost": cost})

    assert accepted == Decimal("0.5")
    assert accounting.provider_cost_usd == accepted
    underflow = RunAccounting()
    underflow.begin_observation(kind="generation", metadata={"cost": Decimal("1e-1000000")})
    assert underflow.provider_cost_usd == Decimal(0)


def test_early_quality_final_export_failure_propagates_once() -> None:
    sink = _FailingRootUpdateTraceSink()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(object()),
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    with pytest.raises(RuntimeError, match="root exporter failed"):
        application.run(
            ResearchCommand(
                ticker="NVDA",
                request="Rank the best semiconductor stocks to buy",
                mode=ResearchMode.QUALITY_SCREEN,
            )
        )

    assert sink.single_root().update_calls == 1


def test_refusal_final_export_failure_returns_refusal_once() -> None:
    sink = _FailingRootUpdateTraceSink()
    application = ResearchApplication(
        NeverRouter(), RuntimeFactory(object()), trace_sink=sink, id_generator=FixedIds()
    )

    result = application.run(
        ResearchCommand(ticker="NVDA", request="Should I buy NVDA now?", mode=ResearchMode.THESIS)
    )

    assert result.status == "refused"
    assert sink.single_root().update_calls == 1


class _FailingChildTraceSink(FakeTraceSink):
    @contextmanager
    def run(
        self, *, run_id: str, input: object, metadata: dict[str, object]
    ) -> Iterator[FakeObservation]:
        root = _FailingChildRoot(
            name="financial-research-agent.run",
            kind="agent",
            input=input,
            metadata=dict(metadata),
            trace_id=f"trace-{run_id}",
        )
        self.roots.append(root)
        root.active = True
        try:
            yield root
        finally:
            root.active = False


class _FailingChildRoot(FakeObservation):
    @contextmanager
    def observation(self, **_: object) -> Iterator[FakeObservation]:
        raise RuntimeError("child exporter failed")
        yield self


class _FailingRootUpdateTraceSink(FakeTraceSink):
    @contextmanager
    def run(
        self, *, run_id: str, input: object, metadata: dict[str, object]
    ) -> Iterator[FakeObservation]:
        root = _FailingRootUpdate(
            name="financial-research-agent.run",
            kind="agent",
            input=input,
            metadata=dict(metadata),
            trace_id=f"trace-{run_id}",
        )
        self.roots.append(root)
        root.active = True
        try:
            yield root
        finally:
            root.active = False


class _FailingRootUpdate(FakeObservation):
    def update(self, **kwargs: object) -> None:
        super().update(**kwargs)  # type: ignore[arg-type]
        raise RuntimeError("root exporter failed")


def test_research_result_satisfies_common_application_result_contract() -> None:
    """Current P0/P1 results must use the same lifecycle contract future results implement."""
    sink = FakeTraceSink()
    application, _, _ = _application(
        trace_sink=sink,
        repository=MemoryRunRepository(),
    )

    result = application.run(_company_command())

    assert hasattr(application_module, "ApplicationResult")
    assert isinstance(result, application_module.ApplicationResult)


def test_conforming_non_research_result_receives_common_final_lifecycle() -> None:
    """The application contract must not hard-code ResearchResult as the only valid result."""

    @dataclass(frozen=True)
    class AlternateResult:
        run_id: str = "pending"
        status: str = "completed"
        rendered_output: str = "# Alternate result"

        def bind_run_id(self, run_id: str) -> AlternateResult:
            return AlternateResult(run_id=run_id)

        def to_run_finish(self, trace_id: str | None) -> RunFinish:
            return RunFinish(
                run_id=self.run_id,
                effective_intent="alternate_result",
                status=self.status,
                corpus_scope=[],
                prompt_version="alternate-v1",
                trace_id=trace_id,
                report_markdown=self.rendered_output,
                claims=[],
            )

        def root_metadata(self) -> dict[str, object]:
            return {"effective_intent": "alternate_result"}

    class AlternateRuntime:
        def execute(self, command: ResearchCommand, decision: RouterDecision) -> AlternateResult:
            del command, decision
            return AlternateResult()

    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(AlternateRuntime()),
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    result = application.run(_company_command())

    assert isinstance(result, AlternateResult)
    assert result.run_id == "run-123"
    assert sink.single_root().output == {"status": "completed"}
    assert repository.finished[0].effective_intent == "alternate_result"  # type: ignore[attr-defined]
    assert sink.flush_calls == 1


def test_wrong_runtime_return_type_becomes_failed_result_and_still_finalizes() -> None:
    """An arbitrary object cannot bypass finish, root status, or exporter flush."""

    class WrongRuntime:
        def execute(self, command: ResearchCommand, decision: RouterDecision) -> object:
            del command, decision
            return object()

    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(WrongRuntime()),
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    result = application.run(_company_command())

    assert hasattr(application_module, "ApplicationResult")
    assert isinstance(result, application_module.ApplicationResult)
    assert result.status == "failed"
    assert result.run_id == "run-123"
    assert result.rendered_output == "Research runtime returned an invalid result."
    assert len(repository.started) == 1
    assert len(repository.finished) == 1
    assert repository.finished[0].status == "failed"  # type: ignore[attr-defined]
    assert sink.single_root().output == {"status": "failed"}
    assert sink.flush_calls == 1


def test_rule_refusal_survives_database_failure_without_dependency_calls() -> None:
    sink = FakeTraceSink()
    repository = MemoryRunRepository(fail=True)
    application, recorder, factory = _application(trace_sink=sink, repository=repository)

    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="Should I buy NVDA?",
            mode=ResearchMode.AUTO,
        )
    )

    assert result.status == "refused"
    assert result.rendered_output == REFUSAL_TEXT
    assert result.run_id == "run-123"
    assert recorder.model_calls == 0
    assert recorder.tool_calls == 0
    assert factory.calls == 0
    assert repository.start_calls == 1
    assert repository.finish_calls == 0
    root = sink.single_root()
    assert root.output == {"status": "refused"}
    assert root.metadata["effective_intent"] == "prohibited_advice"
    assert root.metadata["safety_rule_match"] is True
    assert sink.flush_calls == 1


def test_rule_refusal_creates_trace_and_attempts_start_and_finish_on_first_request() -> None:
    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application, recorder, factory = _application(trace_sink=sink, repository=repository)

    result = application.run(
        ResearchCommand(
            ticker="NVDA",
            request="Should I buy NVDA?",
            mode=ResearchMode.AUTO,
        )
    )

    assert result.status == "refused"
    assert recorder.model_calls == 0
    assert recorder.tool_calls == 0
    assert factory.calls == 0
    assert len(repository.started) == 1
    assert len(repository.finished) == 1
    start = repository.started[0]
    finish = repository.finished[0]
    assert start.run_id == result.run_id  # type: ignore[attr-defined]
    assert start.ticker == "UNKNOWN"  # type: ignore[attr-defined]
    assert finish.run_id == result.run_id  # type: ignore[attr-defined]
    assert finish.status == "refused"  # type: ignore[attr-defined]
    assert finish.trace_id == "trace-run-123"  # type: ignore[attr-defined]
    root = sink.single_root()
    assert root.input == {
        "ticker": "UNKNOWN",
        "ticker_sha256": "426219f2bc19829619cd8e71b51d7b93f6ad4b0634f74d3681a80f34d9162edb",
        "mode": "auto",
        "request_sha256": "7aca22a6ade930efb67eebb4df87d9e8ac9e4c4c5ea0c7b9532182b75bbf1852",
        "request_length": 18,
    }
    assert root.output == {"status": "refused"}
    assert root.metadata["ticker"] == "UNKNOWN"
    assert root.metadata["effective_intent"] == "prohibited_advice"
    assert {child.name for child in _descendants(root)} >= {
        "safety.route",
        "persistence.start",
        "persistence.finish",
    }
    assert "scope.resolve" not in {child.name for child in _descendants(root)}
    assert sink.flush_calls == 1


@pytest.mark.parametrize(
    ("ticker", "peer_tickers", "expected_ticker"),
    [
        ("password=x", (), "UNKNOWN"),
        ("", (), "UNKNOWN"),
        ("TOO-LONG-123", (), "UNKNOWN"),
        ("NVDA", ("password=x",), "UNKNOWN"),
    ],
)
def test_invalid_ticker_inputs_use_safe_traced_persisted_refusal_lifecycle(
    ticker: str,
    peer_tickers: tuple[str, ...],
    expected_ticker: str,
) -> None:
    """Malformed primary/peer values never reach runtime, trace, or persistence raw."""
    private = "PASSWORD=X"
    sink = FakeTraceSink()
    repository = MemoryRunRepository()

    class NeverRuntime:
        def execute(self, command: ResearchCommand, decision: RouterDecision) -> object:
            raise AssertionError(f"invalid ticker reached runtime: {command!r} {decision!r}")

    factory = RuntimeFactory(NeverRuntime())
    application = ResearchApplication(
        NeverRouter(),
        factory,
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )
    result = application.run(
        ResearchCommand(
            ticker=ticker,
            request="Compare exact public peer metrics.",
            mode=ResearchMode.INDUSTRY_RESEARCH,
            peer_tickers=peer_tickers,
            peer_scope="US semiconductors" if peer_tickers else None,
        )
    )

    root = sink.single_root()
    assert result.status == "refused"
    assert result.ticker == expected_ticker
    assert factory.calls == 0
    assert len(repository.started) == len(repository.finished) == 1
    assert repository.started[0].ticker == expected_ticker  # type: ignore[attr-defined]
    assert repository.finished[0].status == "refused"  # type: ignore[attr-defined]
    assert private not in repr((result, root, repository.started, repository.finished))
    assert {child.name for child in _descendants(root)} >= {
        "safety.route",
        "persistence.start",
        "persistence.finish",
    }


def test_ordinary_research_request_is_preserved_at_the_run_start_boundary() -> None:
    """Over-redacting ordinary research would make persisted runs unusable for audit."""
    for command in (
        _company_command(),
        ResearchCommand(
            ticker="NVDA",
            request=(
                "Do institutional investor holdings affect disclosed ownership "
                "concentration?"
            ),
            mode=ResearchMode.THESIS,
        ),
        ResearchCommand(
            ticker="NVDA",
            request=(
                "Taylor owns 100 NVDA shares according to Form 4 and compare NVDA revenue."
            ),
            mode=ResearchMode.THESIS,
        ),
    ):
        sink = FakeTraceSink()
        repository = MemoryRunRepository()
        application, _, _ = _application(trace_sink=sink, repository=repository)

        application.run(command)

        assert repository.started[0].request == command.request  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("command", "expected_status"),
    [
        (
            ResearchCommand(
                ticker="NVDA",
                request="My brokerage account number is ABC123. Should I buy NVDA?",
                mode=ResearchMode.AUTO,
            ),
            "refused",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="My brokerage account number is ABC123. Rank the best stock to buy.",
                mode=ResearchMode.QUALITY_SCREEN,
            ),
            "declined",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request=(
                    "My brokerage account number is ABC123. Does revenue growth have "
                    "supporting evidence?"
                ),
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request=(
                    "Taylor Morgan owns 100 NVDA shares. Form 4 discusses a public "
                    "disclosure."
                ),
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request=(
                    "Taylor Morgan holds NVDA shares according to Form 4 and Alice "
                    "owns NVDA; summarize revenue evidence."
                ),
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="Credentials: ZXCV-1234; summarize revenue growth evidence.",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="Taylor Morgan owns 100 NVDA shares; summarize revenue evidence.",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="Account no. ABC-12345; summarize revenue growth evidence.",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="Alice owns NVDA; summarize revenue growth evidence.",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request=(
                    "Analyze insider ownership, then note Taylor Morgan's portfolio "
                    "is concentrated in NVDA."
                ),
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="She owns 100 NVDA shares; summarize revenue growth evidence.",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request=(
                    "Taylor owns 100 NVDA shares according to Form 4, Alice owns "
                    "50 NVDA shares."
                ),
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
        (
            ResearchCommand(
                ticker="NVDA",
                request="根据Form 4披露，李明持有100股NVDA和王芳持有50股NVDA。",
                mode=ResearchMode.THESIS,
            ),
            "completed",
        ),
    ],
)
def test_sensitive_requests_use_only_a_stable_marker_at_every_run_start_path(
    command: ResearchCommand,
    expected_status: str,
) -> None:
    """Removing one application sanitizer call would durably expose the private request."""
    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application, _, _ = _application(trace_sink=sink, repository=repository)

    result = application.run(command)

    assert result.status == expected_status
    persisted = repository.started[0].request  # type: ignore[attr-defined]
    expected_marker = (
        "[private request redacted; sha256="
        f"{sha256(command.request.encode('utf-8')).hexdigest()}]"
    )
    assert persisted == expected_marker
    durable_material = repr((sink.single_root(), repository.started, repository.finished))
    assert command.request not in durable_material


def test_private_request_is_redacted_before_a_failing_persistence_boundary(caplog) -> None:
    command = ResearchCommand(
        ticker="NVDA",
        request="Credentials: ZXCV-1234; summarize revenue growth evidence.",
        mode=ResearchMode.THESIS,
    )
    sink = FakeTraceSink()
    repository = MemoryRunRepository(fail=True)
    application, _, _ = _application(trace_sink=sink, repository=repository)

    with caplog.at_level("WARNING"):
        result = application.run(command)

    expected_marker = (
        "[private request redacted; sha256="
        f"{sha256(command.request.encode('utf-8')).hexdigest()}]"
    )
    assert result.status == "completed"
    assert repository.attempted_starts[0].request == expected_marker  # type: ignore[attr-defined]
    assert command.request not in repr(sink.single_root())
    assert command.request not in caplog.text


def test_trace_flushes_when_runtime_raises() -> None:
    class ExplodingRuntime:
        def execute(self, command: ResearchCommand, decision: RouterDecision) -> Any:
            del command, decision
            raise RuntimeError("provider failed")

    sink = FakeTraceSink()
    repository = MemoryRunRepository()
    application = ResearchApplication(
        NeverRouter(),
        RuntimeFactory(ExplodingRuntime()),
        run_repository=repository,
        trace_sink=sink,
        id_generator=FixedIds(),
        company_resolver=DeterministicCompanyResolver("NVDA"),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        application.run(_company_command())

    assert sink.single_root().output == {"status": "failed"}
    assert sink.flush_calls == 1
    assert len(repository.finished) == 1
    failed = repository.finished[0]
    assert failed.status == "failed"  # type: ignore[attr-defined]
    assert failed.trace_id == "trace-run-123"  # type: ignore[attr-defined]


def test_observation_metadata_never_contains_raw_queries_or_secrets() -> None:
    sink = FakeTraceSink()
    application, _, _ = _application(
        trace_sink=sink,
        repository=MemoryRunRepository(),
    )

    application.run(_company_command())

    exported = repr(sink.single_root())
    assert "secret support query" not in exported
    assert "secret challenge query" not in exported
    assert "database secret" not in exported
    assert "OPENAI_API_KEY" not in exported


class FakeLangfuseManager:
    def __init__(self, client: FakeLangfuseClient, observation: FakeObservation) -> None:
        self.client = client
        self.observation = observation

    def __enter__(self) -> FakeObservation:
        if self.client.stack:
            self.client.stack[-1].children.append(self.observation)
        else:
            self.client.roots.append(self.observation)
        self.client.stack.append(self.observation)
        self.observation.active = True
        return self.observation

    def __exit__(self, *args: object) -> None:
        del args
        assert self.client.stack.pop() is self.observation
        self.observation.active = False


class FakeLangfuseClient:
    def __init__(self, *, fail_flush: bool = False) -> None:
        self.fail_flush = fail_flush
        self.roots: list[FakeObservation] = []
        self.stack: list[FakeObservation] = []
        self.flush_calls = 0

    def start_as_current_observation(self, **values: object) -> FakeLangfuseManager:
        observation = FakeObservation(
            name=str(values["name"]),
            kind=str(values["as_type"]),
            input=values.get("input"),
            metadata=dict(values.get("metadata", {})),  # type: ignore[arg-type]
            trace_id="langfuse-trace-1" if not self.stack else None,
        )
        return FakeLangfuseManager(self, observation)

    def flush(self) -> None:
        self.flush_calls += 1
        if self.fail_flush:
            raise RuntimeError("exporter secret")


def test_langfuse_v4_exporter_builds_one_agent_tree() -> None:
    client = FakeLangfuseClient()
    sink = LangfuseTraceSink(client)
    application, _, _ = _application(
        trace_sink=sink,  # type: ignore[arg-type]
        repository=MemoryRunRepository(),
    )

    result = application.run(_company_command())

    assert len(client.roots) == 1
    root = client.roots[0]
    assert root.kind == "agent"
    assert root.name == "financial-research-agent.run"
    assert root.metadata["run_id"] == result.run_id
    assert {child.kind for child in _descendants(root)} >= {
        "guardrail",
        "chain",
        "tool",
        "retriever",
    }
    assert root.output == {"status": result.status}
    assert client.flush_calls == 1


def test_langfuse_flush_failure_does_not_change_research_result() -> None:
    client = FakeLangfuseClient(fail_flush=True)
    application, _, _ = _application(
        trace_sink=LangfuseTraceSink(client),  # type: ignore[arg-type]
        repository=MemoryRunRepository(),
    )

    result = application.run(_company_command())

    assert result.status == "completed"
    assert client.flush_calls == 1


def test_langfuse_legacy_record_exports_without_an_application_root() -> None:
    client = FakeLangfuseClient()
    sink = LangfuseTraceSink(client)

    sink.record(
        "evaluation_summary",
        {
            "case_count": 3,
            "raw_query": "secret query must not export",
        },
    )

    assert len(client.roots) == 1
    event = client.roots[0]
    assert event.name == "evaluation_summary"
    assert event.kind == "chain"
    assert event.metadata == {"case_count": 3}


def test_langfuse_legacy_record_stays_inside_active_application_root() -> None:
    client = FakeLangfuseClient()
    sink = LangfuseTraceSink(client)

    with sink.run(run_id="run-legacy", input={}, metadata={}) as run:
        with bind_trace_run(run):
            sink.record("evaluation_case", {"case_id": "case-2"})

    assert len(client.roots) == 1
    root = client.roots[0]
    assert [child.name for child in root.children] == ["evaluation_case"]
    assert root.children[0].metadata == {"case_id": "case-2"}
