"""Command-line entry points for Financial-Research-Agent."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from fra.bootstrap import (
    BootstrapError,
    build_eval_runtime,
    build_filing_repository,
    build_industry_runtime,
    build_market_runtime,
    build_p0_runtime,
    build_p1_runtime,
    build_quality_runtime,
    build_research_application,
)
from fra.config import Settings
from fra.contracts import ResearchCommand, ResearchMode
from fra.domain import (
    DEFAULT_RESEARCH_FORMS,
    FilingForm,
    IntentRoutingError,
)
from fra.market_data.models import MarketDataError
from fra.retrieval.indexing import EmbeddingModelUnavailableError
from fra.safety.router import (
    ARBITRARY_URL_REFUSAL_TEXT,
    PROMPT_INJECTION_TEXT,
    REFUSAL_TEXT,
)

if TYPE_CHECKING:
    from fra.evals.p2_runner import P2EvalSummary
    from fra.evals.runner import EvalSummary

__all__ = [
    "ARBITRARY_URL_REFUSAL_TEXT",
    "EvalSuite",
    "PROMPT_INJECTION_TEXT",
    "REFUSAL_TEXT",
    "app",
]

app = typer.Typer(
    add_completion=False,
    help="Audit public-company research claims against citable evidence.",
    no_args_is_help=True,
)


class EvalSuite(StrEnum):
    """Supported CLI evaluation suite selections."""

    ALL = "all"
    P2 = "p2"


@app.command()
def ingest(
    ticker: Annotated[
        str,
        typer.Option("--ticker", help="US-listed company ticker."),
    ],
    fixture: Annotated[
        Path | None,
        typer.Option("--fixture", help="Offline SEC-like HTML fixture."),
    ] = None,
    form: Annotated[
        str | None,
        typer.Option("--form", help="Single form recorded for fixture ingestion."),
    ] = None,
    forms: Annotated[
        str | None,
        typer.Option("--forms", help="Comma-separated live SEC candidate forms."),
    ] = None,
    as_of_date: Annotated[
        str | None,
        typer.Option("--as-of-date", help="Latest eligible SEC filing date."),
    ] = None,
) -> None:
    """Load supported public-company filings into the local corpus."""
    if fixture is not None and forms is not None:
        raise typer.BadParameter("--fixture and --forms are mutually exclusive")
    if fixture is None and form is not None:
        raise typer.BadParameter("--form is only valid with --fixture")
    if fixture is not None and as_of_date is not None:
        raise typer.BadParameter("--as-of-date is only valid for live SEC ingestion")
    if fixture is not None and form is None:
        raise typer.BadParameter("--form is required with --fixture")
    if fixture is None and forms is None:
        raise typer.BadParameter(
            "choose --fixture for offline ingest or --forms for live SEC ingest"
        )

    from fra.retrieval.indexing import (
        BgeM3EmbeddingProvider,
        HashEmbeddingProvider,
    )
    from fra.retrieval.ingest import (
        ingest_fixture_summary,
        ingest_sec,
    )
    from fra.retrieval.sec import (
        SecProviderError,
        SecRequestRateLimiter,
    )
    from fra.retrieval.xbrl import HttpCompanyFactsGateway, XbrlError
    from fra.storage.cache import build_cache
    from fra.storage.fact_repositories import CompanyFactRepository

    settings = Settings()
    repository = build_filing_repository(settings)
    try:
        if fixture is not None:
            assert form is not None
            summary = ingest_fixture_summary(
                fixture,
                ticker,
                form,
                repository,
                embedding_provider=HashEmbeddingProvider(dimensions=1024),
            )
        else:
            assert forms is not None
            try:
                cutoff = date.fromisoformat(as_of_date) if as_of_date is not None else None
            except ValueError as error:
                raise typer.BadParameter("--as-of-date must use YYYY-MM-DD") from error
            embedding_provider = BgeM3EmbeddingProvider(
                settings.embedding_model,
                cache_dir=settings.embedding_cache_dir,
            )
            sec_rate_limiter = SecRequestRateLimiter()
            summary = ingest_sec(
                ticker,
                forms.split(","),
                cutoff,
                repository=repository,
                user_agent=settings.sec_user_agent,
                embedding_provider=embedding_provider,
                cache=build_cache(settings),
                sec_cache_ttl_seconds=settings.sec_cache_ttl_seconds,
                configured_issuer_domains=settings.web_issuer_domains,
                fact_repository=CompanyFactRepository(repository.engine),
                company_facts_gateway=HttpCompanyFactsGateway(
                    max_response_bytes=settings.xbrl_max_response_bytes,
                    rate_limiter=sec_rate_limiter,
                ),
                rate_limiter=sec_rate_limiter,
            )
    except (
        OSError,
        ValueError,
        EmbeddingModelUnavailableError,
        SecProviderError,
        XbrlError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(summary.model_dump_json(indent=2))


@app.command()
def research(
    ticker: Annotated[str, typer.Argument(help="US-listed company ticker.")],
    thesis: Annotated[
        str | None,
        typer.Option(
            "--thesis",
            help="Verifiable research thesis. Required for thesis mode.",
        ),
    ] = None,
    question: Annotated[
        str | None,
        typer.Option(
            "--question",
            help="Company or earnings question. Required for explicit P1 modes.",
        ),
    ] = None,
    mode: Annotated[
        ResearchMode,
        typer.Option(
            "--mode",
            help=(
                "Research workflow mode: thesis, auto, company-profile, "
                "earnings-review, industry-research, market-snapshot, "
                "or quality-screen."
            ),
        ),
    ] = ResearchMode.THESIS,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Optional future session-memory scope."),
    ] = None,
    forms: Annotated[
        str,
        typer.Option(
            "--forms",
            help="Comma-separated filing forms from 10-K, 10-Q, and 8-K.",
        ),
    ] = ",".join(DEFAULT_RESEARCH_FORMS),
    as_of_date: Annotated[
        str | None,
        typer.Option("--as-of-date", help="Latest filing evidence date (YYYY-MM-DD)."),
    ] = None,
    market: Annotated[
        str,
        typer.Option("--market", help="Market coverage; only US is supported."),
    ] = "US",
    with_context: Annotated[
        bool,
        typer.Option(
            "--with-context",
            help="Include one time-anchored authoritative event search for market mode.",
        ),
    ] = False,
    peer_ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--peer-ticker",
            help="Repeatable explicit peer ticker for industry-research mode.",
        ),
    ] = None,
    peer_scope: Annotated[
        str | None,
        typer.Option(
            "--peer-scope",
            help="Explicit peer-comparison scope description for industry-research mode.",
        ),
    ] = None,
) -> None:
    """Run a controlled filing-research or IEX-only market workflow."""
    request = _validated_research_input(thesis=thesis, question=question, mode=mode)
    selected_forms = _validated_research_forms(forms)
    cutoff = _validated_research_date(as_of_date)
    selected_market = market.strip().upper()
    if selected_market != "US":
        raise typer.BadParameter("--market must be US")
    normalized_session_id = session_id.strip() if session_id is not None else None
    if session_id is not None and not normalized_session_id:
        raise typer.BadParameter("--session-id must not be blank")
    explicit_peers = tuple(peer_ticker or ())
    if any(not peer.strip() for peer in explicit_peers):
        raise typer.BadParameter("--peer-ticker must not be blank")
    if explicit_peers and mode is not ResearchMode.INDUSTRY_RESEARCH:
        raise typer.BadParameter("--peer-ticker is only valid in industry-research mode")
    if peer_scope is not None and mode is not ResearchMode.INDUSTRY_RESEARCH:
        raise typer.BadParameter("--peer-scope is only valid in industry-research mode")
    if peer_scope is not None and explicit_peers == ():
        raise typer.BadParameter("--peer-scope requires at least one --peer-ticker")
    if explicit_peers and (peer_scope is None or not peer_scope.strip()):
        raise typer.BadParameter("--peer-scope is required when --peer-ticker is supplied")
    command = ResearchCommand(
        ticker=ticker,
        request=request,
        mode=mode,
        session_id=normalized_session_id,
        forms=selected_forms,
        as_of_date=cutoff,
        market=selected_market,
        peer_tickers=explicit_peers,
        peer_scope=peer_scope,
        with_context=with_context,
    )
    settings = Settings()
    research_application = build_research_application(
        settings,
        p0_builder=build_p0_runtime,
        p1_builder=build_p1_runtime,
        industry_builder=build_industry_runtime,
        market_builder=build_market_runtime,
        quality_builder=build_quality_runtime,
    )
    try:
        result = research_application.run(command)
    except (BootstrapError, IntentRoutingError, MarketDataError) as error:
        typer.echo(f"{error.code.value}: {error.detail}", err=True)
        raise typer.Exit(2) from error
    except EmbeddingModelUnavailableError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(result.rendered_output, nl=False)


@app.command(name="eval")
def evaluate(
    suite: Annotated[
        EvalSuite,
        typer.Option(
            "--suite",
            help="Evaluation suite: all (legacy P0/P1 plus bundled P2) or p2 only.",
        ),
    ] = EvalSuite.ALL,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Strict mixed P0/P1 JSONL evaluation dataset."),
    ] = None,
    p2_dataset: Annotated[
        Path | None,
        typer.Option("--p2-dataset", help="Strict standalone P2 JSONL evaluation dataset."),
    ] = None,
    langfuse_dataset: Annotated[
        str | None,
        typer.Option(
            "--langfuse-dataset",
            help=(
                "Named Langfuse dataset for explicit canonical evaluation synchronization "
                "and experiments."
            ),
        ),
    ] = None,
    sync_langfuse_dataset: Annotated[
        bool,
        typer.Option(
            "--sync-langfuse-dataset",
            help=(
                "Synchronize canonical cases for the selected evaluation suite into the "
                "named Langfuse dataset."
            ),
        ),
    ] = False,
    langfuse_experiment: Annotated[
        str | None,
        typer.Option(
            "--langfuse-experiment",
            help="Run deterministic application cases through the named Langfuse dataset.",
        ),
    ] = None,
) -> None:
    """Evaluate P0 behavior and additive P1 contracts against local data."""
    if suite is EvalSuite.P2 and dataset is not None:
        raise typer.BadParameter("--dataset is only valid for the legacy P0/P1 dataset")
    if langfuse_dataset is not None and not langfuse_dataset.strip():
        raise typer.BadParameter("--langfuse-dataset must not be blank")
    if langfuse_experiment is not None and not langfuse_experiment.strip():
        raise typer.BadParameter("--langfuse-experiment must not be blank")
    if sync_langfuse_dataset and langfuse_dataset is None:
        raise typer.BadParameter("--sync-langfuse-dataset requires --langfuse-dataset")
    if langfuse_experiment is not None and langfuse_dataset is None:
        raise typer.BadParameter("--langfuse-experiment requires --langfuse-dataset")
    if (
        langfuse_dataset is not None
        and not sync_langfuse_dataset
        and langfuse_experiment is None
    ):
        raise typer.BadParameter("--langfuse-dataset requires a sync and/or experiment flag")

    from fra.evals import (
        bundled_dataset_path,
        bundled_p2_dataset_path,
    )
    from fra.evals.p2_runner import (
        load_p2_eval_cases,
        run_application_experiment,
        run_p2_eval,
    )
    from fra.evals.runner import (
        load_comparable_application_experiment_cases,
        run_comparable_application_experiment,
        run_eval,
    )
    from fra.observability import (
        LangfuseOperationError,
        build_langfuse_operation_client,
    )
    from fra.observability import (
        sync_langfuse_dataset as sync_dataset,
    )

    if suite is EvalSuite.P2:
        with _materialized_eval_path(p2_dataset, bundled_p2_dataset_path) as path:
            summary = run_p2_eval(path)
            if _p2_eval_failed(summary):
                typer.echo(summary.model_dump_json(indent=2))
                raise typer.Exit(1)
            if langfuse_dataset is not None:
                try:
                    cases = load_p2_eval_cases(path)
                    settings = Settings()
                    if not settings.has_complete_langfuse_credentials:
                        raise typer.BadParameter(
                            "complete Langfuse credentials are required "
                            "for dataset synchronization and experiments"
                        )
                    client = build_langfuse_operation_client(settings)
                    if sync_langfuse_dataset:
                        sync_dataset(client, langfuse_dataset.strip(), cases)
                    if langfuse_experiment is not None:
                        run_application_experiment(
                            client,
                            dataset_name=langfuse_dataset.strip(),
                            experiment_name=langfuse_experiment.strip(),
                            cases=cases,
                        )
                except LangfuseOperationError as error:
                    typer.echo(f"{error.code}: {error.detail}", err=True)
                    raise typer.Exit(2) from error
        typer.echo(summary.model_dump_json(indent=2))
        return

    dependencies = build_eval_runtime().dependencies

    with _materialized_eval_path(dataset, bundled_dataset_path) as legacy_path:
        summary = run_eval(legacy_path, dependencies)
    with _materialized_eval_path(p2_dataset, bundled_p2_dataset_path) as resolved_p2_dataset:
        p2_summary = run_p2_eval(resolved_p2_dataset)
    summary = summary.model_copy(
        update={
            "p2_case_count": p2_summary.case_count,
            "p2_passed_count": p2_summary.passed_count,
            "p2_metrics": p2_summary.metrics,
            "p2_results": p2_summary.results,
        }
    )
    typer.echo(summary.model_dump_json(indent=2))
    if _legacy_eval_failed(summary) or _p2_eval_failed(p2_summary):
        raise typer.Exit(1)
    if langfuse_dataset is not None:
        try:
            settings = Settings()
            if not settings.has_complete_langfuse_credentials:
                raise typer.BadParameter(
                    "complete Langfuse credentials are required "
                    "for dataset synchronization and experiments"
                )
            client = build_langfuse_operation_client(settings)
            with (
                _materialized_eval_path(dataset, bundled_dataset_path) as legacy_path,
                _materialized_eval_path(p2_dataset, bundled_p2_dataset_path) as resolved_p2_dataset,
            ):
                cases = load_comparable_application_experiment_cases(
                    legacy_path,
                    resolved_p2_dataset,
                )
                if sync_langfuse_dataset:
                    sync_dataset(client, langfuse_dataset.strip(), cases)
                if langfuse_experiment is not None:
                    run_comparable_application_experiment(
                        client,
                        dataset_name=langfuse_dataset.strip(),
                        experiment_name=langfuse_experiment.strip(),
                        cases=cases,
                    )
        except LangfuseOperationError as error:
            typer.echo(f"{error.code}: {error.detail}", err=True)
            raise typer.Exit(2) from error


def _legacy_eval_failed(summary: EvalSummary) -> bool:
    metrics = summary.p1_metrics
    return (
        summary.passed_count != summary.case_count
        or summary.p1_passed_count != summary.p1_case_count
        or (metrics is not None and (
            metrics.source_policy_violations != 0 or metrics.budget_violations != 0
        ))
    )


def _p2_eval_failed(summary: P2EvalSummary) -> bool:
    metrics = summary.metrics
    return (
        summary.passed_count != summary.case_count
        or metrics.cross_ticker_leakage_count != 0
        or metrics.source_policy_violations != 0
        or metrics.budget_violations != 0
    )


def _validated_research_input(
    *,
    thesis: str | None,
    question: str | None,
    mode: ResearchMode,
) -> str:
    if mode is ResearchMode.THESIS:
        if question is not None:
            raise typer.BadParameter("--question is not valid in thesis mode")
        if thesis is None or not thesis.strip():
            raise typer.BadParameter("--thesis is required in thesis mode")
        return _validated_thesis(thesis)
    if mode in {
        ResearchMode.COMPANY_PROFILE,
        ResearchMode.EARNINGS_REVIEW,
        ResearchMode.INDUSTRY_RESEARCH,
        ResearchMode.MARKET_SNAPSHOT,
        ResearchMode.QUALITY_SCREEN,
    }:
        if thesis is not None:
            raise typer.BadParameter(f"--thesis is not valid in {mode.value} mode")
        if question is None or not question.strip():
            raise typer.BadParameter(f"--question is required in {mode.value} mode")
        return _validated_question(question)
    supplied = [value for value in (thesis, question) if value is not None and value.strip()]
    if len(supplied) != 1:
        raise typer.BadParameter("auto mode requires exactly one of --thesis or --question")
    value = supplied[0]
    return _validated_thesis(value) if thesis is not None else _validated_question(value)


def _validated_thesis(value: str) -> str:
    normalized = " ".join(value.split())
    if not 20 <= len(normalized) <= 500:
        raise typer.BadParameter("--thesis must contain 20 to 500 characters")
    return normalized


def _validated_question(value: str) -> str:
    normalized = " ".join(value.split())
    if not 1 <= len(normalized) <= 2_000:
        raise typer.BadParameter("--question must contain 1 to 2000 characters")
    return normalized


def _validated_research_forms(value: str) -> tuple[FilingForm, ...]:
    parts = tuple(part.strip().upper() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise typer.BadParameter("--forms must contain comma-separated supported forms")
    if any(part not in DEFAULT_RESEARCH_FORMS for part in parts):
        raise typer.BadParameter("--forms only supports 10-K, 10-Q, and 8-K")
    return cast(tuple[FilingForm, ...], parts)


def _validated_research_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter("--as-of-date must use YYYY-MM-DD") from error


@contextmanager
def _materialized_eval_path(
    path: Path | None,
    bundled_path,
):
    if path is not None:
        yield path
        return
    with bundled_path() as default_path:
        yield default_path


def ingest_main() -> None:
    """Run the standalone ingest command."""
    typer.run(ingest)


def research_main() -> None:
    """Run the standalone research command."""
    typer.run(research)


def eval_main() -> None:
    """Run the standalone eval command."""
    typer.run(evaluate)
