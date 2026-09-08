"""Strict MCP client adapters for P1 filing and allowlisted-web collection."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter

from fastmcp.client.client import CallToolResult
from pydantic import ValidationError

from financial_evidence_agent.context import BudgetExhaustedError, BudgetGate
from financial_evidence_agent.domain import EvidenceChunk, WebEvidence
from financial_evidence_agent.graph.models import MCPToolClient, SourceFetchWriter
from financial_evidence_agent.mcp_server.tools import HybridSearchFilingsResponse
from financial_evidence_agent.mcp_server.web_tools import SearchAllowlistedWebResponse
from financial_evidence_agent.observability import ObservationHandle, observe
from financial_evidence_agent.retrieval.collector import (
    EvidenceQuery,
    LocalEvidenceHit,
    LocalSearchResponse,
    RetrievalError,
    RetrievalErrorCode,
    WebEvidenceHit,
    WebSearchRequest,
    WebSearchResponse,
)
from financial_evidence_agent.storage.run_repositories import SourceFetchWrite

_FILING_TOOL = "hybrid_search_filings"
_WEB_TOOL = "search_allowlisted_web"
_PROTOCOL_ERROR = "MCP_PROTOCOL_ERROR"
_CALL_ERROR = "MCP_CALL_ERROR"
logger = logging.getLogger(__name__)


class MCPFilingSearch:
    """Collect filing evidence only through the schema-validated MCP tool."""

    def __init__(
        self,
        client: MCPToolClient,
        *,
        source_fetch_writer: SourceFetchWriter | None = None,
        run_id: str | None = None,
        budget_gate: BudgetGate | None = None,
    ) -> None:
        _validate_fetch_context(source_fetch_writer, run_id)
        self._client = client
        self._source_fetch_writer = source_fetch_writer
        self._run_id = run_id
        self._budget_gate = budget_gate

    def with_budget_gate(self, gate: BudgetGate) -> MCPFilingSearch:
        return MCPFilingSearch(
            self._client,
            source_fetch_writer=self._source_fetch_writer,
            run_id=self._run_id,
            budget_gate=gate,
        )

    async def __call__(self, queries: tuple[EvidenceQuery, ...]) -> LocalSearchResponse:
        if not queries:
            return _local_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "local search requires at least one typed query",
            )
        tickers = {query.ticker.strip().upper() for query in queries}
        if len(tickers) != 1 or "" in tickers:
            return _local_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "local search queries must target exactly one ticker",
            )
        ticker = next(iter(tickers))
        try:
            requests = tuple((query, _filing_arguments(query, ticker)) for query in queries)
        except ValueError:
            return _local_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "local search target metadata exceeds 500 characters",
            )

        hits: list[LocalEvidenceHit] = []
        corpus_version: str | None = None
        for query, arguments in requests:
            chunks, error, returned_corpus = await _call_filing_tool(
                client=self._client,
                writer=self._source_fetch_writer,
                run_id=self._run_id,
                query=str(arguments["query"]),
                ticker=ticker,
                arguments=arguments,
                corpus_version=corpus_version,
                budget_gate=self._budget_gate,
            )
            if error is not None:
                return error
            if returned_corpus is not None:
                corpus_version = returned_corpus
            if not chunks:
                continue
            hits.extend(
                LocalEvidenceHit(
                    evidence=chunk,
                    question_index=query.question_index,
                    side=query.side,
                    facet=query.facet,
                )
                for chunk in chunks
            )
        if not hits:
            return _local_error(
                RetrievalErrorCode.EMPTY_RETRIEVAL,
                "local filing retrieval returned no usable evidence",
            )
        return LocalSearchResponse(evidence=tuple(hits))


class MCPWebSearch:
    """Collect web evidence only through the source-policy-enforcing MCP tool."""

    def __init__(
        self,
        client: MCPToolClient,
        *,
        source_fetch_writer: SourceFetchWriter | None = None,
        run_id: str | None = None,
        budget_gate: BudgetGate | None = None,
    ) -> None:
        _validate_fetch_context(source_fetch_writer, run_id)
        self._client = client
        self._source_fetch_writer = source_fetch_writer
        self._run_id = run_id
        self._budget_gate = budget_gate

    def with_budget_gate(self, gate: BudgetGate) -> MCPWebSearch:
        return MCPWebSearch(
            self._client,
            source_fetch_writer=self._source_fetch_writer,
            run_id=self._run_id,
            budget_gate=gate,
        )

    async def __call__(self, request: WebSearchRequest) -> WebSearchResponse:
        if len(request.missing_pairs) != 1 or len(request.missing_facets) != 1:
            return _web_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "web fallback requires exactly one typed coverage target",
            )
        ticker = request.ticker.strip().upper()
        arguments: dict[str, object] = {
            "ticker": ticker,
            "query": request.query,
            "max_results": request.max_results,
        }
        evidence, error = await _call_web_tool(
            client=self._client,
            writer=self._source_fetch_writer,
            run_id=self._run_id,
            ticker=ticker,
            query=request.query,
            max_results=request.max_results,
            arguments=arguments,
            budget_gate=self._budget_gate,
        )
        if error is not None:
            return error
        question_index, side = request.missing_pairs[0]
        facet = request.missing_facets[0]
        return WebSearchResponse(
            evidence=tuple(
                WebEvidenceHit(
                    evidence=item,
                    question_index=question_index,
                    side=side,
                    facet=facet,
                )
                for item in evidence
            )
        )


async def _call_filing_tool(
    *,
    client: MCPToolClient,
    writer: SourceFetchWriter | None,
    run_id: str | None,
    query: str,
    ticker: str,
    arguments: dict[str, object],
    corpus_version: str | None,
    budget_gate: BudgetGate | None,
) -> tuple[tuple[EvidenceChunk, ...], LocalSearchResponse | None, str | None]:
    attempt = _FetchAttempt(
        writer=writer,
        run_id=run_id,
        source_kind="filing",
        tool=_FILING_TOOL,
        ticker=ticker,
        query=query,
        requested_at=datetime.now(UTC),
    )
    started_at = perf_counter()
    with observe(
        name=f"mcp.{_FILING_TOOL}",
        kind="tool",
        metadata=_safe_tool_metadata(_FILING_TOOL, ticker, query),
    ) as observation:
        try:
            if budget_gate is not None:
                budget_gate.consume(tool_calls=1)
            raw_response = await asyncio.to_thread(
                client.call_tool,
                _FILING_TOOL,
                arguments,
            )
        except BudgetExhaustedError:
            _finish_tool_observation(
                observation,
                started_at,
                error_code="BUDGET_EXHAUSTED",
            )
            raise
        except Exception:
            attempt.finish(fetched_at=None, error_code=_CALL_ERROR)
            _finish_tool_observation(observation, started_at, error_code=_CALL_ERROR)
            return (
                (),
                _local_error(
                    RetrievalErrorCode.OPERATION_ERROR,
                    "local filing retrieval failed",
                ),
                corpus_version,
            )

        fetched_at = datetime.now(UTC)
        try:
            response = HybridSearchFilingsResponse.model_validate(_structured_payload(raw_response))
            if response.error is not None and response.chunks:
                raise _AdapterProtocolError
        except (ValidationError, _AdapterProtocolError):
            attempt.finish(fetched_at=fetched_at, error_code=_PROTOCOL_ERROR)
            _finish_tool_observation(
                observation,
                started_at,
                error_code=_PROTOCOL_ERROR,
            )
            return (
                (),
                _local_error(
                    RetrievalErrorCode.PROTOCOL_ERROR,
                    "local search returned an invalid MCP envelope",
                ),
                corpus_version,
            )

        if response.error is not None:
            error_code = response.error.code
            attempt.finish(fetched_at=fetched_at, error_code=error_code)
            _finish_tool_observation(
                observation,
                started_at,
                error_code=error_code,
                cache_hit=(response.metrics.cache_hit if response.metrics is not None else None),
            )
            if error_code == "EMPTY_RETRIEVAL":
                return (), None, corpus_version
            if error_code == "RERANKER_MODEL_UNAVAILABLE":
                error = _local_error(
                    RetrievalErrorCode.RERANKER_UNAVAILABLE,
                    "RERANKER_MODEL_UNAVAILABLE: configured FlashRank assets are unavailable",
                )
                return (), error, corpus_version
            if error_code in {"UNSUPPORTED_TICKER", "NO_FILINGS"}:
                error = _local_error(
                    RetrievalErrorCode.OPERATION_ERROR,
                    "local filing corpus is unavailable",
                )
            else:
                error = _local_error(
                    RetrievalErrorCode.PROTOCOL_ERROR,
                    "local search returned an invalid MCP tool error",
                )
            return (), error, corpus_version

        response_corpora = {chunk.corpus_version for chunk in response.chunks}
        if (
            len(response.chunks) > int(arguments["k"])
            or any(chunk.ticker.strip().upper() != ticker for chunk in response.chunks)
            or len(response_corpora) > 1
            or (
                corpus_version is not None
                and response_corpora
                and corpus_version not in response_corpora
            )
        ):
            attempt.finish(fetched_at=fetched_at, error_code=_PROTOCOL_ERROR)
            _finish_tool_observation(
                observation,
                started_at,
                error_code=_PROTOCOL_ERROR,
                cache_hit=(response.metrics.cache_hit if response.metrics is not None else None),
            )
            return (
                (),
                _local_error(
                    RetrievalErrorCode.PROTOCOL_ERROR,
                    "local filing retrieval violated its requested scope",
                ),
                corpus_version,
            )

        returned_corpus = next(iter(response_corpora)) if response_corpora else corpus_version
        if not response.chunks:
            attempt.finish(fetched_at=fetched_at, error_code="EMPTY_RETRIEVAL")
            _finish_tool_observation(
                observation,
                started_at,
                error_code="EMPTY_RETRIEVAL",
                cache_hit=(response.metrics.cache_hit if response.metrics is not None else None),
            )
            return (), None, returned_corpus

        attempt.finish(fetched_at=fetched_at, error_code=None)
        metadata: dict[str, object] = {"latency_ms": _latency_ms(started_at)}
        if response.metrics is not None:
            metadata["cache_hit"] = response.metrics.cache_hit
        observation.update(
            output={"status": "completed", "result_count": len(response.chunks)},
            metadata=metadata,
        )
        return tuple(response.chunks), None, returned_corpus


async def _call_web_tool(
    *,
    client: MCPToolClient,
    writer: SourceFetchWriter | None,
    run_id: str | None,
    ticker: str,
    query: str,
    max_results: int,
    arguments: dict[str, object],
    budget_gate: BudgetGate | None,
) -> tuple[tuple[WebEvidence, ...], WebSearchResponse | None]:
    attempt = _FetchAttempt(
        writer=writer,
        run_id=run_id,
        source_kind="web",
        tool=_WEB_TOOL,
        ticker=ticker,
        query=query,
        requested_at=datetime.now(UTC),
    )
    started_at = perf_counter()
    with observe(
        name=f"mcp.{_WEB_TOOL}",
        kind="tool",
        metadata=_safe_tool_metadata(_WEB_TOOL, ticker, query),
    ) as observation:
        try:
            if budget_gate is not None:
                budget_gate.consume(tool_calls=1)
            raw_response = await asyncio.to_thread(
                client.call_tool,
                _WEB_TOOL,
                arguments,
            )
        except BudgetExhaustedError:
            _finish_tool_observation(
                observation,
                started_at,
                error_code="BUDGET_EXHAUSTED",
            )
            raise
        except Exception:
            attempt.finish(fetched_at=None, error_code=_CALL_ERROR)
            _finish_tool_observation(observation, started_at, error_code=_CALL_ERROR)
            return (), _web_error(
                RetrievalErrorCode.OPERATION_ERROR,
                "allowlisted web retrieval failed",
            )

        fetched_at = datetime.now(UTC)
        try:
            response = SearchAllowlistedWebResponse.model_validate(
                _structured_payload(raw_response)
            )
            if response.error is not None and response.evidence:
                raise _AdapterProtocolError
        except (ValidationError, _AdapterProtocolError):
            attempt.finish(fetched_at=fetched_at, error_code=_PROTOCOL_ERROR)
            _finish_tool_observation(
                observation,
                started_at,
                error_code=_PROTOCOL_ERROR,
            )
            return (), _web_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "allowlisted web search returned an invalid MCP envelope",
            )

        if response.error is not None:
            error_code = response.error.code
            attempt.finish(fetched_at=fetched_at, error_code=error_code)
            _finish_tool_observation(observation, started_at, error_code=error_code)
            return (), _web_error(
                RetrievalErrorCode.OPERATION_ERROR,
                "allowlisted web retrieval failed",
            )

        if len(response.evidence) > max_results or any(
            item.ticker.strip().upper() != ticker for item in response.evidence
        ):
            attempt.finish(fetched_at=fetched_at, error_code=_PROTOCOL_ERROR)
            _finish_tool_observation(
                observation,
                started_at,
                error_code=_PROTOCOL_ERROR,
            )
            return (), _web_error(
                RetrievalErrorCode.PROTOCOL_ERROR,
                "allowlisted web retrieval violated its requested scope",
            )

        if not response.evidence:
            attempt.finish(fetched_at=fetched_at, error_code="EMPTY_RETRIEVAL")
            _finish_tool_observation(
                observation,
                started_at,
                error_code="EMPTY_RETRIEVAL",
            )
            return (), _web_error(
                RetrievalErrorCode.EMPTY_RETRIEVAL,
                "allowlisted web retrieval returned no usable evidence",
            )

        attempt.finish(fetched_at=fetched_at, error_code=None)
        observation.update(
            output={"status": "completed", "result_count": len(response.evidence)},
            metadata={"latency_ms": _latency_ms(started_at)},
        )
        return tuple(response.evidence), None


def _validate_fetch_context(
    writer: SourceFetchWriter | None,
    run_id: str | None,
) -> None:
    if (writer is None) != (run_id is None):
        raise ValueError("source_fetch_writer and run_id must be provided together")


def _filing_arguments(query: EvidenceQuery, ticker: str) -> dict[str, object]:
    targeted_query = _targeted_filing_query(query, ticker)
    arguments: dict[str, object] = {
        "ticker": ticker,
        "query": targeted_query,
        "filing_ids": list(query.filing_ids),
        "evidence_side": query.side.value,
        "k": min(query.limit, 8),
    }
    if query.corpus_version is not None:
        arguments["corpus_version"] = query.corpus_version
    return arguments


def _targeted_filing_query(query: EvidenceQuery, ticker: str) -> str:
    retry_suffix = (
        f" | ticker={query.ticker} | facet={query.facet.value} "
        f"| side={query.side.value} | retry=missing_only"
    )
    is_retry = query.query.endswith(retry_suffix)
    free_form_query = query.query[: -len(retry_suffix)] if is_retry else query.query
    target_suffix = (
        f" | ticker={ticker} | question={query.question_index} "
        f"| facet={query.facet.value} | side={query.side.value}"
    )
    if is_retry:
        target_suffix = f"{target_suffix} | retry=missing_only"
    if len(target_suffix) > 500:
        raise ValueError("target metadata exceeds MCP query limit")
    return f"{free_form_query[: 500 - len(target_suffix)]}{target_suffix}"


def _structured_payload(raw_response: object) -> Mapping[str, object]:
    if not isinstance(raw_response, CallToolResult):
        raise _AdapterProtocolError
    if raw_response.is_error or not isinstance(raw_response.structured_content, Mapping):
        raise _AdapterProtocolError
    return raw_response.structured_content


@dataclass(frozen=True, slots=True)
class _FetchAttempt:
    writer: SourceFetchWriter | None
    run_id: str | None
    source_kind: str
    tool: str
    ticker: str
    query: str
    requested_at: datetime

    def finish(self, *, fetched_at: datetime | None, error_code: str | None) -> None:
        if self.writer is None or self.run_id is None:
            return
        query_hash = sha256(self.query.encode("utf-8")).hexdigest()
        try:
            self.writer.record_fetch(
                SourceFetchWrite(
                    run_id=self.run_id,
                    source_kind=self.source_kind,
                    source_ref=f"{self.tool}:{self.ticker}:sha256:{query_hash}",
                    requested_at=self.requested_at,
                    fetched_at=fetched_at,
                    status="completed" if error_code is None else "failed",
                    error_code=error_code,
                )
            )
        except Exception:
            logger.warning("source-fetch persistence failed")


def _local_error(code: RetrievalErrorCode, message: str) -> LocalSearchResponse:
    return LocalSearchResponse(error=RetrievalError(code=code, message=message))


def _web_error(code: RetrievalErrorCode, message: str) -> WebSearchResponse:
    return WebSearchResponse(error=RetrievalError(code=code, message=message))


class _AdapterProtocolError(Exception):
    """An MCP envelope failed before it could become trusted evidence."""


def _safe_tool_metadata(tool: str, ticker: str, query: str) -> dict[str, object]:
    return {
        "tool": tool,
        "ticker": ticker,
        "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
    }


def _latency_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1_000))


def _finish_tool_observation(
    observation: ObservationHandle,
    started_at: float,
    *,
    error_code: str,
    cache_hit: bool | None = None,
) -> None:
    metadata: dict[str, object] = {
        "latency_ms": _latency_ms(started_at),
        "error_code": error_code,
    }
    if cache_hit is not None:
        metadata["cache_hit"] = cache_hit
    observation.update(
        output={"status": "failed", "result_count": 0},
        metadata=metadata,
    )
