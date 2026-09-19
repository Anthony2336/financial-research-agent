"""Pydantic contracts and registrations for the read-only filing tools."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field, field_validator, model_validator

from fra.domain import EvidenceChunk, StrictModel
from fra.retrieval.hybrid import (
    HybridRetriever,
    RetrievalMetrics,
)
from fra.retrieval.rerank import RerankerModelUnavailableError
from fra.storage.repositories import (
    CompanyRead,
    FilingRead,
    FilingRepository,
)

_FORMS = Literal["10-K", "10-Q", "8-K"]
_ERROR_CODES = Literal[
    "UNSUPPORTED_TICKER",
    "NO_FILINGS",
    "EMPTY_RETRIEVAL",
    "RERANKER_MODEL_UNAVAILABLE",
]


class _TickerInput(StrictModel):
    ticker: str = Field(min_length=1, max_length=10)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must not be empty")
        return normalized


class ResolveCompanyInput(_TickerInput):
    """Flat input accepted by the resolve-company MCP tool."""


class FetchRecentFilingsInput(_TickerInput):
    """Flat input accepted by the recent-filings MCP tool."""

    forms: list[_FORMS] = Field(default_factory=list)
    as_of_date: date | None = None
    limit: int = Field(default=4, ge=1, le=4)


class HybridSearchFilingsInput(_TickerInput):
    """Flat input accepted by the filing-search MCP tool."""

    query: str = Field(min_length=1, max_length=500)
    corpus_version: str | None = Field(default=None, min_length=1)
    filing_ids: list[str] = Field(default_factory=list)
    evidence_side: Literal["support", "challenge"] | None = None
    k: int = Field(default=8, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    @field_validator("filing_ids")
    @classmethod
    def normalize_filing_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("filing_ids must not contain empty ids")
        return normalized

    @field_validator("corpus_version")
    @classmethod
    def normalize_corpus_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("corpus_version must not be blank")
        return normalized


class GetSourceSpansInput(StrictModel):
    """Flat input accepted by the source-span MCP tool."""

    chunk_ids: list[str] = Field(min_length=1)

    @field_validator("chunk_ids")
    @classmethod
    def normalize_chunk_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("chunk_ids must not contain empty ids")
        return normalized


class ToolError(StrictModel):
    """Machine-readable recovery information for a valid but empty request."""

    code: _ERROR_CODES
    message: str = Field(min_length=1, max_length=200)


class CompanyOutput(StrictModel):
    ticker: str
    cik: str | None = None
    legal_name: str | None = None
    ir_domain: str | None = None


class FilingOutput(StrictModel):
    id: str
    ticker: str
    form: _FORMS
    filed_at: str
    source_url: str
    accession_no: str
    corpus_version: str = Field(min_length=1)


class QuotedSource(StrictModel):
    chunk_id: str
    content: str
    source_url: str
    form: _FORMS
    filed_at: str
    accession_no: str
    section: str
    raw_start: int
    raw_end: int


class _ToolResponse(StrictModel):
    error: ToolError | None = None


class ResolveCompanyResponse(_ToolResponse):
    company: CompanyOutput | None = None


class FetchRecentFilingsResponse(_ToolResponse):
    corpus_version: str | None = Field(default=None, min_length=1)
    filings: list[FilingOutput] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> FetchRecentFilingsResponse:
        if self.error is not None:
            if self.corpus_version is not None or self.filings:
                raise ValueError("filing response error cannot include success values")
            return self
        if self.corpus_version is None or not self.filings:
            raise ValueError("successful filing response requires version and filings")
        return self


class HybridSearchFilingsResponse(_ToolResponse):
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    metrics: RetrievalMetrics | None = None


class GetSourceSpansResponse(_ToolResponse):
    sources: list[QuotedSource] = Field(default_factory=list)


def register_tools(mcp: FastMCP, repository: FilingRepository, retriever: HybridRetriever) -> None:
    """Register exactly the four P0 tool functions against an injectable server."""

    @mcp.tool(name="resolve_company", output_schema=ResolveCompanyResponse.model_json_schema())
    def resolve_company(
        ticker: Annotated[str, Field(min_length=1, max_length=10)],
    ) -> ResolveCompanyResponse:
        """Resolve one supported US ticker to persisted company metadata."""
        request = ResolveCompanyInput(ticker=ticker)
        company = repository.get_company(request.ticker)
        if company is None:
            return ResolveCompanyResponse(
                error=_error("UNSUPPORTED_TICKER", f"No supported company for {request.ticker}")
            )
        return ResolveCompanyResponse(company=_company_output(company))

    @mcp.tool(
        name="fetch_recent_filings",
        output_schema=FetchRecentFilingsResponse.model_json_schema(),
    )
    def fetch_recent_filings(
        ticker: Annotated[str, Field(min_length=1, max_length=10)],
        forms: list[_FORMS] = [],
        as_of_date: date | None = None,
        limit: Annotated[int, Field(ge=1, le=4)] = 4,
    ) -> FetchRecentFilingsResponse:
        """List up to four recent, allowlisted SEC filings for a supported ticker."""
        request = FetchRecentFilingsInput(
            ticker=ticker,
            forms=forms,
            as_of_date=as_of_date,
            limit=limit,
        )
        if repository.get_company(request.ticker) is None:
            return FetchRecentFilingsResponse(
                error=_error("UNSUPPORTED_TICKER", f"No supported company for {request.ticker}")
            )
        corpus_version = repository.latest_corpus_version(
            request.ticker,
            as_of_date=request.as_of_date,
            forms=request.forms,
        )
        filings = repository.list_recent_filings(
            request.ticker,
            request.forms,
            request.limit,
            as_of_date=request.as_of_date,
            corpus_version=corpus_version,
        )
        if not filings:
            return FetchRecentFilingsResponse(
                error=_error("NO_FILINGS", f"No matching filings for {request.ticker}")
            )
        return FetchRecentFilingsResponse(
            corpus_version=corpus_version,
            filings=[_filing_output(filing) for filing in filings],
        )

    @mcp.tool(
        name="hybrid_search_filings",
        output_schema=HybridSearchFilingsResponse.model_json_schema(),
    )
    def hybrid_search_filings(
        ticker: Annotated[str, Field(min_length=1, max_length=10)],
        query: Annotated[str, Field(min_length=1, max_length=500)],
        corpus_version: str | None = None,
        filing_ids: list[str] = [],
        evidence_side: Literal["support", "challenge"] | None = None,
        k: Annotated[int, Field(ge=1, le=8)] = 8,
    ) -> HybridSearchFilingsResponse:
        """Search the selected P0 filing corpus and return citable chunks only."""
        request = HybridSearchFilingsInput(
            ticker=ticker,
            query=query,
            corpus_version=corpus_version,
            filing_ids=filing_ids,
            evidence_side=evidence_side,
            k=k,
        )
        if repository.get_company(request.ticker) is None:
            return HybridSearchFilingsResponse(
                error=_error("UNSUPPORTED_TICKER", f"No supported company for {request.ticker}")
            )
        selected_scope = _selected_corpus_scope(repository, request)
        if selected_scope is None:
            return HybridSearchFilingsResponse(
                error=_error("NO_FILINGS", f"No selected filings for {request.ticker}")
            )
        selected_version, selected_filing_ids = selected_scope
        try:
            search_with_metrics = getattr(retriever, "search_with_metrics", None)
            if callable(search_with_metrics):
                arguments = (
                    request.ticker,
                    request.query,
                    selected_version,
                    request.k,
                )
                search_options: dict[str, object] = {}
                if selected_filing_ids is not None:
                    search_options["filing_ids"] = selected_filing_ids
                if request.evidence_side is not None:
                    search_options["evidence_side"] = request.evidence_side
                retrieval = search_with_metrics(*arguments, **search_options)
                chunks = list(retrieval.evidence)
                metrics = retrieval.metrics
            else:
                arguments = (
                    request.ticker,
                    request.query,
                    selected_version,
                    request.k,
                )
                search_options = {}
                if selected_filing_ids is not None:
                    search_options["filing_ids"] = selected_filing_ids
                if request.evidence_side is not None:
                    search_options["evidence_side"] = request.evidence_side
                chunks = retriever.search(*arguments, **search_options)
                metrics = None
        except RerankerModelUnavailableError:
            return HybridSearchFilingsResponse(
                error=_error(
                    "RERANKER_MODEL_UNAVAILABLE",
                    "Configured FlashRank assets are unavailable",
                )
            )
        if not chunks:
            return HybridSearchFilingsResponse(
                error=_error("EMPTY_RETRIEVAL", "No evidence matched the selected filing corpus"),
                metrics=metrics,
            )
        return HybridSearchFilingsResponse(chunks=chunks, metrics=metrics)

    @mcp.tool(
        name="get_source_spans",
        output_schema=GetSourceSpansResponse.model_json_schema(),
    )
    def get_source_spans(
        chunk_ids: Annotated[list[str], Field(min_length=1)],
    ) -> GetSourceSpansResponse:
        """Return source text and display metadata for previously selected chunk ids."""
        request = GetSourceSpansInput(chunk_ids=chunk_ids)
        chunks = repository.list_chunks_by_ids(request.chunk_ids)
        if len(chunks) != len(request.chunk_ids):
            return GetSourceSpansResponse(
                error=_error("EMPTY_RETRIEVAL", "One or more source chunk ids were not found")
            )
        return GetSourceSpansResponse(sources=[_quoted_source(chunk) for chunk in chunks])


def _selected_corpus_scope(
    repository: FilingRepository, request: HybridSearchFilingsInput
) -> tuple[str, tuple[str, ...] | None] | None:
    """Resolve one immutable snapshot and an optional exact member subset."""
    if request.corpus_version is not None:
        filings = repository.list_recent_filings(
            request.ticker,
            forms=[],
            limit=4,
            corpus_version=request.corpus_version,
        )
        allowed_ids = {filing.id for filing in filings}
        if not allowed_ids:
            return None
        if request.filing_ids:
            if (
                len(request.filing_ids) != len(set(request.filing_ids))
                or set(request.filing_ids) - allowed_ids
            ):
                return None
            return request.corpus_version, tuple(request.filing_ids)
        return request.corpus_version, None
    if request.filing_ids:
        selected = repository.corpus_version_for_filings(request.ticker, request.filing_ids)
        return (selected, tuple(request.filing_ids)) if selected is not None else None
    selected = repository.latest_corpus_version(request.ticker)
    return (selected, None) if selected is not None else None


def _error(code: _ERROR_CODES, message: str) -> ToolError:
    return ToolError(code=code, message=message)


def _company_output(company: CompanyRead) -> CompanyOutput:
    return CompanyOutput(
        ticker=company.ticker,
        cik=company.cik,
        legal_name=company.legal_name,
        ir_domain=company.ir_domain,
    )


def _filing_output(filing: FilingRead) -> FilingOutput:
    return FilingOutput(
        id=filing.id,
        ticker=filing.ticker,
        form=filing.form,
        filed_at=filing.filed_at.isoformat(),
        source_url=filing.source_url,
        accession_no=filing.accession_no,
        corpus_version=filing.corpus_version,
    )


def _quoted_source(chunk: EvidenceChunk) -> QuotedSource:
    return QuotedSource(
        chunk_id=chunk.id,
        content=chunk.content,
        source_url=chunk.source_url,
        form=chunk.form,
        filed_at=chunk.filed_at.isoformat(),
        accession_no=chunk.accession_no,
        section=chunk.section,
        raw_start=chunk.raw_start,
        raw_end=chunk.raw_end,
    )
