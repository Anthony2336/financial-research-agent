"""FastMCP registrations for allowlisted P1 web evidence."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import ConfigDict, Field, StrictInt, field_validator

from fra.domain import StrictModel, WebEvidence
from fra.storage.web_repositories import WebEvidenceRepository
from fra.web_evidence.gateway import AllowlistedWebGateway, WebGatewayError
from fra.web_evidence.source_policy import (
    PersistedWebEvidenceValidator,
    SourcePolicy,
    WebEvidenceValidationError,
)

_WEB_ERROR_CODE = Literal[
    "INVALID_WEB_REQUEST",
    "SOURCE_NOT_ALLOWED",
    "WEB_PROVIDER_ERROR",
    "WEB_PROVIDER_TIMEOUT",
    "WEB_PROVIDER_UNAVAILABLE",
]


class SearchAllowlistedWebInput(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    ticker: str = Field(min_length=1, max_length=10)
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=3, ge=1, le=3)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class SearchAuthoritativeEventsInput(SearchAllowlistedWebInput):
    """Fixed-policy event search input; callers cannot provide URLs or domains."""

    window_start: datetime
    window_end: datetime

    @field_validator("window_start", "window_end")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("event windows must use timezone-aware UTC timestamps")
        return value


class GetWebEvidenceInput(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def reject_empty_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evidence_ids must not contain empty ids")
        return normalized


class WebToolError(StrictModel):
    """Machine-readable web recovery information."""

    code: _WEB_ERROR_CODE
    message: str = Field(min_length=1, max_length=200)


class SearchAllowlistedWebResponse(StrictModel):
    evidence: list[WebEvidence] = Field(default_factory=list)
    error: WebToolError | None = None


class GetWebEvidenceResponse(StrictModel):
    evidence: list[WebEvidence] = Field(default_factory=list)
    error: WebToolError | None = None


class SearchAuthoritativeEventsResponse(StrictModel):
    evidence: list[WebEvidence] = Field(default_factory=list)
    error: WebToolError | None = None


def register_web_tools(
    mcp: FastMCP,
    gateway: AllowlistedWebGateway | None,
    repository: WebEvidenceRepository,
    source_policy: SourcePolicy | None = None,
) -> None:
    """Register the two injectable P1 tools without altering P0 tool contracts."""

    @mcp.tool(
        name="search_allowlisted_web",
        output_schema=SearchAllowlistedWebResponse.model_json_schema(),
    )
    async def search_allowlisted_web(
        ticker: Annotated[str, Field(min_length=1, max_length=10)],
        query: Annotated[str, Field(min_length=1, max_length=500)],
        max_results: Annotated[StrictInt, Field(ge=1, le=3)] = 3,
    ) -> SearchAllowlistedWebResponse:
        """Search approved issuer, SEC, and Reuters sources and persist snapshots."""
        request = SearchAllowlistedWebInput(
            ticker=ticker, query=query, max_results=max_results
        )
        if gateway is None:
            return SearchAllowlistedWebResponse(
                error=WebToolError(
                    code="WEB_PROVIDER_UNAVAILABLE",
                    message="web search provider is not configured",
                )
            )
        try:
            evidence = await gateway.search(**request.model_dump())
        except WebGatewayError as error:
            return SearchAllowlistedWebResponse(
                error=WebToolError(code=error.code, message=str(error))
            )
        return SearchAllowlistedWebResponse(evidence=evidence)

    validator = (
        PersistedWebEvidenceValidator(source_policy, repository)
        if source_policy is not None
        else None
    )

    @mcp.tool(
        name="search_authoritative_events",
        output_schema=SearchAuthoritativeEventsResponse.model_json_schema(),
    )
    async def search_authoritative_events(
        ticker: Annotated[str, Field(min_length=1, max_length=10)],
        query: Annotated[str, Field(min_length=1, max_length=500)],
        window_start: datetime,
        window_end: datetime,
        max_results: Annotated[StrictInt, Field(ge=1, le=3)] = 3,
    ) -> SearchAuthoritativeEventsResponse:
        """Search frozen approved sources and retain only canonical timed evidence."""
        request = SearchAuthoritativeEventsInput(
            ticker=ticker,
            query=query,
            window_start=window_start,
            window_end=window_end,
            max_results=max_results,
        )
        if request.window_start > request.window_end:
            return SearchAuthoritativeEventsResponse(
                error=WebToolError(
                    code="INVALID_WEB_REQUEST", message="invalid event time window"
                )
            )
        if gateway is None or validator is None:
            return SearchAuthoritativeEventsResponse(
                error=WebToolError(
                    code="WEB_PROVIDER_UNAVAILABLE",
                    message="authoritative event search is not configured",
                )
            )
        try:
            searched = await gateway.search(
                ticker=request.ticker,
                query=request.query,
                max_results=request.max_results,
            )
        except WebGatewayError as error:
            return SearchAuthoritativeEventsResponse(
                error=WebToolError(code=error.code, message=str(error))
            )
        evidence: list[WebEvidence] = []
        for item in searched:
            try:
                canonical = validator.validate(ticker=request.ticker, evidence=item)
            except WebEvidenceValidationError:
                continue
            if (
                canonical.published_at is None
                or canonical.published_at < request.window_start
                or canonical.published_at > request.window_end
            ):
                continue
            evidence.append(canonical)
        return SearchAuthoritativeEventsResponse(evidence=evidence[: request.max_results])

    @mcp.tool(
        name="get_web_evidence",
        output_schema=GetWebEvidenceResponse.model_json_schema(),
    )
    async def get_web_evidence(
        evidence_ids: Annotated[list[str], Field(min_length=1)],
    ) -> GetWebEvidenceResponse:
        """Read previously persisted, citable web snapshots in requested order."""
        request = GetWebEvidenceInput(evidence_ids=evidence_ids)
        return GetWebEvidenceResponse(evidence=repository.get_many(request.evidence_ids))
