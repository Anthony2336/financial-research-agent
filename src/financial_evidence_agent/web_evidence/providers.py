"""Replaceable provider boundary for external web search."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from typing import Protocol
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, HttpUrl

from financial_evidence_agent.web_evidence.source_policy import SourcePolicy

_MAX_REDIRECTS = 5


class RawSearchHit(BaseModel):
    """One provider hit whose URL must be resolved before persistence."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: HttpUrl
    excerpt: str
    published_at: datetime | None = None


class SearchProvider(Protocol):
    """The only external-search behavior used by business logic."""

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        domains: tuple[str, ...],
    ) -> list[RawSearchHit]: ...


class RedirectResolver(Protocol):
    """Resolve redirects while enforcing policy before every network request."""

    async def resolve_with_policy(
        self,
        url: HttpUrl,
        *,
        ticker: str,
        source_policy: SourcePolicy,
    ) -> HttpUrl: ...


class HttpxRedirectResolver:
    """Resolve URLs through bounded, explicit redirects with a fixed timeout."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def resolve(self, url: HttpUrl) -> HttpUrl:
        return await self._resolve(url, validate=lambda _: None)

    async def resolve_with_policy(
        self,
        url: HttpUrl,
        *,
        ticker: str,
        source_policy: SourcePolicy,
    ) -> HttpUrl:
        """Validate every destination before issuing its network request."""
        return await self._resolve(
            url,
            validate=lambda candidate: source_policy.classify(
                ticker=ticker,
                url=candidate,
            ),
        )

    async def _resolve(
        self,
        url: HttpUrl,
        *,
        validate: Callable[[HttpUrl], object],
    ) -> HttpUrl:
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                current = url
                visited: set[str] = set()
                for redirect_count in range(_MAX_REDIRECTS + 1):
                    validate(current)
                    normalized = str(current)
                    if normalized in visited:
                        raise SearchProviderError("web source redirect loop detected")
                    visited.add(normalized)

                    response = await client.get(normalized)
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise SearchProviderError("web source redirect omitted location")
                        if redirect_count == _MAX_REDIRECTS:
                            raise SearchProviderError("web source redirect limit exceeded")
                        current = HttpUrl(urljoin(str(response.url), location))
                        continue
                    response.raise_for_status()
                    return HttpUrl(str(response.url))
                raise SearchProviderError("web source redirect limit exceeded")
        except httpx.TimeoutException as error:
            raise TimeoutError("web source resolution timed out") from error
        except httpx.HTTPStatusError as error:
            raise SearchProviderError(
                "web source resolution failed",
                status_code=error.response.status_code,
            ) from error
        except httpx.HTTPError as error:
            raise SearchProviderError("web source resolution failed") from error


class SearchProviderError(RuntimeError):
    """Provider failure with enough status metadata to decide retry safety."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def transient(self) -> bool:
        return self.status_code == 429 or (
            self.status_code is not None and 500 <= self.status_code <= 599
        )


class TavilyDependencyUnavailableError(RuntimeError):
    """Stable startup failure for a configured provider without its optional package."""


def ensure_tavily_dependency() -> None:
    """Validate the optional Tavily package without constructing a network client."""
    try:
        import_module("tavily")
    except ImportError as error:
        raise TavilyDependencyUnavailableError(
            "WEB_SEARCH_DEPENDENCY_UNAVAILABLE: install the 'web-search' extra"
        ) from error


class TavilySearchProvider:
    """Optional Tavily adapter; importing this module never requires Tavily."""

    def __init__(self, api_key: str) -> None:
        ensure_tavily_dependency()
        module = import_module("tavily")
        self._client = module.AsyncTavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        domains: tuple[str, ...],
    ) -> list[RawSearchHit]:
        if not domains:
            raise ValueError("web provider domains must not be empty")
        try:
            response = await self._client.search(
                query=query,
                max_results=max_results,
                include_domains=list(domains),
            )
        except (TimeoutError, httpx.TimeoutException) as error:
            raise TimeoutError("web provider timed out") from error
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            response_value = getattr(error, "response", None)
            if status_code is None and response_value is not None:
                status_code = getattr(response_value, "status_code", None)
            raise SearchProviderError(str(error), status_code=status_code) from error

        hits: list[RawSearchHit] = []
        for result in response.get("results", []):
            hits.append(
                RawSearchHit(
                    title=result.get("title", "Untitled source"),
                    url=result["url"],
                    excerpt=result.get("content", ""),
                    published_at=result.get("published_date"),
                )
            )
        return hits
