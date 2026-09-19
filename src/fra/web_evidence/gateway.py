"""Bounded allowlisted web search with versioned snapshot persistence."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

from pydantic import HttpUrl, TypeAdapter
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_none

from fra.domain import SourceKind, SourceTier, WebEvidence
from fra.storage.cache import JsonCache, NoopJsonCache
from fra.storage.web_repositories import WebEvidenceRepository
from fra.web_evidence.providers import (
    RawSearchHit,
    RedirectResolver,
    SearchProvider,
    SearchProviderError,
)
from fra.web_evidence.source_policy import (
    SourcePolicy,
    SourcePolicyError,
    canonicalize_source_url,
    content_addressed_web_evidence_id,
)

_MAX_RESULTS = 3
_MAX_EXCERPT_CHARACTERS = 1_200
_CACHE_TTL_SECONDS = 900
_WEB_EVIDENCE_LIST = TypeAdapter(list[WebEvidence])

logger = logging.getLogger(__name__)


class WebGatewayError(RuntimeError):
    """Machine-readable failure raised by the allowlisted gateway."""

    def __init__(
        self,
        code: Literal[
            "INVALID_WEB_REQUEST",
            "SOURCE_NOT_ALLOWED",
            "WEB_PROVIDER_ERROR",
            "WEB_PROVIDER_TIMEOUT",
        ],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class AllowlistedWebGateway:
    """Search through one provider and persist only validated, bounded evidence."""

    def __init__(
        self,
        *,
        provider: SearchProvider,
        redirect_resolver: RedirectResolver,
        source_policy: SourcePolicy,
        repository: WebEvidenceRepository,
        timeout_seconds: float = 10.0,
        cache: JsonCache | None = None,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
        source_policy_version: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        if not callable(getattr(redirect_resolver, "resolve_with_policy", None)):
            raise TypeError("redirect_resolver must implement resolve_with_policy")
        self._provider = provider
        self._redirect_resolver = redirect_resolver
        self._source_policy = source_policy
        self._repository = repository
        self._timeout_seconds = timeout_seconds
        self._cache = NoopJsonCache() if cache is None else cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._source_policy_version = source_policy_version or source_policy.version
        self._clock = clock

    async def search(
        self,
        *,
        ticker: str,
        query: str,
        max_results: int = _MAX_RESULTS,
    ) -> list[WebEvidence]:
        """Return up to three deduplicated, policy-approved evidence snapshots."""
        normalized_ticker = ticker.strip().upper()
        normalized_query = query.strip()
        if not normalized_ticker or not normalized_query or not 1 <= max_results <= _MAX_RESULTS:
            raise WebGatewayError("INVALID_WEB_REQUEST", "invalid web search request")

        cache_key = _web_cache_key(
            ticker=normalized_ticker,
            query=normalized_query,
            source_policy_version=self._source_policy_version,
            max_results=max_results,
        )
        cached = await self._cached_evidence(
            cache_key,
            ticker=normalized_ticker,
            max_results=max_results,
        )
        if cached is not None:
            return cached

        hits = await self._search_provider(
            normalized_query,
            ticker=normalized_ticker,
            max_results=max_results,
        )
        provider_returned_at = _utc_now(self._clock)
        approved = await self._resolve_hits(
            normalized_ticker,
            hits,
            max_results=max_results,
        )
        fetched_at = _utc_now(self._clock)
        if fetched_at < provider_returned_at:
            raise ValueError("web gateway clock moved backwards during source resolution")
        approved = [
            item
            for item in approved
            if _normalized_published_at(item[0].published_at, fetched_at) is not None
        ]
        evidence: list[WebEvidence] = []
        for hit, source_kind, source_tier, content in approved:
            canonical = self._repository.upsert(
                WebEvidence(
                    id="pending",
                    ticker=normalized_ticker,
                    title=hit.title.strip() or "Untitled source",
                    content=content,
                    source_url=hit.url,
                    source_kind=source_kind,
                    source_tier=source_tier,
                    published_at=hit.published_at,
                    fetched_at=fetched_at,
                    content_hash=sha256(content.encode("utf-8")).hexdigest(),
                )
            )
            if canonical is None or not _has_valid_web_times(canonical):
                logger.warning("persisted web evidence timestamp rejected")
                continue
            evidence.append(canonical)
        if evidence:
            try:
                await self._cache.set_json(
                    cache_key,
                    [item.model_dump(mode="json") for item in evidence],
                    ttl_seconds=self._cache_ttl_seconds,
                )
            except Exception:
                logger.warning("web cache write failed")
        return evidence

    async def _cached_evidence(
        self,
        key: str,
        *,
        ticker: str,
        max_results: int,
    ) -> list[WebEvidence] | None:
        try:
            value = await self._cache.get_json(key)
            if not isinstance(value, list) or not value:
                return None
            evidence = _WEB_EVIDENCE_LIST.validate_python(value)
        except Exception:
            logger.warning("web cache read failed")
            return None

        seen_urls: set[str] = set()
        for item in evidence:
            if item.ticker.upper() != ticker:
                logger.warning("web cache value rejected for %s", key)
                return None
            if not _has_valid_web_times(item):
                logger.warning("web cache timestamp rejected for %s", key)
                return None
            try:
                source_kind, source_tier = self._source_policy.classify(
                    ticker=ticker,
                    url=item.source_url,
                )
            except SourcePolicyError:
                logger.warning("web cache source rejected for %s", key)
                return None
            normalized_url = str(item.source_url)
            if (
                item.source_kind is not source_kind
                or item.source_tier is not source_tier
                or normalized_url in seen_urls
            ):
                logger.warning("web cache value rejected for %s", key)
                return None
            seen_urls.add(normalized_url)
        canonical = self._repository.get_many([item.id for item in evidence])
        if len(canonical) != len(evidence) or any(
            cached_item != stored_item or not _has_content_addressed_identity(stored_item)
            for cached_item, stored_item in zip(evidence, canonical, strict=True)
        ):
            logger.warning("web cache provenance mismatch for %s", key)
            return None
        return canonical[:max_results]

    async def _search_provider(
        self,
        query: str,
        *,
        ticker: str,
        max_results: int,
    ) -> list[RawSearchHit]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_none(),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await asyncio.wait_for(
                        self._provider.search(
                            query,
                            max_results=max_results,
                            domains=self._source_policy.provider_domains(ticker),
                        ),
                        timeout=self._timeout_seconds,
                    )
        except TimeoutError as error:
            raise WebGatewayError("WEB_PROVIDER_TIMEOUT", "web provider timed out") from error
        except SearchProviderError as error:
            raise WebGatewayError("WEB_PROVIDER_ERROR", "web provider failed") from error
        except Exception as error:
            raise WebGatewayError("WEB_PROVIDER_ERROR", "web provider failed") from error
        raise WebGatewayError("WEB_PROVIDER_ERROR", "web provider returned no result")

    async def _resolve_hits(
        self,
        ticker: str,
        hits: list[RawSearchHit],
        *,
        max_results: int,
    ) -> list[tuple[RawSearchHit, SourceKind, SourceTier, str]]:
        candidates: list[RawSearchHit] = []
        seen_requested_urls: set[str] = set()
        for hit in hits:
            if len(candidates) == max_results:
                break
            published_at = _aware_published_at(hit.published_at)
            if published_at is None:
                continue
            try:
                self._source_policy.classify(ticker=ticker, url=hit.url)
            except SourcePolicyError:
                continue
            requested_url = str(hit.url)
            if requested_url in seen_requested_urls:
                continue
            seen_requested_urls.add(requested_url)
            candidates.append(hit.model_copy(update={"published_at": published_at}))

        approved: list[tuple[RawSearchHit, SourceKind, SourceTier, str]] = []
        seen_final_urls: set[str] = set()
        for hit in candidates:
            try:
                final_url = canonicalize_source_url(
                    await self._resolve_url(hit.url, ticker=ticker)
                )
            except WebGatewayError as error:
                if error.code == "SOURCE_NOT_ALLOWED":
                    continue
                raise
            try:
                source_kind, source_tier = self._source_policy.classify(
                    ticker=ticker, url=final_url
                )
            except SourcePolicyError:
                continue
            normalized_url = str(final_url)
            if normalized_url in seen_final_urls:
                continue
            seen_final_urls.add(normalized_url)
            approved.append(
                (
                    hit.model_copy(update={"url": final_url}),
                    source_kind,
                    source_tier,
                    hit.excerpt[:_MAX_EXCERPT_CHARACTERS],
                )
            )
        return approved

    async def _resolve_url(self, url: HttpUrl, *, ticker: str) -> HttpUrl:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_none(),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    operation = self._redirect_resolver.resolve_with_policy(
                        url,
                        ticker=ticker,
                        source_policy=self._source_policy,
                    )
                    return await asyncio.wait_for(operation, timeout=self._timeout_seconds)
        except SourcePolicyError as error:
            raise WebGatewayError(error.code, str(error)) from error
        except TimeoutError as error:
            raise WebGatewayError(
                "WEB_PROVIDER_TIMEOUT", "web source resolution timed out"
            ) from error
        except SearchProviderError as error:
            raise WebGatewayError(
                "WEB_PROVIDER_ERROR", "web source resolution failed"
            ) from error
        except Exception as error:
            raise WebGatewayError(
                "WEB_PROVIDER_ERROR", "web source resolution failed"
            ) from error
        raise WebGatewayError("WEB_PROVIDER_ERROR", "web source resolution failed")


def _is_transient(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or (
        isinstance(error, SearchProviderError) and error.transient
    )


def _web_cache_key(
    *, ticker: str, query: str, source_policy_version: str, max_results: int
) -> str:
    query_hash = sha256(query.encode("utf-8")).hexdigest()
    return f"web:{ticker}:{query_hash}:{source_policy_version}:{max_results}"


def _has_content_addressed_identity(evidence: WebEvidence) -> bool:
    content_hash = sha256(evidence.content.encode("utf-8")).hexdigest()
    expected_id = content_addressed_web_evidence_id(
        evidence.ticker,
        evidence.source_url,
        content_hash,
    )
    return evidence.content_hash == content_hash and evidence.id == expected_id


def _has_valid_web_times(evidence: WebEvidence) -> bool:
    return _normalized_published_at(evidence.published_at, evidence.fetched_at) is not None


def _aware_published_at(published_at: datetime | None) -> datetime | None:
    if published_at is None or published_at.tzinfo is None or published_at.utcoffset() is None:
        return None
    return published_at.astimezone(UTC)


def _normalized_published_at(
    published_at: datetime | None,
    fetched_at: datetime,
) -> datetime | None:
    if (
        published_at is None
        or published_at.tzinfo is None
        or published_at.utcoffset() is None
        or fetched_at.tzinfo is None
        or fetched_at.utcoffset() is None
    ):
        return None
    normalized_published = published_at.astimezone(UTC)
    normalized_fetched = fetched_at.astimezone(UTC)
    return normalized_published if normalized_published <= normalized_fetched else None


def _utc_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("web gateway clock must return an aware datetime")
    return value.astimezone(UTC)
