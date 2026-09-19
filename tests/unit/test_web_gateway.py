"""Allowlisted gateway behavior with a deterministic fake search provider."""

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pytest
from pydantic import HttpUrl
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fra.domain import WebEvidence
from fra.storage.database import create_schema
from fra.storage.models import WebEvidenceRecord
from fra.storage.web_repositories import WebEvidenceRepository
from fra.web_evidence.gateway import AllowlistedWebGateway, WebGatewayError
from fra.web_evidence.providers import (
    HttpxRedirectResolver,
    RawSearchHit,
    SearchProviderError,
    TavilySearchProvider,
)
from fra.web_evidence.source_policy import SourcePolicy


class FakeProvider:
    def __init__(self, outcomes: Sequence[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def search(
        self, query: str, *, max_results: int, domains: tuple[str, ...]
    ) -> list[RawSearchHit]:
        del query, max_results, domains
        self.calls += 1
        outcome = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        assert isinstance(outcome, list)
        return outcome


class DomainRecordingProvider:
    def __init__(self, hits: list[RawSearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        domains: tuple[str, ...],
    ) -> list[RawSearchHit]:
        self.calls.append((query, max_results, domains))
        return self.hits


class TimeoutClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def search(self, **kwargs):
        raise self.error


class FakeRedirectResolver:
    def __init__(self, final_urls: dict[str, str] | None = None) -> None:
        self.final_urls = final_urls or {}
        self.calls: list[str] = []

    async def resolve(self, url: HttpUrl) -> HttpUrl:
        source_url = str(url)
        self.calls.append(source_url)
        return HttpUrl(self.final_urls.get(source_url, source_url))

    async def resolve_with_policy(
        self,
        url: HttpUrl,
        *,
        ticker: str,
        source_policy: SourcePolicy,
    ) -> HttpUrl:
        source_policy.classify(ticker=ticker, url=url)
        final_url = await self.resolve(url)
        source_policy.classify(ticker=ticker, url=final_url)
        return final_url


class LegacyRedirectResolver:
    """A resolver that can perform I/O without the gateway's source policy."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve(self, url: HttpUrl) -> HttpUrl:
        self.calls.append(str(url))
        return url


class SequencedHttpTransport:
    def __init__(self, outcomes: Sequence[int | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=request)


class RecordingRedirectTransport:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.urls: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        return self.handler(request)


class RecordingAsyncCache:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, object, int]] = []

    async def get_json(self, key: str):
        self.get_calls.append(key)
        return self.values.get(key)

    async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.set_calls.append((key, value, ttl_seconds))
        self.values[key] = value


def _hit(
    url: str,
    *,
    excerpt: str = "Revenue increased.",
) -> RawSearchHit:
    return RawSearchHit(
        title="Quarterly update",
        url=HttpUrl(url),
        excerpt=excerpt,
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


@pytest.fixture
def repository():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    return WebEvidenceRepository(engine), engine


def _gateway(
    provider: FakeProvider,
    repository: WebEvidenceRepository,
    *,
    redirect_resolver: FakeRedirectResolver | None = None,
    **kwargs,
):
    return AllowlistedWebGateway(
        provider=provider,
        redirect_resolver=redirect_resolver or FakeRedirectResolver(),
        source_policy=SourcePolicy(
            issuer_domains={"NVDA": frozenset({"investor.nvidia.com"})}
        ),
        repository=repository,
        **kwargs,
    )


def test_gateway_rejects_redirect_resolver_without_policy_boundary(repository) -> None:
    """An injected resolver must not perform network I/O before policy validation."""
    web_repository, _ = repository
    resolver = LegacyRedirectResolver()

    with pytest.raises(TypeError, match="resolve_with_policy"):
        _gateway(
            FakeProvider([[_hit("https://www.reuters.com/article")]]),
            web_repository,
            redirect_resolver=resolver,
        )

    assert resolver.calls == []


async def test_gateway_retries_timeouts_then_returns_typed_error(repository) -> None:
    """A hung external provider must remain bounded and report a stable recovery code."""

    async def hangs():
        await asyncio.sleep(1)
        return []

    web_repository, _ = repository
    provider = FakeProvider([hangs])

    with pytest.raises(WebGatewayError) as raised:
        await _gateway(provider, web_repository, timeout_seconds=0.001).search(
            ticker="NVDA", query="revenue"
        )

    assert raised.value.code == "WEB_PROVIDER_TIMEOUT"
    assert provider.calls == 3


@pytest.mark.parametrize(
    "provider_error",
    [TimeoutError("provider request timed out"), httpx.ReadTimeout("read timed out")],
)
async def test_tavily_adapter_preserves_timeout_for_gateway_retry(
    provider_error: Exception,
) -> None:
    """Wrapping a provider timeout as permanent would disable the required retry path."""
    provider = object.__new__(TavilySearchProvider)
    provider._client = TimeoutClient(provider_error)

    with pytest.raises(TimeoutError):
        await provider.search("revenue", max_results=3, domains=("reuters.com",))


async def test_tavily_adapter_uses_the_provider_domain_filter_parameter() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        async def search(self, **kwargs: object) -> dict[str, object]:
            self.kwargs = kwargs
            return {"results": []}

    client = RecordingClient()
    provider = object.__new__(TavilySearchProvider)
    provider._client = client

    hits = await provider.search(
        "NVIDIA quarterly results",
        max_results=3,
        domains=("investor.nvidia.com", "reuters.com", "sec.gov"),
    )

    assert hits == []
    assert client.kwargs == {
        "query": "NVIDIA quarterly results",
        "max_results": 3,
        "include_domains": ["investor.nvidia.com", "reuters.com", "sec.gov"],
    }


async def test_gateway_passes_the_exact_ticker_policy_domains_to_the_provider(
    repository,
) -> None:
    web_repository, _ = repository
    provider = DomainRecordingProvider([_hit("https://www.reuters.com/approved")])

    evidence = await _gateway(provider, web_repository).search(
        ticker="NVDA",
        query="quarterly results",
    )

    assert len(evidence) == 1
    assert provider.calls == [
        (
            "quarterly results",
            3,
            (
                "investor.nvidia.com",
                "reuters.com",
                "sec.gov",
                "www.reuters.com",
                "www.sec.gov",
            ),
        )
    ]


async def test_gateway_skips_one_off_policy_hit_without_erasing_a_valid_hit(
    repository,
) -> None:
    web_repository, _ = repository
    provider = FakeProvider(
        [
            [
                _hit("https://reuters.com.example.org/phishing"),
                _hit("https://www.reuters.com/approved"),
            ]
        ]
    )
    resolver = FakeRedirectResolver()

    evidence = await _gateway(
        provider,
        web_repository,
        redirect_resolver=resolver,
    ).search(ticker="NVDA", query="quarterly results")

    assert [str(item.source_url) for item in evidence] == [
        "https://www.reuters.com/approved"
    ]
    assert resolver.calls == ["https://www.reuters.com/approved"]


async def test_gateway_does_not_retry_non_transient_provider_errors(repository) -> None:
    """Authentication and other permanent provider failures must not consume retry attempts."""
    web_repository, _ = repository
    provider = FakeProvider([SearchProviderError("bad credentials", status_code=401)])

    with pytest.raises(WebGatewayError) as raised:
        await _gateway(provider, web_repository).search(ticker="NVDA", query="revenue")

    assert raised.value.code == "WEB_PROVIDER_ERROR"
    assert provider.calls == 1


async def test_gateway_retries_transient_provider_errors(repository) -> None:
    """Rate limits and server failures should recover within the bounded retry policy."""
    web_repository, _ = repository
    provider = FakeProvider(
        [
            SearchProviderError("rate limited", status_code=429),
            SearchProviderError("upstream unavailable", status_code=503),
            [_hit("https://www.reuters.com/technology/nvidia")],
        ]
    )

    evidence = await _gateway(provider, web_repository).search(
        ticker="NVDA", query="revenue"
    )

    assert len(evidence) == 1
    assert provider.calls == 3


async def test_gateway_caps_results_excerpt_hashes_and_persists_provenance(repository) -> None:
    """Unbounded provider output or unhashed/unpersisted content would break evidence provenance."""
    web_repository, engine = repository
    long_excerpt = "x" * 1_500
    provider = FakeProvider(
        [
            [
                _hit("https://www.reuters.com/a", excerpt=long_excerpt),
                _hit("https://www.reuters.com/a", excerpt="duplicate"),
                _hit("https://investor.nvidia.com/b"),
                _hit("https://www.sec.gov/c"),
                _hit("https://www.reuters.com/d"),
            ]
        ]
    )

    evidence = await _gateway(provider, web_repository).search(
        ticker="nvda", query="  revenue trend  ", max_results=3
    )

    assert len(evidence) == 3
    assert len(evidence[0].content) == 1_200
    assert evidence[0].content_hash == sha256(("x" * 1_200).encode()).hexdigest()
    assert evidence[0].fetched_at.tzinfo == UTC
    assert all(item.ticker == "NVDA" for item in evidence)
    assert len({str(item.source_url) for item in evidence}) == 3
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 3


async def test_gateway_prefilters_missing_naive_and_drops_future_after_resolution(
    repository,
) -> None:
    web_repository, engine = repository
    resolver = FakeRedirectResolver()
    fetched_at = datetime(2026, 8, 2, tzinfo=UTC)
    hits = [
        RawSearchHit(
            title="Missing date",
            url="https://www.reuters.com/missing",
            excerpt="Missing publication metadata.",
            published_at=None,
        ),
        RawSearchHit(
            title="Naive date",
            url="https://www.reuters.com/naive",
            excerpt="Naive publication metadata.",
            published_at=datetime(2026, 8, 1),
        ),
        RawSearchHit(
            title="Future date",
            url="https://www.reuters.com/future",
            excerpt="Future publication metadata.",
            published_at=datetime(2026, 8, 3, tzinfo=UTC),
        ),
        _hit("https://www.reuters.com/valid"),
    ]

    evidence = await _gateway(
        FakeProvider([hits]),
        web_repository,
        redirect_resolver=resolver,
        clock=lambda: fetched_at,
    ).search(ticker="NVDA", query="dated evidence")

    assert [str(item.source_url) for item in evidence] == [
        "https://www.reuters.com/valid"
    ]
    assert evidence[0].published_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert evidence[0].fetched_at == fetched_at
    assert resolver.calls == [
        "https://www.reuters.com/future",
        "https://www.reuters.com/valid",
    ]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 1


async def test_gateway_records_fetch_time_after_source_resolution(repository) -> None:
    web_repository, _ = repository
    before_resolution = datetime(2026, 8, 2, 12, tzinfo=UTC)
    after_resolution = datetime(2026, 8, 2, 12, 1, tzinfo=UTC)
    times = iter([before_resolution, after_resolution])

    evidence = await _gateway(
        FakeProvider([[_hit("https://www.reuters.com/dated")]]),
        web_repository,
        clock=lambda: next(times),
    ).search(ticker="NVDA", query="dated evidence")

    assert evidence[0].fetched_at == after_resolution


async def test_gateway_accepts_publication_between_provider_and_post_resolution_times(
    repository,
) -> None:
    web_repository, _ = repository
    provider_time = datetime(2026, 8, 2, 12, tzinfo=UTC)
    published_at = datetime(2026, 8, 2, 12, 0, 30, tzinfo=UTC)
    fetched_at = datetime(2026, 8, 2, 12, 1, tzinfo=UTC)
    times = iter([provider_time, fetched_at])
    resolver = FakeRedirectResolver()
    hit = RawSearchHit(
        title="Published during resolution",
        url="https://www.reuters.com/during-resolution",
        excerpt="Dated evidence.",
        published_at=published_at,
    )

    evidence = await _gateway(
        FakeProvider([[hit]]),
        web_repository,
        redirect_resolver=resolver,
        clock=lambda: next(times),
    ).search(ticker="NVDA", query="resolution timing")

    assert len(evidence) == 1
    assert evidence[0].published_at == published_at
    assert evidence[0].fetched_at == fetched_at
    assert resolver.calls == ["https://www.reuters.com/during-resolution"]


@pytest.mark.parametrize(
    "legacy_published_at",
    [None, datetime(2026, 8, 1), datetime(2026, 8, 3, tzinfo=UTC)],
)
async def test_gateway_rejects_invalid_actual_canonical_row_returned_by_upsert(
    repository,
    legacy_published_at: datetime | None,
) -> None:
    _, engine = repository
    fetched_at = datetime(2026, 8, 2, tzinfo=UTC)
    content = "Canonical legacy evidence."
    content_hash = sha256(content.encode()).hexdigest()
    legacy = WebEvidence(
        id="legacy-id",
        ticker="NVDA",
        title="Legacy row",
        content=content,
        source_url="https://www.reuters.com/legacy",
        source_kind="authoritative_web",
        source_tier="authoritative_secondary",
        published_at=legacy_published_at,
        fetched_at=fetched_at,
        content_hash=content_hash,
    )

    class LegacyCanonicalRepository:
        def upsert(self, evidence: WebEvidence) -> WebEvidence:
            del evidence
            return legacy

        def get_many(self, evidence_ids):
            return [legacy] if evidence_ids == [legacy.id] else []

    cache = RecordingAsyncCache()
    hit = RawSearchHit(
        title="Current provider hit",
        url="https://www.reuters.com/legacy",
        excerpt=content,
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    gateway = AllowlistedWebGateway(
        provider=FakeProvider([[hit]]),
        redirect_resolver=FakeRedirectResolver(),
        source_policy=SourcePolicy(issuer_domains={}),
        repository=LegacyCanonicalRepository(),  # type: ignore[arg-type]
        cache=cache,
        clock=lambda: fetched_at,
    )

    evidence = await gateway.search(ticker="NVDA", query="legacy collision")

    assert evidence == []
    assert cache.set_calls == []
    assert legacy.published_at is legacy_published_at
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


async def test_gateway_does_not_rewrite_or_return_colliding_legacy_database_row(
    repository,
) -> None:
    web_repository, engine = repository
    content = "Colliding immutable legacy evidence."
    content_hash = sha256(content.encode()).hexdigest()
    legacy_id = sha256(
        f"NVDA\nhttps://www.reuters.com/legacy-collision\n{content_hash}".encode()
    ).hexdigest()
    with Session(engine) as session, session.begin():
        session.add(
            WebEvidenceRecord(
                id=legacy_id,
                ticker="NVDA",
                title="Legacy",
                content=content,
                source_url="https://www.reuters.com/legacy-collision",
                source_kind="authoritative_web",
                source_tier="authoritative_secondary",
                published_at=None,
                fetched_at=datetime(2026, 7, 20),
                content_hash=content_hash,
                time_metadata_validated=False,
            )
        )
    cache = RecordingAsyncCache()
    gateway = _gateway(
        FakeProvider(
            [
                [
                    RawSearchHit(
                        title="Current",
                        url="https://www.reuters.com/legacy-collision",
                        excerpt=content,
                        published_at=datetime(2026, 8, 1, tzinfo=UTC),
                    )
                ]
            ]
        ),
        web_repository,
        cache=cache,
        clock=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )

    evidence = await gateway.search(ticker="NVDA", query="legacy collision")

    assert evidence == []
    assert cache.set_calls == []
    assert web_repository.get_many([legacy_id]) == []
    with Session(engine) as session:
        row = session.get(WebEvidenceRecord, legacy_id)
        assert row is not None
        assert row.published_at is None


async def test_gateway_rejects_unapproved_final_redirect_without_retry(repository) -> None:
    """An approved starting URL must not authorize a redirect to an attacker-controlled host."""
    web_repository, engine = repository
    provider = FakeProvider([[_hit("https://www.reuters.com/article")]])
    resolver = FakeRedirectResolver(
        {
            "https://www.reuters.com/article": (
                "https://reuters.com.example.org/phishing"
            )
        }
    )

    evidence = await _gateway(
        provider, web_repository, redirect_resolver=resolver
    ).search(ticker="NVDA", query="revenue")

    assert evidence == []
    assert provider.calls == 1
    assert resolver.calls == ["https://www.reuters.com/article"]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


async def test_gateway_rejects_unapproved_result_before_redirect_fetch(repository) -> None:
    """An unapproved search URL must never cross the injected fetch boundary."""
    web_repository, _ = repository
    provider = FakeProvider([[_hit("https://reuters.com.example.org/phishing")]])
    resolver = FakeRedirectResolver()

    evidence = await _gateway(
        provider, web_repository, redirect_resolver=resolver
    ).search(ticker="NVDA", query="revenue")

    assert evidence == []
    assert resolver.calls == []


async def test_gateway_prevalidates_each_result_before_its_own_fetch(repository) -> None:
    """An unapproved result is skipped without erasing a separately valid result."""
    web_repository, _ = repository
    provider = FakeProvider(
        [
            [
                _hit("https://www.reuters.com/approved"),
                _hit("https://reuters.com.example.org/phishing"),
            ]
        ]
    )
    resolver = FakeRedirectResolver()

    evidence = await _gateway(
        provider, web_repository, redirect_resolver=resolver
    ).search(ticker="NVDA", query="revenue")

    assert [str(item.source_url) for item in evidence] == [
        "https://www.reuters.com/approved"
    ]
    assert resolver.calls == ["https://www.reuters.com/approved"]


async def test_gateway_keeps_shared_authority_evidence_ticker_scoped(repository) -> None:
    """Searching one Reuters page for two issuers must not reuse binding provenance."""
    web_repository, engine = repository
    provider = FakeProvider([[_hit("https://www.reuters.com/sector-update")]])
    gateway = _gateway(provider, web_repository)

    nvda = (await gateway.search(ticker="NVDA", query="sector update"))[0]
    amd = (await gateway.search(ticker="AMD", query="sector update"))[0]

    assert nvda.id != amd.id
    assert nvda.ticker == "NVDA"
    assert amd.ticker == "AMD"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 2


async def test_gateway_rejects_httpx_actual_redirect_url_before_persistence(repository) -> None:
    """The production resolver path must not persist an allowlisted redirect origin."""
    web_repository, engine = repository

    def redirect_chain(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.reuters.com":
            return httpx.Response(
                302,
                headers={"location": "https://reuters.com.example.org/phishing"},
            )
        return httpx.Response(200)

    resolver = HttpxRedirectResolver(
        timeout_seconds=0.1,
        transport=httpx.MockTransport(redirect_chain),
    )
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=resolver,
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert evidence == []
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


@pytest.mark.parametrize(
    "location",
    ["http://127.0.0.1/private", "https://evil.example/private"],
)
async def test_gateway_never_requests_an_unapproved_redirect_target(
    repository, location: str
) -> None:
    """Validating only response.url would permit an SSRF request before rejection."""
    web_repository, _ = repository
    transport = RecordingRedirectTransport(
        lambda request: httpx.Response(302, headers={"location": location})
    )
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(transport.handle),
        ),
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert evidence == []
    assert transport.urls == ["https://www.reuters.com/article"]


async def test_httpx_resolver_supports_safe_relative_redirects(repository) -> None:
    """Relative redirects on one approved authority must remain usable."""
    web_repository, _ = repository

    def safe_chain(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/article":
            return httpx.Response(302, headers={"location": "/article/final"})
        return httpx.Response(200)

    transport = RecordingRedirectTransport(safe_chain)
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(transport.handle),
        ),
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert str(evidence[0].source_url) == "https://www.reuters.com/article/final"
    assert transport.urls == [
        "https://www.reuters.com/article",
        "https://www.reuters.com/article/final",
    ]


@pytest.mark.parametrize("mode", ["loop", "too-long"])
async def test_httpx_resolver_rejects_circular_or_excessive_redirects(
    repository, mode: str
) -> None:
    """A redirect chain must stop deterministically instead of looping or escaping a bound."""
    web_repository, _ = repository

    def chain(request: httpx.Request) -> httpx.Response:
        if mode == "loop":
            location = "/b" if request.url.path == "/a" else "/a"
        else:
            index = int(request.url.path.rsplit("/", 1)[-1])
            location = f"/{index + 1}"
        return httpx.Response(302, headers={"location": location})

    start = "https://www.reuters.com/a" if mode == "loop" else "https://www.reuters.com/0"
    transport = RecordingRedirectTransport(chain)
    gateway = _gateway(
        FakeProvider([[_hit(start)]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(transport.handle),
        ),
    )

    with pytest.raises(WebGatewayError) as raised:
        await gateway.search(ticker="NVDA", query="revenue")

    assert raised.value.code == "WEB_PROVIDER_ERROR"
    assert len(transport.urls) <= 6


async def test_gateway_retries_httpx_resolver_timeout_to_exact_attempt_cap(repository) -> None:
    """Resolver timeouts must retain their typed code and stop after three attempts."""
    web_repository, engine = repository
    sequence = SequencedHttpTransport([httpx.ReadTimeout("read timed out")])
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(sequence.handle),
        ),
    )

    with pytest.raises(WebGatewayError) as raised:
        await gateway.search(ticker="NVDA", query="revenue")

    assert raised.value.code == "WEB_PROVIDER_TIMEOUT"
    assert sequence.calls == 3
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


async def test_gateway_retries_httpx_resolver_429_then_persists(repository) -> None:
    """A transient rate limit should recover without weakening final URL validation."""
    web_repository, engine = repository
    sequence = SequencedHttpTransport([429, 200])
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(sequence.handle),
        ),
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert len(evidence) == 1
    assert sequence.calls == 2
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 1


async def test_gateway_retries_httpx_resolver_5xx_to_exact_attempt_cap(repository) -> None:
    """Persistent server failures must retry exactly three times without partial persistence."""
    web_repository, engine = repository
    sequence = SequencedHttpTransport([503])
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(sequence.handle),
        ),
    )

    with pytest.raises(WebGatewayError) as raised:
        await gateway.search(ticker="NVDA", query="revenue")

    assert raised.value.code == "WEB_PROVIDER_ERROR"
    assert sequence.calls == 3
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


async def test_gateway_does_not_retry_httpx_resolver_4xx(repository) -> None:
    """Permanent client errors must fail once without consuming transient retries."""
    web_repository, engine = repository
    sequence = SequencedHttpTransport([404])
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/article")]]),
        web_repository,
        redirect_resolver=HttpxRedirectResolver(
            timeout_seconds=0.1,
            transport=httpx.MockTransport(sequence.handle),
        ),
    )

    with pytest.raises(WebGatewayError) as raised:
        await gateway.search(ticker="NVDA", query="revenue")

    assert raised.value.code == "WEB_PROVIDER_ERROR"
    assert sequence.calls == 1
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 0


async def test_gateway_cache_key_includes_ticker_query_hash_and_policy_version(
    repository,
) -> None:
    """Cross-ticker or stale-policy reuse would violate the source ownership boundary."""
    web_repository, _ = repository
    provider = FakeProvider([[_hit("https://www.reuters.com/nvidia")]])
    cache = RecordingAsyncCache()
    gateway = _gateway(
        provider,
        web_repository,
        cache=cache,
        cache_ttl_seconds=75,
        source_policy_version="allowlist-v3",
    )

    evidence = await gateway.search(ticker="nvda", query="  revenue  ")

    query_hash = sha256(b"revenue").hexdigest()
    expected_key = f"web:NVDA:{query_hash}:allowlist-v3:3"
    assert cache.get_calls == [expected_key]
    assert cache.set_calls == [
        (expected_key, [item.model_dump(mode="json") for item in evidence], 75)
    ]


async def test_gateway_cache_hit_preserves_snapshot_freshness_and_skips_provider(
    repository,
) -> None:
    """Serving a cached snapshot must retain its original timestamps and avoid live search."""
    web_repository, _ = repository
    fetched_at = datetime(2026, 7, 20, 12, 30, tzinfo=UTC)
    published_at = datetime(2026, 7, 19, 8, 0, tzinfo=UTC)
    cached_input = WebEvidence(
        id="pending",
        ticker="NVDA",
        title="Cached result",
        content="Cached revenue evidence",
        source_url="https://www.reuters.com/nvidia",
        source_kind="authoritative_web",
        source_tier="authoritative_secondary",
        published_at=published_at,
        fetched_at=fetched_at,
        content_hash=sha256(b"Cached revenue evidence").hexdigest(),
    )
    cached = web_repository.upsert(cached_input)
    query_hash = sha256(b"revenue").hexdigest()
    key = f"web:NVDA:{query_hash}:allowlist-v3:3"
    cache = RecordingAsyncCache({key: [cached.model_dump(mode="json")]})
    provider = FakeProvider([AssertionError("provider must not run on a cache hit")])
    gateway = _gateway(
        provider,
        web_repository,
        cache=cache,
        source_policy_version="allowlist-v3",
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert evidence == [cached]
    assert evidence[0].published_at == published_at
    assert evidence[0].fetched_at == fetched_at
    assert provider.calls == 0


async def test_gateway_cache_hit_rejects_missing_or_tampered_authoritative_snapshot(
    repository,
) -> None:
    """Cached web content must exactly match its persisted content-addressed snapshot."""
    web_repository, _ = repository
    canonical = web_repository.upsert(
        WebEvidence(
            id="pending",
            ticker="NVDA",
            title="Persisted result",
            content="Persisted revenue evidence",
            source_url="https://www.reuters.com/nvidia",
            source_kind="authoritative_web",
            source_tier="authoritative_secondary",
            published_at=datetime(2026, 7, 19, tzinfo=UTC),
            fetched_at=datetime(2026, 7, 20, tzinfo=UTC),
            content_hash=sha256(b"Persisted revenue evidence").hexdigest(),
        )
    )
    tampered = canonical.model_copy(update={"content": "Forged cached evidence"})
    query_hash = sha256(b"revenue").hexdigest()
    key = f"web:NVDA:{query_hash}:allowlist-v3:3"
    cache = RecordingAsyncCache({key: [tampered.model_dump(mode="json")]})
    provider = FakeProvider([[_hit("https://www.reuters.com/fresh")]])
    gateway = _gateway(
        provider,
        web_repository,
        cache=cache,
        source_policy_version="allowlist-v3",
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert provider.calls == 1
    assert evidence[0].content == "Revenue increased."


async def test_gateway_revalidates_cached_evidence_against_current_allowlist(
    repository,
) -> None:
    """A cached URL removed from policy must become a miss, never trusted evidence."""
    web_repository, _ = repository
    cached = WebEvidence(
        id="cached-id",
        ticker="NVDA",
        title="Removed source",
        content="Stale source content",
        source_url="https://old-authority.example/nvidia",
        source_kind="authoritative_web",
        source_tier="authoritative_secondary",
        published_at=None,
        fetched_at=datetime(2026, 7, 20, tzinfo=UTC),
        content_hash=sha256(b"Stale source content").hexdigest(),
    )
    query_hash = sha256(b"revenue").hexdigest()
    key = f"web:NVDA:{query_hash}:allowlist-v3:3"
    cache = RecordingAsyncCache({key: [cached.model_dump(mode="json")]})
    provider = FakeProvider([[_hit("https://www.reuters.com/current")]])
    gateway = _gateway(
        provider,
        web_repository,
        cache=cache,
        source_policy_version="allowlist-v3",
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert provider.calls == 1
    assert str(evidence[0].source_url) == "https://www.reuters.com/current"


async def test_gateway_treats_empty_cached_results_as_a_miss(repository) -> None:
    """An empty cache entry must not suppress the live evidence provider."""
    web_repository, _ = repository
    query_hash = sha256(b"revenue").hexdigest()
    key = f"web:NVDA:{query_hash}:allowlist-v3:3"
    cache = RecordingAsyncCache({key: []})
    provider = FakeProvider([[_hit("https://www.reuters.com/current")]])
    gateway = _gateway(
        provider,
        web_repository,
        cache=cache,
        source_policy_version="allowlist-v3",
    )

    evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert provider.calls == 1
    assert len(evidence) == 1


async def test_gateway_cache_read_failure_returns_live_evidence_without_exception_detail(
    repository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache outage must not suppress live evidence or expose its exception text."""
    private = "password=private-password-value"

    class FailingReadCache:
        async def get_json(self, key: str) -> None:
            del key
            raise RuntimeError(private)

        async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
            del key, value, ttl_seconds

    web_repository, engine = repository
    provider = FakeProvider([[_hit("https://www.reuters.com/current")]])
    gateway = _gateway(provider, web_repository, cache=FailingReadCache())

    with caplog.at_level(logging.WARNING):
        evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert [item.content for item in evidence] == ["Revenue increased."]
    assert provider.calls == 1
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 1
    assert "web cache read failed" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


async def test_gateway_cache_write_failure_keeps_persisted_evidence_without_exception_detail(
    repository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache write outage must retain evidence while logging only a fixed message."""
    private = "password=private-password-value"

    class FailingWriteCache:
        async def get_json(self, key: str) -> None:
            del key
            return None

        async def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
            del key, value, ttl_seconds
            raise RuntimeError(private)

    web_repository, engine = repository
    gateway = _gateway(
        FakeProvider([[_hit("https://www.reuters.com/current")]]),
        web_repository,
        cache=FailingWriteCache(),
    )

    with caplog.at_level(logging.WARNING):
        evidence = await gateway.search(ticker="NVDA", query="revenue")

    assert [item.content for item in evidence] == ["Revenue increased."]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(WebEvidenceRecord)) == 1
    assert "web cache write failed" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


async def test_gateway_derives_a_deterministic_default_policy_version(repository) -> None:
    """Equivalent allowlists must not fragment cache keys by dictionary insertion order."""
    web_repository, _ = repository
    first_cache = RecordingAsyncCache()
    second_cache = RecordingAsyncCache()
    first = AllowlistedWebGateway(
        provider=FakeProvider([[]]),
        redirect_resolver=FakeRedirectResolver(),
        source_policy=SourcePolicy(
            issuer_domains={
                "NVDA": frozenset({"investor.nvidia.com"}),
                "AMD": frozenset({"ir.amd.com"}),
            }
        ),
        repository=web_repository,
        cache=first_cache,
    )
    second = AllowlistedWebGateway(
        provider=FakeProvider([[]]),
        redirect_resolver=FakeRedirectResolver(),
        source_policy=SourcePolicy(
            issuer_domains={
                "AMD": frozenset({"ir.amd.com"}),
                "NVDA": frozenset({"investor.nvidia.com"}),
            }
        ),
        repository=web_repository,
        cache=second_cache,
    )

    await first.search(ticker="NVDA", query="revenue")
    await second.search(ticker="NVDA", query="revenue")

    assert first_cache.get_calls == second_cache.get_calls


async def test_small_cached_search_does_not_truncate_a_later_larger_request(repository):
    web_repository, _ = repository
    provider = FakeProvider([[
        _hit(f"https://www.sec.gov/Archives/filing-{index}") for index in range(3)
    ]])
    gateway = _gateway(provider, web_repository, cache=RecordingAsyncCache())
    small = await gateway.search(ticker="NVDA", query="revenue", max_results=1)
    large = await gateway.search(ticker="NVDA", query="revenue", max_results=3)
    repeated = await gateway.search(ticker="NVDA", query="revenue", max_results=3)
    assert len(small) == 1
    assert len(large) == 3
    assert repeated == large
    assert provider.calls == 2
