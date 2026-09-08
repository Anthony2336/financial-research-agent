"""Thin, validated transport for the SEC public JSON and filing endpoints."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from datetime import date
from enum import StrEnum
from hashlib import sha256
from threading import Lock
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, ValidationError, field_validator

from financial_evidence_agent.domain import StrictModel
from financial_evidence_agent.storage.cache import SyncJsonCache

SUPPORTED_FORMS = ("10-Q", "10-K", "8-K")
_FORM = Literal["10-Q", "10-K", "8-K"]
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"
_ALLOWED_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_CIK = re.compile(r"^\d{1,10}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_PRIMARY_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.(?:htm|html|txt)$")
_IDENTIFYING_USER_AGENT = re.compile(
    r"^(?=.{8,200}$)[^\r\n]*[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}[^\r\n]*$",
    flags=re.IGNORECASE,
)
SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.11
SEC_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
logger = logging.getLogger(__name__)


class SecRequestRateLimiter:
    """Thread-safe SEC scheduler that never holds its lock while sleeping."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._next_request_at: float | None = None
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            scheduled = (
                now
                if self._next_request_at is None
                else max(now, self._next_request_at)
            )
            self._next_request_at = scheduled + SEC_MIN_REQUEST_INTERVAL_SECONDS
            delay = scheduled - now
        if delay > 0:
            self._sleep(delay)


class SecProviderErrorCode(StrEnum):
    """Stable SEC provider failures safe to expose to callers."""

    UNAVAILABLE = "SEC_PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "SEC_RATE_LIMITED"
    MALFORMED_RESPONSE = "SEC_MALFORMED_RESPONSE"
    SCOPE_MISMATCH = "SEC_SCOPE_MISMATCH"


class SecProviderError(RuntimeError):
    """Typed SEC provider failure without upstream response details."""

    def __init__(self, code: SecProviderErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class SecFilingCandidate(StrictModel):
    """Validated SEC submission metadata needed to fetch one primary document."""

    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    resolved_cik: str = Field(pattern=r"^\d{1,10}$")
    cik: str = Field(pattern=r"^\d{1,10}$")
    legal_name: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    )
    accession_no: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    form: _FORM
    filed_at: date
    primary_document: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.(?:htm|html|txt)$")

    @field_validator("resolved_cik", "cik")
    @classmethod
    def reject_all_zero_cik(cls, value: str) -> str:
        if int(value) == 0:
            raise ValueError("SEC CIK must not be all zero")
        return value


class SecFilingDocument(StrictModel):
    """Original SEC response bytes and their canonical archive URL."""

    source_url: str = Field(min_length=1, max_length=2_000)
    raw_bytes: bytes = Field(min_length=1)


class SecRawCachePayload(StrictModel):
    """Fully bound SEC HTML payload stored at ``sec:{accession_no}``."""

    accession_no: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    cik: str = Field(pattern=r"^\d{10}$")
    legal_name: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    )
    form: _FORM
    filed_at: date
    source_url: str = Field(min_length=1, max_length=2_000)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_html: str = Field(min_length=1)

    @field_validator("cik")
    @classmethod
    def reject_all_zero_cik(cls, value: str) -> str:
        if int(value) == 0:
            raise ValueError("SEC CIK must not be all zero")
        return value


class SecRawResponseCache:
    """Accession-addressed SEC HTML cache that treats every invalid value as a miss."""

    def __init__(self, cache: SyncJsonCache, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds <= 0:
            raise ValueError("SEC cache TTL must be positive")
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    def get(self, candidate: SecFilingCandidate) -> SecFilingDocument | None:
        selected = validate_filing_candidate(candidate)
        key = _sec_cache_key(selected.accession_no)
        try:
            value = self._cache.get_json_sync(key)
        except Exception:
            logger.warning("SEC cache read failed")
            return None
        if value is None:
            return None
        try:
            payload = SecRawCachePayload.model_validate(value)
            raw_bytes = payload.raw_html.encode("utf-8")
            expected = (
                selected.accession_no,
                selected.ticker,
                selected.cik.zfill(10),
                selected.legal_name,
                selected.form,
                selected.filed_at,
                filing_url(selected),
            )
            actual = (
                payload.accession_no,
                payload.ticker,
                payload.cik,
                payload.legal_name,
                payload.form,
                payload.filed_at,
                payload.source_url,
            )
            if actual != expected or sha256(raw_bytes).hexdigest() != payload.content_hash:
                raise ValueError("cached SEC document provenance mismatch")
            return validate_filing_document(
                SecFilingDocument(source_url=payload.source_url, raw_bytes=raw_bytes),
                selected,
            )
        except (TypeError, ValidationError, ValueError):
            self._delete(key)
            return None

    def set(
        self,
        candidate: SecFilingCandidate,
        document: SecFilingDocument,
    ) -> None:
        selected = validate_filing_candidate(candidate)
        validated = validate_filing_document(document, selected)
        try:
            raw_html = validated.raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return
        payload = SecRawCachePayload(
            accession_no=selected.accession_no,
            ticker=selected.ticker,
            cik=selected.cik.zfill(10),
            legal_name=selected.legal_name,
            form=selected.form,
            filed_at=selected.filed_at,
            source_url=validated.source_url,
            content_hash=sha256(validated.raw_bytes).hexdigest(),
            raw_html=raw_html,
        )
        try:
            self._cache.set_json_sync(
                _sec_cache_key(selected.accession_no),
                payload.model_dump(mode="json"),
                ttl_seconds=self._ttl_seconds,
            )
        except Exception:
            logger.warning("SEC cache write failed")

    def _delete(self, key: str) -> None:
        try:
            self._cache.delete_json_sync(key)
        except Exception:
            logger.warning("SEC cache invalidation failed")


class SecGateway(Protocol):
    """Injectable boundary used to keep normal ingestion tests offline."""

    def list_filings(self, ticker: str, user_agent: str) -> list[SecFilingCandidate]:
        """Return supported recent filing candidates for a normalized ticker."""

    def fetch_document(
        self,
        filing: SecFilingCandidate,
        user_agent: str,
    ) -> SecFilingDocument:
        """Fetch exactly one validated candidate's primary filing document."""


class HttpSecGateway:
    """Synchronous SEC gateway with fixed hosts, timeout, identity, and rate policy."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rate_limiter: SecRequestRateLimiter | None = None,
    ) -> None:
        self._transport = transport
        self._rate_limiter = rate_limiter or SecRequestRateLimiter(
            clock=clock,
            sleep=sleep,
        )

    def list_filings(self, ticker: str, user_agent: str) -> list[SecFilingCandidate]:
        normalized_ticker = validate_ticker(ticker)
        validated_user_agent = validate_sec_user_agent(user_agent)
        tickers = self._get_json(_TICKERS_URL, validated_user_agent)
        cik, legal_name = _resolve_company(tickers, normalized_ticker)
        submissions = self._get_json(
            _SUBMISSIONS_URL.format(cik=cik.zfill(10)),
            validated_user_agent,
        )
        return _recent_candidates(
            submissions,
            cik,
            normalized_ticker,
            legal_name=legal_name,
        )

    def fetch_document(
        self,
        filing: SecFilingCandidate,
        user_agent: str,
    ) -> SecFilingDocument:
        candidate = validate_filing_candidate(filing)
        validated_user_agent = validate_sec_user_agent(user_agent)
        source_url = filing_url(candidate)
        return SecFilingDocument(
            source_url=source_url,
            raw_bytes=self._get(source_url, validated_user_agent).content,
        )

    def _get_json(self, url: str, user_agent: str) -> Mapping[str, object]:
        try:
            payload = self._get(url, user_agent).json()
        except ValueError as error:
            raise SecProviderError(
                SecProviderErrorCode.MALFORMED_RESPONSE,
                "SEC provider returned malformed JSON",
            ) from error
        if not isinstance(payload, Mapping):
            raise SecProviderError(
                SecProviderErrorCode.MALFORMED_RESPONSE,
                "SEC provider returned a non-object JSON response",
            )
        return payload

    def _get(self, url: str, user_agent: str) -> httpx.Response:
        _validate_sec_url(url)
        self._throttle()
        try:
            with httpx.Client(transport=self._transport, follow_redirects=False) as client:
                response = client.get(
                    url,
                    headers={
                        "User-Agent": user_agent,
                        "Accept-Encoding": "gzip, deflate",
                        "Accept": "application/json, text/html, text/plain",
                    },
                    timeout=SEC_TIMEOUT,
                )
        except Exception as error:
            raise SecProviderError(
                SecProviderErrorCode.UNAVAILABLE,
                "SEC provider request was unavailable",
            ) from error
        if response.status_code == 429:
            raise SecProviderError(
                SecProviderErrorCode.RATE_LIMITED,
                "SEC provider request was rate limited",
            )
        if not 200 <= response.status_code < 300:
            raise SecProviderError(
                SecProviderErrorCode.UNAVAILABLE,
                "SEC provider request was unavailable",
            )
        return response

    def _throttle(self) -> None:
        self._rate_limiter.wait()


def validate_ticker(ticker: str) -> str:
    """Normalize one ticker while rejecting path or host-like input."""
    if not isinstance(ticker, str):
        raise ValueError("ticker must be a string")
    normalized = ticker.strip().upper()
    if not _TICKER.fullmatch(normalized):
        raise ValueError("ticker must be 1-10 letters, digits, dots, or hyphens")
    return normalized


def validate_forms(forms: list[str]) -> list[str]:
    """Normalize a non-empty candidate allowlist without reordering it."""
    if not isinstance(forms, list) or not forms:
        raise ValueError("forms must contain at least one supported form")
    normalized: list[str] = []
    for form in forms:
        if not isinstance(form, str):
            raise ValueError("forms must contain strings")
        value = form.strip().upper()
        if value not in SUPPORTED_FORMS:
            raise ValueError(f"unsupported SEC form: {form}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def validate_sec_user_agent(user_agent: str) -> str:
    """Require SEC-request identity with a contact email and no header controls."""
    if not isinstance(user_agent, str):
        raise ValueError("SEC user agent must be a string")
    if any(not 0x20 <= ord(character) <= 0x7E for character in user_agent):
        raise ValueError("SEC user agent must contain printable ASCII only")
    normalized = user_agent.strip()
    if not _IDENTIFYING_USER_AGENT.fullmatch(normalized):
        raise ValueError("SEC user agent must identify a contact email")
    return normalized


def validate_filing_candidate(candidate: object) -> SecFilingCandidate:
    """Revalidate injected gateway data before it can affect a document URL."""
    try:
        if isinstance(candidate, SecFilingCandidate):
            candidate = candidate.model_dump(warnings=False)
        validated = SecFilingCandidate.model_validate(candidate)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ValueError("invalid SEC filing candidate") from error
    if (
        not _TICKER.fullmatch(validated.ticker)
        or not _CIK.fullmatch(validated.resolved_cik)
        or not _CIK.fullmatch(validated.cik)
        or not _ACCESSION.fullmatch(validated.accession_no)
        or not _PRIMARY_DOCUMENT.fullmatch(validated.primary_document)
        or validated.accession_no[:10] != validated.cik.zfill(10)
    ):
        raise ValueError("invalid SEC filing candidate")
    return validated


def validate_filing_document(
    document: object,
    candidate: SecFilingCandidate,
) -> SecFilingDocument:
    """Require non-empty bytes from the exact canonical URL for the selected filing."""
    try:
        if isinstance(document, SecFilingDocument):
            document = document.model_dump(warnings=False)
        validated = SecFilingDocument.model_validate(document)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ValueError("invalid SEC filing document") from error
    expected_url = filing_url(candidate)
    if validated.source_url != expected_url:
        raise ValueError("SEC filing document URL does not match the selected filing")
    _validate_sec_url(validated.source_url, archive_only=True)
    return validated


def filing_url(candidate: SecFilingCandidate) -> str:
    """Build the sole allowed archive URL after strict field validation."""
    validated = validate_filing_candidate(candidate)
    return _ARCHIVE_URL.format(
        cik=str(int(validated.cik)),
        accession=validated.accession_no.replace("-", ""),
        primary_document=validated.primary_document,
    )


def _resolve_company(payload: Mapping[str, object], ticker: str) -> tuple[str, str]:
    for record in payload.values():
        if not isinstance(record, Mapping):
            continue
        if str(record.get("ticker", "")).upper() != ticker:
            continue
        raw_cik = str(record.get("cik_str", ""))
        legal_name = " ".join(str(record.get("title", "")).split())
        if (
            not _CIK.fullmatch(raw_cik)
            or int(raw_cik) == 0
            or not legal_name
            or len(legal_name) > 255
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in legal_name)
        ):
            break
        return raw_cik, legal_name
    raise SecProviderError(
        SecProviderErrorCode.MALFORMED_RESPONSE,
        f"SEC ticker mapping did not contain requested ticker {ticker}",
    )


def _recent_candidates(
    payload: Mapping[str, object],
    cik: str,
    ticker: str,
    *,
    legal_name: str,
) -> list[SecFilingCandidate]:
    try:
        submissions_cik = str(payload["cik"])
    except KeyError as error:
        raise SecProviderError(
            SecProviderErrorCode.MALFORMED_RESPONSE,
            "SEC submissions response is missing its CIK",
        ) from error
    if not _CIK.fullmatch(submissions_cik) or int(submissions_cik) != int(cik):
        raise SecProviderError(
            SecProviderErrorCode.SCOPE_MISMATCH,
            "SEC submissions CIK does not match the ticker mapping",
        )
    try:
        filings = payload["filings"]
        if not isinstance(filings, Mapping):
            raise TypeError
        recent = filings["recent"]
        if not isinstance(recent, Mapping):
            raise TypeError
        accessions = recent["accessionNumber"]
        dates = recent["filingDate"]
        forms = recent["form"]
        documents = recent["primaryDocument"]
        if not all(isinstance(values, list) for values in (accessions, dates, forms, documents)):
            raise TypeError
        rows = zip(accessions, dates, forms, documents, strict=True)
        candidates = []
        for accession, filed_at, form, document in rows:
            normalized_form = str(form).upper()
            if normalized_form not in SUPPORTED_FORMS:
                continue
            candidates.append(
                SecFilingCandidate(
                    ticker=ticker,
                    resolved_cik=cik,
                    cik=cik,
                    legal_name=legal_name,
                    accession_no=accession,
                    form=normalized_form,
                    filed_at=filed_at,
                    primary_document=document,
                )
            )
    except (KeyError, TypeError, ValidationError, ValueError) as error:
        raise SecProviderError(
            SecProviderErrorCode.MALFORMED_RESPONSE,
            "SEC provider returned malformed recent filing metadata",
        ) from error
    return candidates


def _sec_cache_key(accession_no: str) -> str:
    return f"sec:{accession_no}"


def _validate_sec_url(url: str, *, archive_only: bool = False) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SEC URL must use an allowlisted HTTPS host")
    if archive_only and (
        parsed.hostname != "www.sec.gov" or not parsed.path.startswith("/Archives/edgar/data/")
    ):
        raise ValueError("filing document must use the SEC archives path")
