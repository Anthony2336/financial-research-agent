"""Strict SEC Company Facts transport and exact normalized fact contracts."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Literal, Protocol

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from financial_evidence_agent.domain import StrictModel
from financial_evidence_agent.retrieval.sec import (
    SEC_TIMEOUT,
    SecRequestRateLimiter,
    validate_sec_user_agent,
    validate_ticker,
)

_SUPPORTED_FORMS = frozenset({"10-K", "10-Q", "8-K"})
_FORM = Literal["10-K", "10-Q", "8-K"]
_CIK = re.compile(r"^\d{1,10}$")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,254}$")
_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./*:-]{0,63}$")
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_NUMERIC_MAX_ABS = Decimal("1E+20")
_NUMERIC_SCALE = 18


class XbrlErrorCode(StrEnum):
    """Stable Company Facts failures safe to expose to callers."""

    PROVIDER_UNAVAILABLE = "XBRL_PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "XBRL_RATE_LIMITED"
    MALFORMED_RESPONSE = "XBRL_MALFORMED_RESPONSE"
    SCOPE_MISMATCH = "XBRL_SCOPE_MISMATCH"
    UNSUPPORTED_FORM = "XBRL_UNSUPPORTED_FORM"
    RESPONSE_TOO_LARGE = "XBRL_RESPONSE_TOO_LARGE"


class XbrlError(RuntimeError):
    """Typed XBRL failure that never includes raw provider content."""

    def __init__(self, code: XbrlErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class SecCompanyFactsDocument(StrictModel):
    """Raw bytes from the one fixed SEC Company Facts endpoint."""

    source_url: str = Field(min_length=1, max_length=2_000)
    raw_bytes: bytes = Field(min_length=1)
    fetched_at: datetime

    @field_validator("fetched_at")
    @classmethod
    def require_utc_fetched_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Company Facts fetched_at must be timezone-aware UTC")
        return value


class CompanyFactsGateway(Protocol):
    """Injectable fixed-endpoint boundary used by explicit ingestion."""

    def fetch_company_facts(
        self,
        ticker: str,
        cik: str,
        user_agent: str,
    ) -> SecCompanyFactsDocument:
        """Fetch one issuer's raw SEC Company Facts document."""


class CompanyFact(StrictModel):
    """One exact, source-addressable SEC Company Fact observation."""

    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    cik: str = Field(pattern=r"^\d{10}$")
    taxonomy: str = Field(min_length=1, max_length=255)
    concept: str = Field(min_length=1, max_length=255)
    period_start: date | None = None
    period_end: date | None = None
    instant: date | None = None
    unit: str = Field(min_length=1, max_length=64)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    value: Decimal
    form: _FORM
    filed_at: date
    accession_no: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    source_url: str = Field(min_length=1, max_length=2_000)
    frame: str | None = Field(default=None, min_length=1, max_length=64)
    raw_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: datetime

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("cik", mode="before")
    @classmethod
    def normalize_cik(cls, value: object) -> object:
        return _normalized_cik(value)

    @field_validator("taxonomy", "concept")
    @classmethod
    def require_safe_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("taxonomy and concept must use SEC name syntax")
        return value

    @field_validator("unit")
    @classmethod
    def require_safe_unit(cls, value: str) -> str:
        if not _UNIT.fullmatch(value):
            raise ValueError("unit must use supported SEC unit syntax")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def require_exact_decimal(cls, value: object) -> Decimal:
        if isinstance(value, bool) or isinstance(value, float):
            raise ValueError("Company Facts value must not use float conversion")
        if isinstance(value, int):
            value = Decimal(value)
        if not isinstance(value, Decimal):
            raise ValueError("Company Facts value must be an exact Decimal or integer")
        if not value.is_finite():
            raise ValueError("Company Facts value must be finite")
        if value.copy_abs() >= _NUMERIC_MAX_ABS:
            raise ValueError("Company Facts value must fit NUMERIC(38,18) exactly")
        if not value.is_zero():
            digits = value.as_tuple().digits
            trailing_zeros = 0
            for digit in reversed(digits):
                if digit != 0:
                    break
                trailing_zeros += 1
            if value.as_tuple().exponent + trailing_zeros < -_NUMERIC_SCALE:
                raise ValueError("Company Facts value must fit NUMERIC(38,18) exactly")
        return value

    @field_validator("fetched_at")
    @classmethod
    def require_utc_fetched_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Company Facts fetched_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_source_identity(self) -> CompanyFact:
        duration = (
            self.period_start is not None and self.period_end is not None and self.instant is None
        )
        instant = self.instant is not None and self.period_start is None and self.period_end is None
        if duration == instant:
            raise ValueError("fact period must be exactly one duration or instant")
        if duration and self.period_start > self.period_end:  # type: ignore[operator]
            raise ValueError("fact period start must not be later than period end")
        if self.accession_no[:10] != self.cik:
            raise ValueError("fact accession CIK must match fact CIK")
        if self.source_url != company_facts_url(self.cik):
            raise ValueError("fact source URL must be the fixed SEC Company Facts endpoint")
        expected_currency = self.unit if re.fullmatch(r"[A-Z]{3}", self.unit) else None
        if self.currency != expected_currency:
            raise ValueError("fact currency must be derived exactly from its SEC unit")
        if self.id != company_fact_id(self):
            raise ValueError("fact id does not match its stable source identity")
        return self


class CompanyFactsSnapshot(StrictModel):
    """One normalized immutable view of a fetched Company Facts response."""

    ticker: str
    cik: str
    legal_name: str
    source_url: str
    raw_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: datetime
    facts: tuple[CompanyFact, ...]


class HttpCompanyFactsGateway:
    """Fetch one issuer's Company Facts from a fixed SEC endpoint."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        request_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_response_bytes: int = 25 * 1024 * 1024,
        rate_limiter: SecRequestRateLimiter | None = None,
    ) -> None:
        if max_response_bytes <= 0:
            raise ValueError("Company Facts response limit must be positive")
        self._transport = transport
        self._clock = clock
        self._rate_limiter = rate_limiter or SecRequestRateLimiter(
            clock=request_clock,
            sleep=sleep,
        )
        self._max_response_bytes = max_response_bytes

    def fetch_company_facts(
        self,
        ticker: str,
        cik: str,
        user_agent: str,
    ) -> SecCompanyFactsDocument:
        validate_ticker(ticker)
        normalized_cik = _normalized_cik(cik)
        identity = validate_sec_user_agent(user_agent)
        source_url = company_facts_url(normalized_cik)
        self._throttle()
        try:
            with httpx.Client(transport=self._transport, follow_redirects=False) as client:
                with client.stream(
                    "GET",
                    source_url,
                    headers={
                        "User-Agent": identity,
                        "Accept-Encoding": "gzip, deflate",
                        "Accept": "application/json",
                    },
                    timeout=SEC_TIMEOUT,
                ) as response:
                    if response.status_code == 429:
                        raise XbrlError(
                            XbrlErrorCode.RATE_LIMITED,
                            "SEC Company Facts request was rate limited",
                        )
                    if not 200 <= response.status_code < 300:
                        raise XbrlError(
                            XbrlErrorCode.PROVIDER_UNAVAILABLE,
                            "SEC Company Facts provider was unavailable",
                        )
                    content_length = _content_length(response)
                    if content_length is not None and content_length > self._max_response_bytes:
                        raise XbrlError(
                            XbrlErrorCode.RESPONSE_TOO_LARGE,
                            "SEC Company Facts response exceeded the configured size limit",
                        )
                    raw_bytes = _bounded_response_bytes(
                        response,
                        max_bytes=self._max_response_bytes,
                    )
        except XbrlError:
            raise
        except Exception as error:
            raise XbrlError(
                XbrlErrorCode.PROVIDER_UNAVAILABLE,
                "SEC Company Facts provider was unavailable",
            ) from error
        if not raw_bytes:
            raise XbrlError(
                XbrlErrorCode.PROVIDER_UNAVAILABLE,
                "SEC Company Facts provider was unavailable",
            )
        try:
            return SecCompanyFactsDocument(
                source_url=source_url,
                raw_bytes=raw_bytes,
                fetched_at=_utc_now(self._clock),
            )
        except ValidationError as error:
            raise XbrlError(
                XbrlErrorCode.PROVIDER_UNAVAILABLE,
                "SEC Company Facts fetch metadata was invalid",
            ) from error

    def _throttle(self) -> None:
        self._rate_limiter.wait()


def company_facts_url(cik: str) -> str:
    """Build the only URL accepted by the Company Facts adapter."""
    return _COMPANY_FACTS_URL.format(cik=_normalized_cik(cik))


def normalize_company_facts(
    document: SecCompanyFactsDocument,
    *,
    ticker: str,
    cik: str,
) -> CompanyFactsSnapshot:
    """Normalize a strict Company Facts response without ever constructing floats."""
    normalized_ticker = validate_ticker(ticker)
    normalized_cik = _normalized_cik(cik)
    try:
        validated_document = SecCompanyFactsDocument.model_validate(document)
    except ValidationError as error:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts document metadata was malformed",
        ) from error
    expected_url = company_facts_url(normalized_cik)
    if validated_document.source_url != expected_url:
        raise XbrlError(
            XbrlErrorCode.SCOPE_MISMATCH,
            "SEC Company Facts source did not match the requested issuer",
        )
    try:
        payload = json.loads(
            validated_document.raw_bytes,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as error:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts provider returned malformed JSON",
        ) from error
    if not isinstance(payload, Mapping):
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts provider returned a non-object payload",
        )
    try:
        payload_cik = _normalized_cik(payload["cik"])
    except (KeyError, TypeError, ValueError) as error:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts payload omitted a valid CIK",
        ) from error
    if payload_cik != normalized_cik:
        raise XbrlError(
            XbrlErrorCode.SCOPE_MISMATCH,
            "SEC Company Facts CIK did not match the requested issuer",
        )

    try:
        legal_name = _legal_name(payload["entityName"])
        raw_facts = payload["facts"]
        if not isinstance(raw_facts, Mapping) or not raw_facts:
            raise TypeError
        raw_content_hash = sha256(validated_document.raw_bytes).hexdigest()
        by_id: dict[str, CompanyFact] = {}
        saw_unsupported_form = False
        for taxonomy in sorted(raw_facts):
            concepts = raw_facts[taxonomy]
            if not isinstance(taxonomy, str) or not _NAME.fullmatch(taxonomy):
                raise ValueError
            if not isinstance(concepts, Mapping) or not concepts:
                raise TypeError
            for concept in sorted(concepts):
                concept_payload = concepts[concept]
                if not isinstance(concept, str) or not _NAME.fullmatch(concept):
                    raise ValueError
                if not isinstance(concept_payload, Mapping):
                    raise TypeError
                units = concept_payload.get("units")
                if not isinstance(units, Mapping) or not units:
                    raise TypeError
                for unit in sorted(units):
                    entries = units[unit]
                    if not isinstance(unit, str) or not _UNIT.fullmatch(unit):
                        raise ValueError
                    if not isinstance(entries, list) or not entries:
                        raise TypeError
                    for entry in entries:
                        if not isinstance(entry, Mapping):
                            raise TypeError
                        form = entry.get("form")
                        if not isinstance(form, str):
                            raise TypeError
                        normalized_form = form.strip().upper()
                        if normalized_form not in _SUPPORTED_FORMS:
                            saw_unsupported_form = True
                            continue
                        fact = _normalized_fact(
                            ticker=normalized_ticker,
                            cik=normalized_cik,
                            taxonomy=taxonomy,
                            concept=concept,
                            unit=unit,
                            entry=entry,
                            form=normalized_form,
                            source_url=expected_url,
                            raw_content_hash=raw_content_hash,
                            fetched_at=validated_document.fetched_at,
                        )
                        existing = by_id.get(fact.id)
                        if existing is not None and existing != fact:
                            raise ValueError("conflicting duplicate fact")
                        by_id[fact.id] = fact
    except XbrlError:
        raise
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts provider returned malformed fact data",
        ) from error
    if not by_id:
        code = (
            XbrlErrorCode.UNSUPPORTED_FORM
            if saw_unsupported_form
            else XbrlErrorCode.MALFORMED_RESPONSE
        )
        raise XbrlError(code, "SEC Company Facts contained no supported facts")
    facts = tuple(
        sorted(
            by_id.values(),
            key=lambda fact: (
                fact.taxonomy,
                fact.concept,
                fact.period_start or fact.instant,
                fact.period_end or fact.instant,
                fact.unit,
                fact.accession_no,
            ),
        )
    )
    return CompanyFactsSnapshot(
        ticker=normalized_ticker,
        cik=normalized_cik,
        legal_name=legal_name,
        source_url=expected_url,
        raw_content_hash=raw_content_hash,
        fetched_at=validated_document.fetched_at,
        facts=facts,
    )


def company_fact_id(fact: CompanyFact | Mapping[str, object]) -> str:
    """Hash the immutable source identity, excluding fetch time and exact value."""
    value = fact if isinstance(fact, Mapping) else fact.model_dump(mode="python")
    payload = json.dumps(
        {
            "ticker": value["ticker"],
            "cik": value["cik"],
            "taxonomy": value["taxonomy"],
            "concept": value["concept"],
            "period_start": _date_text(value.get("period_start")),
            "period_end": _date_text(value.get("period_end")),
            "instant": _date_text(value.get("instant")),
            "unit": value["unit"],
            "form": value["form"],
            "filed_at": _date_text(value["filed_at"]),
            "accession_no": value["accession_no"],
            "source_url": value["source_url"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _normalized_fact(
    *,
    ticker: str,
    cik: str,
    taxonomy: str,
    concept: str,
    unit: str,
    entry: Mapping[str, object],
    form: str,
    source_url: str,
    raw_content_hash: str,
    fetched_at: datetime,
) -> CompanyFact:
    end = _date(entry["end"])
    start_value = entry.get("start")
    period_start = _date(start_value) if start_value is not None else None
    period_end = end if period_start is not None else None
    instant = None if period_start is not None else end
    frame_value = entry.get("frame")
    if frame_value is not None and (
        not isinstance(frame_value, str) or not frame_value.strip() or len(frame_value.strip()) > 64
    ):
        raise ValueError("invalid frame")
    values: dict[str, object] = {
        "ticker": ticker,
        "cik": cik,
        "taxonomy": taxonomy,
        "concept": concept,
        "period_start": period_start,
        "period_end": period_end,
        "instant": instant,
        "unit": unit,
        "currency": unit if re.fullmatch(r"[A-Z]{3}", unit) else None,
        "value": entry["val"],
        "form": form,
        "filed_at": _date(entry["filed"]),
        "accession_no": entry["accn"],
        "source_url": source_url,
        "frame": frame_value.strip() if isinstance(frame_value, str) else None,
        "raw_content_hash": raw_content_hash,
        "fetched_at": fetched_at,
    }
    return CompanyFact(id=company_fact_id(values), **values)


def _normalized_cik(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("CIK must be an integer or digit string")
    if isinstance(value, int):
        raw = str(value)
    elif isinstance(value, str):
        raw = value.strip()
    else:
        raise ValueError("CIK must be an integer or digit string")
    if not _CIK.fullmatch(raw) or int(raw) == 0:
        raise ValueError("CIK must contain one to ten digits")
    return raw.zfill(10)


def _legal_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("entityName must be text")
    normalized = " ".join(value.split())
    if (
        not normalized
        or len(normalized) > 255
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized)
    ):
        raise ValueError("entityName is invalid")
    return normalized


def _date(value: object) -> date:
    if not isinstance(value, str):
        raise TypeError("SEC fact date must be text")
    return date.fromisoformat(value)


def _date_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("fact identity date must be a date")
    return value.isoformat()


def _reject_json_constant(value: str) -> Decimal:
    raise ValueError(f"unsupported JSON numeric constant: {value}")


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts response had invalid length metadata",
        ) from error
    if length < 0:
        raise XbrlError(
            XbrlErrorCode.MALFORMED_RESPONSE,
            "SEC Company Facts response had invalid length metadata",
        )
    return length


def _bounded_response_bytes(response: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise XbrlError(
                XbrlErrorCode.RESPONSE_TOO_LARGE,
                "SEC Company Facts response exceeded the configured size limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _utc_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Company Facts clock must return an aware datetime")
    return value.astimezone(UTC)
