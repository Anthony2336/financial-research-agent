"""Strict offline SEC Company Facts normalization and persistence contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from fra.retrieval.sec import SecRequestRateLimiter
from fra.retrieval.xbrl import (
    CompanyFact,
    HttpCompanyFactsGateway,
    SecCompanyFactsDocument,
    XbrlError,
    XbrlErrorCode,
    company_fact_id,
    company_facts_url,
    normalize_company_facts,
)
from fra.storage.database import create_schema
from fra.storage.fact_repositories import CompanyFactRepository
from fra.storage.models import Company, CompanyFactRecord
from fra.storage.repositories import FilingRepository

FIXTURE = Path("tests/fixtures/sec/companyfacts_nvda.json")
FETCHED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
CIK = "0001045810"


def _document(
    raw_bytes: bytes | None = None,
    *,
    cik: str = CIK,
    fetched_at: datetime = FETCHED_AT,
) -> SecCompanyFactsDocument:
    return SecCompanyFactsDocument(
        source_url=company_facts_url(cik),
        raw_bytes=raw_bytes or FIXTURE.read_bytes(),
        fetched_at=fetched_at,
    )


def _repository():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    FilingRepository(engine).upsert_company_metadata(
        ticker="NVDA",
        cik=CIK,
        legal_name="NVIDIA CORPORATION",
        ir_domain="investor.nvidia.com",
    )
    return CompanyFactRepository(engine), engine


def test_companyfacts_normalization_preserves_exact_values_periods_and_source_metadata() -> None:
    snapshot = normalize_company_facts(_document(), ticker="nvda", cik="1045810")

    assert snapshot.ticker == "NVDA"
    assert snapshot.cik == CIK
    assert snapshot.legal_name == "NVIDIA CORPORATION"
    assert snapshot.source_url == company_facts_url(CIK)
    assert snapshot.raw_content_hash == sha256(FIXTURE.read_bytes()).hexdigest()
    assert snapshot.fetched_at == FETCHED_AT
    assert len(snapshot.facts) == 3
    assert {fact.form for fact in snapshot.facts} == {"10-K", "10-Q"}

    facts = {fact.concept: fact for fact in snapshot.facts}
    revenue = facts["RevenueFromContractWithCustomerExcludingAssessedTax"]
    assert revenue.value == Decimal("130497000000.125")
    assert type(revenue.value) is Decimal
    assert revenue.period_start == date(2024, 1, 29)
    assert revenue.period_end == date(2025, 1, 26)
    assert revenue.instant is None
    assert revenue.unit == "USD"
    assert revenue.currency == "USD"
    assert revenue.taxonomy == "us-gaap"
    assert revenue.frame == "CY2024"
    assert revenue.accession_no == "0001045810-25-000023"
    assert revenue.source_url == company_facts_url(CIK)
    assert revenue.raw_content_hash == snapshot.raw_content_hash
    assert revenue.fetched_at == FETCHED_AT

    assets = facts["Assets"]
    assert assets.instant == date(2025, 1, 26)
    assert assets.period_start is None
    assert assets.period_end is None


def test_companyfact_ids_are_stable_across_refetch_time_without_float_conversion() -> None:
    first = normalize_company_facts(_document(), ticker="NVDA", cik=CIK)
    later = normalize_company_facts(
        _document(fetched_at=FETCHED_AT + timedelta(hours=1)),
        ticker="NVDA",
        cik=CIK,
    )

    assert [fact.id for fact in first.facts] == [fact.id for fact in later.facts]
    with pytest.raises(ValidationError, match="float"):
        CompanyFact.model_validate(first.facts[0].model_dump(mode="python") | {"value": 1.25})


def test_companyfact_contract_rejects_unsupported_forms_and_invalid_period_shape() -> None:
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    with pytest.raises(ValidationError, match="form"):
        CompanyFact.model_validate(fact.model_dump(mode="python") | {"form": "DEF 14A"})
    with pytest.raises(ValidationError, match="period"):
        CompanyFact.model_validate(
            fact.model_dump(mode="python")
            | {
                "period_start": None,
                "period_end": fact.period_end,
                "instant": None,
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "1E-18",
        "1E+19",
        "99999999999999999999.999999999999999999",
        "-99999999999999999999.999999999999999999",
    ],
)
def test_companyfact_accepts_exact_numeric_38_18_boundaries(value: str) -> None:
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    validated = CompanyFact.model_validate(
        fact.model_dump(mode="python") | {"value": Decimal(value)}
    )

    assert validated.value == Decimal(value)


@pytest.mark.parametrize(
    "value",
    [
        "1E-19",
        "-1E-19",
        "100000000000000000000",
        "-100000000000000000000",
        "99999999999999999999.9999999999999999999",
    ],
)
def test_companyfact_rejects_values_outside_exact_numeric_38_18(value: str) -> None:
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    with pytest.raises(ValidationError, match=r"NUMERIC\(38,18\)"):
        CompanyFact.model_validate(
            fact.model_dump(mode="python") | {"value": Decimal(value)}
        )


@pytest.mark.parametrize(
    "value",
    [0, 99_999_999_999_999_999_999, -99_999_999_999_999_999_999],
)
def test_companyfact_integer_boundaries_use_the_same_numeric_validation(value: int) -> None:
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    validated = CompanyFact.model_validate(
        fact.model_dump(mode="python") | {"value": value}
    )

    assert validated.value == Decimal(value)


@pytest.mark.parametrize("value", [10**20, -(10**20)])
def test_companyfact_rejects_integer_overflow(value: int) -> None:
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    with pytest.raises(ValidationError, match=r"NUMERIC\(38,18\)"):
        CompanyFact.model_validate(
            fact.model_dump(mode="python") | {"value": value}
        )


def test_companyfacts_json_integer_overflow_is_rejected_before_fact_identity() -> None:
    payload = json.loads(FIXTURE.read_text())
    payload["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"] = 10**20

    with pytest.raises(XbrlError) as raised:
        normalize_company_facts(
            _document(json.dumps(payload).encode()),
            ticker="NVDA",
            cik=CIK,
        )

    assert raised.value.code is XbrlErrorCode.MALFORMED_RESPONSE


def test_companyfact_persisted_row_reconstruction_rejects_numeric_overflow() -> None:
    repository, engine = _repository()
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]
    repository.save_facts([fact])
    with Session(engine) as session, session.begin():
        row = session.get(CompanyFactRecord, fact.id)
        assert row is not None
        row.value = Decimal("1E+20")

    with pytest.raises(ValidationError, match=r"NUMERIC\(38,18\)"):
        repository.list_facts("NVDA", limit=10)


def test_companyfact_orm_numeric_contract_is_precision_38_scale_18() -> None:
    numeric = CompanyFactRecord.__table__.c.value.type.impl

    assert numeric.precision == 38
    assert numeric.scale == 18


def test_companyfacts_rejects_cross_cik_and_conflicting_duplicate_facts() -> None:
    payload = json.loads(FIXTURE.read_text())
    payload["cik"] = 320193
    with pytest.raises(XbrlError) as cross_cik:
        normalize_company_facts(
            _document(json.dumps(payload).encode()),
            ticker="NVDA",
            cik=CIK,
        )
    assert cross_cik.value.code is XbrlErrorCode.SCOPE_MISMATCH

    conflicting = json.loads(FIXTURE.read_text())
    entries = conflicting["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    entries.append(entries[0] | {"val": 999})
    with pytest.raises(XbrlError) as duplicate:
        normalize_company_facts(
            _document(json.dumps(conflicting).encode()),
            ticker="NVDA",
            cik=CIK,
        )
    assert duplicate.value.code is XbrlErrorCode.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json private-provider-token",
        b"{}",
        b'{"cik":1045810,"entityName":"NVIDIA","facts":[]}',
    ],
)
def test_companyfacts_malformed_provider_payload_is_typed_and_sanitized(
    payload: bytes,
) -> None:
    with pytest.raises(XbrlError) as raised:
        normalize_company_facts(_document(payload), ticker="NVDA", cik=CIK)

    assert raised.value.code is XbrlErrorCode.MALFORMED_RESPONSE
    assert "private-provider-token" not in str(raised.value)


def test_companyfact_repository_is_separate_idempotent_exact_and_narrowly_readable() -> None:
    repository, engine = _repository()
    first_snapshot = normalize_company_facts(_document(), ticker="NVDA", cik=CIK)
    later_snapshot = normalize_company_facts(
        _document(fetched_at=FETCHED_AT + timedelta(hours=1)),
        ticker="NVDA",
        cik=CIK,
    )

    first = repository.save_facts(first_snapshot.facts)
    repeated = repository.save_facts(later_snapshot.facts)
    revenue = repository.list_facts(
        "NVDA",
        concepts=["RevenueFromContractWithCustomerExcludingAssessedTax"],
        limit=10,
    )

    assert repeated == first
    assert len(revenue) == 1
    assert revenue[0].value == Decimal("130497000000.125")
    assert type(revenue[0].value) is Decimal
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(CompanyFactRecord)) == 3


def test_companyfact_repository_rejects_company_scope_and_duplicate_conflicts() -> None:
    repository, _ = _repository()
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]

    with pytest.raises(ValueError, match="conflicting"):
        repository.save_facts([fact, fact.model_copy(update={"value": fact.value + Decimal("1")})])

    repository.save_facts([fact])
    amd_values = fact.model_dump(mode="python") | {"ticker": "AMD"}
    amd_values["id"] = company_fact_id(amd_values)
    with pytest.raises(ValueError, match="ticker"):
        repository.save_facts([CompanyFact.model_validate(amd_values)])


def test_companyfact_read_revalidates_current_ticker_cik_ownership() -> None:
    repository, engine = _repository()
    fact = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts[0]
    repository.save_facts([fact])
    with Session(engine) as session, session.begin():
        company = session.scalar(select(Company).where(Company.ticker == "NVDA"))
        assert company is not None
        company.cik = "0000320193"

    with pytest.raises(ValueError, match="CIK"):
        repository.list_facts("NVDA", limit=10)


def test_http_companyfacts_gateway_uses_only_fixed_sec_url_identity_and_timeout() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=FIXTURE.read_bytes(), request=request)

    gateway = HttpCompanyFactsGateway(
        transport=httpx.MockTransport(handler),
        clock=lambda: FETCHED_AT,
    )
    document = gateway.fetch_company_facts(
        "NVDA",
        CIK,
        "Example Research Operator research-operator@example.com",
    )

    assert document == _document()
    assert [str(request.url) for request in requests] == [company_facts_url(CIK)]
    assert requests[0].headers["user-agent"] == (
        "Example Research Operator research-operator@example.com"
    )
    assert requests[0].extensions["timeout"]["read"] == 30.0


def test_http_companyfacts_gateway_enforces_sub_ten_requests_per_second() -> None:
    current_time = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    gateway = HttpCompanyFactsGateway(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=FIXTURE.read_bytes(),
                request=request,
            )
        ),
        clock=lambda: FETCHED_AT,
        request_clock=lambda: current_time[0],
        sleep=sleep,
    )

    for _ in range(2):
        gateway.fetch_company_facts(
            "NVDA",
            CIK,
            "Example Research Operator research-operator@example.com",
        )

    assert len(sleeps) == 1
    assert sleeps[0] > 0.1


def test_sec_rate_limiter_reserves_distinct_slots_for_concurrent_calls() -> None:
    sleeps: list[float] = []
    barrier = Barrier(4)
    limiter = SecRequestRateLimiter(
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )

    def wait_concurrently(_: int) -> None:
        barrier.wait()
        limiter.wait()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(wait_concurrently, range(4)))

    assert sorted(sleeps) == pytest.approx([0.11, 0.22, 0.33])


def test_http_companyfacts_rejects_oversized_content_length_before_streaming() -> None:
    class TrackingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            yield b"must-not-be-read"

    stream = TrackingStream()
    gateway = HttpCompanyFactsGateway(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Length": "11"},
                stream=stream,
                request=request,
            )
        ),
        max_response_bytes=10,
    )

    with pytest.raises(XbrlError) as raised:
        gateway.fetch_company_facts(
            "NVDA",
            CIK,
            "Example Research Operator research-operator@example.com",
        )

    assert raised.value.code is XbrlErrorCode.RESPONSE_TOO_LARGE
    assert stream.iterated is False


def test_http_companyfacts_aborts_accumulation_at_response_size_limit() -> None:
    class ChunkedStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.chunks_yielded = 0

        def __iter__(self):
            for chunk in (b"123456", b"78901", b"must-not-be-read"):
                self.chunks_yielded += 1
                yield chunk

    stream = ChunkedStream()
    gateway = HttpCompanyFactsGateway(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream, request=request)
        ),
        max_response_bytes=10,
    )

    with pytest.raises(XbrlError) as raised:
        gateway.fetch_company_facts(
            "NVDA",
            CIK,
            "Example Research Operator research-operator@example.com",
        )

    assert raised.value.code is XbrlErrorCode.RESPONSE_TOO_LARGE
    assert stream.chunks_yielded == 2


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [
        (429, XbrlErrorCode.RATE_LIMITED),
        (503, XbrlErrorCode.PROVIDER_UNAVAILABLE),
        (httpx.ReadTimeout("private-provider-token"), XbrlErrorCode.PROVIDER_UNAVAILABLE),
        (RuntimeError("private-provider-token"), XbrlErrorCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_http_companyfacts_provider_errors_are_typed_and_sanitized(
    outcome: int | Exception,
    expected_code: XbrlErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, text="private-provider-token", request=request)

    gateway = HttpCompanyFactsGateway(transport=httpx.MockTransport(handler))

    with pytest.raises(XbrlError) as raised:
        gateway.fetch_company_facts(
            "NVDA",
            CIK,
            "Example Research Operator research-operator@example.com",
        )

    assert raised.value.code is expected_code
    assert "private-provider-token" not in str(raised.value)


def test_companyfacts_batch_uses_one_insert_without_changing_exact_values() -> None:
    from sqlalchemy import event

    repository, engine = _repository()
    facts = normalize_company_facts(_document(), ticker="NVDA", cik=CIK).facts
    inserts = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO company_facts"):
            inserts.append(statement)

    saved = repository.save_facts(facts)
    assert saved == list(facts)
    assert len(repository.list_facts("NVDA")) == len(facts)
    assert len(inserts) == 1
    assert repository.save_facts(facts) == saved
    assert len(inserts) == 1
