from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_evidence_agent.application import ResearchCommand, ResearchMode
from financial_evidence_agent.config import Settings
from financial_evidence_agent.domain import (
    Claim,
    ClaimKind,
    Confidence,
    EvidenceChunk,
    Intent,
    ResearchMemo,
    ResearchQuestion,
    RouterDecision,
    SourceRef,
    SourceRefKind,
)


def test_verified_fact_requires_evidence_id() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        Claim(
            kind=ClaimKind.VERIFIED_FACT,
            text="Revenue increased.",
            confidence=Confidence.HIGH,
        )


def test_verified_fact_with_sec_evidence_remains_valid() -> None:
    claim = Claim(
        kind=ClaimKind.VERIFIED_FACT,
        text="Revenue increased.",
        confidence=Confidence.HIGH,
        evidence_chunk_ids=["sec-chunk-1"],
    )

    assert claim.evidence_chunk_ids == ["sec-chunk-1"]


def test_inference_can_be_recorded_without_evidence_id() -> None:
    claim = Claim(
        kind=ClaimKind.INFERENCE,
        text="Demand may remain elevated.",
        confidence=Confidence.LOW,
    )

    assert claim.evidence_chunk_ids == []


def test_router_decision_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        RouterDecision(intent="trade_now", reason="unsupported")


def test_source_ref_reserves_market_namespaces_for_future_market_provenance() -> None:
    snapshot = SourceRef(ticker="NVDA", kind=SourceRefKind.MARKET_SNAPSHOT, source_id="quote-1")
    bar = SourceRef(ticker="NVDA", kind=SourceRefKind.MARKET_BAR, source_id="bar-1")

    assert snapshot.encode() == "NVDA:market_snapshot:quote-1"
    assert bar.encode() == "NVDA:market_bar:bar-1"


def test_research_question_rejects_unsupported_filing_form() -> None:
    with pytest.raises(ValidationError):
        ResearchQuestion(
            question="What supports growth?",
            support_query="revenue growth demand",
            challenge_query="revenue risks headwinds",
            forms=["20-F"],
        )


def test_evidence_chunk_rejects_reversed_raw_span() -> None:
    with pytest.raises(ValidationError, match="raw_end"):
        EvidenceChunk(
            id="chunk-1",
            ticker="NVDA",
            corpus_version="NVDA-v1",
            content="Data center demand remained strong.",
            source_url="https://www.sec.gov/Archives/example.htm",
            form="10-Q",
            filed_at=date(2026, 5, 20),
            accession_no="0001045810-26-000001",
            section="MD&A",
            raw_start=50,
            raw_end=25,
        )


def test_settings_do_not_require_live_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "OPENAI_API_KEY",
        "FAST_MODEL",
        "ANALYST_MODEL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.fast_model is None
    assert settings.analyst_model is None
    assert settings.offline_demo is False
    assert settings.context_compressor_model == (
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    )
    assert settings.sec_user_agent == "Example Research Operator research-operator@example.com"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_cache_dir is None
    assert settings.reranker_cache_dir is None
    assert settings.tokenizer_cache_dir is None
    assert settings.context_compressor_cache_dir is None
    assert settings.sec_cache_ttl_seconds == 86_400
    assert settings.xbrl_max_response_bytes == 25 * 1024 * 1024
    assert settings.alpaca_api_key_id is None
    assert settings.alpaca_api_secret_key is None


def test_sec_cache_ttl_uses_the_exact_environment_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_CACHE_TTL_SECONDS", "43200")

    assert Settings(_env_file=None).sec_cache_ttl_seconds == 43_200
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sec_cache_ttl_seconds=0)


def test_model_asset_cache_settings_use_explicit_operator_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "/models/embedding")
    monkeypatch.setenv("TOKENIZER_CACHE_DIR", "/models/tokenizer")
    monkeypatch.setenv("CONTEXT_COMPRESSOR_CACHE_DIR", "/models/llmlingua")
    monkeypatch.setenv("RERANKER_CACHE_DIR", "/models/flashrank")

    settings = Settings(_env_file=None)

    assert settings.embedding_cache_dir == "/models/embedding"
    assert settings.tokenizer_cache_dir == "/models/tokenizer"
    assert settings.context_compressor_cache_dir == "/models/llmlingua"
    assert settings.reranker_cache_dir == "/models/flashrank"


def test_xbrl_response_size_uses_the_bounded_environment_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XBRL_MAX_RESPONSE_BYTES", "1048576")

    assert Settings(_env_file=None).xbrl_max_response_bytes == 1_048_576

    with pytest.raises(ValidationError):
        Settings(_env_file=None, xbrl_max_response_bytes=0)


def test_offline_demo_requires_an_explicit_true_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_DEMO", "true")

    assert Settings(_env_file=None).offline_demo is True


@pytest.mark.parametrize("feed", ["sip", "delayed_sip", ""])
def test_settings_reject_non_iex_feed(feed: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, alpaca_data_feed=feed)


@pytest.mark.parametrize("environment", ["sandbox", "production", ""])
def test_settings_reject_unsupported_alpaca_trading_environment(
    environment: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, alpaca_trading_environment=environment)


def test_settings_use_bounded_iex_market_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.market_data_provider == "alpaca"
    assert settings.alpaca_data_feed == "iex"
    assert settings.alpaca_trading_environment == "paper"
    assert settings.market_data_timeout_seconds == 5.0
    assert settings.market_data_max_bars == 5
    assert settings.market_data_max_staleness_seconds == 90
    assert settings.market_abnormal_move_threshold == Decimal("0.05")
    assert isinstance(settings.market_abnormal_move_threshold, Decimal)
    assert settings.research_quality_max_source_age_days == 365
    assert settings.web_industry_authority_domains == frozenset(
        {"ftc.gov", "justice.gov", "commerce.gov", "federalregister.gov"}
    )
    assert Intent.MARKET_SNAPSHOT_REQUEST.value == "market_snapshot_request"
    assert Intent.INDUSTRY_RESEARCH_REQUEST.value == "industry_research_request"
    assert Intent.RESEARCH_QUALITY_SCREEN_REQUEST.value == "research_quality_screen_request"


def test_research_command_normalizes_the_fixed_product_scope() -> None:
    command = ResearchCommand(
        ticker=" nvda ",
        request="  A factual thesis with   normalized spacing.  ",
        mode=ResearchMode.THESIS,
        session_id=" session-1 ",
        forms=("10-q", " 8-k "),
        as_of_date=date(2099, 1, 1),
        market="us",
    )

    assert command.ticker == "NVDA"
    assert command.request == "A factual thesis with normalized spacing."
    assert command.session_id == "session-1"
    assert command.forms == ("10-Q", "8-K")
    assert command.as_of_date == date(2099, 1, 1)
    assert command.market == "US"


def test_research_command_uses_exact_default_forms() -> None:
    command = ResearchCommand(
        ticker="NVDA",
        request="A factual thesis long enough to validate.",
        mode=ResearchMode.THESIS,
    )

    assert command.session_id is None
    assert command.forms == ("10-K", "10-Q", "8-K")
    assert command.as_of_date is None
    assert command.market == "US"


@pytest.mark.parametrize("length", [19, 501])
def test_research_command_rejects_out_of_bounds_normalized_thesis(length: int) -> None:
    with pytest.raises(ValidationError):
        ResearchCommand(ticker="NVDA", request="x" * length, mode=ResearchMode.THESIS)


@pytest.mark.parametrize("length", [20, 500])
def test_research_command_accepts_thesis_boundaries(length: int) -> None:
    command = ResearchCommand(
        ticker="NVDA", request=f"  {'x' * length}  ", mode=ResearchMode.THESIS
    )

    assert len(command.request) == length


def test_explicit_question_modes_retain_the_documented_2000_character_limit() -> None:
    accepted = ResearchCommand(
        ticker="NVDA", request="x" * 2_000, mode=ResearchMode.COMPANY_PROFILE
    )
    with pytest.raises(ValidationError):
        ResearchCommand(ticker="NVDA", request="x" * 2_001, mode=ResearchMode.COMPANY_PROFILE)

    assert len(accepted.request) == 2_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "   "),
        ("forms", ("20-F",)),
        ("forms", ()),
        ("market", "CA"),
    ],
)
def test_research_command_rejects_invalid_scope_values(field: str, value: object) -> None:
    values = {
        "ticker": "NVDA",
        "request": "A factual thesis long enough to validate.",
        "mode": ResearchMode.THESIS,
        field: value,
    }

    with pytest.raises(ValidationError):
        ResearchCommand(**values)


def test_settings_parse_abnormal_move_threshold_as_exact_decimal() -> None:
    settings = Settings(_env_file=None, market_abnormal_move_threshold="0.075")

    assert settings.market_abnormal_move_threshold == Decimal("0.075")
    assert isinstance(settings.market_abnormal_move_threshold, Decimal)


@pytest.mark.parametrize("threshold", ["0", "-0.01", "1.01"])
def test_settings_reject_invalid_abnormal_move_threshold(threshold: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, market_abnormal_move_threshold=threshold)


def test_settings_read_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_MODEL", "fast-model-test")
    monkeypatch.setenv("ANALYST_MODEL", "analyst-model-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "/models/cache")
    monkeypatch.setenv("WEB_INDUSTRY_AUTHORITY_DOMAINS", '["gao.gov", "treasury.gov"]')

    settings = Settings(_env_file=None)

    assert settings.fast_model == "fast-model-test"
    assert settings.analyst_model == "analyst-model-test"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_cache_dir == "/models/cache"
    assert settings.web_industry_authority_domains == frozenset({"gao.gov", "treasury.gov"})
    assert Intent.RESEARCH_REQUEST.value == "research_request"


def test_research_memo_rejects_unknown_output_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchMemo(
            research_question="Does demand support growth?",
            information_sufficiency="B",
            confidence=Confidence.MEDIUM,
            recommendation="buy",
        )
