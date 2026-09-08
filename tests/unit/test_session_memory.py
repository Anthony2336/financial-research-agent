"""Bounded, privacy-preserving Redis session memory contracts."""

from __future__ import annotations

import json
import logging

import pytest

from financial_evidence_agent.memory.models import ConversationTurn, SessionMemory
from financial_evidence_agent.memory.privacy import contains_private_financial_or_secret
from financial_evidence_agent.memory.privacy_normalization import privacy_text_views
from financial_evidence_agent.memory.session import (
    SESSION_TTL_SECONDS,
    SessionMemoryStore,
    build_rolling_summary,
    is_session_memory_eligible_request,
    load_session_memory_for_ticker,
)
from financial_evidence_agent.storage.cache import InMemoryTtlJsonCache, RedisJsonCache

PRIVATE_REQUESTS = (
    "Credentials: ZXCV-1234",
    "Authorization: Bearer abc.def.ghi",
    "api_key=private-api-value",
    "password: private-password-value",
    "email=investor@example.com",
    "account number: ABC-12345",
    "My holdings include 100 NVDA shares.",
    "You own a position in NVDA.",
    "She holds 50 NVDA shares.",
    "Taylor Morgan owns 100 NVDA shares.",
    "Taylor Morgan's personal holdings include 100 NVDA shares.",
    "My risk tolerance is aggressive.",
    "My portfolio is concentrated in semiconductors.",
    "I plan to buy NVDA after reviewing the filing.",
    "我的持仓包括100股NVDA。",
    "你的投资组合包括NVDA。",
    "她持有50股NVDA。",
    "我的风险偏好较高。",
    "我打算买入NVDA。",
)


SAFE_PUBLIC_REQUESTS = (
    "What credentials terminology appears in public filing access controls?",
    "How did NVDA's corporate account balances change this quarter?",
    "What do public Form 4 filings show about Jensen Huang's insider ownership?",
    "Jensen Huang holds NVDA shares according to public Form 4 filings.",
    "How does NVDA sell accelerators to cloud providers?",
    "Does NVDA's product portfolio support revenue growth?",
)

FIX_ROUND_PRIVATE_REQUESTS = (
    "Account no. ABC-12345",
    "account # ABC-12345",
    "Alice owns 100 NVDA shares.",
    "Alice owns NVDA shares in a retirement fund.",
    "José García owns 100 NVDA shares.",
    "李明 owns 100 NVDA shares.",
    "Taylor Morgan owns NVDA.",
    "Alice's risk tolerance is aggressive.",
    "Taylor Morgan's risk profile is aggressive.",
    "李明的风险偏好较高。",
    "Analyze insider ownership, then note Taylor Morgan's portfolio is concentrated "
    "in NVDA.",
)

FIX_ROUND_SAFE_REQUESTS = (
    "Analyze Taylor Morgan's beneficial ownership disclosed on Form 4.",
    "Jensen Huang holds NVDA shares according to public Form 4 filings.",
    "Analyze NVDA's bearer bonds and debt maturity profile.",
    "Analyze NVDA's risk tolerance and enterprise risk management disclosures.",
    "Compare the issuer risk profile with disclosed risk-management controls.",
    "Acme Holdings LLC owns NVDA shares in its disclosed corporate account.",
    "发行人持有NVDA股份作为公司投资。",
    "发行人的风险偏好反映企业风险管理政策。",
)

FIX_ROUND_2_PRIVATE_REQUESTS = (
    "Analyze Taylor Morgan's insider ownership on Form 4 and Alice's portfolio is "
    "concentrated in NVDA.",
    "Taylor Morgan holds NVDA shares according to Form 4 and Alice owns NVDA.",
    "Taylor Morgan holds NVDA shares per Form 4 or Alice's risk tolerance is low.",
    "Taylor Morgan holds NVDA shares per Form 4 Alice owns NVDA.",
    "Form 4 shows Taylor Morgan's portfolio is concentrated in NVDA.",
    "Analyze insider ownership and Taylor Morgan owns NVDA.",
    "Analyze insider ownership, Taylor Morgan owns NVDA.",
    "Taylor Morgan holds NVDA shares per Form 4以及李明的投资组合包括NVDA。",
    "My risk appetite: aggressive.",
    "Her risk tolerance: low.",
    "Taylor Morgan's risk profile: aggressive.",
    "李明的风险偏好：较高。",
)

FIX_ROUND_2_SAFE_REQUESTS = (
    "According to Form 4, Taylor Morgan holds NVDA shares.",
    "Taylor Morgan holds NVDA shares according to public Form 4 filings.",
    "According to Form 4, Taylor Morgan's holdings include NVDA shares.",
    "NVDA risk tolerance: conservative.",
    "Issuer risk appetite: conservative.",
    "Acme Holdings LLC risk profile: conservative.",
    "发行人风险偏好：保守。",
    "Risk tolerance: conservative.",
)

FIX_ROUND_3_PRIVATE_REQUESTS = (
    "Taylor Morgan owns 100 NVDA shares. Form 4 discusses a public disclosure.",
    "Taylor Morgan owns 100 NVDA shares. Schedule 13D discusses filing mechanics.",
    "Taylor Morgan owns 100 NVDA shares. A public disclosure discusses governance.",
    "Form 4 discusses a public disclosure. Taylor Morgan owns 100 NVDA shares.",
    "Taylor Morgan owns 100 NVDA shares and review Form 4.",
    "Taylor Morgan owns NVDA while Alice owns NVDA according to Form 4.",
    "Taylor Morgan owns NVDA according to Form 4 and Alice owns NVDA.",
    "李明持有NVDA股份，王芳持有NVDA股份 according to Form 4.",
    "李明持有NVDA股份。Form 4 discusses a public disclosure.",
    "李明持有NVDA股份。公开披露讨论公司治理。",
)

FIX_ROUND_3_SAFE_REQUESTS = (
    "Taylor Morgan owns 100 NVDA shares according to Form 4.",
    "Taylor Morgan owns NVDA as disclosed in Schedule 13D.",
    "Form 4 shows Taylor Morgan owns 100 NVDA shares.",
    "Schedule 13G reports Taylor Morgan holds NVDA shares.",
    "Public disclosure reports Taylor Morgan owns 100 NVDA shares.",
    "Taylor Morgan owns NVDA as disclosed in a public filing.",
    "根据Form 4披露，李明持有100股NVDA。",
    "公开披露显示，李明持有100股NVDA。",
)

FIX_ROUND_4_PRIVATE_REQUESTS = (
    "Taylor Morgan owns NVDA and Alice owns 100 NVDA shares according to Form 4.",
    "李明持有NVDA，王芳持有NVDA股票，根据Form 4披露。",
    "Taylor Morgan owns 100 NVDA shares according to Form 4 and Alice owns NVDA.",
    "根据Form 4披露，李明持有100股NVDA，王芳持有NVDA股票。",
    "Taylor Morgan owns NVDA and Alice owns NVDA and Robert Chen owns 100 NVDA "
    "shares according to Form 4.",
    "李明持有NVDA，王芳持有NVDA，赵强持有NVDA股票，根据Form 4披露。",
    "According to Form 4, Taylor Morgan owns NVDA, Alice owns NVDA, and according "
    "to Schedule 13D, Robert Chen owns 100 shares of NVDA.",
    "根据Form 4披露，李明持有NVDA，王芳持有NVDA；根据Schedule 13D披露，"
    "赵强持有NVDA股票。",
)

FIX_ROUND_4_SAFE_REQUESTS = (
    "Taylor Morgan owns 100 shares of NVDA according to Form 4.",
    "According to Form 4, Taylor Morgan owns 100 shares of NVDA.",
    "李明持有100股NVDA，根据Form 4披露。",
    "李明持有NVDA股票100股，根据Form 4披露。",
    "根据Form 4披露，李明持有100股NVDA。",
    "According to Form 4, Taylor Morgan owns NVDA, and according to Schedule 13D, "
    "Alice owns 100 shares of NVDA.",
)

FIX_ROUND_5_PRIVATE_REQUESTS = (
    "根据Form 4披露，李明持有100股NVDA和王芳持有50股NVDA。",
    "根据Form 4披露，李明持有100股NVDA或王芳持有50股NVDA。",
    "根据Form 4披露，李明持有100股NVDA、王芳持有50股NVDA、赵强持有25股NVDA。",
    "根据Form 4披露，李明持有100股NVDA。王芳持有50股NVDA。",
    "根据Form 4披露，李明持有100股NVDA，王芳持有50股NVDA，赵强持有25股NVDA。",
    "根据Form 4披露，李明持有100股NVDA王芳持有50股NVDA赵强持有25股NVDA。",
    "Taylor owns 100 NVDA shares according to Form 4, Alice owns 50 NVDA shares.",
    "李明持有100股NVDA，根据Form 4披露，王芳持有50股NVDA。",
    "Taylor owns 100 NVDA shares, Form 4 shows Alice owns 50 NVDA shares.",
    "李明持有100股NVDA，Form 4显示王芳持有50股NVDA。",
    "Taylor owns 100 NVDA shares. Form 4 discusses filing mechanics later.",
    "李明持有100股NVDA。Form 4随后讨论披露规则。",
)

FIX_ROUND_5_SAFE_REQUESTS = (
    "According to Form 4, Taylor owns 100 NVDA shares, Alice owns 50 NVDA shares "
    "as disclosed in Schedule 13D.",
    "根据Form 4披露，李明持有100股NVDA，王芳持有50股NVDA，根据Schedule 13D披露。",
    "Form 4 shows Taylor Morgan owns 100 shares of NVDA.",
    "Taylor Morgan owns 100 NVDA shares according to Form 4.",
    "根据Form 4披露，李明持有100股NVDA。",
    "根据Form 4披露，李明持有NVDA股票100股。",
    "李明持有100股NVDA，根据Form 4披露。",
    "李明持有NVDA股票100股，根据Form 4披露。",
    "Taylor owns 100 NVDA shares according to Form 4 and compare NVDA revenue.",
    "李明持有100股NVDA，根据Form 4披露，并比较NVDA收入。",
)

PREREQUISITE_COMPANY_CONTEXT_PRIVATE_REQUESTS = (
    "Acme Company employee Alice owns 50 NVDA shares.",
    "Acme Company employee José García owns 50 NVDA shares.",
    "Acme Company employee 王小明 owns 50 NVDA shares.",
    "Acme公司员工王小明持有100股NVDA。",
)

PREREQUISITE_LEGAL_ENTITY_SAFE_REQUESTS = (
    "Acme Company owns 50 NVDA shares.",
    "Société Générale Bank owns 50 NVDA shares.",
    "腾讯公司持有100股NVDA。",
)

PREREQUISITE_CHINESE_SUBJECT_PRIVATE_REQUESTS = (
    "王小明持有100股NVDA。",
    "欧阳娜娜持有100股NVDA。",
)

PREREQUISITE_CHINESE_SUBJECT_SAFE_REQUESTS = (
    "根据Form 4披露，王小明持有100股NVDA。",
    "根据Form 4披露，欧阳娜娜持有100股NVDA。",
)

PROFILE_COMPANY_CONTEXT_PRIVATE_REQUESTS = (
    "Acme Company employee Alice's holdings include 50 NVDA shares.",
    "Acme Company employee Alice's portfolio is concentrated in NVDA.",
    "Acme Company employee Alice's risk tolerance is high.",
    "Acme Company employee José García's holdings include 50 NVDA shares.",
    "Acme Company employee José García's portfolio is concentrated in NVDA.",
    "Acme Company employee José García's risk tolerance is high.",
    "Acme Company employee 王小明's holdings include 50 NVDA shares.",
    "Acme Company employee 王小明's portfolio is concentrated in NVDA.",
    "Acme Company employee 王小明's risk tolerance is high.",
    "Acme公司员工王小明的持仓包括50股NVDA。",
    "Acme公司员工王小明的投资组合集中于NVDA。",
    "Acme公司员工王小明的风险偏好较高。",
)

PROFILE_LEGAL_ENTITY_SAFE_REQUESTS = (
    "Acme Company's holdings include 50 NVDA shares.",
    "Société Générale Bank's portfolio includes NVDA.",
    "腾讯公司的风险偏好反映企业政策。",
)

PROFILE_PUBLIC_ATTRIBUTION_SAFE_REQUESTS = (
    "According to Form 4, Alice's holdings include 50 NVDA shares.",
    "According to Form 4, José García's holdings include 50 NVDA shares.",
    "根据Form 4披露，王小明的持仓包括100股NVDA。",
)

NORMALIZED_VARIANT_PRIVATE_REQUESTS = (
    "Credentials are ZXCV-1234.",
    "CREDENTIALS ARE ZXCV-1234.",
    "Ｃｒｅｄｅｎｔｉａｌｓ：ＺＸＣＶ－１２３４",
    "API key＝private-api-value",
    "Creden\u200btials： ZXCV-1234",
    "pass\u2060word＝private-password-value",
    "Credentials\u200bare ZXCV-1234.",
    "Account\u2060ID: ABC-12345",
    "Account ID: ABC-12345",
    "account id = abc-12345",
    "Alice\u200bowns 100 NVDA shares.",
    "Taylor\u200bowns msft.",
    "I\u2060own MSFT.",
    "alice owns 100 NVDA shares.",
    "ALICE OWNS 100 NVDA SHARES.",
    "ALICE MORGAN OWNS 100 NVDA SHARES.",
    "Alice has a high risk tolerance.",
    "alice has a high risk tolerance.",
    "ALICE MORGAN HAS A HIGH RISK TOLERANCE.",
    "Alice's stake in MSFT is 12%.",
    "alice's ownership of MSFT is 12%.",
    "ALICE MORGAN'S HOLDINGS INCLUDE MSFT.",
    "I own MSFT.",
    "You hold AMD.",
    "She sold TSLA.",
    "We plan to buy MSFT.",
    "They have a stake in AMD.",
)

PUBLIC_AGGREGATE_REQUESTS = (
    "Compare institutional investors' holdings in NVDA.",
    "Summarize client account balances disclosed by the company.",
    "Compare institutional investor holdings reported in public filings.",
    "Credentials are required for public filing access controls.",
    "Public filing credentials are discussed as access-control terminology.",
    "According to Form 4, alice's ownership of NVDA is 5%.",
    "According to Form 4, ALICE MORGAN'S HOLDINGS INCLUDE NVDA.",
    "Analyze NVDA's institutional ownership.",
    "The company holds a minority stake in a supplier.",
    "Microsoft Corporation owns a disclosed stake in OpenAI.",
    "NVIDIA holds a stake in CoreWeave for strategic purposes.",
)

AGGREGATE_CLAUSE_PRIVATE_REQUESTS = (
    "Compare institutional investors' holdings; Alice owns 50 NVDA shares.",
    "Summarize client account balances disclosed by the company; Account ID: ABC-12345.",
    "Institutional investor Alice's holdings include 50 NVDA shares.",
    "Client Alice's account balance is $10,000.",
    "Institutional investors’ holdings in Alice’s account",
    "NVIDIA holds a stake in CoreWeave for Alice",
)


def _turn(index: int, *, ticker: str = "NVDA") -> ConversationTurn:
    return ConversationTurn(
        question=f"Question {index}?",
        answer_summary=f"Guarded answer {index}.",
        run_id=f"run-{index}",
        ticker=ticker,
        open_questions=(f"Open question {index}?",),
    )


class _SyncRedisBoundary:
    def __init__(self, *, stored: object = None, error: Exception | None = None) -> None:
        self.stored = stored
        self.error = error
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str, int]] = []
        self.delete_calls: list[str] = []

    def get(self, key: str) -> object:
        self.get_calls.append(key)
        if self.error is not None:
            raise self.error
        return self.stored

    def set(self, key: str, value: str, *, ex: int) -> None:
        self.set_calls.append((key, value, ex))
        if self.error is not None:
            raise self.error
        self.stored = value

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        if self.error is not None:
            raise self.error
        self.stored = None


class _UnusedAsyncRedis:
    pass


def test_privacy_views_cover_inside_word_and_word_boundary_format_characters() -> None:
    """Dropping either canonical view misses a distinct Unicode format-character evasion."""
    views = privacy_text_views("Creden\u200btials： ZXCV-1234; Taylor\u2060owns msft")

    assert "Credentials: ZXCV-1234" in views.joined
    assert "Taylor owns msft" in views.separated


@pytest.mark.parametrize("text", PRIVATE_REQUESTS)
def test_shared_privacy_classifier_rejects_private_financial_or_secret_text(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", SAFE_PUBLIC_REQUESTS)
def test_shared_privacy_classifier_allows_public_company_research(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", FIX_ROUND_PRIVATE_REQUESTS)
def test_shared_classifier_covers_reviewed_private_identifiers_and_profiles(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", FIX_ROUND_SAFE_REQUESTS)
def test_shared_classifier_preserves_reviewed_public_issuer_research(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", FIX_ROUND_2_PRIVATE_REQUESTS)
def test_public_ownership_context_never_exempts_a_different_private_match(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", FIX_ROUND_2_SAFE_REQUESTS)
def test_public_ownership_and_unqualified_issuer_risk_research_remains_safe(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", FIX_ROUND_3_PRIVATE_REQUESTS)
def test_unrelated_public_sources_never_exempt_named_private_holdings(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", FIX_ROUND_3_SAFE_REQUESTS)
def test_explicit_source_to_ownership_bindings_remain_public_safe(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", FIX_ROUND_4_PRIVATE_REQUESTS)
def test_each_named_ownership_relation_requires_its_own_public_source(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", FIX_ROUND_4_SAFE_REQUESTS)
def test_complete_ownership_targets_keep_explicit_public_sources_bound(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", FIX_ROUND_5_PRIVATE_REQUESTS)
def test_directional_relation_scanner_rejects_any_unsourced_named_ownership(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", FIX_ROUND_5_SAFE_REQUESTS)
def test_directional_relation_scanner_preserves_independently_sourced_ownership(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", PREREQUISITE_COMPANY_CONTEXT_PRIVATE_REQUESTS)
def test_company_context_never_hides_an_employee_ownership_relation(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", PREREQUISITE_LEGAL_ENTITY_SAFE_REQUESTS)
def test_exact_legal_entity_ownership_subjects_remain_public_safe(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", PREREQUISITE_CHINESE_SUBJECT_PRIVATE_REQUESTS)
def test_unsourced_full_chinese_names_remain_private(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", PREREQUISITE_CHINESE_SUBJECT_SAFE_REQUESTS)
def test_source_before_binding_uses_the_full_chinese_name(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", PROFILE_COMPANY_CONTEXT_PRIVATE_REQUESTS)
def test_company_context_never_hides_a_named_person_profile(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", PROFILE_LEGAL_ENTITY_SAFE_REQUESTS)
def test_exact_legal_entity_profile_subjects_remain_public_safe(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", PROFILE_PUBLIC_ATTRIBUTION_SAFE_REQUESTS)
def test_exact_public_holdings_attribution_remains_safe(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", NORMALIZED_VARIANT_PRIVATE_REQUESTS)
def test_shared_classifier_normalizes_and_rejects_ordinary_private_variants(
    text: str,
) -> None:
    """Removing compatibility normalization or a bounded relation must expose a literal."""
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize("text", PUBLIC_AGGREGATE_REQUESTS)
def test_shared_classifier_preserves_public_corporate_aggregate_research(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


@pytest.mark.parametrize("text", AGGREGATE_CLAUSE_PRIVATE_REQUESTS)
def test_public_aggregate_clause_never_exempts_identified_private_material(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize(
    ("text", "private"),
    [
        ("NVIDIA holds a stake in CoreWeave for strategic purposes", False),
        ("NVIDIA's holdings increased", False),
        ("NVIDIA's holdings of common stock", False),
        ("NVIDIA's holdings of strategic importance", False),
        ("NVIDIA's holdings held by institutional investors", False),
        ("NVIDIA's holdings belonging to public shareholders", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors", False),
        ("NVIDIA's holdings of common stock increased", False),
        ("NVIDIA's holdings held by institutional investors worldwide", False),
        ("NVIDIA holds a stake in CoreWeave held by institutional investors increased", False),
        ("NVIDIA's holdings belonging to public shareholders of record", False),
        ("NVIDIA's holdings of Class A common stock", False),
        ("NVIDIA's holdings of Class A common stock and preferred stock", False),
        (
            "NVIDIA's holdings held by institutional investors or public shareholders worldwide",
            False,
        ),
        ("NVIDIA's holdings of common stock increased and remained outstanding", False),
        ("NVIDIA holds a stake in CoreWeave for strategic growth", False),
        ("NVIDIA holds a stake in CoreWeave for long-term investment", False),
        ("NVIDIA holds a stake in CoreWeave for disclosure purposes", False),
        ("NVIDIA's holdings held by Rose Morgan", True),
        ("NVDA's holdings belonging to Rose Morgan", True),
        ("NVIDIA holds a stake in CoreWeave of Rose Morgan", True),
        ("NVIDIA's holdings held by institutional investor Alice", True),
        ("NVDA's holdings belonging to institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by institutional investor Alice", True),
        ("NVIDIA's holdings for institutional investor Alice", True),
        ("NVDA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA holds a stake in CoreWeave for institutional investor Alice", True),
        ("NVIDIA's holdings held by institutional investors, Alice", True),
        ("NVIDIA's holdings held by institutional investors or Alice", True),
        ("NVIDIA's holdings held by institutional investors as well as Alice", True),
        ("NVIDIA's holdings held by institutional investors including Alice", True),
        ("NVIDIA's holdings held by the client Alice and institutional investors", True),
        ("NVIDIA's holdings held by institutional investors and Alice", True),
        ("NVIDIA's holdings of common stock held by Alice", True),
        ("NVIDIA's holdings in Alice's account", True),
        ("NVIDIA holds a stake in CoreWeave for Alice", True),
        ("NVIDIA's holdings held by Alice", True),
        ("NVIDIA's holdings belonging to Alice", True),
        ("NVIDIA's holdings of Alice", True),
        ("NVDA's holdings held by Alice", True),
        ("NVDA's holdings belonging to Alice", True),
        ("NVDA's holdings of Alice", True),
        ("NVIDIA holds a stake in CoreWeave held by Alice", True),
        ("NVIDIA holds a stake in CoreWeave belonging to Alice", True),
        ("NVIDIA holds a stake in CoreWeave of Alice", True),
        ("Institutional investors' holdings in Alice's account", True),
        ("NVDA holds a stake in CoreWeave for strategic purposes", False),
        ("NVDA holds a stake in CoreWeave for Alice", True),
    ],
)
def test_complete_relation_privacy_is_enforced_at_session_memory_boundary(
    text: str, private: bool
) -> None:
    assert is_session_memory_eligible_request(text, current_ticker="NVDA") is not private


@pytest.mark.parametrize(
    "text",
    [
        "ALICE holds a stake in CoreWeave.",
        "TAYLOR owns a stake in CoreWeave.",
    ],
)
def test_corporate_stake_exception_never_exempts_uppercase_people(text: str) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


@pytest.mark.parametrize(
    "text",
    [
        "Client account balances disclosed by the issuer for client Alice.",
        "Client account balances disclosed by the issuer for account holder Alice.",
        "Client account balances disclosed by the issuer for alice.",
        "Client account balances disclosed by the issuer for account id abc-123.",
    ],
)
def test_issuer_aggregate_never_exempts_identified_client_or_account(
    text: str,
) -> None:
    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is True


def test_issuer_aggregate_does_not_capture_a_name_from_a_later_relation() -> None:
    text = (
        "Client account balances disclosed by the issuer. "
        "Compare public revenue for Alice."
    )

    assert contains_private_financial_or_secret(text, current_ticker="NVDA") is False


def test_public_issuer_alias_holdings_remain_non_private() -> None:
    """Treating the configured issuer alias as a person would over-redact public research."""
    assert (
        contains_private_financial_or_secret(
            "NVIDIA’s holdings increased",
            current_ticker="NVDA",
        )
        is False
    )


def test_store_writes_exact_session_key_as_json_with_24_hour_ttl() -> None:
    """A namespace, pickle payload, or wrong expiry breaks the Redis contract."""
    redis = _SyncRedisBoundary()
    cache = RedisJsonCache(_UnusedAsyncRedis(), sync_client=redis, namespace="")  # type: ignore[arg-type]
    store = SessionMemoryStore(cache)

    store.append("session-a", _turn(1))

    assert len(redis.set_calls) == 1
    key, payload, ttl = redis.set_calls[0]
    assert key == "session:session-a"
    assert ttl == SESSION_TTL_SECONDS == 86_400
    assert json.loads(payload) == {
        "summary": "[NVDA] Question 1? — Guarded answer 1.",
        "turns": [
            {
                "answer_summary": "Guarded answer 1.",
                "open_questions": ["Open question 1?"],
                "question": "Question 1?",
                "run_id": "run-1",
                "ticker": "NVDA",
            }
        ],
    }


def test_store_isolates_sessions_preserves_tickers_and_trims_to_five_turns() -> None:
    """A shared key or unbounded list could leak or retain the wrong conversation."""
    store = SessionMemoryStore(InMemoryTtlJsonCache())

    for index in range(1, 7):
        store.append("session-a", _turn(index, ticker="NVDA" if index < 6 else "AMD"))
    store.append("session-b", _turn(9, ticker="MSFT"))

    first = store.load("session-a")
    second = store.load("session-b")
    assert [turn.run_id for turn in first.turns] == [
        "run-2",
        "run-3",
        "run-4",
        "run-5",
        "run-6",
    ]
    assert [turn.ticker for turn in first.turns] == ["NVDA", "NVDA", "NVDA", "NVDA", "AMD"]
    assert [turn.run_id for turn in second.turns] == ["run-9"]
    assert "Question 1?" not in first.summary
    assert "Question 6?" in first.summary


def test_store_clear_removes_only_the_requested_session() -> None:
    store = SessionMemoryStore(InMemoryTtlJsonCache())
    store.append("session-a", _turn(1))
    store.append("session-b", _turn(2))

    store.clear("session-a")

    assert store.load("session-a") == SessionMemory()
    assert [turn.run_id for turn in store.load("session-b").turns] == ["run-2"]


def test_store_rejects_malformed_or_extra_cache_data_without_raising(caplog) -> None:
    redis = _SyncRedisBoundary(stored='{"summary":"unsafe","turns":[],"extra":"x"}')
    cache = RedisJsonCache(_UnusedAsyncRedis(), sync_client=redis, namespace="")  # type: ignore[arg-type]
    store = SessionMemoryStore(cache)

    with caplog.at_level(logging.WARNING):
        memory = store.load("session-a")

    assert memory == SessionMemory()
    assert "session memory" in caplog.text


def test_store_degrades_redis_outage_to_empty_memory_and_best_effort_write(caplog) -> None:
    redis = _SyncRedisBoundary(error=ConnectionError("redis unavailable"))
    cache = RedisJsonCache(_UnusedAsyncRedis(), sync_client=redis, namespace="")  # type: ignore[arg-type]
    store = SessionMemoryStore(cache)

    with caplog.at_level(logging.WARNING):
        assert store.load("session-a") == SessionMemory()
        store.append("session-a", _turn(1))
        store.clear("session-a")

    assert "cache read failed" in caplog.text
    assert "cache write failed" in caplog.text
    assert "cache invalidation failed" in caplog.text
    assert "redis unavailable" not in caplog.text


def test_session_memory_boundary_never_logs_raw_exception_detail(caplog) -> None:
    """Best-effort session failures log only fixed operation codes."""
    private = "password=private-password-value"

    class RaisingCache:
        def get_json_sync(self, key: str) -> object:
            del key
            raise RuntimeError(private)

        def set_json_sync(self, key: str, value: object, *, ttl_seconds: int) -> None:
            del key, value, ttl_seconds
            raise RuntimeError(private)

        def delete_json_sync(self, key: str) -> None:
            del key
            raise RuntimeError(private)

    store = SessionMemoryStore(RaisingCache())  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        assert store.load("session-a") == SessionMemory()
        store.append("session-a", _turn(1))
        store.clear("session-a")

    assert "session memory load failed" in caplog.text
    assert "session memory write failed" in caplog.text
    assert "session memory clear failed" in caplog.text
    assert private not in caplog.text
    assert private not in repr(caplog.records)


@pytest.mark.parametrize(
    "request_text",
    [
        "How does NVDA sell accelerators to cloud providers?",
        "What is the effect of trade restrictions on revenue?",
        "Explain the company's share repurchase disclosure.",
        "What is NVDA's secret sauce?",
    ],
)
def test_business_research_language_remains_session_memory_eligible(request_text: str) -> None:
    assert is_session_memory_eligible_request(request_text)


@pytest.mark.parametrize(
    "request_text",
    [
        "I have 100 NVDA shares.",
        "I own a large position in NVDA.",
        "My portfolio is concentrated in semiconductors.",
        "brokerage_account_number=12345678",
        "My risk appetite is aggressive.",
        "My risk tolerance is low.",
        "email=investor@example.com",
        "phone: +1-416-555-0199",
        "full_name=Private Person",
        "home_address=123 Private Street",
        "password=hunter2",
        "password is hunter2",
        "api_secret=top-secret",
        "API key is top-secret",
        "Authorization: Bearer abc.def.ghi",
        "Bearer abc.def.ghi",
        "access token is private-access-token",
        "refresh_token=secret-token",
        "refresh token is secret-token",
        "session token: private-session-token",
        "session token is private-session-token",
        "I plan to buy 100 NVDA shares; summarize the latest filing.",
        "I intend to sell 25 NVDA shares; summarize revenue growth.",
        "I want to hold 10 NVDA shares; explain disclosed risks.",
        "我的持仓是英伟达 100 股。",
        "我的风险偏好较高。",
        "我打算买入 100 股英伟达；请总结最新财报。",
        "我想卖出英伟达股票；请分析收入增长。",
        "我计划持有这些股票；请解释风险披露。",
        "券商账户号码=12345678",
        "邮箱=investor@example.com",
        "姓名=私人用户",
        "密码是 hunter2",
        "访问令牌是 private-token",
        "密钥是 top-secret",
    ],
)
def test_personal_or_secret_requests_are_not_session_memory_eligible(
    request_text: str,
) -> None:
    assert not is_session_memory_eligible_request(request_text)


@pytest.mark.parametrize(
    ("request_text", "eligible"),
    [
        ("secret is credential-value", False),
        ("secret: credential-value", False),
        ("secret=credential-value", False),
        ("秘密是 credential-value", False),
        ("机密: credential-value", False),
        ("I plan to buy NVDA; summarize the latest filing.", False),
        ("I intend to sell NVDA; summarize revenue growth.", False),
        ("I'm going to hold NVDA; explain disclosed risks.", False),
        ("I will buy NVDA; summarize the latest filing.", False),
        ("We would sell NVDA if guidance weakens.", False),
        ("I shall hold NVDA while reviewing disclosed risks.", False),
        ("We'll buy NVDA after the earnings call.", False),
        ("I'd sell NVDA if margins contract.", False),
        ("We'd hold NVDA through the next filing.", False),
        ("I might buy NVDA after reviewing the filing.", False),
        ("We should sell NVDA based on the disclosed risks.", False),
        ("I intend buying NVDA; summarize revenue growth.", False),
        ("We plan on selling NVDA; explain the latest filing.", False),
        ("I want holding NVDA to be my next position.", False),
        ("I'm planning to buy NVDA after the earnings call.", False),
        ("We're going to sell NVDA after the filing.", False),
        ("I would like to hold NVDA through the next quarter.", False),
        ("I am buying NVDA after the earnings call.", False),
        ("I will be buying NVDA after the filing.", False),
        ("We would be selling NVDA if guidance weakens.", False),
        ("I'll be holding NVDA through the quarter.", False),
        ("I'm gonna buy NVDA after the call.", False),
        ("I wanna hold NVDA for a year.", False),
        ("I'm considering buying NVDA after the filing.", False),
        ("We are thinking of selling NVDA.", False),
        ("我会买入NVDA；请总结最新财报。", False),
        ("我们将卖出NVDA；请分析收入增长。", False),
        ("我准备持有NVDA；请解释风险披露。", False),
        ("我們會買入NVDA；請總結最新財報。", False),
        ("我将会持有NVDA；请解释风险披露。", False),
        ("NVDA plans to buy components from suppliers.", True),
        ("How does NVDA sell accelerators?", True),
        ("I want to research how NVDA buys components.", True),
        ("I will research whether NVDA buys components.", True),
        ("I intend to analyze NVDA buying behavior.", True),
        ("We would analyze whether NVDA should sell more accelerators.", True),
        ("I will buy components and research NVDA afterward.", True),
        ("I will buy NVDA GPUs for a workstation.", True),
        ("We plan on buying NVDA's new systems.", True),
        ("I will hold NVDA management accountable for the disclosure.", True),
        ("I want to understand NVDA's trade restrictions.", True),
        ("The company's secret sauce is its software ecosystem.", True),
    ],
)
def test_ticker_aware_trade_intent_and_generic_secret_eligibility(
    request_text: str,
    eligible: bool,
) -> None:
    assert is_session_memory_eligible_request(request_text, current_ticker="NVDA") is eligible


@pytest.mark.parametrize("modal", ["will", "would", "might", "may", "could", "shall"])
@pytest.mark.parametrize(
    "wrapper",
    ["consider", "be contemplating", "be thinking about"],
)
@pytest.mark.parametrize("action", ["buying", "selling", "holding"])
def test_modal_deliberation_about_ticker_is_not_session_memory_eligible(
    modal: str,
    wrapper: str,
    action: str,
) -> None:
    request_text = f"I {modal} {wrapper} {action} NVDA after reviewing the filing."

    assert not is_session_memory_eligible_request(request_text, current_ticker="NVDA")


@pytest.mark.parametrize(
    "request_text",
    [
        "I contemplate buying NVDA after the filing.",
        "I am considering selling NVDA after the filing.",
        "I think about holding NVDA through the quarter.",
        "I am looking to buy NVDA after the filing.",
        "I prepare to sell NVDA after the earnings call.",
        "I would prepare to hold NVDA through the quarter.",
        "我会考虑买入NVDA。",
        "我将会考虑卖出NVDA。",
        "我可能会考虑持有NVDA。",
        "我们也许会考虑购买NVDA。",
        "我或许会考虑出售NVDA。",
        "我们应该会考虑持仓NVDA。",
        "我正在考虑买入NVDA。",
        "我們將會考慮賣出NVDA。",
        "我們可能會考慮持有NVDA。",
    ],
)
def test_natural_deliberative_trade_forms_are_not_session_memory_eligible(
    request_text: str,
) -> None:
    assert not is_session_memory_eligible_request(request_text, current_ticker="NVDA")


@pytest.mark.parametrize(
    "request_text",
    [
        "I would consider not buying NVDA.",
        "I might consider buying or selling NVDA.",
        "I may contemplate holding or selling NVDA.",
        "我会考虑不买入NVDA。",
        "我可能会考虑买入或卖出NVDA。",
        "我們會考慮不持有NVDA。",
    ],
)
def test_post_wrapper_negation_and_coordinated_trade_actions_are_not_eligible(
    request_text: str,
) -> None:
    """Dropping either suffix form would retain current-ticker deliberation."""
    assert not is_session_memory_eligible_request(request_text, current_ticker="NVDA")


@pytest.mark.parametrize(
    "request_text",
    [
        "I will consider whether NVDA buys components.",
        "I might analyze how NVDA sells GPUs.",
        "I would consider buying NVDA GPUs for a workstation.",
        "I will contemplate buying NVDA components for a prototype.",
        "I might be thinking about buying NVDA's new systems.",
        "I would prepare to hold NVDA management accountable.",
        "我会考虑NVDA是否购买组件。",
        "我可能会分析NVDA如何销售GPU。",
        "我会考虑购买NVDA芯片用于工作站。",
        "I will consider buying about around approximately roughly up to another "
        "an additional some more all any the my NVDA.",
    ],
)
def test_deliberative_company_product_and_bounded_relation_cases_remain_eligible(
    request_text: str,
) -> None:
    assert is_session_memory_eligible_request(request_text, current_ticker="NVDA")


@pytest.mark.parametrize(
    "stored",
    [
        {"summary": "x" * 8_001, "turns": []},
        {
            "summary": "oversized turn",
            "turns": [
                {
                    "question": "Question?",
                    "answer_summary": "x" * 1_201,
                    "run_id": "run-1",
                    "ticker": "NVDA",
                    "open_questions": [],
                }
            ],
        },
        {
            "summary": "oversized open question",
            "turns": [
                {
                    "question": "Question?",
                    "answer_summary": "Guarded answer.",
                    "run_id": "run-1",
                    "ticker": "NVDA",
                    "open_questions": ["x" * 501],
                }
            ],
        },
    ],
)
def test_store_rejects_oversized_model_fields_as_empty_memory(stored: dict[str, object]) -> None:
    redis = _SyncRedisBoundary(stored=json.dumps(stored))
    cache = RedisJsonCache(_UnusedAsyncRedis(), sync_client=redis, namespace="")  # type: ignore[arg-type]

    assert SessionMemoryStore(cache).load("session-a") == SessionMemory()


def test_store_and_public_rolling_summary_helper_have_identical_bounds() -> None:
    store = SessionMemoryStore(InMemoryTtlJsonCache())
    for index in range(1, 7):
        store.append("session-a", _turn(index))

    memory = store.load("session-a")

    assert memory.summary == build_rolling_summary(memory.turns)
    assert len(memory.summary) <= 8_000


def test_same_ticker_view_reuses_the_public_rolling_summary_helper() -> None:
    store = SessionMemoryStore(InMemoryTtlJsonCache())
    store.append("session-a", _turn(1, ticker="NVDA"))
    store.append("session-a", _turn(2, ticker="AMD"))

    memory = load_session_memory_for_ticker(store, "session-a", "NVDA")

    assert [turn.ticker for turn in memory.turns] == ["NVDA"]
    assert memory.summary == build_rolling_summary(memory.turns)
    assert "AMD" not in memory.summary


@pytest.mark.parametrize(
    ("ticker", "text", "allowed"),
    [
        ("AMD", "AMD disclosed a change.", False),
        ("AMD", "amd disclosed a change.", False),
        ("AMD", "AMD. disclosed a change.", False),
        ("AMD", "AMD, disclosed a change.", False),
        ("BRK.B", "BRK.B disclosed a change.", False),
        ("BRK.B", "brk.b disclosed a change.", False),
        ("ON", "ON disclosed a change.", False),
        ("ON", "Demand remained on trend.", True),
        ("A", "A company disclosed a change.", True),
        ("A", "a company disclosed a change.", True),
    ],
)
def test_peer_ticker_sanitizer_uses_punctuation_safe_conservative_boundaries(
    ticker: str,
    text: str,
    allowed: bool,
) -> None:
    from financial_evidence_agent import application as application_module

    safety = application_module._SummarySafety(  # noqa: SLF001
        forbidden_values=(),
        other_tickers=(ticker,),
    )

    assert safety.allows(text) is allowed
