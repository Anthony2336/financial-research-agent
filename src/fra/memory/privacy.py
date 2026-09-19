"""Shared request privacy classification."""

from __future__ import annotations

import re
from collections.abc import Iterator

from fra.memory.privacy_normalization import privacy_text_views
from fra.memory.privacy_relations import (
    contains_private_named_finance,
    contains_security_ticker,
    is_public_aggregate_profile,
    is_public_institutional_actor,
)

_PERSONAL_FINANCE = re.compile(
    r"\b(?:my\s+(?:shares?|positions?|portfolio|holdings?|risk\s+(?:appetite|tolerance))|"
    r"i\s+(?:have|own|hold)\b.{0,40}\b(?:shares?|positions?|portfolio|holdings?))\b|"
    r"(?:我的持仓|我的持倉|我的仓位|我的倉位|我的投资组合|我的投資組合|"
    r"我的风险偏好|我的風險偏好|我的风险承受能力|我的風險承受能力|"
    r"我(?:有|持有|拥有|擁有).{0,20}(?:股|仓位|倉位|投资组合|投資組合))",
    re.IGNORECASE,
)
_PERSONAL_TRADE_INTENT = re.compile(
    r"\bi\s+(?:plan|intend|want)(?:\s+to)?\s+(?:buy|sell|hold)\b.{0,60}"
    r"\b(?:shares?|stocks?|positions?|securities)\b|"
    r"我(?:计划|計劃|打算|想要|想)\s*(?:买入|買入|买|買|卖出|賣出|卖|賣|持有)"
    r".{0,40}(?:股|股票|仓位|倉位)",
    re.IGNORECASE,
)
_ENGLISH_MODAL = (
    r"(?:will|would|shall|should|could|can|may|might|won['’]t|wouldn['’]t|"
    r"shouldn['’]t|couldn['’]t|can['’]t|mightn['’]t|shan['’]t)"
)
_ENGLISH_FIRST_PERSON = (
    rf"(?:i(?:['’](?:m|ll|d)|\s+(?:am|{_ENGLISH_MODAL}))?|"
    rf"we(?:['’](?:re|ll|d)|\s+(?:are|{_ENGLISH_MODAL}))?)"
)
_ENGLISH_MODAL_MODIFIER = r"(?:not|probably|definitely|possibly|still)"
_ENGLISH_INTENT_WRAPPER = (
    r"(?:consider(?:ed|ing)?|contemplat(?:e|ed|ing)|"
    r"(?:think(?:ing)?|thought)\s+(?:about|of)|"
    r"(?:look(?:ed|ing)?|prepar(?:e|ed|ing))\s+to|"
    r"(?:plan|planned|planning)(?:\s+(?:to|on))?|"
    r"(?:intend(?:ed|ing)?|want(?:ed|ing)?)(?:\s+to)?|"
    r"(?:like|prefer)\s+to|rather|(?:going|about)\s+to|gonna|wanna)"
)
_ENGLISH_TRADE_ACTION = r"(?:buy(?:ing)?|sell(?:ing)?|hold(?:ing)?)"
_ENGLISH_TRADE_ACTIONS = (
    rf"{_ENGLISH_TRADE_ACTION}"
    rf"(?:\s+(?:or|and(?:/or)?)\s+{_ENGLISH_TRADE_ACTION})*"
)
_ENGLISH_PERSONAL_TRADE_ACTION = re.compile(
    rf"\b{_ENGLISH_FIRST_PERSON}\s+"
    rf"(?:{_ENGLISH_MODAL_MODIFIER}\s+)*"
    rf"(?:be\s+(?:{_ENGLISH_MODAL_MODIFIER}\s+)*)?"
    rf"(?:{_ENGLISH_INTENT_WRAPPER}\s+)?"
    rf"(?:{_ENGLISH_MODAL_MODIFIER}\s+)*"
    rf"{_ENGLISH_TRADE_ACTIONS}\b",
    re.IGNORECASE,
)
_CHINESE_MODAL = (
    r"(?:可能会|可能會|也许会|也許會|或许会|或許會|应该会|應該會|"
    r"将会|將會|将要|將要|可能|也许|也許|或许|或許|会|會|将|將|要)"
)
_CHINESE_MODAL_MODIFIER = r"(?:仍然|不会|不會|不|再|也)"
_CHINESE_INTENT_WRAPPER = (
    r"(?:正在考虑|正在考慮|正在思考|正考虑|正考慮|考虑|考慮|思考|"
    r"准备|準備|打算|计划|計劃|希望|有意|想要|想)"
)
_CHINESE_TRADE_ACTION = (
    r"(?:买入|買入|购买|購買|卖出|賣出|出售|持有|持仓|持倉|买|買|卖|賣)"
)
_CHINESE_TRADE_ACTIONS = (
    rf"{_CHINESE_TRADE_ACTION}(?:(?:或|或者|還是|还是|和|及){_CHINESE_TRADE_ACTION})*"
)
_CHINESE_PERSONAL_TRADE_ACTION = re.compile(
    rf"我(?:们|們)?\s*(?:{_CHINESE_MODAL}\s*)?"
    rf"(?:{_CHINESE_MODAL_MODIFIER}\s*)*"
    rf"(?:{_CHINESE_INTENT_WRAPPER}\s*)?"
    rf"(?:{_CHINESE_MODAL_MODIFIER}\s*)*"
    rf"{_CHINESE_TRADE_ACTIONS}"
)
_SENSITIVE_IDENTITY = re.compile(
    r"\b(?:brokerage[_ -]?account(?:[_ -]?(?:number|no\.?))?|"
    r"account[_ -]?(?:number|no\.?|#|id)|"
    r"email|e-mail|phone|telephone|mobile|ssn|sin|credit[_ -]?card|full[_ -]?name|"
    r"home[_ -]?address|mailing[_ -]?address|date[_ -]?of[_ -]?birth|dob)\s*"
    r"(?::|=|\b(?:is|are)\b)\s*\S+|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"(?:券商账户(?:号码|号)?|券商帳戶(?:號碼|號)?|账户号码|帳戶號碼|邮箱|郵箱|"
    r"电子邮件|電子郵件|手机号|手機號|电话号码|電話號碼|身份证号|身份證號|"
    r"姓名|住址|家庭地址|出生日期)\s*(?:[:=：]|是|为|為)\s*\S+",
    re.IGNORECASE,
)
_ACCOUNT_IDENTIFIER = re.compile(
    r"\b(?:brokerage[_ -]?account(?:[_ -]?(?:number|no\.?|#))?|"
    r"account[_ -]?(?:number|no\.?|#|id))\s*"
    r"(?:(?:is|equals?)\b|[:=])?\s*"
    r"(?=[A-Z0-9-]{3,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]{3,}\b",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:password|passcode|secret|credentials?|api[_ -]?(?:secret|key)|"
    r"client[_ -]?secret|access[_ -]?token|refresh[_ -]?token|session[_ -]?token|"
    r"bearer[_ -]?token|tokens?|private[_ -]?key)\s*"
    r"(?::|=|\b(?:is|are)\b)\s*\S+|"
    r"\bauthorization\s*:\s*bearer\s+\S+|"
    r"\bbearer\s+(?=[A-Z0-9._~-]{8,}\b)(?=[A-Z0-9._~-]*[._~-])"
    r"[A-Z0-9._~-]+|"
    r"(?:密码|密碼|口令|秘密|机密|機密|凭据|憑據|API密钥|API密鑰|API秘密|"
    r"客户端密钥|客戶端密鑰|密钥|密鑰|访问令牌|訪問令牌|刷新令牌|"
    r"会话令牌|會話令牌|令牌|私钥|私鑰)\s*(?:[:=：]|是|为|為)\s*\S+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)
_PERSONAL_SUBJECT_EN = (
    r"(?:\b(?:i|me|my|mine|we|us|our|ours|you|your|yours|"
    r"he|she|him|her|his|hers|they|them|their|theirs)\b|"
    r"\b(?:(?:the|an?)\s+)?(?:users?|clients?|investors?|account\s+holders?)"
    r"(?:['’]s)?\b)"
)
_PERSONAL_POSSESSOR_EN = (
    r"(?:\b(?:my|our|your|his|her|hers|their|theirs)\b|"
    r"\b(?:(?:the|an?)\s+)?(?:users?|clients?|investors?|account\s+holders?)"
    r"(?:['’]s|['’]))"
)
_PROFILE_EN = (
    r"\b(?:holdings?|positions?|portfolios?|brokerage(?:\s+accounts?)?|accounts?|"
    r"stakes?|ownership|risk\s+(?:appetite|tolerance|profile)|trade\s+intent)\b"
)
_FINANCIAL_OBJECT_EN = (
    r"\b(?:shares?|stocks?|trades?|positions?|holdings?|portfolios?|"
    r"stakes?|ownership|brokerage(?:\s+accounts?)?|accounts?|"
    r"security\s+interests?)\b"
)
_PERSONAL_OWNER_EN = (
    r"(?:\b(?:mine|ours|yours|his|hers|theirs)\b|"
    r"\b(?:(?:the|an?)\s+)?(?:users?|clients?|investors?|account\s+holders?)"
    r"(?:['’]s\b|['’]))"
)
_PERSONAL_SUBJECT_ZH = (
    r"(?:我(?:的|们|們|们的|們的)?|你(?:的)?|您(?:的)?|"
    r"他(?:的|们|們|们的|們的)?|她(?:的|们|們|们的|們的)?|本人|"
    r"(?:该|該)?(?:用户|用戶)|(?:该|該)?(?:客户|客戶)|"
    r"投资者|投資者|账户持有人|帳戶持有人)"
)
_PROFILE_ZH = (
    r"(?:持仓|持倉|仓位|倉位|投资组合|投資組合|券商账户|券商帳戶|"
    r"经纪账户|經紀帳戶|账户|帳戶|风险偏好|風險偏好|"
    r"风险承受能力|風險承受能力|交易意图|交易意圖)"
)
_FINANCIAL_OBJECT_ZH = (
    r"(?:股票交易|证券交易|證券交易|交易|持仓|持倉|仓位|倉位|"
    r"持股|投资组合|投資組合|券商账户|券商帳戶|经纪账户|經紀帳戶|"
    r"账户|帳戶|证券权益|證券權益|股票|股份)"
)
_SEGMENT = r"[^\n.!?。！？]{0,120}"
_SCALAR_STATEMENT_BOUNDARY = re.compile(r"[.!?。！？;；]")
_PUBLIC_CREDENTIAL_DESCRIPTION = re.compile(
    r"\b(?:(?:public|sec)\s+filing\s+)?credentials?\s+(?:is|are)\s+"
    r"(?:described|discussed|defined|mentioned|required|used)\b"
    r"[^.!?。！？;；]{0,80}\b(?:access(?:-control)?|filings?|terminology)\b",
    re.IGNORECASE,
)


def _iter_personal_trade_action_ends(text: str) -> Iterator[int]:
    for grammar in (_ENGLISH_PERSONAL_TRADE_ACTION, _CHINESE_PERSONAL_TRADE_ACTION):
        for match in grammar.finditer(text):
            yield match.end()


def _is_public_credential_description(
    text: str,
    match: re.Match[str],
) -> bool:
    clause_start = max(
        (
            boundary.end()
            for boundary in _SCALAR_STATEMENT_BOUNDARY.finditer(text, 0, match.start())
        ),
        default=0,
    )
    boundary = _SCALAR_STATEMENT_BOUNDARY.search(text, match.end())
    clause_end = len(text) if boundary is None else boundary.start()
    return any(
        candidate.start() < match.end() - clause_start
        and candidate.end() > match.start() - clause_start
        for candidate in _PUBLIC_CREDENTIAL_DESCRIPTION.finditer(
            text[clause_start:clause_end]
        )
    )


def _all_personal_subjects_are_public_institutional(
    text: str,
    match: re.Match[str],
) -> bool:
    subjects = list(re.finditer(_PERSONAL_SUBJECT_EN, match.group(), re.IGNORECASE))
    if not subjects:
        return False
    for subject in subjects:
        subject_start = match.start() + subject.start()
        if not is_public_institutional_actor(
            text,
            subject_start=subject_start,
            subject=subject.group(),
        ):
            return False
    return True


def _contains_private_scalar(
    text: str,
    *,
    current_ticker: str | None = None,
) -> bool:
    normalized = text
    if not normalized:
        return False
    if any(
        pattern.search(normalized)
        for pattern in (
            _PERSONAL_FINANCE,
            _PERSONAL_TRADE_INTENT,
            _SENSITIVE_IDENTITY,
            _ACCOUNT_IDENTIFIER,
        )
    ):
        return True
    for secret_match in _SECRET_ASSIGNMENT.finditer(normalized):
        if _is_public_credential_description(normalized, secret_match):
            continue
        return True
    normalized_ticker = (current_ticker or "").strip().upper()
    personal_profile = re.compile(
        rf"{_PERSONAL_SUBJECT_EN}{_SEGMENT}{_PROFILE_EN}|"
        rf"{_PROFILE_EN}{_SEGMENT}{_PERSONAL_SUBJECT_EN}",
        re.IGNORECASE,
    )
    for match in personal_profile.finditer(normalized):
        if is_public_aggregate_profile(
            normalized,
            start=match.start(),
            end=match.end(),
        ) or _all_personal_subjects_are_public_institutional(normalized, match):
            continue
        return True
    if re.search(
        rf"{_PERSONAL_POSSESSOR_EN}{_SEGMENT}\b(?:shares|stocks?)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True
    personal_owner_relation = re.compile(
        rf"{_FINANCIAL_OBJECT_EN}{_SEGMENT}"
        rf"(?:belong(?:s|ed)?\s+to|owned\s+by|held\s+by){_SEGMENT}"
        rf"(?P<subject>{_PERSONAL_SUBJECT_EN})",
        re.IGNORECASE,
    )
    for relation in personal_owner_relation.finditer(normalized):
        if is_public_institutional_actor(
            normalized,
            subject_start=relation.start("subject"),
            subject=relation.group("subject"),
        ):
            continue
        return True
    if re.search(
        rf"{_FINANCIAL_OBJECT_EN}\s+(?:is|are|was|were)\s+"
        rf"(?:(?:exclusively|entirely|personally|solely|wholly)\s+)?"
        rf"{_PERSONAL_OWNER_EN}",
        normalized,
        re.IGNORECASE,
    ):
        return True
    security_target = r"\b(?:shares?|stocks?|securities|positions?|stakes?|ownership)\b"
    ownership_or_trade = (
        r"(?:own(?:s|ed|ing)?|has|have|hold(?:s|ing)?|buy(?:s|ing)?|"
        r"sell(?:s|ing)?|invest(?:s|ed|ing)?|trad(?:e|es|ed|ing))"
    )
    personal_security_relation = re.compile(
        rf"(?P<subject>{_PERSONAL_SUBJECT_EN}){_SEGMENT}{ownership_or_trade}{_SEGMENT}"
        rf"{security_target}",
        re.IGNORECASE,
    )
    for relation in personal_security_relation.finditer(normalized):
        if is_public_institutional_actor(
            normalized,
            subject_start=relation.start("subject"),
            subject=relation.group("subject"),
        ):
            continue
        return True
    personal_relation = re.compile(
        rf"(?P<subject>{_PERSONAL_SUBJECT_EN}){_SEGMENT}{ownership_or_trade}",
        re.IGNORECASE,
    )
    for action in personal_relation.finditer(
        normalized,
    ):
        if is_public_institutional_actor(
            normalized,
            subject_start=action.start("subject"),
            subject=action.group("subject"),
        ):
            continue
        tail = normalized[action.end() : action.end() + 60]
        if contains_security_ticker(tail, normalized_ticker):
            return True
    if re.search(
        rf"{_PERSONAL_SUBJECT_ZH}{_SEGMENT}{_PROFILE_ZH}|"
        rf"{_PROFILE_ZH}{_SEGMENT}{_PERSONAL_SUBJECT_ZH}",
        normalized,
    ):
        return True
    if re.search(rf"{_PERSONAL_SUBJECT_ZH}{_SEGMENT}(?:股票|股份)", normalized):
        return True
    if re.search(
        rf"{_FINANCIAL_OBJECT_ZH}\s*(?:(?:属于|屬於|归|歸)\s*{_PERSONAL_SUBJECT_ZH}|"
        rf"(?:是|为|為)\s*{_PERSONAL_SUBJECT_ZH}\s*(?:的|所有))",
        normalized,
    ):
        return True
    chinese_security_target = (
        rf"(?:{re.escape(normalized_ticker)}"
        r"(?!\s*(?:芯片|產品|产品|組件|组件|管理层|管理層))|"
        r"股票|股份|股|持仓|持倉|仓位|倉位)"
        if normalized_ticker
        else r"(?:股票|股份|股|持仓|持倉|仓位|倉位)"
    )
    if re.search(
        rf"{_PERSONAL_SUBJECT_ZH}{_SEGMENT}"
        rf"(?:持有|拥有|擁有|买入|買入|购买|購買|卖出|賣出|投资|投資)"
        rf"{_SEGMENT}{chinese_security_target}",
        normalized,
    ):
        return True
    if normalized_ticker:
        for action_end in _iter_personal_trade_action_ends(normalized):
            tail = normalized[action_end : action_end + 60]
            if contains_security_ticker(tail, normalized_ticker):
                return True
    return False


def contains_private_financial_or_secret(
    text: str,
    *,
    current_ticker: str | None = None,
) -> bool:
    """Return whether text contains private financial or secret material."""
    views = privacy_text_views(text)
    return any(
        _contains_private_scalar(value, current_ticker=current_ticker)
        or contains_private_named_finance(
            value,
            current_ticker=current_ticker or "",
        )
        for value in dict.fromkeys((views.joined, views.separated))
        if value
    )
