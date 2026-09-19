"""Deterministic, bounded named-finance relation classification."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

_TRADE_OBJECT_PREFIX = re.compile(
    r"[\s\"'“‘(\[]*"
    r"(?:(?:about|around|approximately|roughly|up\s+to|another|an\s+additional|"
    r"some|more|all|any|the|my)\s+)*"
    r"(?:(?:[$€£¥]\s*)?\d[\d,.]*(?:\s+(?:shares?|stocks?|units?))?\s+)?"
    r"(?:(?:shares?|stocks?|securities|a\s+position)\s+(?:of|in)\s+)?"
    r"(?:(?:约|約|大约|大約|大概|近|最多|至少|再|更多|全部|所有)\s*)?"
    r"(?:\d+(?:[,.]\d+)*(?:万|萬|千|百)?股?\s*)?",
    re.IGNORECASE,
)
_NON_SECURITY_TICKER_SUFFIX = re.compile(
    r"(?:['’]s)?\s+(?:(?:new|latest|next|upcoming)\s+)?"
    r"(?:(?:[a-z0-9-]+\s+){0,3})?"
    r"(?:accelerators?|components?|gpus?|chips?|products?|systems?|hardware|software|"
    r"platforms?|servers?|services?|cards?|boards?|management|executives?|board)\b|"
    r"\s+(?:(?:[a-z]*\d[a-z0-9-]*|dgx)"
    r"(?:\s+(?:rtx|pro|ultra|super|\d+)){0,2}|geforce\s+rtx\s+\d+)\b|"
    r"\s+(?:accountable|responsible)\b|"
    r"(?:的)?\s*(?:(?:新|最新|下一代)\s*)?"
    r"(?:加速器|组件|組件|零部件|芯片|晶片|产品|產品|系统|系統|硬件|硬體|软件|軟體|"
    r"平台|服务器|伺服器|服务|服務|显卡|顯卡|板卡|管理层|管理層|高管|董事会|董事會)",
    re.IGNORECASE,
)
_PUBLIC_SOURCE = (
    r"(?:(?:public\s+)?Form\s+[345](?![A-Z0-9])(?:\s+filings?)?|"
    r"Schedule\s+13[DG](?![A-Z0-9])|proxy\s+(?:filing|statement)|"
    r"(?:an?\s+)?public\s+(?:filing|disclosure)|公开披露|公開披露)"
)
_SOURCE_BEFORE_OWNERSHIP = re.compile(
    rf"(?:according\s+to\s+{_PUBLIC_SOURCE}|"
    rf"(?:as\s+)?(?:disclosed|reported)\s+in\s+{_PUBLIC_SOURCE}|"
    rf"{_PUBLIC_SOURCE}\s+(?:shows?|reports?|discloses?|lists?|states?)|"
    rf"(?:根据|根據|据|據)\s*{_PUBLIC_SOURCE}\s*"
    rf"(?:披露|显示|顯示|报告|報告)?|"
    rf"{_PUBLIC_SOURCE}\s*(?:披露|显示|顯示|报告|報告|表明))\s*[,，:：-]?\s*$",
    re.IGNORECASE,
)
_SOURCE_AFTER_CONNECTOR = (
    rf"(?:according\s+to\s+{_PUBLIC_SOURCE}|"
    rf"(?:as\s+)?(?:disclosed|reported)\s+in\s+{_PUBLIC_SOURCE}|"
    rf"(?:根据|根據|据|據)\s*{_PUBLIC_SOURCE}\s*"
    rf"(?:披露|显示|顯示|报告|報告)?)"
)
_SOURCE_AFTER_OWNERSHIP = re.compile(
    rf"^\s*[,，:：-]?\s*{_SOURCE_AFTER_CONNECTOR}", re.IGNORECASE
)
_SOURCE_AFTER_CANDIDATE = re.compile(
    rf"\s*[,，:：-]?\s*{_SOURCE_AFTER_CONNECTOR}", re.IGNORECASE
)
_STATEMENT_BOUNDARY = re.compile(r"[.!?。！？;；]")
_CORPORATE_ENTITY = re.compile(
    r"\b(?:company|corporation|corp\.?|inc\.?|issuer|llc|ltd\.?|plc|bank|partners|"
    r"capital|fund)\b|(?:公司|企业|企業|集团|集團|银行|銀行|基金|发行人|發行人)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)
_ENGLISH_NAMED_OWNERSHIP = re.compile(
    r"(?P<subject>[^\n.!?。！？;；,]{1,80}?)\s+"
    r"(?:owns?|holds?|has|bought|sold|buys?|sells?)\b",
    re.IGNORECASE,
)
_ENGLISH_NAMED_PROFILE = re.compile(
    r"(?P<subject>[^\n.!?。！？;；,]{1,80}?)['’]s\s+"
    r"(?:personal\s+)?(?P<profile>holdings?|positions?|portfolios?|"
    r"stakes?|ownership|brokerage\s+accounts?|risk\s+(?:appetite|tolerance|profile))\b",
    re.IGNORECASE,
)
_ENGLISH_NAMED_RISK = re.compile(
    r"(?P<subject>[^\n.!?。！？;；,]{1,80}?)\s+(?:has|have)\s+"
    r"(?:an?\s+)?(?:very\s+)?(?:high|low|aggressive|conservative|moderate)\s+"
    r"risk\s+(?:appetite|tolerance|profile)\b",
    re.IGNORECASE,
)
_CHINESE_NAMED_PROFILE = re.compile(
    r"(?P<subject>[\u3400-\u9fff]{2,4})\s*的\s*"
    r"(?P<profile>持仓|持倉|仓位|倉位|投资组合|投資組合|风险偏好|風險偏好|"
    r"风险承受能力|風險承受能力|风险概况|風險概況)"
)
_CHINESE_NAMED_OWNERSHIP = re.compile(
    r"(?=(?P<subject>[\u3400-\u9fff]{2,4})\s*(?:的\s*)?"
    r"(?P<relation>持有|拥有|擁有|买入|買入|购买|購買|卖出|賣出|投资|投資))"
)
_NAME_STOPWORDS = {
    "analyze", "can", "client", "clients", "compare", "customer", "customers",
    "could", "does", "employee", "employees", "explain", "fact", "investor",
    "investors", "it", "its", "management", "may", "might", "must", "note",
    "please", "research", "shall", "should", "summarize", "the", "then", "this",
    "user", "users", "what", "whether", "will", "would",
}
_CJK_NON_PERSON_SUBJECT_PARTS = ("我", "你", "您", "他", "她", "考虑", "考慮", "计划", "計劃")
_GENERIC_TICKER = re.compile(
    r"(?<![A-Za-z0-9])(?P<ticker>[A-Z]{2,5}(?:[.-][A-Z]{1,2})?)(?![A-Za-z0-9])"
)
_LOWERCASE_TICKER = re.compile(
    r"(?<![A-Za-z0-9])(?P<ticker>[a-z]{2,5})(?![A-Za-z0-9])"
)
_NON_TICKER_WORDS = frozenset(
    {"ACCOUNT", "API", "ARE", "CORP", "FORM", "HOLD", "HOLDS", "ID", "INC",
     "LLC", "LTD", "PLC", "SELL", "SHARE", "SHARES", "STOCK", "STOCKS",
     "THE", "USD"}
)
_PUBLIC_AGGREGATE_PROFILE = re.compile(
    r"(?:institutional\s+investors?['’]?\s+holdings?\b|"
    r"clients?['’]?\s+account\s+balances?\b[^.!?。！？;；]{0,80}"
    r"\b(?:disclosed|reported)\s+by\s+(?:the\s+)?(?:company|issuer)\b)",
    re.IGNORECASE,
)
_IDENTIFIED_AGGREGATE_PERSON = re.compile(
    r"\b(?P<connector>for|of|held\s+by|belonging\s+to)\s+"
    r"(?:(?:(?:the|a|an)\s+)?"
    r"(?P<role>clients?|customers?|account\s+holders?)\s+)?"
    r"(?!(?:account|aggregate|all|analysis|clients?|comparison|customers?|"
    r"reporting|the)\b)"
    r"(?P<subject>[^\W\d_]+(?:[-'’][^\W\d_]+)*(?:\s+[^\W\d_]+){0,2})\b",
    re.IGNORECASE,
)
_IDENTIFIED_AGGREGATE_ACCOUNT = re.compile(
    r"\baccount(?:\s+(?:number|no\.?|id|#))?\s*[:=#-]?\s*"
    r"(?=[a-z0-9-]{3,}\b)(?=[a-z0-9-]*\d)[a-z0-9-]{3,}\b",
    re.IGNORECASE,
)
_IDENTIFIED_AGGREGATE_POSSESSIVE_ACCOUNT = re.compile(
    r"\bin\s+[^\W\d_]+(?:[-'’][^\W\d_]+)*(?:\s+[^\W\d_]+){0,2}"
    r"['’]s\s+(?:accounts?|portfolios?|holdings?|positions?)\b",
    re.IGNORECASE,
)
_CORPORATE_BENEFICIARY = re.compile(
    r"\b(?:for|on\s+behalf\s+of)\s+(?P<subject>[^\n.!?。！？;；,，]{1,60})",
    re.IGNORECASE,
)
_BENEFICIARY_ROLES = frozenset({"client", "customer", "person", "account holder"})
_PUBLIC_PURPOSE_NOUNS = frozenset(
    {
        "analysis",
        "comparison",
        "disclosure",
        "growth",
        "investment",
        "purposes",
        "reasons",
        "reporting",
        "research",
        "strategy",
    }
)
_PUBLIC_RELATION_SUBJECT_PATTERN = (
    r"(?:"
    r"(?:class\s+[a-z0-9]+\s+)?"
    r"(?:common|preferred|voting)\s+(?:stock|shares?|securities)|"
    r"(?:institutional|public|retail)\s+(?:investors?|shareholders?)|"
    r"(?:strategic|economic|public)\s+(?:importance|purposes?|reasons?|interest)"
    r")"
)
_PUBLIC_RELATION_SUBJECT = re.compile(_PUBLIC_RELATION_SUBJECT_PATTERN, re.IGNORECASE)
_PUBLIC_RELATION_CONJUNCTION = re.compile(
    rf"^\s*(?:and|or|as\s+well\s+as)\s+{_PUBLIC_RELATION_SUBJECT_PATTERN}",
    re.IGNORECASE,
)
_CONJOINED_RELATION_SUBJECT = re.compile(
    r"(?:,|\b(?:and|or|but|with|alongside|including|as\s+well\s+as)\b)\s+"
    r"(?:(?:(?:the|a|an)\s+)?"
    r"(?P<role>clients?|customers?|account\s+holders?)\s+)?"
    r"(?P<subject>[^\W\d_]+(?:[-'’][^\W\d_]+)*(?:\s+[^\W\d_]+){0,2})\b",
    re.IGNORECASE,
)
_PUBLIC_RELATION_QUALIFIER = re.compile(
    r"^\s*(?:of\s+record|worldwide|globally|outstanding)\b\s*",
    re.IGNORECASE,
)
_PUBLIC_RELATION_PREDICATE = re.compile(
    r"^\s*(?:increased|decreased|rose|fell|remained|remains?|"
    r"represented|represents?|accounted|accounts?|was|were|is|are)\b"
)
_PUBLIC_PURPOSE_MODIFIERS = frozenset(
    {"disclosure", "economic", "long-term", "public", "reporting", "research", "strategic"}
)
_CORPORATE_TO_CORPORATE_STAKE = re.compile(
    r"(?P<relation>(?<![A-Za-z0-9])(?P<actor>[A-Z]{3,})\s+"
    r"(?:holds?|owns?|has)\s+"
    r"(?:an?\s+)?stake\s+in\s+"
    r"(?P<target>[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)\b)"
)
_ISSUER_ALIASES_BY_TICKER = {
    "NVDA": frozenset({"NVIDIA"}),
}


@dataclass(frozen=True, slots=True)
class _NamedFinanceRelation:
    name_start: int
    relation_end: int
    public_eligible: bool
    target_pattern: str | None = None
    requires_target: bool = False
    public_subject: bool = False


def _iter_security_ticker_matches(
    text: str, current_ticker: str
) -> Iterator[re.Match[str]]:
    seen: set[tuple[int, int]] = set()
    if current_ticker:
        pattern = rf"(?<![A-Z0-9]){re.escape(current_ticker)}(?![A-Z0-9])"
        flags = re.IGNORECASE if len(current_ticker) >= 3 else 0
        for match in re.finditer(pattern, text, flags):
            seen.add(match.span())
            yield match
    for match in _GENERIC_TICKER.finditer(text):
        if match.span() in seen or match.group("ticker") in _NON_TICKER_WORDS:
            continue
        yield match


def contains_security_ticker(text: str, current_ticker: str) -> bool:
    """Return whether a bounded relation tail begins with a security ticker."""
    return any(
        _TRADE_OBJECT_PREFIX.fullmatch(text[: match.start()])
        and _NON_SECURITY_TICKER_SUFFIX.match(text[match.end() :]) is None
        for match in _iter_security_ticker_matches(text, current_ticker)
    )


def _is_cjk_name(token: str) -> bool:
    return 2 <= len(token) <= 4 and all("\u3400" <= char <= "\u9fff" for char in token)


def _is_title_name(token: str) -> bool:
    if token.casefold() in _NAME_STOPWORDS:
        return False
    parts = re.split(r"[-'’]", token)
    return all(len(part) >= 2 and part[0].isupper() and part[1:].islower() for part in parts)


def _person_name_span(subject: str) -> tuple[int, int] | None:
    words = list(_WORD.finditer(subject))
    if not words:
        return None
    if _is_cjk_name(words[-1].group()):
        return words[-1].span()
    named_tail = 0
    for word in reversed(words):
        if not _is_title_name(word.group()):
            break
        named_tail += 1
    if not 1 <= named_tail <= 3:
        case_independent_tail = 0
        for word in reversed(words):
            token = word.group()
            if (
                token.casefold() in _NAME_STOPWORDS
                or _CORPORATE_ENTITY.fullmatch(token) is not None
                or not all(character.isalpha() or character in "-'’" for character in token)
            ):
                break
            case_independent_tail += 1
            if case_independent_tail == 3:
                break
        if not 1 <= case_independent_tail <= 3:
            return None
        named_tail = case_independent_tail
    span = words[-named_tail].start(), words[-1].end()
    return None if _CORPORATE_ENTITY.search(subject[slice(*span)]) else span


def _is_current_ticker_subject(
    subject: str, person_span: tuple[int, int], ticker: str
) -> bool:
    return bool(ticker) and subject[slice(*person_span)].strip().upper() == ticker


def _is_current_issuer_subject(
    subject: str,
    person_span: tuple[int, int],
    ticker: str,
) -> bool:
    aliases = _ISSUER_ALIASES_BY_TICKER.get(ticker, ())
    return subject[slice(*person_span)].strip().upper() in aliases


def _ownership_target_end(
    text: str,
    *,
    relation_end: int,
    ticker: str,
    target_pattern: str,
    forward_end: int,
) -> int | None:
    tail = text[relation_end:forward_end]
    source_candidate = _SOURCE_AFTER_CANDIDATE.search(tail)
    target_region = tail[: source_candidate.start()] if source_candidate else tail
    target_matches = list(re.finditer(target_pattern, target_region, re.IGNORECASE))
    ticker_matches = list(_iter_security_ticker_matches(target_region, ticker))
    if target_matches:
        target_end = target_matches[-1].end()
        for ticker_match in ticker_matches:
            if ticker_match.start() < target_end:
                continue
            bridge = target_region[target_end : ticker_match.start()]
            if re.fullmatch(r"\s*(?:of|in)?\s*", bridge, re.IGNORECASE):
                target_end = ticker_match.end()
            break
        return relation_end + target_end
    if contains_security_ticker(target_region, ticker):
        return relation_end + ticker_matches[0].end()
    for lowercase_match in _LOWERCASE_TICKER.finditer(target_region):
        if (
            _TRADE_OBJECT_PREFIX.fullmatch(target_region[: lowercase_match.start()])
            and _NON_SECURITY_TICKER_SUFFIX.match(
                target_region[lowercase_match.end() :]
            )
            is None
        ):
            return relation_end + lowercase_match.end()
    return None


def is_public_aggregate_profile(text: str, *, start: int, end: int) -> bool:
    """Return whether a profile match overlaps a public aggregate clause."""
    clause_start = max(
        (match.end() for match in _STATEMENT_BOUNDARY.finditer(text, 0, start)), default=0
    )
    boundary = _STATEMENT_BOUNDARY.search(text, start)
    clause_end = len(text) if boundary is None else boundary.start()
    return any(
        match.start() < end - clause_start and match.end() > start - clause_start
        for match in _PUBLIC_AGGREGATE_PROFILE.finditer(text[clause_start:clause_end])
    )


def is_public_institutional_actor(
    text: str, *, subject_start: int, subject: str
) -> bool:
    """Return whether a generic investor subject is institutionally qualified."""
    if re.search(r"\binvestors?\b", subject, re.IGNORECASE) is None:
        return False
    clause_start = max(
        (match.end() for match in _STATEMENT_BOUNDARY.finditer(text, 0, subject_start)),
        default=0,
    )
    return re.search(
        r"\binstitutional\s*$", text[clause_start:subject_start], re.IGNORECASE
    ) is not None


def _after_source_end(text: str, *, relation_end: int, forward_end: int) -> int | None:
    match = _SOURCE_AFTER_OWNERSHIP.match(text[relation_end:forward_end])
    return None if match is None else relation_end + match.end()


def _without_public_corporate_stakes(text: str, current_ticker: str) -> str:
    characters = list(text)
    issuer_aliases = _ISSUER_ALIASES_BY_TICKER.get(current_ticker, ())
    for match in _CORPORATE_TO_CORPORATE_STAKE.finditer(text):
        if match.group("actor") not in issuer_aliases:
            continue
        start, end = match.span("relation")
        boundary = _STATEMENT_BOUNDARY.search(text, end)
        relation_end = len(text) if boundary is None else boundary.start()
        relation_text = text[start:relation_end]
        if (
            _contains_identified_corporate_beneficiary(relation_text)
            or _contains_identified_non_beneficiary_person(relation_text)
            or _IDENTIFIED_AGGREGATE_ACCOUNT.search(relation_text)
            or _IDENTIFIED_AGGREGATE_POSSESSIVE_ACCOUNT.search(relation_text)
        ):
            continue
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _contains_identified_corporate_beneficiary(text: str) -> bool:
    for match in _CORPORATE_BENEFICIARY.finditer(text):
        subject = re.split(
            r"\b(?:and|but|while)\b",
            match.group("subject"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        folded = " ".join(word.group().casefold() for word in _WORD.finditer(subject))
        if not folded:
            continue
        if any(folded.startswith(role) for role in _BENEFICIARY_ROLES):
            return True
        words = list(_WORD.finditer(subject))
        if words[-1].group().casefold() in _PUBLIC_PURPOSE_NOUNS:
            continue
        if len(words) == 1 or all(_is_title_name(word.group()) for word in words[:3]):
            return True
    return False


def _is_identified_relation_person(subject: str, role: str | None) -> bool:
    if role is not None:
        return True
    candidate = subject.strip()
    if _PUBLIC_RELATION_SUBJECT.match(candidate) is not None:
        return False
    return _person_name_span(candidate) is not None


def _contains_conjoined_relation_person(text: str) -> bool:
    for match in _CONJOINED_RELATION_SUBJECT.finditer(text):
        subject = match.group("subject")
        if _PUBLIC_RELATION_PREDICATE.match(subject) is not None:
            continue
        if _is_identified_relation_person(subject, match.group("role")):
            return True
    return False


def _is_public_purpose_subject(subject: str) -> bool:
    words = [word.group().casefold() for word in _WORD.finditer(subject)]
    if not words or words[-1] not in _PUBLIC_PURPOSE_NOUNS:
        return False
    return len(words) == 1 or all(
        word in _PUBLIC_PURPOSE_MODIFIERS for word in words[:-1]
    )


def _public_relation_remainder_has_person(text: str) -> bool:
    remainder = _PUBLIC_RELATION_QUALIFIER.sub("", text, count=1)
    if not remainder.strip():
        return False
    public_conjunction = _PUBLIC_RELATION_CONJUNCTION.match(remainder)
    if public_conjunction is not None:
        return _public_relation_remainder_has_person(
            remainder[public_conjunction.end() :]
        )
    if (
        _contains_identified_corporate_beneficiary(remainder)
        or _contains_identified_non_beneficiary_person(remainder)
        or _contains_conjoined_relation_person(remainder)
        or _IDENTIFIED_AGGREGATE_ACCOUNT.search(remainder)
        or _IDENTIFIED_AGGREGATE_POSSESSIVE_ACCOUNT.search(remainder)
    ):
        return True
    if _PUBLIC_RELATION_PREDICATE.match(remainder) is not None:
        return False
    words = list(_WORD.finditer(remainder))[:3]
    if not words:
        return False
    subject = remainder[words[0].start() : words[-1].end()]
    return _is_identified_relation_person(subject, None)


def _contains_identified_non_beneficiary_person(text: str) -> bool:
    for match in _IDENTIFIED_AGGREGATE_PERSON.finditer(text):
        if (
            match.group("connector").casefold() == "for"
            and _is_public_purpose_subject(match.group("subject"))
        ):
            continue
        public_subject = _PUBLIC_RELATION_SUBJECT.match(
            text,
            match.start("subject"),
        )
        if public_subject is not None:
            if _public_relation_remainder_has_person(text[public_subject.end() :]):
                return True
            continue
        if _is_identified_relation_person(
            match.group("subject"),
            match.group("role"),
        ):
            return True
    return False


def _contains_identified_public_aggregate(text: str) -> bool:
    for aggregate in _PUBLIC_AGGREGATE_PROFILE.finditer(text):
        relation_start = max(
            (
                boundary.end()
                for boundary in _STATEMENT_BOUNDARY.finditer(
                    text,
                    0,
                    aggregate.start(),
                )
            ),
            default=0,
        )
        boundary = _STATEMENT_BOUNDARY.search(text, aggregate.end())
        relation_end = len(text) if boundary is None else boundary.start()
        relation_text = text[relation_start:relation_end]
        if (
            _contains_identified_corporate_beneficiary(relation_text)
            or _contains_identified_non_beneficiary_person(relation_text)
            or _IDENTIFIED_AGGREGATE_ACCOUNT.search(relation_text)
            or _IDENTIFIED_AGGREGATE_POSSESSIVE_ACCOUNT.search(relation_text)
        ):
            return True
    return False


def _contains_bounded_named_relation(text: str, ticker: str) -> bool:
    relations: list[_NamedFinanceRelation] = []
    for match in _ENGLISH_NAMED_PROFILE.finditer(text):
        subject = match.group("subject")
        person_span = _person_name_span(subject)
        if person_span is None:
            continue
        profile = match.group("profile").lower()
        public_eligible = profile.startswith(("holding", "position", "stake", "ownership"))
        public_subject = _is_current_ticker_subject(
            subject, person_span, ticker
        ) or _is_current_issuer_subject(subject, person_span, ticker)
        if (
            (public_subject and not public_eligible)
            or _CORPORATE_ENTITY.search(subject[person_span[0] : person_span[1]])
        ):
            continue
        relations.append(
            _NamedFinanceRelation(
                name_start=match.start("subject") + person_span[0],
                relation_end=match.end(),
                public_eligible=public_eligible,
                target_pattern=(
                    r"\b(?:shares?|stocks?|positions?|holdings?|stakes?|ownership)\b"
                    if public_eligible else None
                ),
                public_subject=public_subject,
            )
        )
    for match in _ENGLISH_NAMED_RISK.finditer(text):
        subject = match.group("subject")
        person_span = _person_name_span(subject)
        if person_span is None or _is_current_ticker_subject(subject, person_span, ticker):
            continue
        relations.append(
            _NamedFinanceRelation(
                name_start=match.start("subject") + person_span[0],
                relation_end=match.end(),
                public_eligible=False,
            )
        )
    for match in _ENGLISH_NAMED_OWNERSHIP.finditer(text):
        subject = match.group("subject")
        person_span = _person_name_span(subject)
        if person_span is None:
            continue
        public_subject = _is_current_ticker_subject(subject, person_span, ticker)
        if _CORPORATE_ENTITY.search(subject[person_span[0] : person_span[1]]):
            continue
        relations.append(
            _NamedFinanceRelation(
                name_start=match.start("subject") + person_span[0],
                relation_end=match.end(),
                public_eligible=True,
                target_pattern=(r"\b(?:shares?|stocks?|positions?|holdings?|portfolios?|stakes?|ownership)\b"),
                requires_target=True,
                public_subject=public_subject,
            )
        )
    for match in _CHINESE_NAMED_PROFILE.finditer(text):
        subject = match.group("subject")
        if _CORPORATE_ENTITY.search(subject) or any(
            part in subject for part in _CJK_NON_PERSON_SUBJECT_PARTS
        ):
            continue
        public_eligible = match.group("profile") in {"持仓", "持倉", "仓位", "倉位"}
        relations.append(
            _NamedFinanceRelation(
                name_start=match.start("subject"),
                relation_end=match.end(),
                public_eligible=public_eligible,
                target_pattern=(
                    r"(?:股票|股份|股|持仓|持倉|仓位|倉位)"
                    if public_eligible
                    else None
                ),
            )
        )
    chinese_matches_by_relation_end: dict[int, list[re.Match[str]]] = {}
    for match in _CHINESE_NAMED_OWNERSHIP.finditer(text):
        chinese_matches_by_relation_end.setdefault(match.end("relation"), []).append(match)
    for relation_end, matches in chinese_matches_by_relation_end.items():
        subjects = [match.group("subject") for match in matches]
        if any(_CORPORATE_ENTITY.search(subject) for subject in subjects) or any(
            part in subject for subject in subjects for part in _CJK_NON_PERSON_SUBJECT_PARTS
        ):
            continue
        match = min(
            matches,
            key=lambda candidate: (-len(candidate.group("subject")), candidate.start("subject")),
        )
        relations.append(
            _NamedFinanceRelation(
                name_start=match.start("subject"),
                relation_end=relation_end,
                public_eligible=True,
                target_pattern=r"(?:股票|股份|股|持仓|持倉|仓位|倉位)",
                requires_target=True,
            )
        )
    relations.sort(key=lambda relation: relation.name_start)
    previous_relation_end = 0
    for index, relation in enumerate(relations):
        sentence_start = max(
            (match.end() for match in _STATEMENT_BOUNDARY.finditer(text, 0, relation.name_start)),
            default=0,
        )
        sentence_end_match = _STATEMENT_BOUNDARY.search(text, relation.relation_end)
        sentence_end = len(text) if sentence_end_match is None else sentence_end_match.start()
        next_relation_start = (
            relations[index + 1].name_start if index + 1 < len(relations) else len(text)
        )
        forward_end = min(sentence_end, next_relation_start)
        relation_end = relation.relation_end
        if relation.target_pattern is not None:
            target_end = _ownership_target_end(
                text,
                relation_end=relation_end,
                ticker=ticker,
                target_pattern=relation.target_pattern,
                forward_end=forward_end,
            )
            if target_end is None and relation.requires_target:
                previous_relation_end = max(previous_relation_end, relation_end)
                continue
            relation_end = target_end or relation_end
        after_source_end = (
            _after_source_end(text, relation_end=relation_end, forward_end=forward_end)
            if relation.public_eligible else None
        )
        before_start = max(sentence_start, previous_relation_end)
        has_before_source = bool(
            _SOURCE_BEFORE_OWNERSHIP.search(text[before_start:relation.name_start])
        )
        previous_relation_end = after_source_end or relation_end
        if relation.public_subject:
            relation_suffix = text[relation.relation_end:forward_end]
            if (
                _contains_identified_corporate_beneficiary(relation_suffix)
                or _contains_identified_non_beneficiary_person(relation_suffix)
                or _IDENTIFIED_AGGREGATE_ACCOUNT.search(relation_suffix)
                or _IDENTIFIED_AGGREGATE_POSSESSIVE_ACCOUNT.search(relation_suffix)
            ):
                return True
            continue
        if relation.public_eligible and (after_source_end or has_before_source):
            continue
        return True
    return False


def contains_private_named_finance(
    text: str, *, current_ticker: str = ""
) -> bool:
    """Return whether text contains a private named financial relation."""
    if _contains_identified_public_aggregate(text):
        return True
    normalized_ticker = current_ticker.strip().upper()
    relation_text = _without_public_corporate_stakes(text, normalized_ticker)
    return _contains_bounded_named_relation(
        relation_text,
        normalized_ticker,
    )
