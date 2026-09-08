"""Deterministic, zero-token safety checks that run before the research graph."""

import re
import unicodedata

from financial_evidence_agent.domain import Intent, RouterDecision

REFUSAL_TEXT = (
    "我不能提供买卖、仓位、价格目标或个性化投资建议。"
    "我可以协助你把问题改写成可验证的研究观点，例如："
    "“最新 10-Q 中有哪些事实支持或挑战该公司的收入增长假设？”"
)

PROMPT_INJECTION_TEXT = "检测到试图改变系统规则的指令。请重新提交 ticker 和研究观点。"
ARBITRARY_URL_REFUSAL_TEXT = (
    "Arbitrary URL requests are not supported. Research uses only configured, "
    "allowlisted sources for the requested ticker."
)

_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b"),
    re.compile(r"\b(?:show|reveal|print)\s+(?:me\s+)?(?:the\s+)?system\s+prompt\b"),
    re.compile(r"\b(?:override|ignore|reveal)\s+(?:the\s+)?developer\s+message\b"),
    re.compile(r"(?:忽略|无视)(?:之前|先前|以上|所有).{0,8}(?:指令|提示|规则)"),
    re.compile(r"(?:显示|泄露|告诉我).{0,8}系统提示(?:词)?"),
    re.compile(r"(?:忽略|覆盖).{0,8}开发者(?:消息|指令)"),
)

_BASE_TRADE_ACTION = r"(?:buy|sell|hold)"
_ENGLISH_TRADE_ACTION = r"(?:buy(?:ing)?|sell(?:ing)?|hold(?:ing)?)"
_ARBITRARY_URL_PATTERN = re.compile(
    r"(?:\b(?:https?|ftp)://\S+|(?<!:)//(?:[a-z0-9-]+\.)+[a-z]{2,63}(?:/\S*)?|"
    r"\bwww\.(?:[a-z0-9-]+\.)+[a-z]{2,63}(?:/\S*)?|"
    r"\b[a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,63}(?:/\S*)?|"
    r"\b(?:[a-z0-9-]+\.)+[a-z]{2,63}/\S*)"
)
_BARE_HOST_PATTERN = re.compile(
    r"(?<![@a-z0-9.-])(?:"
    r"(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]{1,62}|"
    r"(?:\d{1,3}\.){3}\d{1,3}|localhost"
    r")(?::\d{1,5})?(?:/[^\s;；,，!?！？]*)?"
)
_SOURCE_REQUEST_PATTERN = re.compile(
    r"(?:summarize|open|read|review|fetch|browse|visit|介绍|总结|打开|读取|访问|浏览)"
    r"[^;；.!?。！？]{0,32}$"
)
_PRICE_PREDICTION_PATTERNS = (
    re.compile(
        r"\b(?:predict|forecast|project|estimate)\b.{0,48}"
        r"\b(?:stock\s+|share\s+)?price\b"
    ),
    re.compile(
        r"\b(?:stock\s+|share\s+)?price\b.{0,48}"
        r"\b(?:will|would|could|may)\b.{0,24}\b(?:reach|hit|be|trade)\b"
    ),
    re.compile(
        r"\b(?:will|would|could|may)\b.{0,48}"
        r"\b(?:stock\s+|share\s+)?price\b.{0,24}\b(?:reach|hit|be|trade)\b"
    ),
    re.compile(r"\b(?:target\s+price|price\s+target)\b"),
    re.compile(r"(?:预测|预估|预计|估算).{0,24}(?:股价|股票价格|目标价)"),
    re.compile(r"(?:股价|股票价格).{0,24}(?:会|将|能).{0,12}(?:达到|涨到|跌到|是多少|多少)"),
)
_ADVICE_PATTERNS = (
    re.compile(r"\bshould\s+i\s+(?:buy|sell|hold)\b"),
    re.compile(r"\bis\s+[a-z0-9.-]+\s+a\s+buy\b"),
    re.compile(r"\btarget\s+price\b"),
    re.compile(r"\b(?:position\s+siz(?:e|ing)|stop\s+loss|take\s+profit)\b"),
    re.compile(r"\bhow\s+much\s+of\s+my\s+portfolio\b"),
    re.compile(r"\bwhen\s+should\s+i\s+enter\b"),
    re.compile(r"\bwhat\s+trade\s+to\s+place\b"),
    re.compile(r"\btell\s+me\s+exactly\s+what\s+trade\b"),
    re.compile(
        r"\b(?:would|could|should|can|do|will)\s+(?:you|i|we)\s+"
        rf"(?:ever\s+)?(?:(?:recommend|suggest)\s+|be\s+)?{_ENGLISH_TRADE_ACTION}\b"
    ),
    re.compile(
        rf"\b(?:do|would|could|can)\s+you\s+think\b.{{0,40}}\bworth\s+"
        rf"{_ENGLISH_TRADE_ACTION}\b"
    ),
    re.compile(
        rf"\b(?:recommend|suggest)\s+{_ENGLISH_TRADE_ACTION}\s+"
        r"(?:(?:this|the|a)\s+)?(?:stock|shares?|securities?|position)\b"
    ),
    re.compile(
        rf"\b(?:do|would|could|can|will)\s+(?:analysts?|advisers?|brokers?)\s+"
        rf"(?:recommend|suggest)\s+{_ENGLISH_TRADE_ACTION}(?:\s+now)?(?=$|[?.,!;:])"
    ),
    re.compile(rf"\bworth\s+{_ENGLISH_TRADE_ACTION}\b"),
    re.compile(
        rf"\bwhether\s+(?:(?:i|you|we)\s+)?(?:should\s+|to\s+)?"
        rf"(?:be\s+)?{_ENGLISH_TRADE_ACTION}\b"
    ),
    re.compile(r"(?:^|[,:;.!?]\s*)\s*(?:please\s+)?(?:buy|sell|hold)\b"),
    re.compile(r"(?:能买吗|该不该买|现在能上车吗|卖掉|目标价|仓位|配仓|满仓|止损|止盈)"),
    re.compile(r"(?:是否|应该|该不该|要不要|建议|请|可以|能否).{0,10}(?:买入|卖出|持有)"),
    re.compile(r"(?:值得|推荐).{0,10}(?:买入|卖出|持有)"),
    re.compile(r"(?:买入|卖出|持有).{0,10}(?:值得|推荐)"),
    re.compile(r"^(?:请)?(?:买入|卖出|持有)(?:\b|[，,。.!?？])"),
)
_QUALITY_SCREEN_OUT_OF_SCOPE_PATTERNS = (
    re.compile(
        r"\b(?:rank|screen|screener|sort|list|filter)\b.{0,48}"
        r"\b(?:stock|stocks|shares?|securities?|companies?)\b"
    ),
    re.compile(
        r"\b(?:best|top)\b.{0,24}\b(?:stock|stocks|shares?|securities?)\b"
    ),
    re.compile(r"\b(?:portfolio|allocation|weight|weights)\b"),
    re.compile(
        r"\b(?:auto|automatic(?:ally)?|find|identify|pick|select|generate)\b.{0,24}"
        r"\b(?:peer|peers|competitors?|comps)\b"
    ),
    re.compile(
        r"\b(?:peer|peers|competitors?|comps)\b.{0,24}"
        r"\b(?:auto|automatic(?:ally)?|find|identify|pick|select|generate)\b"
    ),
    re.compile(r"(?:排名|筛选|选股|配仓|仓位|目标价|同行|可比公司)"),
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _contains_arbitrary_url_request(text: str) -> bool:
    if _ARBITRARY_URL_PATTERN.search(text):
        return True
    return any(
        _SOURCE_REQUEST_PATTERN.search(text[max(0, match.start() - 40) : match.start()])
        for match in _BARE_HOST_PATTERN.finditer(text)
    )


def _ticker_advice_patterns(normalized_ticker: str) -> tuple[re.Pattern[str], ...]:
    ticker_pattern = re.escape(normalized_ticker)
    security_target = rf"{ticker_pattern}(?:\s+(?:stock|shares?|securities?))?"
    recommendation_tail = (
        r"(?=$|[?.,!;:]|\s+based\s+on\b|\s+for\s+"
        r"(?:(?:my|your|our|the|a)\s+)?(?:portfolio|position|allocation|exposure)\b)"
    )
    return (
        re.compile(rf"\b{_BASE_TRADE_ACTION}\s+(?:shares?\s+(?:of\s+)?)?{ticker_pattern}\b"),
        re.compile(rf"\b{ticker_pattern}\s+(?:stock\s+)?{_BASE_TRADE_ACTION}\b"),
        re.compile(
            rf"\b(?:would|do|does|could|can|will)\s+"
            rf"(?:you|analysts?|advisers?|brokers?)\s+(?:recommend|suggest)\s+"
            rf"(?:the\s+)?{security_target}{recommendation_tail}"
        ),
        re.compile(
            rf"\b(?:would|do|does|could|can|will)\s+(?:you|analysts?|advisers?|brokers?)"
            rf"\s+(?:recommend|suggest)\s+(?:investing|(?:an?\s+)?investment)\s+in\s+"
            rf"{security_target}(?=$|[?.,!;:])"
        ),
        re.compile(
            rf"\b(?:recommend|suggest)\s+{_ENGLISH_TRADE_ACTION}\s+"
            rf"(?:shares?\s+(?:of\s+)?)?{security_target}\b"
        ),
        re.compile(
            rf"\b(?:would|could|should|can|do|will)\s+(?:you|i|we)\s+"
            rf"invest\s+in\s+{security_target}(?=$|[?.,!;:])"
        ),
        re.compile(rf"\bis\s+{security_target}\s+(?:a\s+)?good\s+investment\b"),
        re.compile(rf"\b{security_target}\s+is\s+(?:a\s+)?recommended\s+purchase\b"),
        re.compile(rf"(?:买入|卖出|持有)\s*{ticker_pattern}\b"),
        re.compile(rf"\b{ticker_pattern}\s*(?:买入|卖出|持有)"),
        re.compile(
            rf"(?:你|您|分析师|顾问)\s*(?:会|是否|能否|可以)?\s*"
            rf"(?:推荐|建议)(?:买(?:入)?|购买|卖出|持有)?\s*{security_target}\s*"
            r"(?:吗|呢)?(?=$|[?。.!])"
        ),
        re.compile(rf"(?:你|您)\s*(?:会|是否|能否|可以)?\s*投资\s*{security_target}"),
        re.compile(rf"{security_target}\s*是?\s*值得投资的?(?:股票|证券)?(?:吗)?(?=$|[?。.!])"),
    )


def contains_prohibited_investment_advice(text: str, ticker: str) -> bool:
    """Return whether the normalized request asks for prohibited investment advice."""

    normalized = _normalize(text)
    normalized_ticker = _normalize(ticker)
    return contains_price_prediction(normalized, normalized_ticker) or any(
        pattern.search(normalized) for pattern in _ADVICE_PATTERNS
    ) or any(pattern.search(normalized) for pattern in _ticker_advice_patterns(normalized_ticker))


def contains_quality_screen_out_of_scope_request(
    text: str,
    *,
    ticker: str | None = None,
) -> bool:
    """Return whether a quality-screen request widens into ranking, advice, or auto-peers."""

    normalized = _normalize(text)
    if any(pattern.search(normalized) for pattern in _QUALITY_SCREEN_OUT_OF_SCOPE_PATTERNS):
        return True
    if ticker is not None and contains_prohibited_investment_advice(normalized, ticker):
        return True
    return any(pattern.search(normalized) for pattern in _ADVICE_PATTERNS)


def route_quality_request(ticker: str, thesis: str) -> RouterDecision | None:
    """Apply only prompt-injection and arbitrary-source checks for explicit quality mode."""

    normalized_thesis = _normalize(thesis)

    if any(pattern.search(normalized_thesis) for pattern in _PROMPT_INJECTION_PATTERNS):
        return RouterDecision(
            intent=Intent.PROMPT_INJECTION,
            reason="deterministic prompt-injection match",
        )

    if _contains_arbitrary_url_request(normalized_thesis):
        return RouterDecision(
            intent=Intent.UNSAFE_SOURCE_REQUEST,
            reason="arbitrary source URL match",
        )

    del ticker
    return None


def contains_price_prediction(text: str, ticker: str) -> bool:
    """Detect requests or claims that predict a security's future value."""
    normalized = _normalize(text)
    ticker_pattern = re.escape(_normalize(ticker))
    contextual_patterns = (
        re.compile(
            r"\b(?:stocks?|shares?|securities?)\b.{0,24}"
            r"\b(?:will|would|could|may|can)\b.{0,24}"
            r"\b(?:trade|reach|hit|go|rise|fall|be)\b"
        ),
        re.compile(
            r"\b(?:will|would|could|may|can)\b.{0,32}"
            r"\b(?:stocks?|shares?|securities?)\b.{0,24}"
            r"\b(?:trade|reach|hit|go|rise|fall|be)\b"
        ),
        re.compile(
            rf"\b{ticker_pattern}\b.{{0,32}}\b(?:will|would|could|may|can)\b"
            r".{0,24}\b(?:trade|reach|hit|go|rise|fall)\b"
        ),
        re.compile(
            rf"\b{ticker_pattern}\b.{{0,32}}\b(?:will|would|could|may|can)\b"
            r".{0,16}\bbe\b.{0,20}(?:\$|\b\d+(?:\.\d+)?\s*(?:usd|dollars?)\b)"
            r".{0,32}(?:\bnext\b|\b20\d{2}\b|\byear[- ]end\b)"
        ),
        re.compile(
            r"\bwhere\s+do\s+you\s+see\b.{0,48}"
            r"\b(?:trad(?:e|ing)|go(?:ing)?|reach(?:ing)?)\b"
        ),
        re.compile(
            r"\bhow\s+(?:high|low)\s+(?:can|could|will|would|may)\b"
            r".{0,40}\b(?:go|trade|reach|rise|fall)\b"
        ),
        re.compile(r"\b(?:give|calculate|estimate|project|predict)\b.{0,48}\bfair\s+value\b"),
        re.compile(
            r"\bfair\s+value\b.{0,64}"
            r"(?:\$|\b20\d{2}\b|\bnext\b|\byear[- ]end\b)"
        ),
    )
    return any(pattern.search(normalized) for pattern in _PRICE_PREDICTION_PATTERNS) or any(
        pattern.search(normalized) for pattern in contextual_patterns
    )


def route_request(ticker: str, thesis: str) -> RouterDecision | None:
    """Return an immediate safety decision, or ``None`` for later model routing."""

    normalized_thesis = _normalize(thesis)
    normalized_ticker = _normalize(ticker)

    if any(pattern.search(normalized_thesis) for pattern in _PROMPT_INJECTION_PATTERNS):
        return RouterDecision(
            intent=Intent.PROMPT_INJECTION,
            reason="deterministic prompt-injection match",
        )

    if _contains_arbitrary_url_request(normalized_thesis):
        return RouterDecision(
            intent=Intent.UNSAFE_SOURCE_REQUEST,
            reason="arbitrary source URL match",
        )

    if contains_prohibited_investment_advice(normalized_thesis, normalized_ticker):
        return RouterDecision(
            intent=Intent.PROHIBITED_ADVICE,
            reason="deterministic investment-advice match",
        )

    ticker_pattern = re.escape(normalized_ticker)
    ticker_earnings_request = re.search(
        rf"(?:分析|解读).{{0,16}}{ticker_pattern}.{{0,16}}(?:最近一期|最新).{{0,16}}(?:财报|业绩)",
        normalized_thesis,
    )
    natural_earnings_request = re.search(
        r"(?:分析|解读).{0,16}(?:最近一期|最新).{0,16}(?:财报|业绩)",
        normalized_thesis,
    )
    if ticker_earnings_request or natural_earnings_request:
        return RouterDecision(
            intent=Intent.EARNINGS_REVIEW_REQUEST,
            reason="deterministic earnings-review match",
        )

    ticker_company_request = re.search(
        rf"(?:介绍|简介).{{0,8}}{ticker_pattern}\b",
        normalized_thesis,
    )
    natural_company_request = re.search(
        r"(?:介绍|简介)(?:一下)?(?:这家|该)?公司",
        normalized_thesis,
    )
    if ticker_company_request or natural_company_request:
        return RouterDecision(
            intent=Intent.COMPANY_PROFILE_REQUEST,
            reason="deterministic company-profile match",
        )

    return None
