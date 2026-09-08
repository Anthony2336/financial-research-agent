"""One deterministic privacy contract for persistable report output."""

from collections.abc import Iterable

from financial_evidence_agent.memory.privacy import contains_private_financial_or_secret

PRIVATE_TEXT_REDACTION = "Removed by privacy guard."
PRIVATE_DETAIL_CODE = "PRIVATE_OUTPUT_DETAIL_REDACTED"
_PRIVATE_REFERENCE_LABEL = "[redacted]"


def redact_private_text(text: str, *, ticker: str) -> tuple[str, bool]:
    """Replace private narrative with fixed text without echoing the input."""
    if contains_private_financial_or_secret(text, current_ticker=ticker):
        return PRIVATE_TEXT_REDACTION, True
    return text, False


def retain_public_text(text: str, *, ticker: str) -> bool:
    """Return whether text may cross a persistable report boundary unchanged."""
    return not contains_private_financial_or_secret(text, current_ticker=ticker)


def safe_reference_label(value: str, *, ticker: str) -> str:
    """Return a fixed label when a persistable reference contains private text."""
    return value if retain_public_text(value, ticker=ticker) else _PRIVATE_REFERENCE_LABEL


def collapse_private_detail_errors(
    errors: Iterable[str],
) -> list[str]:
    """Deduplicate diagnostics after trust boundaries have made them code-only."""
    return list(
        dict.fromkeys(
            PRIVATE_DETAIL_CODE
            if PRIVATE_DETAIL_CODE in error
            else error
            for error in errors
        )
    )
