"""Canonical text views for deterministic privacy classification."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrivacyTextViews:
    joined: str
    separated: str


def privacy_text_views(text: str) -> PrivacyTextViews:
    """Return canonical joined and separated privacy views."""
    compatible = unicodedata.normalize("NFKC", text)
    joined = _collapse(
        "".join(
            character
            for character in compatible
            if unicodedata.category(character) != "Cf"
        )
    )
    separated = _collapse(
        "".join(
            " " if unicodedata.category(character) == "Cf" else character
            for character in compatible
        )
    )
    return PrivacyTextViews(joined=joined, separated=separated)


def _collapse(text: str) -> str:
    return " ".join(text.split())
