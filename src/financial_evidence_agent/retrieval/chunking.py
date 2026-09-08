"""Section-aware filing parsing and source-span-preserving chunking."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

TOKEN_PATTERN = re.compile(r"\S+")
TARGET_TOKENS = 500
MIN_TOKENS = 350
OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class ParsedSection:
    """One supported filing section within the persisted normalized source text."""

    name: str
    raw_start: int
    raw_end: int


@dataclass(frozen=True)
class ParsedFiling:
    """Normalized text and source ranges for sections allowed into the corpus."""

    raw_text: str
    sections: list[ParsedSection]


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk with offsets into ``ParsedFiling.raw_text``."""

    section: str
    content: str
    token_count: int
    raw_start: int
    raw_end: int


def parse_supported_sections(html: str) -> ParsedFiling:
    """Keep only the four explicitly supported filing section boundaries."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    parts: list[str] = []
    sections: list[ParsedSection] = []
    current_name: str | None = None
    current_start = 0
    offset = 0

    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        line = _normalized_text(element)
        if not line:
            continue
        line_start = offset
        parts.append(line)
        offset += len(line) + 1

        heading = _section_name(element, line)
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            if current_name is not None and current_start < line_start:
                sections.append(ParsedSection(current_name, current_start, line_start - 1))
            current_name = heading
            current_start = offset

    if current_name is not None and current_start < offset:
        sections.append(ParsedSection(current_name, current_start, offset - 1))
    return ParsedFiling(raw_text="\n".join(parts), sections=sections)


def chunk_sections(parsed: ParsedFiling) -> list[ChunkSpan]:
    """Split supported sections into 350--500-token spans with 80-token overlap."""
    chunks: list[ChunkSpan] = []
    for section in parsed.sections:
        tokens = list(TOKEN_PATTERN.finditer(parsed.raw_text, section.raw_start, section.raw_end))
        if not tokens:
            continue
        start_index = 0
        for chunk_size in _chunk_token_lengths(len(tokens)):
            end_index = start_index + chunk_size
            raw_start = tokens[start_index].start()
            raw_end = tokens[end_index - 1].end()
            chunks.append(
                ChunkSpan(
                    section=section.name,
                    content=parsed.raw_text[raw_start:raw_end],
                    token_count=end_index - start_index,
                    raw_start=raw_start,
                    raw_end=raw_end,
                )
            )
            start_index = end_index - OVERLAP_TOKENS
    return chunks


def _chunk_token_lengths(token_count: int) -> list[int]:
    if token_count <= TARGET_TOKENS:
        return [token_count]
    unique_step = TARGET_TOKENS - OVERLAP_TOKENS
    chunk_count = (token_count - TARGET_TOKENS + unique_step - 1) // unique_step + 1
    appearances = token_count + (chunk_count - 1) * OVERLAP_TOKENS
    if appearances < chunk_count * MIN_TOKENS:
        return [MIN_TOKENS] * (chunk_count - 1) + [
            appearances - (chunk_count - 1) * MIN_TOKENS
        ]
    lengths = [MIN_TOKENS] * chunk_count
    remaining = appearances - chunk_count * MIN_TOKENS
    for index in range(chunk_count):
        added = min(TARGET_TOKENS - MIN_TOKENS, remaining)
        lengths[index] += added
        remaining -= added
    return lengths


def _normalized_text(element: Tag) -> str:
    return " ".join(element.get_text(" ", strip=True).split())


def _section_name(element: Tag, text: str) -> str | None:
    if element.name not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return None
    normalized = text.casefold()
    if normalized == "md&a" or "management's discussion and analysis" in normalized:
        return "MD&A"
    if normalized == "risk factors":
        return "Risk Factors"
    if "financial statements" in normalized:
        return "Financial Statements"
    if normalized == "earnings release":
        return "Earnings Release"
    return None
