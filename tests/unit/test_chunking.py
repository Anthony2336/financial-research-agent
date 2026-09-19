import pytest

from fra.retrieval.chunking import chunk_sections, parse_supported_sections


@pytest.mark.parametrize(
    ("heading", "expected_section"),
    [
        ("MD&A", "MD&A"),
        ("Management's Discussion and Analysis", "MD&A"),
        ("Risk Factors", "Risk Factors"),
        ("Financial Statements", "Financial Statements"),
        ("Earnings Release", "Earnings Release"),
    ],
)
def test_supported_heading_aliases_are_parsed(heading: str, expected_section: str) -> None:
    """A supported SEC heading must make its following text available for citation."""
    parsed = parse_supported_sections(f"<h2>{heading}</h2><p>ALLOWED_MARKER</p>")

    assert [
        (section.name, parsed.raw_text[section.raw_start : section.raw_end])
        for section in parsed.sections
    ] == [(expected_section, "ALLOWED_MARKER")]


def test_unsupported_heading_closes_the_active_supported_section() -> None:
    """Unrecognized disclosure text must not inherit the preceding supported section label."""
    parsed = parse_supported_sections(
        "<h2>MD&A</h2><p>ALLOWED_MARKER</p>"
        "<h2>Other Disclosure</h2><p>UNSUPPORTED_MARKER</p>"
        "<h2>Risk Factors</h2><p>RISK_MARKER</p>"
    )

    allowed_content = [
        parsed.raw_text[section.raw_start : section.raw_end] for section in parsed.sections
    ]

    assert [section.name for section in parsed.sections] == ["MD&A", "Risk Factors"]
    assert allowed_content == ["ALLOWED_MARKER", "RISK_MARKER"]
    assert "UNSUPPORTED_MARKER" not in "\n".join(allowed_content)


def test_earnings_release_chunks_keep_exact_raw_spans_and_eighty_token_overlap() -> None:
    tokens = [f"token-{index}" for index in range(920)]
    parsed = parse_supported_sections(
        f"<h2>Earnings Release</h2><p>{' '.join(tokens)}</p>"
    )

    chunks = chunk_sections(parsed)

    assert [chunk.section for chunk in chunks] == [
        "Earnings Release",
        "Earnings Release",
    ]
    assert all(350 <= chunk.token_count <= 500 for chunk in chunks)
    assert all(
        parsed.raw_text[chunk.raw_start : chunk.raw_end] == chunk.content
        for chunk in chunks
    )
    assert chunks[0].content.split()[-80:] == chunks[1].content.split()[:80]


@pytest.mark.parametrize("token_total", [500, 501, 550, 619, 620, 920])
def test_chunk_sizes_cover_source_once_with_exact_eighty_token_adjacent_overlap(
    token_total: int,
) -> None:
    tokens = [f"token-{index}" for index in range(token_total)]
    parsed = parse_supported_sections(
        f"<h2>Earnings Release</h2><p>{' '.join(tokens)}</p>"
    )

    chunks = chunk_sections(parsed)
    chunk_tokens = [chunk.content.split() for chunk in chunks]

    assert all(
        chunk.token_count == len(values) <= 500
        for chunk, values in zip(chunks, chunk_tokens, strict=True)
    )
    assert all(chunk.token_count >= 350 for chunk in chunks[:-1])
    if token_total >= 620:
        assert chunks[-1].token_count >= 350
    assert chunk_tokens[0][0] == "token-0"
    assert chunk_tokens[-1][-1] == f"token-{token_total - 1}"
    for previous, current in zip(chunk_tokens, chunk_tokens[1:], strict=False):
        assert previous[-80:] == current[:80]
        previous_end = int(previous[-1].split("-")[1])
        current_start = int(current[0].split("-")[1])
        assert current_start == previous_end - 79
    assert all(
        parsed.raw_text[chunk.raw_start : chunk.raw_end] == chunk.content
        for chunk in chunks
    )
