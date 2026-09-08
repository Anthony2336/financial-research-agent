"""Namespace legacy filing and web source references.

Revision ID: 20260830_0004
Revises: 20260830_0003
Create Date: 2026-08-30 00:00:03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0004"
down_revision: str | Sequence[str] | None = "20260830_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    companies = sa.table("companies", sa.column("id"), sa.column("ticker"))
    filings = sa.table("filings", sa.column("id"), sa.column("company_id"))
    chunks = sa.table("chunks", sa.column("id"), sa.column("filing_id"))
    web_evidence = sa.table("web_evidence", sa.column("id"), sa.column("ticker"))
    research_runs = sa.table("research_runs", sa.column("id"), sa.column("ticker"))
    claims = sa.table(
        "claims",
        sa.column("id"),
        sa.column("run_id"),
        sa.column("evidence_chunk_ids", sa.JSON()),
        sa.column("source_refs", sa.JSON()),
    )
    skill_runs = sa.table(
        "skill_runs",
        sa.column("id"),
        sa.column("ticker"),
        sa.column("source_ids", sa.JSON()),
    )

    filing_keys = {
        (str(row.ticker).upper(), str(row.id))
        for row in connection.execute(
            sa.select(chunks.c.id, companies.c.ticker).select_from(
                chunks.join(filings, chunks.c.filing_id == filings.c.id).join(
                    companies, filings.c.company_id == companies.c.id
                )
            )
        )
    }
    web_keys = {
        (str(row.ticker).upper(), str(row.id))
        for row in connection.execute(sa.select(web_evidence.c.id, web_evidence.c.ticker))
    }

    claim_rows = connection.execute(
        sa.select(
            claims.c.id,
            research_runs.c.ticker,
            claims.c.evidence_chunk_ids,
            claims.c.source_refs,
        ).select_from(claims.join(research_runs, claims.c.run_id == research_runs.c.id))
    )
    for row in claim_rows:
        ticker = str(row.ticker).upper()
        converted = [
            _filing_or_unresolved(ticker, str(source_id), filing_keys)
            for source_id in _json_list(row.evidence_chunk_ids)
        ]
        converted.extend(
            _resolve_legacy(ticker, str(source_id), filing_keys, web_keys)
            for source_id in _json_list(row.source_refs)
        )
        connection.execute(
            sa.update(claims)
            .where(claims.c.id == row.id)
            .values(source_refs=_deduplicate(converted))
        )

    for row in connection.execute(
        sa.select(skill_runs.c.id, skill_runs.c.ticker, skill_runs.c.source_ids)
    ):
        ticker = str(row.ticker).upper()
        converted = [
            _resolve_legacy(ticker, str(source_id), filing_keys, web_keys)
            for source_id in _json_list(row.source_ids)
        ]
        connection.execute(
            sa.update(skill_runs)
            .where(skill_runs.c.id == row.id)
            .values(source_ids=_deduplicate(converted))
        )


def downgrade() -> None:
    # Resolution is intentionally not discarded: these strings remain valid JSON data
    # for revision 0003 readers, while stripping them would recreate the ambiguity.
    pass


def _resolve_legacy(
    ticker: str,
    value: str,
    filing_keys: set[tuple[str, str]],
    web_keys: set[tuple[str, str]],
) -> str:
    parts = value.split(":")
    if len(parts) == 3:
        encoded_ticker, kind, source_id = parts
        if encoded_ticker == ticker and kind == "unresolved":
            return value
        if encoded_ticker == ticker and kind == "filing" and (
            ticker,
            source_id,
        ) in filing_keys:
            return value
        if encoded_ticker == ticker and kind == "web" and (ticker, source_id) in web_keys:
            return value
        if kind in {"filing", "web", "unresolved"}:
            return _unresolved(ticker, source_id)

    is_filing = (ticker, value) in filing_keys
    is_web = (ticker, value) in web_keys
    if is_filing == is_web:
        return _unresolved(ticker, value)
    kind = "filing" if is_filing else "web"
    return f"{ticker}:{kind}:{value}"


def _filing_or_unresolved(
    ticker: str,
    source_id: str,
    filing_keys: set[tuple[str, str]],
) -> str:
    if (ticker, source_id) in filing_keys:
        return f"{ticker}:filing:{source_id}"
    return _unresolved(ticker, source_id)


def _unresolved(ticker: str, source_id: str) -> str:
    return f"{ticker}:unresolved:{source_id}"


def _json_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
