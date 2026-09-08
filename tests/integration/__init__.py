"""Integration test package and shared composition helpers."""

from sqlalchemy import create_engine

from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.repositories import FilingRepository


def seed_supported_companies(database_url: str, *tickers: str) -> None:
    """Populate explicit local company metadata for accepted-path CLI tests."""
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    for index, ticker in enumerate(tickers, start=1):
        repository.upsert_company_metadata(
            ticker=ticker,
            cik=f"{index:010d}",
            legal_name=f"{ticker} Test Company",
            ir_domain=f"investor.{ticker.lower()}.example",
        )
