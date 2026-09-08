"""Provider protocol for normalized market-data retrieval."""

from typing import Literal, Protocol

from financial_evidence_agent.market_data.models import MarketBar, MarketSnapshot


class MarketDataProvider(Protocol):
    """Fetch normalized market values without exposing provider payloads upstream."""

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return one latest snapshot for an uppercase ticker symbol."""

    async def get_bars(
        self,
        symbol: str,
        *,
        interval: Literal["1Day"],
        limit: int,
    ) -> list[MarketBar]:
        """Return bounded chronological bars for an uppercase ticker symbol."""
