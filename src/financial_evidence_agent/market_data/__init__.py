"""Normalized, provider-independent market-data contracts."""

from financial_evidence_agent.market_data.alpaca import AlpacaMarketDataProvider
from financial_evidence_agent.market_data.gateway import (
    MarketCache,
    MarketDataGateway,
    MarketFetchWrite,
    MarketFetchWriter,
    NoopMarketCache,
    NoopMarketFetchWriter,
)
from financial_evidence_agent.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketDataError,
    MarketDataErrorCode,
    MarketSnapshot,
    MarketStatus,
)
from financial_evidence_agent.market_data.providers import MarketDataProvider

__all__ = [
    "AlpacaMarketDataProvider",
    "MarketBar",
    "MarketCache",
    "MarketDataBundle",
    "MarketDataError",
    "MarketDataErrorCode",
    "MarketDataGateway",
    "MarketDataProvider",
    "MarketFetchWrite",
    "MarketFetchWriter",
    "MarketSnapshot",
    "MarketStatus",
    "NoopMarketCache",
    "NoopMarketFetchWriter",
]
