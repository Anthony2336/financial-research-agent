"""Opt-in live Alpaca Basic smoke tests for the bounded IEX-only adapter."""

from __future__ import annotations

import os

import httpx
import pytest

from financial_evidence_agent.market_data.alpaca import AlpacaMarketDataProvider
from financial_evidence_agent.market_data.models import MarketStatus


def _require_live_values(*names: str) -> list[str]:
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail(
            "live Alpaca smoke requires environment variables: "
            + ", ".join(sorted(missing))
        )
    return [os.environ[name].strip() for name in names]


@pytest.mark.live_provider
@pytest.mark.asyncio
async def test_real_alpaca_snapshot_and_bars_remain_iex_only() -> None:
    key_id, secret_key = _require_live_values(
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
    )
    trading_environment = os.environ.get("ALPACA_TRADING_ENVIRONMENT", "paper").strip() or "paper"

    async with httpx.AsyncClient() as client:
        provider = AlpacaMarketDataProvider(
            key_id=key_id,
            secret_key=secret_key,
            client=client,
            trading_environment=trading_environment,  # type: ignore[arg-type]
        )
        snapshot = await provider.get_snapshot("NVDA")
        bars = await provider.get_bars("NVDA", interval="1Day", limit=2)

    assert snapshot.provider == "alpaca"
    assert snapshot.feed == "iex"
    assert snapshot.coverage == "IEX-only"
    assert snapshot.symbol == "NVDA"
    assert snapshot.exchange
    assert snapshot.currency == "USD"
    assert snapshot.as_of.tzinfo is not None
    assert snapshot.fetched_at.tzinfo is not None
    assert snapshot.market_status in {
        MarketStatus.OPEN,
        MarketStatus.CLOSED,
        MarketStatus.UNKNOWN,
    }
    assert bars
    assert len(bars) <= 2
    assert [bar.timestamp for bar in bars] == sorted(bar.timestamp for bar in bars)
    assert all(bar.provider == "alpaca" for bar in bars)
    assert all(bar.feed == "iex" for bar in bars)
    assert all(bar.coverage == "IEX-only" for bar in bars)
    assert all(bar.symbol == "NVDA" for bar in bars)
