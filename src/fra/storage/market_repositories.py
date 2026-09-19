"""Exact, immutable persistence mapping for normalized market data."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from fra.market_data.models import (
    MarketBar,
    MarketDataBundle,
    MarketSnapshot,
)
from fra.storage.models import (
    MarketBarRecord,
    MarketBundleRecord,
    MarketSnapshotRecord,
)


class MarketDataRepository:
    """Persist and resolve market source IDs without crossing provider scope."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        return self._engine

    def save_bundle(self, bundle: MarketDataBundle) -> None:
        """Store one validated bundle as immutable source records."""
        with Session(self._engine) as session, session.begin():
            self._save_snapshot(
                session,
                bundle.snapshot,
                status=bundle.status,
                freshness_label=bundle.freshness_label,
                errors=bundle.errors,
            )
            for bar in bundle.bars:
                self._save_bar(session, bar)
            self._save_bundle_record(session, bundle)

    def save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Persist a snapshot fetched independently by the snapshot MCP tool."""
        with Session(self._engine) as session, session.begin():
            self._save_snapshot(
                session,
                snapshot,
                status="partial",
                freshness_label=_freshness_label(snapshot),
                errors=[],
            )

    def save_bars(self, bars: list[MarketBar]) -> None:
        """Persist independently fetched bars without inventing a snapshot relation."""
        with Session(self._engine) as session, session.begin():
            for bar in bars:
                self._save_bar(session, bar)

    def latest_bundle(
        self, provider: str, feed: str, symbol: str
    ) -> MarketDataBundle | None:
        """Return the latest exact-scope snapshot with chronological exact-scope bars."""
        with Session(self._engine) as session:
            bundle_record = session.scalar(
                select(MarketBundleRecord)
                .where(
                    MarketBundleRecord.provider == provider,
                    MarketBundleRecord.feed == feed,
                    MarketBundleRecord.symbol == symbol,
                )
                .order_by(
                    MarketBundleRecord.observation_at.desc(),
                    MarketBundleRecord.snapshot_id.desc(),
                    MarketBundleRecord.id.desc(),
                )
                .limit(1)
            )
            if bundle_record is not None:
                return self._bundle_model(session, bundle_record)
            snapshot_record = session.scalar(
                select(MarketSnapshotRecord)
                .where(
                    MarketSnapshotRecord.provider == provider,
                    MarketSnapshotRecord.feed == feed,
                    MarketSnapshotRecord.symbol == symbol,
                )
                .order_by(
                    MarketSnapshotRecord.fetched_at.desc(),
                    MarketSnapshotRecord.as_of.desc(),
                    MarketSnapshotRecord.id.desc(),
                )
                .limit(1)
            )
            if snapshot_record is None:
                return None
            snapshot = _snapshot_model(snapshot_record)
            latest_bar_fetch = select(func.max(MarketBarRecord.fetched_at)).where(
                MarketBarRecord.provider == provider,
                MarketBarRecord.feed == feed,
                MarketBarRecord.symbol == symbol,
                MarketBarRecord.exchange == snapshot.exchange,
                MarketBarRecord.currency == snapshot.currency,
                MarketBarRecord.coverage == snapshot.coverage,
            )
            bar_records = session.scalars(
                select(MarketBarRecord)
                .where(
                    MarketBarRecord.provider == provider,
                    MarketBarRecord.feed == feed,
                    MarketBarRecord.symbol == symbol,
                    MarketBarRecord.exchange == snapshot.exchange,
                    MarketBarRecord.currency == snapshot.currency,
                    MarketBarRecord.coverage == snapshot.coverage,
                    MarketBarRecord.fetched_at == latest_bar_fetch.scalar_subquery(),
                )
                .order_by(MarketBarRecord.timestamp.asc(), MarketBarRecord.id.desc())
                .limit(20)
            ).all()
            seen_timestamps: set[datetime] = set()
            deduped_bar_records: list[MarketBarRecord] = []
            for record in bar_records:
                if record.timestamp in seen_timestamps:
                    continue
                seen_timestamps.add(record.timestamp)
                deduped_bar_records.append(record)
            return MarketDataBundle(
                snapshot=snapshot,
                bars=[_bar_model(record) for record in deduped_bar_records],
                status=snapshot_record.bundle_status,
                freshness_label=snapshot_record.freshness_label,
                errors=list(snapshot_record.errors or []),
            )

    def get_snapshot(self, snapshot_id: str) -> MarketSnapshot | None:
        """Resolve only a snapshot ID from the snapshot namespace."""
        with Session(self._engine) as session:
            record = session.get(MarketSnapshotRecord, snapshot_id)
            return None if record is None else _snapshot_model(record)

    def get_bar(self, bar_id: str) -> MarketBar | None:
        """Resolve only a bar ID from the bar namespace."""
        with Session(self._engine) as session:
            record = session.get(MarketBarRecord, bar_id)
            return None if record is None else _bar_model(record)

    def _save_snapshot(
        self,
        session: Session,
        snapshot: MarketSnapshot,
        *,
        status: Literal["completed", "partial"],
        freshness_label: Literal[
            "open-iex", "latest-available-iex", "market-status-unknown"
        ],
        errors: list[str],
    ) -> None:
        existing = session.get(MarketSnapshotRecord, snapshot.id)
        if existing is not None:
            if not _same_snapshot_identity(existing, snapshot):
                raise ValueError("immutable market snapshot id reused for different payload")
            return
        values = snapshot.model_dump()
        values["market_status"] = snapshot.market_status.value
        session.add(
            MarketSnapshotRecord(
                **values,
                bundle_status=status,
                freshness_label=freshness_label,
                errors=list(errors),
            )
        )

    def _save_bar(self, session: Session, bar: MarketBar) -> None:
        existing = session.get(MarketBarRecord, bar.id)
        if existing is not None:
            if not _same_bar_identity(existing, bar):
                raise ValueError("immutable market bar id reused for different payload")
            return
        session.add(MarketBarRecord(**bar.model_dump()))

    def _save_bundle_record(self, session: Session, bundle: MarketDataBundle) -> None:
        identity = _bundle_identity(bundle)
        if session.get(MarketBundleRecord, identity) is not None:
            return
        session.add(
            MarketBundleRecord(
                id=identity,
                provider=bundle.snapshot.provider,
                feed=bundle.snapshot.feed,
                symbol=bundle.snapshot.symbol,
                snapshot_id=bundle.snapshot.id,
                bar_ids=[bar.id for bar in bundle.bars],
                status=bundle.status,
                freshness_label=bundle.freshness_label,
                errors=list(bundle.errors),
                observation_at=bundle.snapshot.fetched_at,
                created_at=datetime.now(UTC),
            )
        )

    def _bundle_model(
        self, session: Session, record: MarketBundleRecord
    ) -> MarketDataBundle:
        snapshot_record = session.get(MarketSnapshotRecord, record.snapshot_id)
        if snapshot_record is None:
            raise ValueError("market bundle snapshot is missing")
        bar_records = [session.get(MarketBarRecord, bar_id) for bar_id in record.bar_ids]
        if any(bar is None for bar in bar_records):
            raise ValueError("market bundle bar is missing")
        return MarketDataBundle(
            snapshot=_snapshot_model(snapshot_record),
            bars=[_bar_model(bar) for bar in bar_records if bar is not None],
            status=record.status,
            freshness_label=record.freshness_label,
            errors=list(record.errors or []),
        )


def _snapshot_model(record: MarketSnapshotRecord) -> MarketSnapshot:
    return MarketSnapshot(
        id=record.id,
        provider=record.provider,
        feed=record.feed,
        coverage=record.coverage,
        symbol=record.symbol,
        exchange=record.exchange,
        currency=record.currency,
        price=record.price,
        open=record.open,
        day_high=record.day_high,
        day_low=record.day_low,
        previous_close=record.previous_close,
        as_of=_utc(record.as_of),
        fetched_at=_utc(record.fetched_at),
        market_status=record.market_status,
        delayed_by_seconds=record.delayed_by_seconds,
        raw_payload_hash=record.raw_payload_hash,
    )


def _bar_model(record: MarketBarRecord) -> MarketBar:
    return MarketBar(
        id=record.id,
        provider=record.provider,
        feed=record.feed,
        coverage=record.coverage,
        symbol=record.symbol,
        exchange=record.exchange,
        currency=record.currency,
        interval=record.interval,
        timestamp=_utc(record.timestamp),
        open=record.open,
        high=record.high,
        low=record.low,
        close=record.close,
        volume=record.volume,
        fetched_at=_utc(record.fetched_at),
        raw_payload_hash=record.raw_payload_hash,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _freshness_label(
    snapshot: MarketSnapshot,
) -> Literal["open-iex", "latest-available-iex", "market-status-unknown"]:
    if snapshot.market_status.value == "open":
        return "open-iex"
    if snapshot.market_status.value == "closed":
        return "latest-available-iex"
    return "market-status-unknown"


def _bundle_identity(bundle: MarketDataBundle) -> str:
    payload = json.dumps(
        {
            "snapshot_id": bundle.snapshot.id,
            "bar_ids": [bar.id for bar in bundle.bars],
            "status": bundle.status,
            "freshness_label": bundle.freshness_label,
            "errors": bundle.errors,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _same_snapshot_identity(
    record: MarketSnapshotRecord,
    snapshot: MarketSnapshot,
) -> bool:
    return (
        record.id == snapshot.id
        and record.provider == snapshot.provider
        and record.feed == snapshot.feed
        and record.coverage == snapshot.coverage
        and record.symbol == snapshot.symbol
        and record.exchange == snapshot.exchange
        and record.currency == snapshot.currency
        and record.price == snapshot.price
        and record.open == snapshot.open
        and record.day_high == snapshot.day_high
        and record.day_low == snapshot.day_low
        and record.previous_close == snapshot.previous_close
        and _utc(record.as_of) == snapshot.as_of
        and _utc(record.fetched_at) == snapshot.fetched_at
        and record.market_status == snapshot.market_status.value
        and record.delayed_by_seconds == snapshot.delayed_by_seconds
        and record.raw_payload_hash == snapshot.raw_payload_hash
    )


def _same_bar_identity(record: MarketBarRecord, bar: MarketBar) -> bool:
    return (
        record.id == bar.id
        and record.provider == bar.provider
        and record.feed == bar.feed
        and record.coverage == bar.coverage
        and record.symbol == bar.symbol
        and record.exchange == bar.exchange
        and record.currency == bar.currency
        and record.interval == bar.interval
        and _utc(record.timestamp) == bar.timestamp
        and record.open == bar.open
        and record.high == bar.high
        and record.low == bar.low
        and record.close == bar.close
        and record.volume == bar.volume
        and _utc(record.fetched_at) == bar.fetched_at
        and record.raw_payload_hash == bar.raw_payload_hash
    )
