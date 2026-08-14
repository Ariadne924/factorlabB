from __future__ import annotations

import sqlite3

import pytest

from trading.ledger import TradingLedger
from trading.models import (
    FillRecord,
    MarketType,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSnapshot,
)


def intent() -> OrderIntent:
    return OrderIntent(
        client_order_id="trend-btc-001",
        strategy_id="trend-v1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
    )


def test_ledger_is_idempotent_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "trading.sqlite3"
    ledger = TradingLedger(path)
    assert ledger.create_order(intent()) is True
    assert ledger.create_order(intent()) is False
    ledger.update_order(
        "trend-btc-001",
        OrderStatus.SUBMITTED,
        exchange_order_id="991",
        raw_response={"status": "NEW"},
    )
    fill = FillRecord(
        exchange_trade_id="fill-1",
        client_order_id="trend-btc-001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        price=50_001.0,
        quantity=0.01,
        fee=0.1,
    )
    assert ledger.record_fill(fill) is True
    assert ledger.record_fill(fill) is False

    reopened = TradingLedger(path)
    order = reopened.get_order("trend-btc-001")
    assert order is not None
    assert order["status"] == "SUBMITTED"
    assert reopened.health()["counts"] == {
        "orders": 1,
        "fills": 1,
        "position_snapshots": 0,
        "events": 0,
    }


def test_ledger_records_position_and_event(tmp_path) -> None:
    ledger = TradingLedger(tmp_path / "trading.sqlite3")
    row_id = ledger.record_position(
        PositionSnapshot(
            symbol="ETHUSDT",
            market=MarketType.USD_M_FUTURES,
            quantity=-1.0,
            entry_price=3_000.0,
            mark_price=2_950.0,
            leverage=2.0,
            unrealized_pnl=50.0,
            observed_at="2026-08-14T00:00:00+00:00",
        )
    )
    event_id = ledger.append_event("risk_triggered", {"rule": "max_drawdown"})
    assert row_id == 1
    assert event_id == 1
    assert ledger.health()["status"] == "ok"


def test_fill_requires_existing_order(tmp_path) -> None:
    ledger = TradingLedger(tmp_path / "trading.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.record_fill(
            FillRecord(
                exchange_trade_id="missing-fill",
                client_order_id="missing-order",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                price=1.0,
                quantity=1.0,
            )
        )


def test_order_intent_rejects_unsafe_shapes() -> None:
    with pytest.raises(ValueError, match="limit order"):
        OrderIntent(
            client_order_id="x",
            strategy_id="s",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1,
        ).validate()
