"""SQLite event ledger for restart-safe testnet and backtest state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading.models import FillRecord, OrderIntent, OrderStatus, PositionSnapshot

SCHEMA_VERSION = 1


class TradingLedger:
    """Small local ledger with idempotent order/fill keys and WAL durability."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1_000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    reduce_only INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    exchange_order_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    raw_response TEXT
                );
                CREATE TABLE IF NOT EXISTS fills (
                    exchange_trade_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    fee REAL NOT NULL,
                    fee_asset TEXT,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY(client_order_id) REFERENCES orders(client_order_id)
                );
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL,
                    mark_price REAL,
                    leverage REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    observed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(client_order_id);
                CREATE INDEX IF NOT EXISTS idx_positions_symbol
                    ON position_snapshots(symbol, observed_at);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create_order(self, intent: OrderIntent) -> bool:
        """Insert an intent once; False means the idempotency key already exists."""
        intent.validate()
        now = self._now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO orders (
                    client_order_id, strategy_id, environment, market, symbol, side,
                    order_type, quantity, price, reduce_only, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.client_order_id,
                    intent.strategy_id,
                    intent.environment.value,
                    intent.market.value,
                    intent.symbol,
                    intent.side.value,
                    intent.order_type.value,
                    intent.quantity,
                    intent.price,
                    int(intent.reduce_only),
                    OrderStatus.CREATED.value,
                    now,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def update_order(
        self,
        client_order_id: str,
        status: OrderStatus,
        *,
        exchange_order_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE orders
                SET status = ?, exchange_order_id = COALESCE(?, exchange_order_id),
                    raw_response = COALESCE(?, raw_response), updated_at = ?
                WHERE client_order_id = ?
                """,
                (
                    status.value,
                    exchange_order_id,
                    json.dumps(raw_response, ensure_ascii=False) if raw_response else None,
                    self._now(),
                    client_order_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown client_order_id: {client_order_id}")

    def record_fill(self, fill: FillRecord) -> bool:
        fill.validate()
        executed_at = fill.executed_at or self._now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO fills (
                    exchange_trade_id, client_order_id, symbol, side, price,
                    quantity, fee, fee_asset, executed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.exchange_trade_id,
                    fill.client_order_id,
                    fill.symbol,
                    fill.side.value,
                    fill.price,
                    fill.quantity,
                    fill.fee,
                    fill.fee_asset,
                    executed_at,
                ),
            )
            return cursor.rowcount == 1

    def record_position(self, snapshot: PositionSnapshot) -> int:
        snapshot.validate()
        values = asdict(snapshot)
        values["market"] = snapshot.market.value
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO position_snapshots (
                    symbol, market, quantity, entry_price, mark_price,
                    leverage, unrealized_pnl, observed_at
                ) VALUES (:symbol, :market, :quantity, :entry_price, :mark_price,
                          :leverage, :unrealized_pnl, :observed_at)
                """,
                values,
            )
            if cursor.lastrowid is None:
                raise RuntimeError("position insert did not return a row id")
            return int(cursor.lastrowid)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> int:
        if not event_type.strip():
            raise ValueError("event_type cannot be empty")
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO events(event_type, payload, created_at) VALUES (?, ?, ?)",
                (event_type, json.dumps(payload, ensure_ascii=False), self._now()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("event insert did not return a row id")
            return int(cursor.lastrowid)

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_orders(self, *, strategy_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM orders"
        params: tuple[str, ...] = ()
        if strategy_id is not None:
            query += " WHERE strategy_id = ?"
            params = (strategy_id,)
        query += " ORDER BY created_at, client_order_id"
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def health(self) -> dict[str, Any]:
        with self._connection() as connection:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            counts = {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("orders", "fills", "position_snapshots", "events")
            }
        return {
            "status": "ok" if integrity == "ok" else "error",
            "integrity": integrity,
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "path": str(self.path),
        }
