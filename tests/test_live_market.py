from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data.live_market as live_market
from data.live_market import (
    DEFAULT_LIVE_SYMBOLS,
    BinanceLiveCollector,
    FanInWebSocketStream,
    FinalizedKlineStore,
    LiveMarketState,
    combined_stream_names,
    live_snapshot_status,
    load_live_snapshot,
)


def test_combined_stream_names_contains_public_feeds() -> None:
    names = combined_stream_names(("BTCUSDT",), "1m")
    assert names == (
        "btcusdt@aggTrade/btcusdt@bookTicker/btcusdt@depth5@500ms/"
        "btcusdt@kline_1m/btcusdt@markPrice@1s"
    )


def test_default_live_universe_matches_six_core_research_assets() -> None:
    assert DEFAULT_LIVE_SYMBOLS == (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    )
    assert live_market.spot_book_stream_names(("BTCUSDT",)) == "btcusdt@bookTicker"


def test_fan_in_tags_source_market() -> None:
    spot = FakeStream([{"e": "spot-event"}])
    futures = FakeStream([{"e": "futures-event"}])
    combined = FanInWebSocketStream(  # type: ignore[arg-type]
        {"spot": spot, "futures": futures}
    )
    messages = list(combined.stream())
    assert {message["_source_market"] for message in messages} == {"spot", "futures"}
    combined.close()
    assert spot.closed and futures.closed


def test_live_state_normalizes_trade_book_derivatives_and_closed_kline() -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    state = LiveMarketState(("BTCUSDT",), clock=lambda: now)
    state.apply(
        {
            "data": {
                "e": "aggTrade",
                "E": 1_765_000_000_000,
                "T": 1_765_000_000_000,
                "s": "BTCUSDT",
                "a": 7,
                "p": "100.5",
                "q": "2",
                "m": False,
            }
        }
    )
    state.apply(
        {
            "e": "bookTicker",
            "E": 1_765_000_000_001,
            "s": "BTCUSDT",
            "b": "100",
            "B": "3",
            "a": "101",
            "A": "4",
        }
    )
    state.apply(
        {
            "e": "markPriceUpdate",
            "E": 1_765_000_000_002,
            "s": "BTCUSDT",
            "p": "102",
            "i": "100",
            "r": "0.0001",
            "T": 1_765_010_000_000,
        }
    )
    closed = state.apply(
        {
            "e": "kline",
            "E": 1_765_000_000_003,
            "s": "BTCUSDT",
            "k": {
                "t": 1_765_000_000_000,
                "T": 1_765_000_059_999,
                "i": "1m",
                "o": "100",
                "h": "103",
                "l": "99",
                "c": "102",
                "v": "20",
                "q": "2020",
                "n": 10,
                "x": True,
                "V": "12",
                "Q": "1212",
            },
        }
    )

    symbol = state.snapshot["symbols"]["BTCUSDT"]
    assert symbol["last_trade"]["side"] == "buy"
    assert symbol["book"]["spread_bps"] == pytest.approx(10_000 / 100.5)
    assert symbol["derivatives"]["basis"] == pytest.approx(0.02)
    assert closed is not None
    assert closed["number_of_trades"] == 10
    assert state.snapshot["message_count"] == 4


def test_book_ticker_event_can_be_inferred_from_combined_stream_name() -> None:
    state = LiveMarketState(("BTCUSDT",))
    state.apply(
        {
            "stream": "btcusdt@bookTicker",
            "data": {
                "s": "BTCUSDT",
                "b": "100",
                "B": "2",
                "a": "101",
                "A": "3",
            },
        }
    )
    assert state.snapshot["symbols"]["BTCUSDT"]["book"]["bid_price"] == 100
    assert state.snapshot["event_counts"]["bookTicker"] == 1


def test_snapshot_status_detects_stale_and_invalid_payload() -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    snapshot = {"status": "live", "updated_at": (now - timedelta(seconds=12)).isoformat()}
    assert live_snapshot_status(snapshot, now=now, stale_after_seconds=10) == ("stale", 12.0)
    assert live_snapshot_status({"updated_at": "bad"}, now=now) == ("offline", None)


class FakeStream:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.closed = False

    def stream(self):  # type: ignore[no-untyped-def]
        yield from self.messages

    def close(self) -> None:
        self.closed = True


class FakeStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> Path:
        self.records.append(record)
        return Path("unused")


def _closed_kline_message() -> dict[str, Any]:
    return {
        "e": "kline",
        "E": 1_765_000_000_003,
        "s": "BTCUSDT",
        "k": {
            "t": 1_765_000_000_000,
            "T": 1_765_000_059_999,
            "i": "1m",
            "o": "100",
            "h": "103",
            "l": "99",
            "c": "102",
            "v": "20",
            "q": "2020",
            "n": 10,
            "x": True,
            "V": "12",
            "Q": "1212",
        },
    }


def test_collector_writes_snapshot_and_only_persists_closed_bar(tmp_path: Path) -> None:
    stream = FakeStream([_closed_kline_message()])
    store = FakeStore()
    snapshot_path = tmp_path / "live.json"
    collector = BinanceLiveCollector(
        symbols=("BTCUSDT",),
        snapshot_path=snapshot_path,
        stream=stream,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
    )
    assert collector.run(max_messages=1) == 1
    assert stream.closed
    assert len(store.records) == 1
    assert load_live_snapshot(snapshot_path)["last_persisted_bar"]["symbol"] == "BTCUSDT"


def test_finalized_store_is_idempotent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "klines.parquet"
    monkeypatch.setattr(live_market, "silver_klines_path", lambda **kwargs: path)
    store = FinalizedKlineStore()
    record = _closed_kline_message()["k"]
    normalized = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "open_time": record["t"],
        "close_time": record["T"],
        "open": record["o"],
        "high": record["h"],
        "low": record["l"],
        "close": record["c"],
        "volume": record["v"],
        "quote_asset_volume": record["q"],
        "number_of_trades": record["n"],
        "taker_buy_base_volume": record["V"],
        "taker_buy_quote_volume": record["Q"],
    }
    store.append(normalized)
    store.append(normalized)
    frame = pd.read_parquet(path)
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == 102


def test_load_snapshot_tolerates_partial_json(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"status": "live"}), encoding="utf-8")
    assert load_live_snapshot(path) == {"status": "live"}
    path.write_text("{", encoding="utf-8")
    assert load_live_snapshot(path) == {}
