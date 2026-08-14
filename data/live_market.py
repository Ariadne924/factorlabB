"""Binance 公开 WebSocket 行情的轻量快照与已收盘 K 线落盘。"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.binance_client import BinanceClient
from data.binance_websocket import BinanceWebSocketStream
from data.market_alerts import evaluate_market_alerts
from data.schema import KlineRaw, bronze_to_silver, raw_to_dataframe
from data.silver import silver_klines_path
from utils.io_utils import merge_parquet_safe

LIVE_SNAPSHOT_VERSION = "1.0"
DEFAULT_LIVE_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
)


def _iso_from_ms(value: int | str | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat()


def _float(value: Any) -> float:
    return float(value)


def combined_stream_names(symbols: tuple[str, ...], interval: str = "1m") -> str:
    """构造 USD-M Futures combined stream 名称。"""
    streams: list[str] = []
    for symbol in symbols:
        normalized = symbol.strip().lower()
        if not normalized:
            raise ValueError("实时交易对不能为空")
        streams.extend(
            [
                f"{normalized}@aggTrade",
                f"{normalized}@bookTicker",
                f"{normalized}@depth5@500ms",
                f"{normalized}@kline_{interval}",
                f"{normalized}@markPrice@1s",
            ]
        )
    return "/".join(streams)


def spot_stream_names(symbols: tuple[str, ...], interval: str = "1m") -> str:
    """现货成交与 K 线在部分网络环境下比 Futures 同名流更稳定。"""
    return "/".join(
        stream
        for symbol in symbols
        for stream in (
            f"{symbol.lower()}@aggTrade",
            f"{symbol.lower()}@kline_{interval}",
        )
    )


def spot_book_stream_names(symbols: tuple[str, ...]) -> str:
    """单独连接现货 best bid/ask，避免盘口高频更新阻塞成交/K 线子流。"""
    return "/".join(f"{symbol.lower()}@bookTicker" for symbol in symbols)


def futures_stream_names(symbols: tuple[str, ...]) -> str:
    """USD-M 盘口与永续衍生指标流。"""
    return "/".join(
        stream
        for symbol in symbols
        for stream in (
            f"{symbol.lower()}@depth5@500ms",
            f"{symbol.lower()}@markPrice@1s",
        )
    )


class FanInWebSocketStream:
    """将现货与 USD-M 两条公开连接汇入一个同步消息迭代器。"""

    def __init__(self, streams: dict[str, BinanceWebSocketStream]) -> None:
        self.streams = streams
        self.url = " | ".join(
            f"{name}: {getattr(stream, 'url', 'injected')}"
            for name, stream in streams.items()
        )
        self._closed = False
        self._queue: queue.Queue[dict[str, Any] | BaseException | tuple[str, str]] = (
            queue.Queue()
        )

    def _consume(self, market: str, source: BinanceWebSocketStream) -> None:
        try:
            for message in source.stream():
                tagged = dict(message)
                tagged["_source_market"] = market
                self._queue.put(tagged)
        except BaseException as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(("done", market))

    def stream(self) -> Iterator[dict[str, Any]]:
        active = set(self.streams)
        for market, source in self.streams.items():
            threading.Thread(
                target=self._consume,
                args=(market, source),
                daemon=True,
                name=f"binance-live-{market}",
            ).start()
        while active and not self._closed:
            item = self._queue.get()
            if isinstance(item, BaseException):
                raise ConnectionError("实时子流异常退出") from item
            if isinstance(item, tuple):
                active.discard(item[1])
                continue
            yield item

    def close(self) -> None:
        self._closed = True
        for source in self.streams.values():
            source.close()


class LiveMarketState:
    """将 Binance combined stream 事件归一化为稳定的前端 JSON 契约。"""

    def __init__(
        self,
        symbols: tuple[str, ...] = DEFAULT_LIVE_SYMBOLS,
        *,
        interval: str = "1m",
        recent_trade_limit: int = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if recent_trade_limit < 1:
            raise ValueError("recent_trade_limit 必须大于 0")
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.interval = interval
        self.recent_trade_limit = recent_trade_limit
        self._clock = clock or (lambda: datetime.now(UTC))
        now = self._clock().isoformat()
        self.snapshot: dict[str, Any] = {
            "version": LIVE_SNAPSHOT_VERSION,
            "source": "binance_usdm_public_websocket",
            "status": "connecting",
            "started_at": now,
            "updated_at": None,
            "message_count": 0,
            "event_counts": {},
            "interval": interval,
            "symbols": {symbol: {"recent_trades": []} for symbol in self.symbols},
        }

    def apply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """应用一个原始/combined 事件，并在 K 线收盘时返回标准落盘记录。"""
        source_market = str(message.get("_source_market", "futures"))
        storage_market = "spot" if source_market.startswith("spot") else source_market
        payload = message.get("data", message)
        if not isinstance(payload, dict):
            return None
        event = str(payload.get("e", ""))
        if not event:
            stream_name = str(message.get("stream", ""))
            stream_event = stream_name.partition("@")[2]
            if stream_event.startswith("bookTicker"):
                event = "bookTicker"
        symbol = str(payload.get("s", "")).upper()
        if symbol not in self.snapshot["symbols"]:
            return None
        symbol_state = self.snapshot["symbols"][symbol]
        event_counts = self.snapshot["event_counts"]
        event_counts[event or "unknown"] = int(event_counts.get(event or "unknown", 0)) + 1
        event_time = _iso_from_ms(payload.get("E")) or self._clock().isoformat()
        closed_kline: dict[str, Any] | None = None

        if event == "aggTrade":
            trade = {
                "timestamp": _iso_from_ms(payload.get("T", payload.get("E"))),
                "trade_id": int(payload["a"]),
                "price": _float(payload["p"]),
                "quantity": _float(payload["q"]),
                "side": "sell" if bool(payload["m"]) else "buy",
            }
            symbol_state["last_trade"] = trade
            trades = symbol_state["recent_trades"]
            trades.append(trade)
            del trades[: -self.recent_trade_limit]
        elif event == "bookTicker":
            bid = _float(payload["b"])
            ask = _float(payload["a"])
            mid = (bid + ask) / 2
            symbol_state["book"] = {
                "timestamp": event_time,
                "bid_price": bid,
                "bid_quantity": _float(payload["B"]),
                "ask_price": ask,
                "ask_quantity": _float(payload["A"]),
                "spread": ask - bid,
                "spread_bps": (ask - bid) / mid * 10_000 if mid else None,
            }
        elif event == "depthUpdate":
            bid_levels = [[_float(row[0]), _float(row[1])] for row in payload.get("b", [])]
            ask_levels = [[_float(row[0]), _float(row[1])] for row in payload.get("a", [])]
            symbol_state["depth"] = {
                "timestamp": event_time,
                "bids": bid_levels,
                "asks": ask_levels,
            }
            if bid_levels and ask_levels:
                bid, ask = bid_levels[0][0], ask_levels[0][0]
                mid = (bid + ask) / 2
                symbol_state["book"] = {
                    "timestamp": event_time,
                    "bid_price": bid,
                    "bid_quantity": bid_levels[0][1],
                    "ask_price": ask,
                    "ask_quantity": ask_levels[0][1],
                    "spread": ask - bid,
                    "spread_bps": (ask - bid) / mid * 10_000 if mid else None,
                }
        elif event == "markPriceUpdate":
            mark_price = _float(payload["p"])
            index_price = _float(payload["i"])
            symbol_state["derivatives"] = {
                "timestamp": event_time,
                "mark_price": mark_price,
                "index_price": index_price,
                "basis": (mark_price - index_price) / index_price,
                "funding_rate": _float(payload["r"]),
                "next_funding_time": _iso_from_ms(payload.get("T")),
            }
        elif event == "kline":
            kline = payload["k"]
            normalized = {
                "open_time": _iso_from_ms(kline["t"]),
                "close_time": _iso_from_ms(kline["T"]),
                "open": _float(kline["o"]),
                "high": _float(kline["h"]),
                "low": _float(kline["l"]),
                "close": _float(kline["c"]),
                "volume": _float(kline["v"]),
                "quote_volume": _float(kline["q"]),
                "num_trades": int(kline["n"]),
                "taker_buy_volume": _float(kline["V"]),
                "taker_buy_quote_volume": _float(kline["Q"]),
                "closed": bool(kline["x"]),
            }
            symbol_state["current_kline"] = normalized
            if normalized["closed"]:
                symbol_state["last_closed_kline"] = normalized
                closed_kline = {
                    "symbol": symbol,
                    "interval": str(kline["i"]),
                    "market": storage_market,
                    "open_time": int(kline["t"]),
                    "close_time": int(kline["T"]),
                    "open": normalized["open"],
                    "high": normalized["high"],
                    "low": normalized["low"],
                    "close": normalized["close"],
                    "volume": normalized["volume"],
                    "quote_asset_volume": normalized["quote_volume"],
                    "number_of_trades": normalized["num_trades"],
                    "taker_buy_base_volume": normalized["taker_buy_volume"],
                    "taker_buy_quote_volume": normalized["taker_buy_quote_volume"],
                }
        else:
            return None

        symbol_state["updated_at"] = event_time
        self.snapshot["updated_at"] = self._clock().isoformat()
        self.snapshot["message_count"] += 1
        self.snapshot["status"] = "live"
        return closed_kline

    def mark_stopped(self, error: str | None = None) -> None:
        self.snapshot["status"] = "error" if error else "stopped"
        self.snapshot["error"] = error
        self.snapshot["updated_at"] = self._clock().isoformat()


def write_live_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """使用同目录临时文件原子替换，避免前端读到半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_live_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def live_snapshot_status(
    snapshot: dict[str, Any],
    *,
    now: datetime | None = None,
    stale_after_seconds: float = 10.0,
) -> tuple[str, float | None]:
    """返回前端状态与快照年龄；旧快照不会被误标成在线。"""
    updated_at = snapshot.get("updated_at")
    if not updated_at:
        return "offline", None
    try:
        updated = datetime.fromisoformat(str(updated_at)).astimezone(UTC)
    except ValueError:
        return "offline", None
    age = max(0.0, ((now or datetime.now(UTC)) - updated).total_seconds())
    if snapshot.get("status") == "error":
        return "error", age
    return ("live" if age <= stale_after_seconds else "stale"), age


class FinalizedKlineStore:
    """将已收盘 K 线幂等合并到现有 Binance Futures Silver 文件。"""

    def append(self, record: dict[str, Any]) -> Path:
        raw = KlineRaw(
            open_time=int(record["open_time"]),
            close_time=int(record["close_time"]),
            open=float(record["open"]),
            high=float(record["high"]),
            low=float(record["low"]),
            close=float(record["close"]),
            volume=float(record["volume"]),
            quote_asset_volume=float(record["quote_asset_volume"]),
            number_of_trades=int(record["number_of_trades"]),
            taker_buy_base_volume=float(record["taker_buy_base_volume"]),
            taker_buy_quote_volume=float(record["taker_buy_quote_volume"]),
        )
        new_frame = bronze_to_silver(
            raw_to_dataframe(
                [raw], symbol=str(record["symbol"]), interval=str(record["interval"])
            )
        )
        path = silver_klines_path(
            exchange="binance",
            market=str(record.get("market", "futures")),
            symbol=str(record["symbol"]),
            interval=str(record["interval"]),
        )
        merge_parquet_safe(path, new_frame, key="open_time_utc")
        return path


class BinanceLiveCollector:
    """独立实时采集器；前端只读快照，不持有 WebSocket。"""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...] = DEFAULT_LIVE_SYMBOLS,
        interval: str = "1m",
        snapshot_path: Path,
        persist_closed_bars: bool = True,
        stream: BinanceWebSocketStream | FanInWebSocketStream | None = None,
        state: LiveMarketState | None = None,
        store: FinalizedKlineStore | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        snapshot_interval: float = 1.0,
        derivative_poll_seconds: float | None = 60.0,
    ) -> None:
        self.state = state or LiveMarketState(symbols, interval=interval)
        self.snapshot_path = snapshot_path
        self.persist_closed_bars = persist_closed_bars
        self.store = store or FinalizedKlineStore()
        self._monotonic = monotonic
        self.snapshot_interval = snapshot_interval
        self.derivative_poll_seconds = derivative_poll_seconds if stream is None else None
        self._last_derivative_poll = float("-inf")
        self.stream = stream or FanInWebSocketStream(
            {
                "spot": BinanceWebSocketStream(
                    spot_stream_names(symbols, interval),
                    base_url="wss://stream.binance.com:9443/stream?streams=",
                    timeout=30,
                    backoff_base=0.5,
                    backoff_cap=8.0,
                ),
                "spot_book": BinanceWebSocketStream(
                    spot_book_stream_names(symbols),
                    base_url="wss://stream.binance.com:9443/stream?streams=",
                    timeout=30,
                    backoff_base=0.5,
                    backoff_cap=8.0,
                ),
                "futures": BinanceWebSocketStream(
                    futures_stream_names(symbols),
                    base_url="wss://fstream.binance.com/stream?streams=",
                    timeout=30,
                    backoff_base=0.5,
                    backoff_cap=8.0,
                ),
            }
        )
        self.state.snapshot["stream_url"] = getattr(self.stream, "url", "injected")

    def _poll_derivatives(self) -> None:
        """公开 REST 低频补充 Funding/Basis；失败不会中断主行情流。"""
        errors: list[dict[str, str]] = []
        for symbol in self.state.symbols:
            try:
                client = BinanceClient(symbol)
                row = client.fetch_basis().iloc[-1]
                funding = client.fetch_funding_rate(limit=1)
                latest_funding = (
                    float(funding.iloc[-1]["funding_rate"]) if not funding.empty else None
                )
                self.state.snapshot["symbols"][symbol]["derivatives"] = {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "mark_price": float(row["mark_price"]),
                    "index_price": float(row["index_price"]),
                    "basis": float(row["basis"]),
                    "funding_rate": latest_funding,
                    "next_funding_time": None,
                    "source": "binance_public_rest",
                }
            except Exception as exc:
                errors.append({"symbol": symbol, "error": str(exc)})
        self.state.snapshot["derivative_poll_errors"] = errors
        self.state.snapshot["derivatives_polled_at"] = datetime.now(UTC).isoformat()

    def run(self, *, max_messages: int | None = None) -> int:
        write_live_snapshot(self.snapshot_path, self.state.snapshot)
        processed = 0
        last_write = self._monotonic()
        try:
            for message in self.stream.stream():
                closed = self.state.apply(message)
                processed += 1
                self.state.snapshot["active_alerts"] = evaluate_market_alerts(
                    self.state.snapshot
                )
                current = self._monotonic()
                if (
                    self.derivative_poll_seconds is not None
                    and current - self._last_derivative_poll >= self.derivative_poll_seconds
                ):
                    self._poll_derivatives()
                    self._last_derivative_poll = current
                if closed is not None and self.persist_closed_bars:
                    self.store.append(closed)
                    self.state.snapshot["last_persisted_bar"] = {
                        "symbol": closed["symbol"],
                        "interval": closed["interval"],
                        "open_time": _iso_from_ms(closed["open_time"]),
                    }
                if current - last_write >= self.snapshot_interval or closed is not None:
                    write_live_snapshot(self.snapshot_path, self.state.snapshot)
                    last_write = current
                if max_messages is not None and processed >= max_messages:
                    break
        except KeyboardInterrupt:
            self.state.mark_stopped()
        except Exception as exc:
            self.state.mark_stopped(str(exc))
            write_live_snapshot(self.snapshot_path, self.state.snapshot)
            raise
        finally:
            write_live_snapshot(self.snapshot_path, self.state.snapshot)
            self.stream.close()
        return processed
