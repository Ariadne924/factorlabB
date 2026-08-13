from __future__ import annotations

import json

from data.binance_websocket import BinanceWebSocketStream


class FakeSocket:
    def __init__(self, messages: list[object]):
        self.messages = iter(messages)
        self.closed = False

    def recv(self) -> str:
        value = next(self.messages)
        if isinstance(value, Exception):
            raise value
        return str(value)

    def close(self) -> None:
        self.closed = True


def test_websocket_reconnects_and_continues_stream() -> None:
    sockets = [FakeSocket([ConnectionError("drop")]), FakeSocket([json.dumps({"event": "ok"})])]
    sleeps: list[float] = []

    def connector(url: str, timeout: float) -> FakeSocket:
        assert url.endswith("/btcusdt@trade")
        assert timeout == 5
        return sockets.pop(0)

    stream = BinanceWebSocketStream(
        "BTCUSDT@trade",
        connector=connector,
        sleep=sleeps.append,
        timeout=5,
        backoff_base=0.1,
        max_reconnects=2,
    )
    iterator = stream.stream()
    assert next(iterator) == {"event": "ok"}
    stream.close()
    assert sleeps == [0.1]


def test_websocket_preserves_case_sensitive_event_names() -> None:
    stream = BinanceWebSocketStream(
        "BTCUSDT@aggTrade/BTCUSDT@bookTicker/BTCUSDT@markPrice@1s",
        base_url="wss://fstream.binance.com/stream?streams=",
    )
    assert stream.url == (
        "wss://fstream.binance.com/stream?streams="
        "btcusdt@aggTrade/btcusdt@bookTicker/btcusdt@markPrice@1s"
    )
