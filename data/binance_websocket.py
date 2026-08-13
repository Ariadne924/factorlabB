"""Binance WebSocket 的轻量重连封装。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import websocket


class SocketLike(Protocol):
    def recv(self) -> str: ...
    def close(self) -> None: ...


class BinanceWebSocketStream:
    """同步消息生成器，异常后按指数退避重新连接并继续产出消息。

    这是研究环境的基础实现：退避有上限，可注入连接器和 sleep 便于 mock，
    不承担持久化、心跳状态机或 exactly-once 语义。
    """

    def __init__(
        self,
        stream_name: str,
        *,
        connector: Callable[[str, float], SocketLike] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = "wss://stream.binance.com:9443/ws",
        timeout: float = 30.0,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        max_reconnects: int | None = None,
    ) -> None:
        normalized_base = base_url.rstrip("/")
        separator = "" if normalized_base.endswith("=") else "/"
        normalized_streams = "/".join(
            self._normalize_stream_name(name) for name in stream_name.split("/")
        )
        self.url = f"{normalized_base}{separator}{normalized_streams}"
        self._connector = connector or (
            lambda url, timeout: websocket.create_connection(url, timeout=timeout)
        )
        self._sleep = sleep
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.max_reconnects = max_reconnects
        self._closed = False
        self._socket: SocketLike | None = None

    @staticmethod
    def _normalize_stream_name(stream_name: str) -> str:
        """Binance 要求交易对小写，但部分事件名本身大小写敏感。"""
        symbol, separator, event = stream_name.partition("@")
        return f"{symbol.lower()}{separator}{event}"

    def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            self._socket.close()

    def stream(self) -> Iterator[dict[str, Any]]:
        reconnects = 0
        while not self._closed:
            try:
                self._socket = self._connector(self.url, self.timeout)
                while not self._closed:
                    payload = self._socket.recv()
                    if not payload:
                        raise ConnectionError("WebSocket 连接已关闭")
                    message = json.loads(payload)
                    reconnects = 0
                    yield message
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                ValueError,
                websocket.WebSocketException,
            ) as exc:
                if self._closed:
                    break
                reconnects += 1
                if self.max_reconnects is not None and reconnects > self.max_reconnects:
                    raise ConnectionError("WebSocket 重连次数已耗尽") from exc
                self._sleep(min(self.backoff_cap, self.backoff_base * 2 ** (reconnects - 1)))
            finally:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    finally:
                        self._socket = None
