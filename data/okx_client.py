"""OKX 公共 REST 市场数据的最小统一客户端。"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from config.constants import KlineInterval, get_interval_ms
from data.base import ExchangeBase
from data.schema import (
    DepthLevel,
    FundingRateRaw,
    KlineRaw,
    OrderBookRaw,
    TradeRaw,
    order_book_to_dataframe,
    raw_to_dataframe,
    trades_to_dataframe,
)


def _spot_instrument(symbol: str) -> str:
    """将 BTCUSDT/BTC-USDT 规范为 OKX 现货 instrument id。"""
    normalized = symbol.strip().upper()
    if "-" in normalized:
        parts = normalized.split("-")
        return "-".join(parts[:2])
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return f"{normalized[:-len(quote)]}-{quote}"
    raise ValueError(f"无法将 symbol 映射为 OKX instrument: {symbol!r}")


def _swap_instrument(symbol: str) -> str:
    spot = _spot_instrument(symbol)
    return f"{spot}-SWAP"


class OKXClient(ExchangeBase):
    """实现 ExchangeBase 最小公共接口；只访问无需密钥的公共端点。"""

    _BAR_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6Hutc",
        "12h": "12Hutc",
        "1d": "1Dutc",
        "3d": "3Dutc",
        "1w": "1Wutc",
    }

    def __init__(
        self,
        symbol: str,
        *,
        session: requests.Session | None = None,
        base_url: str = "https://www.okx.com",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(symbol)
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep

    def _get(self, path: str, params: dict[str, Any] | None = None) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}{path}", params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self._sleep(self.backoff_base * 2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                self._sleep(self.backoff_base * 2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "0":
                raise requests.HTTPError(
                    f"OKX {path}: code={payload.get('code')} msg={payload.get('msg')}",
                    response=response,
                )
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("OKX 响应缺少 data 数组")
            return data
        raise RuntimeError("OKX 请求重试耗尽") from last_error

    def fetch_klines(
        self,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        if not 1 <= limit <= 300:
            raise ValueError("OKX kline limit 必须在 1..300")
        if interval.value not in self._BAR_MAP:
            raise ValueError(f"OKX 暂不支持周期 {interval.value}")
        params: dict[str, Any] = {
            "instId": _spot_instrument(self.symbol),
            "bar": self._BAR_MAP[interval.value],
            "limit": limit,
        }
        # OKX before/after 的命名方向容易误用；这里请求一页后在本地严格裁剪。
        data = self._get("/api/v5/market/history-candles", params)
        start_ms = int(start_time.astimezone(UTC).timestamp() * 1000) if start_time else None
        end_ms = int(end_time.astimezone(UTC).timestamp() * 1000) if end_time else None
        rows: list[KlineRaw] = []
        interval_ms = get_interval_ms(interval.value)
        for item in data:
            opened = int(item[0])
            if (start_ms is not None and opened < start_ms) or (
                end_ms is not None and opened > end_ms
            ):
                continue
            rows.append(
                KlineRaw(
                    open_time=opened,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    close_time=opened + interval_ms - 1,
                    quote_asset_volume=float(item[7]),
                    number_of_trades=0,
                    taker_buy_base_volume=0.0,
                    taker_buy_quote_volume=0.0,
                )
            )
        if not rows:
            return pd.DataFrame()
        frame = raw_to_dataframe(rows, symbol=self.symbol.replace("-", ""), interval=interval.value)
        return frame.sort_values("open_time").reset_index(drop=True)

    def fetch_trades(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        if not 1 <= limit <= 500:
            raise ValueError("OKX trades limit 必须在 1..500")
        data = self._get(
            "/api/v5/market/trades", {"instId": _spot_instrument(self.symbol), "limit": limit}
        )
        records: list[TradeRaw] = []
        for item in data:
            timestamp = datetime.fromtimestamp(int(item["ts"]) / 1000, tz=UTC)
            if start_time and timestamp < start_time.astimezone(UTC):
                continue
            if end_time and timestamp > end_time.astimezone(UTC):
                continue
            price, quantity = float(item["px"]), float(item["sz"])
            records.append(
                TradeRaw(
                    timestamp=timestamp,
                    symbol=self.symbol.replace("-", "").upper(),
                    trade_id=int(item["tradeId"]),
                    price=price,
                    quantity=quantity,
                    quote_quantity=price * quantity,
                    side=item["side"],
                )
            )
        return trades_to_dataframe(records)

    def fetch_depth(self, limit: int = 100) -> pd.DataFrame:
        if not 1 <= limit <= 400:
            raise ValueError("OKX depth limit 必须在 1..400")
        data = self._get(
            "/api/v5/market/books", {"instId": _spot_instrument(self.symbol), "sz": limit}
        )
        if not data:
            return pd.DataFrame()
        item = data[0]
        timestamp = datetime.fromtimestamp(int(item["ts"]) / 1000, tz=UTC)
        book = OrderBookRaw(
            timestamp=timestamp,
            symbol=self.symbol.replace("-", "").upper(),
            last_update_id=int(item.get("seqId", item["ts"])),
            bids=[
                DepthLevel(price=float(row[0]), quantity=float(row[1]))
                for row in item["bids"]
            ],
            asks=[
                DepthLevel(price=float(row[0]), quantity=float(row[1]))
                for row in item["asks"]
            ],
        )
        return order_book_to_dataframe(book)

    def fetch_funding_rate(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        if not 1 <= limit <= 100:
            raise ValueError("OKX funding limit 必须在 1..100")
        data = self._get(
            "/api/v5/public/funding-rate-history",
            {"instId": _swap_instrument(self.symbol), "limit": limit},
        )
        records: list[dict[str, Any]] = []
        for item in data:
            timestamp = datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=UTC)
            if start_time and timestamp < start_time.astimezone(UTC):
                continue
            if end_time and timestamp > end_time.astimezone(UTC):
                continue
            records.append(
                FundingRateRaw(
                    timestamp=timestamp,
                    funding_rate=float(item["fundingRate"]),
                    symbol=self.symbol.replace("-", "").upper(),
                ).model_dump()
            )
        return pd.DataFrame.from_records(records)

    def ping(self) -> bool:
        self._get("/api/v5/public/time")
        return True

    @property
    def exchange_name(self) -> str:
        return "okx"
