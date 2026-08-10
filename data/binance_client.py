"""
Binance 交易所客户端

实现 ExchangeBase 中定义的所有抽象方法，
对接 Binance REST API 获取历史数据和实时数据。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from config.constants import KlineInterval
from data.base import ExchangeBase
from data.schema import (
    BasisRaw,
    FundingRateRaw,
    HistoricalBasisRaw,
    KlineRaw,
    OpenInterestRaw,
    OrderBookRaw,
    TradeRaw,
    order_book_to_dataframe,
    raw_to_dataframe,
    trades_to_dataframe,
)


class BinanceClient(ExchangeBase):
    """Binance 交易所数据客户端

    封装 Binance API 的 K 线、成交、资金费率等数据获取逻辑。
    """

    def __init__(
        self,
        symbol: str,
        *,
        session: requests.Session | None = None,
        spot_base_url: str = "https://api.binance.com",
        futures_base_url: str = "https://fapi.binance.com",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(symbol)
        self.session = session or requests.Session()
        self.spot_base_url = spot_base_url.rstrip("/")
        self.futures_base_url = futures_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep

    @staticmethod
    def _milliseconds(value: datetime | None) -> int | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Binance 时间参数必须带时区")
        return int(value.astimezone(UTC).timestamp() * 1000)

    def _get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """执行 GET，并对网络错误、429 和 5xx 做有限指数退避。"""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    f"{base_url}{path}", params=params, timeout=self.timeout
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.backoff_base * 2**attempt
                    self._sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self._sleep(self.backoff_base * 2**attempt)
        raise RuntimeError("Binance 请求重试耗尽") from last_error

    # ------------------------------------------------------------------
    # K 线数据
    # ------------------------------------------------------------------
    def fetch_klines(
        self,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 历史 K 线数据"""
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须在 1..1000")
        params: dict[str, Any] = {
            "symbol": self.symbol.upper(),
            "interval": interval.value,
            "limit": limit,
        }
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        rows = self._get(self.spot_base_url, "/api/v3/klines", params)
        raw = [KlineRaw.from_binance_row(row) for row in rows]
        if not raw:
            return pd.DataFrame()
        return raw_to_dataframe(raw, symbol=self.symbol, interval=interval.value)

    def fetch_futures_klines(
        self,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 USDⓈ-M 永续合约 K 线，返回与现货相同的 Bronze Schema。"""
        if not 1 <= limit <= 1500:
            raise ValueError("futures kline limit 必须在 1..1500")
        params: dict[str, Any] = {
            "symbol": self.symbol.upper(),
            "interval": interval.value,
            "limit": limit,
        }
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        rows = self._get(self.futures_base_url, "/fapi/v1/klines", params)
        raw = [KlineRaw.from_binance_row(row) for row in rows]
        if not raw:
            return pd.DataFrame()
        return raw_to_dataframe(raw, symbol=self.symbol, interval=interval.value)

    # ------------------------------------------------------------------
    # 逐笔成交
    # ------------------------------------------------------------------
    def fetch_trades(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 逐笔成交数据"""
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须在 1..1000")
        params: dict[str, Any] = {"symbol": self.symbol.upper(), "limit": limit}
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        payload = self._get(self.spot_base_url, "/api/v3/aggTrades", params)
        return trades_to_dataframe(
            [TradeRaw.from_binance_payload(item, symbol=self.symbol) for item in payload]
        )

    def fetch_depth(self, limit: int = 100) -> pd.DataFrame:
        """获取 Binance 现货盘口快照并展开为逐档统一表。"""
        if limit not in {5, 10, 20, 50, 100, 500, 1000, 5000}:
            raise ValueError("Binance depth limit 不合法")
        payload = self._get(
            self.spot_base_url,
            "/api/v3/depth",
            {"symbol": self.symbol.upper(), "limit": limit},
        )
        book = OrderBookRaw.from_binance_payload(
            payload, symbol=self.symbol, timestamp=datetime.now(UTC)
        )
        return order_book_to_dataframe(book)

    # ------------------------------------------------------------------
    # 资金费率
    # ------------------------------------------------------------------
    def fetch_funding_rate(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 历史资金费率数据"""
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须在 1..1000")
        params: dict[str, Any] = {"symbol": self.symbol.upper(), "limit": limit}
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        payload = self._get(self.futures_base_url, "/fapi/v1/fundingRate", params)
        records = [
            FundingRateRaw(
                timestamp=datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=UTC),
                funding_rate=float(item["fundingRate"]),
                symbol=item["symbol"],
            ).model_dump()
            for item in payload
        ]
        return pd.DataFrame.from_records(records)

    def fetch_open_interest(
        self,
        *,
        period: str = "5m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 30,
    ) -> pd.DataFrame:
        """获取 Binance 永续历史持仓量。"""
        if not 1 <= limit <= 500:
            raise ValueError("open interest limit 必须在 1..500")
        params: dict[str, Any] = {"symbol": self.symbol.upper(), "period": period, "limit": limit}
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        payload = self._get(self.futures_base_url, "/futures/data/openInterestHist", params)
        rows = [
            OpenInterestRaw(
                timestamp=datetime.fromtimestamp(int(item["timestamp"]) / 1000, tz=UTC),
                symbol=item["symbol"],
                open_interest=float(item["sumOpenInterest"]),
            ).model_dump()
            for item in payload
        ]
        return pd.DataFrame.from_records(rows)

    def fetch_basis(self) -> pd.DataFrame:
        """获取当前永续标记价格相对指数价格的基差快照。"""
        item = self._get(
            self.futures_base_url,
            "/fapi/v1/premiumIndex",
            {"symbol": self.symbol.upper()},
        )
        mark_price = float(item["markPrice"])
        index_price = float(item["indexPrice"])
        record = BasisRaw(
            timestamp=datetime.fromtimestamp(int(item["time"]) / 1000, tz=UTC),
            symbol=item["symbol"],
            mark_price=mark_price,
            index_price=index_price,
            basis=(mark_price - index_price) / index_price,
        )
        return pd.DataFrame([record.model_dump()])

    def fetch_historical_basis(
        self,
        *,
        period: str = "1h",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取 Binance 最近约 30 天的永续历史基差。"""
        if not 1 <= limit <= 500:
            raise ValueError("historical basis limit 必须在 1..500")
        params: dict[str, Any] = {
            "pair": self.symbol.upper(),
            "contractType": "PERPETUAL",
            "period": period,
            "limit": limit,
        }
        if (start_ms := self._milliseconds(start_time)) is not None:
            params["startTime"] = start_ms
        if (end_ms := self._milliseconds(end_time)) is not None:
            params["endTime"] = end_ms
        payload = self._get(self.futures_base_url, "/futures/data/basis", params)
        rows = [
            HistoricalBasisRaw(
                timestamp=datetime.fromtimestamp(int(item["timestamp"]) / 1000, tz=UTC),
                symbol=item["pair"],
                futures_price=float(item["futuresPrice"]),
                index_price=float(item["indexPrice"]),
                basis=float(item["basisRate"]),
            ).model_dump()
            for item in payload
        ]
        return pd.DataFrame.from_records(rows)

    # ------------------------------------------------------------------
    # 交易所信息
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """测试 Binance 连接状态"""
        self._get(self.spot_base_url, "/api/v3/ping")
        return True

    @property
    def exchange_name(self) -> str:
        return "binance"
