"""
Binance 交易所客户端

实现 ExchangeBase 中定义的所有抽象方法，
对接 Binance REST API 获取历史数据和实时数据。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from config.constants import KlineInterval
from data.base import ExchangeBase


class BinanceClient(ExchangeBase):
    """Binance 交易所数据客户端

    封装 Binance API 的 K 线、成交、资金费率等数据获取逻辑。
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)

    # ------------------------------------------------------------------
    # K 线数据
    # ------------------------------------------------------------------
    def fetch_klines(
        self,
        interval: KlineInterval,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 历史 K 线数据"""
        # TODO: 实现 Binance API K 线获取逻辑
        raise NotImplementedError("Binance K 线获取方法尚未实现")

    # ------------------------------------------------------------------
    # 逐笔成交
    # ------------------------------------------------------------------
    def fetch_trades(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 逐笔成交数据"""
        # TODO: 实现 Binance API 成交数据获取逻辑
        raise NotImplementedError("Binance 成交获取方法尚未实现")

    # ------------------------------------------------------------------
    # 资金费率
    # ------------------------------------------------------------------
    def fetch_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取 Binance 历史资金费率数据"""
        # TODO: 实现 Binance API 资金费率获取逻辑
        raise NotImplementedError("Binance 资金费率获取方法尚未实现")

    # ------------------------------------------------------------------
    # 交易所信息
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """测试 Binance 连接状态"""
        # TODO: 实现 Binance ping 测试
        raise NotImplementedError("Binance ping 方法尚未实现")

    @property
    def exchange_name(self) -> str:
        return "binance"
