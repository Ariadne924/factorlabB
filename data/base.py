"""
交易所抽象基类模块

定义所有交易所数据接口的统一抽象基类。
每个具体交易所实现须继承此基类并实现所有抽象方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from config.constants import KlineInterval


class ExchangeBase(ABC):
    """交易所数据接口抽象基类

    所有具体交易所实现必须继承此类，保证接口的一致性。
    新增交易所时只需实现对应方法，上层调用逻辑无需修改。
    """

    def __init__(self, symbol: str) -> None:
        """
        Args:
            symbol: 交易对 ID，如 'BTCUSDT'
        """
        self.symbol: str = symbol

    # ------------------------------------------------------------------
    # K 线数据
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch_klines(
        self,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取历史 K 线数据

        Args:
            interval: K 线周期
            start_time: 起始时间（UTC），None 表示不限
            end_time: 结束时间（UTC），None 表示不限
            limit: 最大返回条数

        Returns:
            通过 KlineRaw 校验后生成的 Bronze DataFrame
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 逐笔成交
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch_trades(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取逐笔成交数据

        Args:
            start_time: 起始时间（UTC）
            end_time: 结束时间（UTC）
            limit: 最大返回条数

        Returns:
            通过 TradeRaw 校验后生成的 Bronze DataFrame
        """
        raise NotImplementedError

    def fetch_depth(self, limit: int = 100) -> pd.DataFrame:
        """获取盘口快照。

        该方法不是抽象方法，以免破坏尚未实现盘口接口的第二交易所客户端。
        支持盘口的客户端应返回统一的逐档 DataFrame。
        """
        raise NotImplementedError(f"{self.exchange_name} 尚未实现盘口接口")

    # ------------------------------------------------------------------
    # 资金费率
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch_funding_rate(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取历史资金费率数据

        Args:
            start_time: 起始时间（UTC）
            end_time: 结束时间（UTC）
            limit: 最大返回条数

        Returns:
            通过 FundingRateRaw 校验后生成的 Bronze DataFrame
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 交易所信息
    # ------------------------------------------------------------------
    @abstractmethod
    def ping(self) -> bool:
        """测试交易所连接状态

        Returns:
            True 表示连接正常
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """返回交易所名称"""
        raise NotImplementedError
