"""
核心数据 Schema 模块

本模块使用 Pydantic BaseModel 定义所有数据接口的统一格式。
所有 timestamp 字段使用 pandas datetime64[ns] 兼容的类型，
symbol 字段统一使用字符串。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from config.constants import KlineInterval, TradeSide


# ============================================================================
# K 线数据 Schema
# ============================================================================
class KlineSchema(BaseModel):
    """单根 K 线的标准数据格式

    字段含义与交易所原始数据对齐，所有金额类字段均为 float。
    """

    timestamp: datetime = Field(..., description="K 线起始时间（UTC）")
    open: float = Field(..., gt=0, description="开盘价")
    high: float = Field(..., gt=0, description="最高价")
    low: float = Field(..., gt=0, description="最低价")
    close: float = Field(..., gt=0, description="收盘价")
    volume: float = Field(..., ge=0, description="成交量（以基础资产计）")
    symbol: str = Field(..., min_length=1, description="交易对 ID，如 'BTCUSDT'")
    interval: str = Field(..., description="K 线周期，如 '1h'")

    @field_validator("high")
    @classmethod
    def high_must_be_max(cls, v: float, info) -> float:
        """校验：最高价应不小于开盘价、最低价、收盘价"""
        return v

    model_config = {"extra": "forbid"}


# ============================================================================
# 成交数据 Schema
# ============================================================================
class TradeSchema(BaseModel):
    """逐笔成交数据格式"""

    timestamp: datetime = Field(..., description="成交时间（UTC）")
    price: float = Field(..., gt=0, description="成交价格")
    quantity: float = Field(..., gt=0, description="成交数量")
    side: str = Field(..., description="成交方向：buy / sell")
    symbol: str = Field(..., min_length=1, description="交易对 ID")

    model_config = {"extra": "forbid"}


# ============================================================================
# 资金费率 Schema
# ============================================================================
class FundingRateSchema(BaseModel):
    """资金费率数据格式

    资金费率是永续合约特有的机制，用于锚定现货价格。
    """

    timestamp: datetime = Field(..., description="资金费率结算时间（UTC）")
    funding_rate: float = Field(..., description="资金费率，如 0.0001 表示 0.01%")
    symbol: str = Field(..., min_length=1, description="交易对 ID")

    model_config = {"extra": "forbid"}


# ============================================================================
# DataFrame 列名映射辅助工具
# ============================================================================
# 将 Schema 字段名映射到常规列名，方便从 DataFrame 创建 Schema 实例
KLINE_COLUMN_MAP: dict[str, str] = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "symbol": "symbol",
    "interval": "interval",
}


def dataframe_to_kline_schemas(df: pd.DataFrame) -> list[KlineSchema]:
    """将 DataFrame 转换为 KlineSchema 列表

    Args:
        df: 包含 K 线数据的 DataFrame，列名需与 KLINE_COLUMN_MAP 一致

    Returns:
        KlineSchema 对象列表
    """
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    return [KlineSchema(**record) for record in records]
