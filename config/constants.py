"""
全局常量模块

本模块定义交易所枚举、K 线周期映射等项目中使用的常量。
所有枚举值尽量与交易所 API 原始值保持一致（均为字符串）。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


# ============================================================================
# 交易所枚举
# ============================================================================
class Exchange(StrEnum):
    """支持的交易所列表"""
    BINANCE = "binance"
    OKX = "okx"
    # 预留扩展：
    # BYBIT = "bybit"


# ============================================================================
# K 线周期映射
# ============================================================================
class KlineInterval(StrEnum):
    """K 线周期枚举

    值与 Binance API 的 interval 参数保持一致。
    """
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


# K 线周期 → 毫秒数映射（用于时间计算）
# 注意：仅包含固定长度周期。月份（1M）长度可变，不在此表中，
# 应使用 pd.DateOffset(months=1) 或 pd.tseries.frequencies.to_offset("1M") 处理。
INTERVAL_TO_MS: Final[dict[str, int]] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

# 月份类周期：不能映射为固定毫秒数
MONTHLY_INTERVALS: Final[frozenset[str]] = frozenset({"1M"})


def get_interval_ms(interval: str) -> int:
    """安全地获取非月份周期的毫秒数

    Args:
        interval: K 线周期字符串

    Returns:
        毫秒数

    Raises:
        ValueError: 如果是月份周期或未知周期
    """
    if interval in MONTHLY_INTERVALS:
        raise ValueError(
            f"'{interval}' 是月份级周期，长度可变，无法映射为固定毫秒数。"
            f"请使用 pd.DateOffset 或 pd.Timedelta 以正确处理月份偏移。"
        )
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"未知的 K 线周期: '{interval}'")
    return INTERVAL_TO_MS[interval]

# ============================================================================
# 交易方向
# ============================================================================
class TradeSide(StrEnum):
    """成交方向"""
    BUY = "buy"
    SELL = "sell"


# ============================================================================
# 因子分类
# ============================================================================
class FactorCategory(StrEnum):
    """因子分类标签"""
    MOMENTUM = "动量"
    VOLATILITY = "波动率"
    VOLUME_LIQUIDITY = "成交量与流动性"
    CRYPTO_SPECIFIC = "加密货币特有"
