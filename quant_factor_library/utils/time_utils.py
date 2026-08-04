"""
时间工具模块

统一的时间戳转换工具，所有时间均转换为 UTC。
支持多种输入格式：datetime、字符串、Unix 时间戳（秒/毫秒）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

import pandas as pd

# 支持的输入时间类型
TimeInput = Union[str, int, float, datetime, pd.Timestamp]


def to_utc_timestamp(value: TimeInput) -> pd.Timestamp:
    """将任意输入时间转换为 UTC 时间戳（pd.Timestamp）

    支持的输入格式：
        - str: '2024-01-01'、'2024-01-01 12:00:00'、ISO 8601 格式
        - int/float: Unix 时间戳（秒或毫秒，自动判断）
        - datetime: Python 原生 datetime 对象
        - pd.Timestamp: 直接返回（转换时区）

    Args:
        value: 输入时间值

    Returns:
        带 UTC 时区的 pd.Timestamp

    Raises:
        ValueError: 无法解析输入格式
        TypeError: 不支持的输入类型

    示例：
        >>> to_utc_timestamp('2024-01-01')
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')

        >>> to_utc_timestamp(1704067200)
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')

        >>> to_utc_timestamp(1704067200000)  # 毫秒时间戳
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    """
    # ----------------------------------------------------------------
    # 字符串输入
    # ----------------------------------------------------------------
    if isinstance(value, str):
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"无法解析时间字符串: '{value}'") from exc

        return _ensure_utc(ts)

    # ----------------------------------------------------------------
    # 数值输入（Unix 时间戳）
    # ----------------------------------------------------------------
    if isinstance(value, (int, float)):
        # 自动判断秒/毫秒：大于 1e12 视为毫秒
        if value > 1e12:
            ts = pd.Timestamp(value, unit="ms")
        else:
            ts = pd.Timestamp(value, unit="s")
        return _ensure_utc(ts)

    # ----------------------------------------------------------------
    # datetime 对象
    # ----------------------------------------------------------------
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        return _ensure_utc(ts)

    # ----------------------------------------------------------------
    # pd.Timestamp
    # ----------------------------------------------------------------
    if isinstance(value, pd.Timestamp):
        return _ensure_utc(value)

    raise TypeError(f"不支持的时间输入类型: {type(value)}")


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """确保 Timestamp 带 UTC 时区"""
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_unix_milliseconds(value: TimeInput) -> int:
    """将任意时间输入转换为 Unix 毫秒时间戳（整数）

    常用于构造交易所 API 请求参数。
    """
    ts = to_utc_timestamp(value)
    return int(ts.timestamp() * 1000)


def now_utc() -> pd.Timestamp:
    """获取当前 UTC 时间"""
    return pd.Timestamp.now(tz="UTC")
