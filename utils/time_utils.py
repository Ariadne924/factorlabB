"""
时间工具模块

统一的时间戳转换工具，所有时间均转换为 UTC。
支持多种输入格式：datetime、字符串、Unix 时间戳（秒/毫秒）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd

# 支持的输入时间类型
type TimeInput = str | int | float | datetime | pd.Timestamp


def to_utc_timestamp(
    value: TimeInput,
    *,
    numeric_unit: Literal["s", "ms"] | None = None,
) -> pd.Timestamp:
    """将任意输入时间转换为 UTC 时间戳（pd.Timestamp）

    支持的输入格式：
        - str: '2024-01-01'、'2024-01-01 12:00:00'、ISO 8601 格式
        - int/float: Unix 时间戳，必须通过 numeric_unit 指定秒或毫秒
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

        >>> to_utc_timestamp(1704067200, numeric_unit="s")
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')

        >>> to_utc_timestamp(1704067200000, numeric_unit="ms")
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
        if numeric_unit is None:
            raise ValueError(
                "数值时间戳必须显式指定 numeric_unit='s' 或 numeric_unit='ms'"
            )
        ts = pd.Timestamp(value, unit=numeric_unit)
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


def to_unix_milliseconds(
    value: TimeInput,
    *,
    numeric_unit: Literal["s", "ms"] | None = None,
) -> int:
    """将任意时间输入转换为 Unix 毫秒时间戳（整数）

    常用于构造交易所 API 请求参数。
    """
    ts = to_utc_timestamp(value, numeric_unit=numeric_unit)
    return int(ts.timestamp() * 1000)


def now_utc() -> pd.Timestamp:
    """获取当前 UTC 时间"""
    return pd.Timestamp.now(tz="UTC")
