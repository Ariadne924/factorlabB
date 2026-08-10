"""数字资产特有因子：资金费率、持仓量与永续基差。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor


def _window(params: dict[str, Any] | None, default: int) -> int:
    value = int((params or {}).get("window", default))
    if value < 1:
        raise ValueError("window 必须大于 0")
    return value


def _required(df: pd.DataFrame, column: str, factor_name: str) -> pd.Series:
    validate_factor_input(df)
    if column not in df.columns:
        raise ValueError(f"{factor_name} 需要 {column} 列")
    return pd.to_numeric(df[column], errors="coerce")


@register_factor(
    "funding_rate",
    category="加密货币特有",
    description="最近 window 次已公布资金费率的历史均值",
    default_params={"window": 3},
    data_dependencies=("open", "high", "low", "close", "volume", "funding_rate"),
)
def make_funding_rate(params: dict[str, Any] | None = None):
    window = _window(params, 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _required(df, "funding_rate", "funding_rate")
        return values.rolling(window, min_periods=window).mean().rename("funding_rate")

    return factor


@register_factor(
    "open_interest_change",
    category="加密货币特有",
    description="持仓量相对 window 根 K 线前的变化率",
    default_params={"window": 12},
    data_dependencies=("open", "high", "low", "close", "volume", "open_interest"),
)
def make_open_interest_change(params: dict[str, Any] | None = None):
    window = _window(params, 12)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _required(df, "open_interest", "open_interest_change")
        return (
            values.pct_change(window)
            .replace([np.inf, -np.inf], np.nan)
            .rename("open_interest_change")
        )

    return factor


@register_factor(
    "basis",
    category="加密货币特有",
    description="最近 window 个已观测永续基差的历史均值",
    default_params={"window": 3},
    data_dependencies=("open", "high", "low", "close", "volume", "basis"),
)
def make_basis(params: dict[str, Any] | None = None):
    window = _window(params, 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _required(df, "basis", "basis")
        return values.rolling(window, min_periods=window).mean().rename("basis")

    return factor
