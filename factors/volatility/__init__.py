"""波动率因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor


def _window(params: dict[str, Any] | None, default: int = 20) -> int:
    value = int((params or {}).get("window", default))
    if value < 2:
        raise ValueError("window 必须至少为 2")
    return value


@register_factor(
    "realized_volatility",
    category="波动率",
    description="对数收益的历史滚动标准差",
    default_params={"window": 20},
)
def make_realized_volatility(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        returns = np.log(df["close"]).diff()
        return returns.rolling(window, min_periods=window).std().rename("realized_volatility")

    return factor


@register_factor(
    "range_volatility",
    category="波动率",
    description="历史滚动平均的高低价对数振幅",
    default_params={"window": 20},
)
def make_range_volatility(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        log_range = np.log(df["high"].div(df["low"]))
        return log_range.rolling(window, min_periods=window).mean().rename("range_volatility")

    return factor


@register_factor(
    "downside_volatility",
    category="波动率",
    description="负对数收益的历史滚动波动率",
    default_params={"window": 20},
)
def make_downside_volatility(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        returns = np.log(df["close"]).diff().clip(upper=0)
        return returns.rolling(window, min_periods=window).std().rename("downside_volatility")

    return factor
