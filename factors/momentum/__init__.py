"""动量与反转因子。所有 rolling 计算只使用当前及历史行。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor


def _window(params: dict[str, Any] | None, default: int) -> int:
    value = int((params or {}).get("window", default))
    if value < 1:
        raise ValueError("window 必须大于 0")
    return value


@register_factor(
    "return_momentum",
    category="动量",
    description="过去 window 根 K 线的收盘价收益率",
    default_params={"window": 12},
)
def make_return_momentum(params: dict[str, Any] | None = None):
    window = _window(params, 12)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        return df["close"].pct_change(window).rename("return_momentum")

    return factor


@register_factor(
    "short_term_reversal",
    category="动量",
    description="短周期历史收益率的相反数",
    default_params={"window": 3},
)
def make_short_term_reversal(params: dict[str, Any] | None = None):
    window = _window(params, 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        return (-df["close"].pct_change(window)).rename("short_term_reversal")

    return factor


@register_factor(
    "moving_average_gap",
    category="动量",
    description="收盘价相对历史移动均线的偏离",
    default_params={"window": 20},
)
def make_moving_average_gap(params: dict[str, Any] | None = None):
    window = _window(params, 20)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        average = df["close"].rolling(window, min_periods=window).mean()
        return df["close"].div(average).sub(1).rename("moving_average_gap")

    return factor


@register_factor(
    "rsi", category="动量", description="相对强弱指标（0-100）", default_params={"window": 14}
)
def make_rsi(params: dict[str, Any] | None = None):
    window = _window(params, 14)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
        relative_strength = gain.div(loss.replace(0, float("nan")))
        return (100 - 100 / (1 + relative_strength)).rename("rsi")

    return factor
