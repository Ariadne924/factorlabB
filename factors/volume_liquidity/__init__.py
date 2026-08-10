"""成交量与流动性因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor


def _window(params: dict[str, Any] | None, default: int = 20) -> int:
    value = int((params or {}).get("window", default))
    if value < 1:
        raise ValueError("window 必须大于 0")
    return value


@register_factor(
    "volume_momentum",
    category="成交量与流动性",
    description="成交量相对历史均值的偏离",
    default_params={"window": 20},
)
def make_volume_momentum(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        average = df["volume"].rolling(window, min_periods=window).mean()
        return df["volume"].div(average.replace(0, np.nan)).sub(1).rename("volume_momentum")

    return factor


@register_factor(
    "amihud_illiquidity",
    category="成交量与流动性",
    description="绝对收益除以成交额的历史滚动均值",
    default_params={"window": 20},
)
def make_amihud_illiquidity(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        quote_volume = df.get("quote_volume", df["close"] * df["volume"])
        raw = df["close"].pct_change().abs().div(quote_volume.replace(0, np.nan))
        return raw.rolling(window, min_periods=window).mean().rename("amihud_illiquidity")

    return factor


@register_factor(
    "taker_buy_ratio",
    category="成交量与流动性",
    description="主动买入量占成交量的历史滚动均值",
    default_params={"window": 12},
)
def make_taker_buy_ratio(params: dict[str, Any] | None = None):
    window = _window(params, 12)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        if "taker_buy_volume" not in df.columns:
            raise ValueError("taker_buy_ratio 需要 taker_buy_volume 列")
        ratio = df["taker_buy_volume"].div(df["volume"].replace(0, np.nan))
        return ratio.rolling(window, min_periods=window).mean().rename("taker_buy_ratio")

    return factor
