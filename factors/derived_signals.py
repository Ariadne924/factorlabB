"""数字资产结构、市场状态与交互候选因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor


def _positive(params: dict[str, Any] | None, key: str, default: int) -> int:
    value = int((params or {}).get(key, default))
    if value < 1:
        raise ValueError(f"{key} 必须大于 0")
    return value


def _windows(
    params: dict[str, Any] | None, short_default: int = 6, long_default: int = 24
) -> tuple[int, int]:
    short = _positive(params, "short_window", short_default)
    long = _positive(params, "long_window", long_default)
    if short >= long:
        raise ValueError("short_window 必须小于 long_window")
    return short, long


def _column(df: pd.DataFrame, name: str, factor_name: str) -> pd.Series:
    validate_factor_input(df)
    if name not in df.columns:
        raise ValueError(f"{factor_name} 需要 {name} 列")
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=0)
    return values.sub(mean).div(std.replace(0, np.nan))


def _returns(df: pd.DataFrame) -> pd.Series:
    validate_factor_input(df)
    return pd.to_numeric(df["close"], errors="coerce").pct_change(fill_method=None)


@register_factor(
    "funding_rate_zscore",
    category="加密货币特有",
    description="已公布资金费率相对历史窗口的标准化偏离",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "funding_rate"),
)
def make_funding_rate_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        return _zscore(_column(df, "funding_rate", "funding_rate_zscore"), window).rename(
            "funding_rate_zscore"
        )

    return factor

@register_factor(
    "funding_rate_change",
    category="加密货币特有",
    description="资金费率相对历史窗口前的变化",
    default_params={"window": 3},
    data_dependencies=("open", "high", "low", "close", "volume", "funding_rate"),
)
def make_funding_rate_change(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _column(df, "funding_rate", "funding_rate_change")
        return values.diff(window).rename("funding_rate_change")

    return factor


@register_factor(
    "funding_rate_acceleration",
    category="加密货币特有",
    description="资金费率变化速度的二阶差分",
    default_params={"window": 3},
    data_dependencies=("open", "high", "low", "close", "volume", "funding_rate"),
)
def make_funding_rate_acceleration(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _column(df, "funding_rate", "funding_rate_acceleration")
        return values.diff(window).diff(window).rename("funding_rate_acceleration")

    return factor


@register_factor(
    "funding_price_divergence",
    category="数字资产交互",
    description="资金费率拥挤度与同窗口价格动量之间的标准化差异",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "funding_rate"),
)
def make_funding_price_divergence(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        funding = _column(df, "funding_rate", "funding_price_divergence")
        momentum = df["close"].pct_change(window, fill_method=None)
        return (_zscore(funding, window) - _zscore(momentum, window)).rename(
            "funding_price_divergence"
        )

    return factor


@register_factor(
    "open_interest_zscore",
    category="加密货币特有",
    description="持仓量相对历史窗口的标准化偏离",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "open_interest"),
)
def make_open_interest_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        return _zscore(_column(df, "open_interest", "open_interest_zscore"), window).rename(
            "open_interest_zscore"
        )

    return factor


@register_factor(
    "open_interest_momentum",
    category="加密货币特有",
    description="短期与长期持仓量变化率之差",
    default_params={"short_window": 6, "long_window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "open_interest"),
)
def make_open_interest_momentum(params: dict[str, Any] | None = None):
    short, long = _windows(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        values = _column(df, "open_interest", "open_interest_momentum")
        short_change = values.pct_change(short, fill_method=None)
        long_change = values.pct_change(long, fill_method=None)
        return short_change.sub(long_change).rename("open_interest_momentum")

    return factor


@register_factor(
    "open_interest_price_divergence",
    category="数字资产交互",
    description="持仓量变化与价格变化之间的标准化背离",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "open_interest"),
)
def make_open_interest_price_divergence(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        oi = _column(df, "open_interest", "open_interest_price_divergence")
        oi_change = oi.pct_change(window, fill_method=None)
        price_change = df["close"].pct_change(window, fill_method=None)
        return (_zscore(oi_change, window) - _zscore(price_change, window)).rename(
            "open_interest_price_divergence"
        )

    return factor


@register_factor(
    "open_interest_volume_confirmation",
    category="数字资产交互",
    description="持仓量变化与成交量异常的联合确认",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "open_interest"),
)
def make_open_interest_volume_confirmation(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        oi = _column(df, "open_interest", "open_interest_volume_confirmation")
        oi_change = oi.pct_change(window, fill_method=None)
        volume_shock = _zscore(np.log1p(df["volume"]), window)
        return oi_change.mul(volume_shock).rename("open_interest_volume_confirmation")

    return factor


@register_factor(
    "basis_zscore",
    category="加密货币特有",
    description="已结束统计周期基差相对历史窗口的标准化偏离",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "basis"),
)
def make_basis_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        return _zscore(_column(df, "basis", "basis_zscore"), window).rename("basis_zscore")

    return factor


@register_factor(
    "basis_change",
    category="加密货币特有",
    description="基差相对历史窗口前的变化",
    default_params={"window": 3},
    data_dependencies=("open", "high", "low", "close", "volume", "basis"),
)
def make_basis_change(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 3)

    def factor(df: pd.DataFrame) -> pd.Series:
        return _column(df, "basis", "basis_change").diff(window).rename("basis_change")

    return factor


@register_factor(
    "basis_funding_spread",
    category="数字资产交互",
    description="标准化基差与标准化资金费率之间的价差信号",
    default_params={"window": 24},
    data_dependencies=(
        "open",
        "high",
        "low",
        "close",
        "volume",
        "basis",
        "funding_rate",
    ),
)
def make_basis_funding_spread(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        basis = _column(df, "basis", "basis_funding_spread")
        funding = _column(df, "funding_rate", "basis_funding_spread")
        return (_zscore(basis, window) - _zscore(funding, window)).rename(
            "basis_funding_spread"
        )

    return factor


@register_factor(
    "taker_imbalance",
    category="成交量与流动性",
    description="主动买入与主动卖出成交量的不平衡度",
    default_params={"window": 12},
    data_dependencies=("open", "high", "low", "close", "volume", "taker_buy_volume"),
)
def make_taker_imbalance(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 12)

    def factor(df: pd.DataFrame) -> pd.Series:
        taker_buy = _column(df, "taker_buy_volume", "taker_imbalance")
        imbalance = taker_buy.mul(2).div(df["volume"].replace(0, np.nan)).sub(1)
        return imbalance.rolling(window, min_periods=window).mean().rename("taker_imbalance")

    return factor


@register_factor(
    "taker_imbalance_momentum",
    category="成交量与流动性",
    description="主动成交不平衡度的短长窗口差",
    default_params={"short_window": 6, "long_window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "taker_buy_volume"),
)
def make_taker_imbalance_momentum(params: dict[str, Any] | None = None):
    short, long = _windows(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        taker_buy = _column(df, "taker_buy_volume", "taker_imbalance_momentum")
        imbalance = taker_buy.mul(2).div(df["volume"].replace(0, np.nan)).sub(1)
        return (
            imbalance.rolling(short, min_periods=short).mean()
            - imbalance.rolling(long, min_periods=long).mean()
        ).rename("taker_imbalance_momentum")

    return factor


@register_factor(
    "signed_price_impact",
    category="成交量与流动性",
    description="单位成交额对应的有符号价格变化，经历史窗口平滑",
    default_params={"window": 20},
    data_dependencies=("open", "high", "low", "close", "volume", "quote_volume"),
)
def make_signed_price_impact(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 20)

    def factor(df: pd.DataFrame) -> pd.Series:
        quote_volume = _column(df, "quote_volume", "signed_price_impact")
        raw = _returns(df).div(quote_volume.replace(0, np.nan))
        return raw.rolling(window, min_periods=window).mean().rename("signed_price_impact")

    return factor


@register_factor(
    "volume_shock_zscore",
    category="成交量与流动性",
    description="对数成交量相对历史窗口的标准化异常",
    default_params={"window": 24},
)
def make_volume_shock_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        return _zscore(np.log1p(df["volume"]), window).rename("volume_shock_zscore")

    return factor


@register_factor(
    "volatility_term_ratio",
    category="波动率",
    description="短窗口实现波动率与长窗口实现波动率的比值偏离",
    default_params={"short_window": 6, "long_window": 24},
)
def make_volatility_term_ratio(params: dict[str, Any] | None = None):
    short, long = _windows(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        returns = _returns(df)
        short_vol = returns.rolling(short, min_periods=short).std(ddof=0)
        long_vol = returns.rolling(long, min_periods=long).std(ddof=0)
        return short_vol.div(long_vol.replace(0, np.nan)).sub(1).rename(
            "volatility_term_ratio"
        )

    return factor


@register_factor(
    "momentum_volatility_interaction",
    category="状态交互",
    description="历史价格动量与短长波动率比值的交互",
    default_params={"short_window": 6, "long_window": 24},
)
def make_momentum_volatility_interaction(params: dict[str, Any] | None = None):
    short, long = _windows(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        returns = _returns(df)
        momentum = df["close"].pct_change(long, fill_method=None)
        short_vol = returns.rolling(short, min_periods=short).std(ddof=0)
        long_vol = returns.rolling(long, min_periods=long).std(ddof=0)
        ratio = short_vol.div(long_vol.replace(0, np.nan)).sub(1)
        return momentum.mul(ratio).rename("momentum_volatility_interaction")

    return factor


@register_factor(
    "momentum_liquidity_interaction",
    category="状态交互",
    description="历史价格动量与成交量冲击的交互",
    default_params={"window": 24},
)
def make_momentum_liquidity_interaction(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        momentum = df["close"].pct_change(window, fill_method=None)
        volume_shock = _zscore(np.log1p(df["volume"]), window)
        return momentum.mul(volume_shock).rename("momentum_liquidity_interaction")

    return factor


@register_factor(
    "trade_intensity_zscore",
    category="成交量与流动性",
    description="单位成交量对应成交笔数的历史标准分数",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "num_trades"),
)
def make_trade_intensity_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        trades = _column(df, "num_trades", "trade_intensity_zscore")
        intensity = trades.div(df["volume"].replace(0, np.nan))
        return _zscore(intensity, window).rename("trade_intensity_zscore")

    return factor


@register_factor(
    "average_trade_size_zscore",
    category="成交量与流动性",
    description="平均每笔成交额相对历史窗口的标准化偏离",
    default_params={"window": 24},
    data_dependencies=("open", "high", "low", "close", "volume", "quote_volume", "num_trades"),
)
def make_average_trade_size_zscore(params: dict[str, Any] | None = None):
    window = _positive(params, "window", 24)

    def factor(df: pd.DataFrame) -> pd.Series:
        quote_volume = _column(df, "quote_volume", "average_trade_size_zscore")
        trades = _column(df, "num_trades", "average_trade_size_zscore")
        average_size = quote_volume.div(trades.replace(0, np.nan))
        return _zscore(average_size, window).rename("average_trade_size_zscore")

    return factor
