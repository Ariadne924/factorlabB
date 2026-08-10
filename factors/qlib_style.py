"""Past-only OHLCV factors adapted from Microsoft Qlib's Alpha158 feature set.

The formulas are implemented locally instead of importing Qlib.  Cross-sectional
operators and industry neutralisation are intentionally excluded because this
project currently evaluates one symbol/time series at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor

QLIB_ALPHA158_URL = "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"
QLIB_METADATA: dict[str, Any] = {
    "source": "Microsoft Qlib Alpha158",
    "source_url": QLIB_ALPHA158_URL,
    "scope": "time_series",
    "data_dependencies": ("open", "high", "low", "close", "volume"),
}
EPSILON = 1e-12


def _window(params: dict[str, Any] | None, default: int = 24) -> int:
    value = int((params or {}).get("window", default))
    if value < 2:
        raise ValueError("window must be at least 2")
    return value


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


def _intrabar_factory(name: str, numerator: str, denominator: str):
    def factory(params: dict[str, Any] | None = None):
        del params

        def factor(df: pd.DataFrame) -> pd.Series:
            validate_factor_input(df)
            open_, high, low, close = (df[column] for column in ("open", "high", "low", "close"))
            numerators = {
                "body": close - open_,
                "range": high - low,
                "upper": high - pd.concat([open_, close], axis=1).max(axis=1),
                "lower": pd.concat([open_, close], axis=1).min(axis=1) - low,
                "shift": 2 * close - high - low,
            }
            divisor = open_.replace(0, np.nan) if denominator == "open" else _range(df)
            return numerators[numerator].div(divisor).rename(name)

        return factor

    return factory


_INTRABAR_FACTORS = (
    ("k_mid", "body", "open", "K-line body divided by open"),
    ("k_length", "range", "open", "K-line range divided by open"),
    ("k_mid2", "body", "range", "K-line body divided by high-low range"),
    ("upper_shadow", "upper", "open", "Upper shadow divided by open"),
    ("upper_shadow2", "upper", "range", "Upper shadow divided by high-low range"),
    ("lower_shadow", "lower", "open", "Lower shadow divided by open"),
    ("lower_shadow2", "lower", "range", "Lower shadow divided by high-low range"),
    ("k_shift", "shift", "open", "Close location shift divided by open"),
    ("k_shift2", "shift", "range", "Close location shift divided by high-low range"),
)

for _name, _numerator, _denominator, _description in _INTRABAR_FACTORS:
    register_factor(
        _name,
        category="形态",
        description=_description,
        default_params={},
        **QLIB_METADATA,
    )(_intrabar_factory(_name, _numerator, _denominator))


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _r_squared(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    fitted = np.polyval(np.polyfit(x, values, 1), x)
    total = np.square(values - values.mean()).sum()
    if total <= EPSILON:
        return np.nan
    return float(1 - np.square(values - fitted).sum() / total)


def _last_residual(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    fitted_last = np.polyval(np.polyfit(x, values, 1), x[-1])
    return float(values[-1] - fitted_last)


@register_factor(
    "linear_trend_slope",
    category="趋势",
    description="Rolling close-price linear-regression slope divided by current close",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_linear_trend_slope(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        slope = df["close"].rolling(window, min_periods=window).apply(_slope, raw=True)
        return slope.div(df["close"].replace(0, np.nan)).rename("linear_trend_slope")

    return factor


@register_factor(
    "trend_r_squared",
    category="趋势",
    description="R-squared of the rolling close-price linear trend",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_trend_r_squared(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        return (
            df["close"]
            .rolling(window, min_periods=window)
            .apply(_r_squared, raw=True)
            .rename("trend_r_squared")
        )

    return factor


@register_factor(
    "trend_residual",
    category="趋势",
    description="Current residual from the rolling close-price trend, scaled by close",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_trend_residual(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        residual = df["close"].rolling(window, min_periods=window).apply(_last_residual, raw=True)
        return residual.div(df["close"].replace(0, np.nan)).rename("trend_residual")

    return factor


@register_factor(
    "price_position_rsv",
    category="趋势",
    description="Close position within the rolling high-low range",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_price_position_rsv(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        low = df["low"].rolling(window, min_periods=window).min()
        high = df["high"].rolling(window, min_periods=window).max()
        return df["close"].sub(low).div((high - low).replace(0, np.nan)).rename(
            "price_position_rsv"
        )

    return factor


def _recency_factory(name: str, column: str, find_maximum: bool):
    def factory(params: dict[str, Any] | None = None):
        window = _window(params)

        def recency(values: np.ndarray) -> float:
            position = np.argmax(values) if find_maximum else np.argmin(values)
            return float(position / (len(values) - 1))

        def factor(df: pd.DataFrame) -> pd.Series:
            validate_factor_input(df)
            return (
                df[column]
                .rolling(window, min_periods=window)
                .apply(recency, raw=True)
                .rename(name)
            )

        return factor

    return factory


register_factor(
    "high_recency",
    category="趋势",
    description="Recency of the rolling high; 1 means the newest bar",
    default_params={"window": 24},
    **QLIB_METADATA,
)(_recency_factory("high_recency", "high", True))

register_factor(
    "low_recency",
    category="趋势",
    description="Recency of the rolling low; 1 means the newest bar",
    default_params={"window": 24},
    **QLIB_METADATA,
)(_recency_factory("low_recency", "low", False))


@register_factor(
    "high_low_recency_diff",
    category="趋势",
    description="Rolling high recency minus rolling low recency",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_high_low_recency_diff(params: dict[str, Any] | None = None):
    high_factor = _recency_factory("high", "high", True)(params)
    low_factor = _recency_factory("low", "low", False)(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        return high_factor(df).sub(low_factor(df)).rename("high_low_recency_diff")

    return factor


@register_factor(
    "price_volume_correlation",
    category="量价",
    description="Rolling correlation between close and log volume",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_price_volume_correlation(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        log_volume = np.log1p(df["volume"].clip(lower=0))
        return df["close"].rolling(window, min_periods=window).corr(log_volume).rename(
            "price_volume_correlation"
        )

    return factor


@register_factor(
    "return_volume_correlation",
    category="量价",
    description="Rolling correlation between close return and log-volume change",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_return_volume_correlation(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        returns = df["close"].pct_change(fill_method=None)
        volume_change = np.log1p(df["volume"].clip(lower=0)).diff()
        return returns.rolling(window, min_periods=window).corr(volume_change).rename(
            "return_volume_correlation"
        )

    return factor


@register_factor(
    "volume_weighted_volatility",
    category="量价",
    description="Dispersion of return-volume impact divided by its mean absolute impact",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_volume_weighted_volatility(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        impact = df["close"].pct_change(fill_method=None) * df["volume"]
        numerator = impact.rolling(window, min_periods=window).std(ddof=0)
        denominator = impact.abs().rolling(window, min_periods=window).mean().replace(0, np.nan)
        return numerator.div(denominator).rename("volume_weighted_volatility")

    return factor


@register_factor(
    "volume_change_strength",
    category="量价",
    description="Signed rolling volume-change balance divided by absolute volume change",
    default_params={"window": 24},
    **QLIB_METADATA,
)
def make_volume_change_strength(params: dict[str, Any] | None = None):
    window = _window(params)

    def factor(df: pd.DataFrame) -> pd.Series:
        validate_factor_input(df)
        change = df["volume"].diff()
        signed = change.rolling(window, min_periods=window).sum()
        absolute = change.abs().rolling(window, min_periods=window).sum().replace(0, np.nan)
        return signed.div(absolute).rename("volume_change_strength")

    return factor

