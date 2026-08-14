"""Microsoft Qlib Alpha360-compatible lagged raw-price/volume features.

Qlib Alpha360 consists of 60 lags for each of close, open, high, low,
VWAP and volume.  The formulas are implemented locally and use only current
or past bars.  Binance bar VWAP is quote_volume / volume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.base import validate_factor_input
from factors.registry import register_factor

QLIB_ALPHA360_URL = (
    "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"
)
ALPHA360_FIELDS = ("close", "open", "high", "low", "vwap", "volume")
ALPHA360_LAGS = tuple(range(60))


def _vwap(df: pd.DataFrame) -> pd.Series:
    if "vwap" in df.columns:
        return pd.to_numeric(df["vwap"], errors="coerce")
    if "quote_volume" not in df.columns:
        raise ValueError("Qlib Alpha360 VWAP 特征需要 vwap 或 quote_volume 列")
    quote_volume = pd.to_numeric(df["quote_volume"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    return quote_volume.div(volume.replace(0, np.nan))


def _source(df: pd.DataFrame, field: str) -> pd.Series:
    return _vwap(df) if field == "vwap" else pd.to_numeric(df[field], errors="coerce")


def _alpha360_factory(name: str, field: str, lag: int):
    def factory(params: dict[str, Any] | None = None):
        del params

        def factor(df: pd.DataFrame) -> pd.Series:
            validate_factor_input(df)
            values = _source(df, field)
            denominator = df["volume"] if field == "volume" else df["close"]
            result = values.shift(lag).div(denominator.replace(0, np.nan))
            return result.replace([np.inf, -np.inf], np.nan).rename(name)

        return factor

    return factory


for _field in ALPHA360_FIELDS:
    for _lag in ALPHA360_LAGS:
        _name = f"qlib360_{_field}_{_lag}"
        _dependencies = ["open", "high", "low", "close", "volume"]
        if _field == "vwap":
            _dependencies.append("quote_volume")
        register_factor(
            _name,
            category="Qlib Alpha360",
            description=(
                f"Qlib Alpha360 {_field.upper()} lag {_lag}; "
                "price-like fields are scaled by current close and volume by current volume"
            ),
            default_params={},
            source="Microsoft Qlib Alpha360",
            source_url=QLIB_ALPHA360_URL,
            scope="time_series",
            data_dependencies=tuple(_dependencies),
        )(_alpha360_factory(_name, _field, _lag))
