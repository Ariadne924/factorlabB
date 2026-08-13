"""Point-in-time construction of the LTW cryptocurrency factor returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.panel_analysis import panel_forward_returns
from factors.panel import SYMBOL_LEVEL, TIME_LEVEL, validate_panel


def _per_symbol_return(close: pd.Series, periods: int) -> pd.Series:
    output = pd.Series(np.nan, index=close.index, dtype=float)
    for _, group in close.groupby(level=SYMBOL_LEVEL, sort=False):
        output.loc[group.index] = (group / group.shift(periods) - 1.0).to_numpy(dtype=float)
    return output


def _weighted_return(group: pd.DataFrame) -> float:
    clean = group.dropna(subset=["forward_return", "market_cap"])
    clean = clean.loc[clean["market_cap"] > 0]
    total = clean["market_cap"].sum()
    if clean.empty or total <= 0:
        return float("nan")
    return float((clean["forward_return"] * clean["market_cap"]).sum() / total)


def build_ltw_factor_returns(
    panel: pd.DataFrame,
    *,
    momentum_lookback: int,
    forward_horizon: int = 1,
    risk_free: pd.Series | None = None,
) -> pd.DataFrame:
    """Build CMKT, CSMB, and CMOM using point-in-time market capitalization.

    The paper uses weekly observations and three-week momentum. Callers must map
    those horizons to bars explicitly for their selected frequency.
    """
    validate_panel(panel)
    if "market_cap" not in panel.columns:
        raise ValueError("LTW factor construction requires point-in-time market_cap")
    if momentum_lookback < 1 or forward_horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    close = panel["close"].astype(float)
    data = pd.DataFrame(
        {
            "forward_return": panel_forward_returns(close, forward_horizon),
            "market_cap": panel["market_cap"].astype(float),
            "momentum": _per_symbol_return(close, momentum_lookback),
        }
    )
    records: list[dict[str, Any]] = []
    for timestamp, group in data.groupby(level=TIME_LEVEL, sort=True):
        clean = group.dropna()
        if len(clean) < 4:
            records.append(
                {
                    "timestamp": timestamp,
                    "cmkt": np.nan,
                    "csmb": np.nan,
                    "cmom": np.nan,
                }
            )
            continue
        size_rank = clean["market_cap"].rank(pct=True, method="average")
        momentum_rank = clean["momentum"].rank(pct=True, method="average")
        small = _weighted_return(clean.loc[size_rank <= 0.3])
        big = _weighted_return(clean.loc[size_rank > 0.7])
        low_momentum = _weighted_return(clean.loc[momentum_rank <= 0.3])
        high_momentum = _weighted_return(clean.loc[momentum_rank > 0.7])
        records.append(
            {
                "timestamp": timestamp,
                "cmkt": _weighted_return(clean),
                "csmb": small - big,
                "cmom": high_momentum - low_momentum,
            }
        )
    result = pd.DataFrame(records).set_index("timestamp")
    if risk_free is not None:
        result["cmkt"] = result["cmkt"] - risk_free.reindex(result.index).fillna(0.0)
    return result
