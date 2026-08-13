"""Cross-sectional factor evaluation for canonical multi-symbol panels."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from factors.panel import (
    SYMBOL_LEVEL,
    TIME_LEVEL,
    cs_rank,
    cs_scale,
    panel_coverage,
    validate_panel,
)


def panel_forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Align each asset's t-to-t+h return to timestamp t."""
    validate_panel(close)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    output = pd.Series(np.nan, index=close.index, name=f"forward_return_{horizon}", dtype=float)
    for _, group in close.astype(float).groupby(level=SYMBOL_LEVEL, sort=False):
        output.loc[group.index] = (group.shift(-horizon) / group - 1.0).to_numpy(dtype=float)
    return output


def cross_sectional_ic_series(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Compute Pearson IC and Spearman RankIC independently per timestamp."""
    validate_panel(factor_values)
    validate_panel(forward_returns)
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")
    aligned = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("forward_return")], axis=1
    )
    rows: list[dict[str, Any]] = []
    for timestamp, group in aligned.groupby(level=TIME_LEVEL, sort=True):
        clean = group.dropna()
        ic = float("nan")
        rank_ic = float("nan")
        if len(clean) >= min_assets:
            ic = float(clean["factor"].corr(clean["forward_return"]))
            rank_ic = float(
                clean["factor"].rank().corr(clean["forward_return"].rank())
            )
        rows.append(
            {
                "timestamp": timestamp,
                "n_assets": int(len(clean)),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


def panel_positions(factor_values: pd.Series) -> pd.Series:
    """Build same-time market-neutral rank weights with unit gross exposure."""
    ranks = cs_rank(factor_values)
    centered = ranks - ranks.groupby(level=TIME_LEVEL, sort=False).transform("mean")
    return cs_scale(centered).rename("position")


def panel_strategy_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Calculate gross/net long-short returns and one-way turnover by timestamp."""
    if fee_rate < 0 or slippage < 0:
        raise ValueError("cost assumptions cannot be negative")
    positions = panel_positions(factor_values)
    aligned = pd.concat(
        [positions, forward_returns.rename("forward_return")], axis=1
    ).dropna()
    gross = (aligned["position"] * aligned["forward_return"]).groupby(
        level=TIME_LEVEL
    ).sum()
    wide = positions.unstack(SYMBOL_LEVEL).fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1).mul(0.5).fillna(0.0)
    result = pd.DataFrame({"gross_return": gross, "turnover": turnover.reindex(gross.index)})
    result["net_return"] = result["gross_return"] - result["turnover"] * (
        fee_rate + slippage
    )
    return result


def panel_group_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Equal-weight same-time quantile returns and top-minus-bottom series."""
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")
    aligned = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("forward_return")], axis=1
    )
    records: list[dict[str, Any]] = []
    spreads: dict[pd.Timestamp, float] = {}
    for timestamp, group in aligned.groupby(level=TIME_LEVEL, sort=True):
        clean = group.dropna().copy()
        groups = min(n_groups, len(clean))
        if groups < 2:
            continue
        clean["group"] = pd.qcut(
            clean["factor"].rank(method="first"), q=groups, labels=False
        ).astype(int) + 1
        means = clean.groupby("group")["forward_return"].mean()
        spreads[pd.Timestamp(timestamp)] = float(means.iloc[-1] - means.iloc[0])
        for group_number, mean_return in means.items():
            records.append(
                {
                    "timestamp": timestamp,
                    "group": int(group_number),
                    "mean_return": float(mean_return),
                    "count": int((clean["group"] == group_number).sum()),
                }
            )
    detail = pd.DataFrame(records)
    if detail.empty:
        summary = pd.DataFrame(columns=["group", "mean_return", "count"])
    else:
        summary = (
            detail.groupby("group", as_index=False)
            .agg(mean_return=("mean_return", "mean"), count=("count", "sum"))
            .sort_values("group")
        )
    spread_series = pd.Series(spreads, name="top_bottom_spread", dtype=float).sort_index()
    return summary, spread_series


def panel_ic_decay(
    factor_values: pd.Series,
    close: pd.Series,
    *,
    horizons: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    min_assets: int = 3,
) -> pd.DataFrame:
    """Mean cross-sectional IC and RankIC over explicit future horizons."""
    rows = []
    for horizon in horizons:
        ic_series = cross_sectional_ic_series(
            factor_values,
            panel_forward_returns(close, horizon),
            min_assets=min_assets,
        )
        rows.append(
            {
                "horizon": horizon,
                "mean_ic": float(ic_series["ic"].mean()),
                "mean_rank_ic": float(ic_series["rank_ic"].mean()),
                "n_periods": int(ic_series["rank_ic"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def check_panel_truncation_invariance(
    factor: Callable[[pd.DataFrame], pd.Series],
    panel: pd.DataFrame,
    *,
    sample_positions: tuple[int, ...] | None = None,
) -> bool:
    """Recompute selected timestamps after removing every later cross section."""
    validate_panel(panel)
    timestamps = panel.index.get_level_values(TIME_LEVEL).unique()
    if len(timestamps) == 0:
        raise ValueError("panel cannot be empty")
    positions = sample_positions or tuple(
        sorted({len(timestamps) // 3, 2 * len(timestamps) // 3, len(timestamps) - 1})
    )
    full = factor(panel)
    for position in positions:
        if position < 0 or position >= len(timestamps):
            raise ValueError("sample_positions out of bounds")
        timestamp = timestamps[position]
        truncated = panel.loc[panel.index.get_level_values(TIME_LEVEL) <= timestamp]
        expected = full.xs(timestamp, level=TIME_LEVEL).sort_index()
        actual = factor(truncated).xs(timestamp, level=TIME_LEVEL).sort_index()
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
            return False
    return True


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _normal_p_value(values: pd.Series) -> float | None:
    clean = values.dropna()
    if len(clean) < 3:
        return None
    standard_error = clean.std(ddof=1) / math.sqrt(len(clean))
    if not np.isfinite(standard_error) or standard_error <= 0:
        return None
    statistic = abs(float(clean.mean() / standard_error))
    return float(math.erfc(statistic / math.sqrt(2.0)))


def build_panel_factor_report(
    factor_name: str,
    factor_values: pd.Series,
    close: pd.Series,
    *,
    interval: str,
    metadata: dict[str, Any],
    lookahead_status: str,
    horizon: int = 1,
    min_assets: int = 3,
    rolling_window: int = 20,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
) -> dict[str, Any]:
    """Build JSON-ready cross-sectional diagnostics without overstating evidence."""
    forward = panel_forward_returns(close, horizon)
    ic_series = cross_sectional_ic_series(
        factor_values, forward, min_assets=min_assets
    )
    valid_ic = ic_series["rank_ic"].dropna()
    if len(valid_ic) < 3:
        return {
            "factor_name": factor_name,
            "scope": "cross_sectional",
            "status": "insufficient_data",
            "reason": "fewer than three valid cross-sectional IC periods",
            "interval": interval,
            "metrics": {
                "ic": None,
                "rank_ic": None,
                "icir": None,
                "turnover": None,
            },
            "provenance": metadata,
            "lookahead_status": lookahead_status,
            "research_note": "Insufficient real panel data; no alpha conclusion is available.",
        }

    strategy = panel_strategy_returns(
        factor_values,
        forward,
        fee_rate=fee_rate,
        slippage=slippage,
    )
    groups, spread = panel_group_returns(factor_values, forward)
    decay = panel_ic_decay(factor_values, close, min_assets=min_assets)
    rolling = valid_ic.rolling(min(rolling_window, len(valid_ic)), min_periods=3).mean()
    rank_std = valid_ic.std(ddof=1)
    icir = valid_ic.mean() / rank_std if rank_std > 0 else float("nan")
    coverage = panel_coverage(factor_values.to_frame("factor"))
    cost_grid = []
    for total_cost in (0.0, 0.0005, 0.001, 0.002, 0.005):
        net = strategy["gross_return"] - strategy["turnover"] * total_cost
        cost_grid.append(
            {
                "total_one_way_cost": total_cost,
                "mean_net_return": _number(net.mean()),
                "cumulative_net_return": _number((1 + net).prod() - 1),
            }
        )
    return {
        "factor_name": factor_name,
        "scope": "cross_sectional",
        "status": "computed_short_sample",
        "interval": interval,
        "sample": {
            "start": str(valid_ic.index.min()),
            "end": str(valid_ic.index.max()),
            "n_periods": int(len(valid_ic)),
            "median_assets": _number(ic_series["n_assets"].median()),
            "mean_coverage": _number(coverage["coverage_ratio"].mean()),
        },
        "metrics": {
            "ic": _number(ic_series["ic"].mean()),
            "rank_ic": _number(valid_ic.mean()),
            "icir": _number(icir),
            "turnover": _number(strategy["turnover"].mean()),
            "rank_ic_p_value": _normal_p_value(valid_ic),
            "mean_gross_return": _number(strategy["gross_return"].mean()),
            "mean_net_return": _number(strategy["net_return"].mean()),
            "top_bottom_mean": _number(spread.mean()),
        },
        "provenance": metadata,
        "ic_series": [
            {
                "time": str(index),
                "n_assets": int(row["n_assets"]),
                "ic": _number(row["ic"]),
                "rank_ic": _number(row["rank_ic"]),
            }
            for index, row in ic_series.iterrows()
        ],
        "rolling_ic": [
            {"time": str(index), "value": _number(value)}
            for index, value in rolling.dropna().items()
        ],
        "ic_decay": [
            {
                "horizon": int(row["horizon"]),
                "ic": _number(row["mean_ic"]),
                "rank_ic": _number(row["mean_rank_ic"]),
                "n_periods": int(row["n_periods"]),
            }
            for _, row in decay.iterrows()
        ],
        "group_returns": [
            {
                "group": int(row["group"]),
                "mean_return": _number(row["mean_return"]),
                "count": int(row["count"]),
            }
            for _, row in groups.iterrows()
        ],
        "cost_sensitivity": cost_grid,
        "lookahead_status": lookahead_status,
        "cost_assumptions": {
            "fee_rate": fee_rate,
            "slippage": slippage,
            "units": "one-way",
        },
        "research_note": (
            "Current-sample candidate diagnostics only; not six-month OOS and not "
            "evidence of validated alpha."
        ),
    }
