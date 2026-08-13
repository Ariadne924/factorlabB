"""Robustness diagnostics for serialized single-asset strategy results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from evaluation.time_series_strategy import (
    TimeSeriesStrategyConfig,
    run_time_series_strategy,
)


def _returns_frame(result: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("returns", []))
    required = {"time", "gross_return", "turnover", "net_return"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"strategy result is missing return fields: {missing}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    frame = frame.set_index("time").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("strategy return timestamps must be unique")
    for column in required.difference({"time"}):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _path_metrics(returns: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "n_periods": 0,
            "total_return": None,
            "mean_bar_return": None,
            "bar_sharpe": None,
            "max_drawdown": None,
            "positive_bar_ratio": None,
        }
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = clean.std(ddof=1)
    sharpe = clean.mean() / std if std > 0 else np.nan
    return {
        "n_periods": int(len(clean)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mean_bar_return": float(clean.mean()),
        "bar_sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "max_drawdown": float(drawdown.min()),
        "positive_bar_ratio": float(clean.gt(0).mean()),
    }


def reprice_cost_grid(
    result: dict[str, Any],
    *,
    one_way_costs: Iterable[float] = (0.0, 0.0005, 0.001, 0.0015, 0.0025),
) -> list[dict[str, Any]]:
    """Reprice the identical gross path at alternative one-way costs."""
    frame = _returns_frame(result)
    rows: list[dict[str, Any]] = []
    for raw_cost in one_way_costs:
        cost = float(raw_cost)
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("one-way costs must be finite and non-negative")
        net = frame["gross_return"] - frame["turnover"] * cost
        metrics = _path_metrics(net)
        rows.append(
            {
                "one_way_cost": cost,
                **metrics,
                "mean_turnover": float(frame["turnover"].mean()),
            }
        )
    return rows


def monthly_performance(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate net and gross returns by UTC calendar month."""
    frame = _returns_frame(result)
    months = frame.index.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby(months):
        net_metrics = _path_metrics(group["net_return"])
        gross_metrics = _path_metrics(group["gross_return"])
        rows.append(
            {
                "month": str(month),
                "n_periods": int(len(group)),
                "gross_return": gross_metrics["total_return"],
                "net_return": net_metrics["total_return"],
                "bar_sharpe": net_metrics["bar_sharpe"],
                "max_drawdown": net_metrics["max_drawdown"],
                "mean_turnover": float(group["turnover"].mean()),
            }
        )
    return rows


def rolling_performance(
    result: dict[str, Any],
    *,
    window: int = 168,
) -> list[dict[str, Any]]:
    """Compute trailing return, raw bar Sharpe and drawdown for a fixed bar window."""
    if window < 5:
        raise ValueError("rolling window must be at least 5")
    frame = _returns_frame(result)
    net = frame["net_return"]
    rolling_return = (1.0 + net).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0
    rolling_mean = net.rolling(window, min_periods=window).mean()
    rolling_std = net.rolling(window, min_periods=window).std(ddof=1)
    rolling_sharpe = rolling_mean.div(rolling_std.where(rolling_std > 0))

    def max_drawdown(values: np.ndarray) -> float:
        equity = np.cumprod(1.0 + values)
        return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))

    rolling_drawdown = net.rolling(window, min_periods=window).apply(
        max_drawdown, raw=True
    )
    combined = pd.concat(
        [
            rolling_return.rename("rolling_return"),
            rolling_sharpe.rename("rolling_bar_sharpe"),
            rolling_drawdown.rename("rolling_max_drawdown"),
        ],
        axis=1,
    ).dropna()
    return [
        {
            "time": str(index),
            "window_bars": window,
            **{column: float(row[column]) for column in combined.columns},
        }
        for index, row in combined.iterrows()
    ]


def benchmark_comparison(result: dict[str, Any]) -> dict[str, Any]:
    """Compare the strategy with buy-and-hold over the identical aligned bars."""
    frame = _returns_frame(result)
    if "bar_return" not in frame.columns:
        return {"status": "unavailable", "reason": "bar_return missing from legacy result"}
    benchmark = _path_metrics(frame["bar_return"])
    strategy = _path_metrics(frame["net_return"])
    strategy_total = strategy["total_return"]
    benchmark_total = benchmark["total_return"]
    excess = (
        float(strategy_total - benchmark_total)
        if strategy_total is not None and benchmark_total is not None
        else None
    )
    return {
        "status": "computed",
        "strategy": strategy,
        "buy_and_hold": benchmark,
        "excess_total_return": excess,
        "same_aligned_period": True,
    }


def parameter_perturbation(
    frame: pd.DataFrame,
    config: TimeSeriesStrategyConfig,
    *,
    thresholds: Iterable[float],
    rebalance_values: Iterable[int],
    standardize_windows: Iterable[int],
) -> list[dict[str, Any]]:
    """Rerun a bounded grid around the chosen strategy configuration."""
    rows: list[dict[str, Any]] = []
    for threshold in dict.fromkeys(float(value) for value in thresholds):
        for rebalance in dict.fromkeys(int(value) for value in rebalance_values):
            for window in dict.fromkeys(int(value) for value in standardize_windows):
                varied = replace(
                    config,
                    score_threshold=threshold,
                    rebalance_every=rebalance,
                    standardize_window=window,
                )
                varied.validate()
                result = run_time_series_strategy(frame, varied)
                rows.append(
                    {
                        "score_threshold": threshold,
                        "rebalance_every": rebalance,
                        "standardize_window": window,
                        **result["metrics"],
                        "status": result["status"],
                    }
                )
    return rows


def build_strategy_robustness_report(
    result: dict[str, Any],
    *,
    rolling_window: int = 168,
    one_way_costs: Iterable[float] = (0.0, 0.0005, 0.001, 0.0015, 0.0025),
    perturbations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the serializable diagnostics used by reports and the front end."""
    monthly = monthly_performance(result)
    positive_months = [row for row in monthly if float(row["net_return"] or 0.0) > 0]
    cost_grid = reprice_cost_grid(result, one_way_costs=one_way_costs)
    return {
        "status": "computed" if result.get("returns") else "insufficient_data",
        "cost_grid": cost_grid,
        "monthly_performance": monthly,
        "rolling_performance": rolling_performance(result, window=rolling_window),
        "benchmark": benchmark_comparison(result),
        "parameter_perturbation": perturbations or [],
        "stability_summary": {
            "month_count": len(monthly),
            "positive_month_ratio": (
                len(positive_months) / len(monthly) if monthly else None
            ),
            "all_cost_scenarios_positive": all(
                float(row["total_return"] or 0.0) > 0 for row in cost_grid
            ),
        },
        "lookahead_status": result.get("lookahead_status", "not_run"),
        "note": (
            "Diagnostics reuse the current sample and are not a substitute for a locked "
            "six-month OOS evaluation. Parameter grids are sensitivity checks, not tuning proof."
        ),
    }
