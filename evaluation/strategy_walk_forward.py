"""Expanding walk-forward validation for single-asset strategy parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evaluation.strategy_robustness import _path_metrics, _returns_frame
from evaluation.time_series_strategy import (
    TimeSeriesStrategyConfig,
    run_time_series_strategy,
)


@dataclass(frozen=True)
class StrategyWalkForwardConfig:
    """Expanding training and non-overlapping test windows, expressed in bars."""

    min_train_size: int
    test_size: int
    embargo: int = 1
    min_score_periods: int = 20

    def validate(self) -> None:
        if self.min_train_size < 20:
            raise ValueError("min_train_size must be at least 20 bars")
        if self.test_size < 1:
            raise ValueError("test_size must be positive")
        if self.embargo < 0:
            raise ValueError("embargo cannot be negative")
        if self.min_score_periods < 5:
            raise ValueError("min_score_periods must be at least 5")


def _candidate_configs(
    base_config: TimeSeriesStrategyConfig,
    *,
    thresholds: Iterable[float],
    rebalance_values: Iterable[int],
    standardize_windows: Iterable[int],
) -> list[TimeSeriesStrategyConfig]:
    candidates: list[TimeSeriesStrategyConfig] = []
    for threshold in dict.fromkeys(float(value) for value in thresholds):
        for rebalance in dict.fromkeys(int(value) for value in rebalance_values):
            for window in dict.fromkeys(int(value) for value in standardize_windows):
                candidate = replace(
                    base_config,
                    score_threshold=threshold,
                    rebalance_every=rebalance,
                    standardize_window=window,
                )
                candidate.validate()
                candidates.append(candidate)
    if not candidates:
        raise ValueError("walk-forward requires at least one parameter candidate")
    return candidates


def _selection_row(
    returns: pd.DataFrame,
    config: TimeSeriesStrategyConfig,
) -> dict[str, Any]:
    metrics = _path_metrics(returns["net_return"])
    sharpe = metrics["bar_sharpe"]
    total_return = metrics["total_return"]
    turnover = returns["turnover"].mean() if not returns.empty else None
    return {
        "score_threshold": config.score_threshold,
        "rebalance_every": config.rebalance_every,
        "standardize_window": config.standardize_window,
        "n_periods": int(metrics["n_periods"]),
        "train_bar_sharpe": float(sharpe) if sharpe is not None else None,
        "train_total_return": float(total_return) if total_return is not None else None,
        "train_mean_turnover": float(turnover) if turnover is not None else None,
    }


def _selection_key(row: dict[str, Any], min_score_periods: int) -> tuple[float, float, float]:
    if int(row["n_periods"]) < min_score_periods:
        return (-np.inf, -np.inf, -np.inf)
    sharpe = row["train_bar_sharpe"]
    total_return = row["train_total_return"]
    turnover = row["train_mean_turnover"]
    return (
        float(sharpe) if sharpe is not None and np.isfinite(sharpe) else -np.inf,
        float(total_return)
        if total_return is not None and np.isfinite(total_return)
        else -np.inf,
        -float(turnover) if turnover is not None and np.isfinite(turnover) else -np.inf,
    )


def walk_forward_strategy(
    frame: pd.DataFrame,
    base_config: TimeSeriesStrategyConfig,
    *,
    config: StrategyWalkForwardConfig,
    thresholds: Iterable[float],
    rebalance_values: Iterable[int],
    standardize_windows: Iterable[int],
) -> dict[str, Any]:
    """Select parameters on expanding train prefixes and freeze them for each test fold."""
    config.validate()
    base_config.validate()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("walk-forward input requires a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("walk-forward input index must be unique and increasing")
    first_test = config.min_train_size + config.embargo
    if first_test >= len(frame):
        raise ValueError("sample is too short for the requested walk-forward windows")

    candidates = _candidate_configs(
        base_config,
        thresholds=thresholds,
        rebalance_values=rebalance_values,
        standardize_windows=standardize_windows,
    )
    folds: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    # Every candidate path is causal and can therefore be computed once. Each fold
    # only reads its training prefix for selection and its own test slice for output.
    # This keeps minute-data use practical without changing the information boundary.
    candidate_returns = [
        _returns_frame(run_time_series_strategy(frame, candidate))
        for candidate in candidates
    ]

    for fold_id, test_start in enumerate(
        range(first_test, len(frame), config.test_size), start=1
    ):
        test_end = min(test_start + config.test_size, len(frame))
        train_end = test_start - config.embargo
        train_end_time = frame.index[train_end - 1]
        score_rows = [
            _selection_row(
                returns.loc[returns.index <= train_end_time],
                candidate,
            )
            for candidate, returns in zip(candidates, candidate_returns, strict=True)
        ]
        best_index = max(
            range(len(score_rows)),
            key=lambda index: _selection_key(
                score_rows[index], config.min_score_periods
            ),
        )
        best_row = score_rows[best_index]
        if not np.isfinite(_selection_key(best_row, config.min_score_periods)[0]):
            continue
        selected_config = candidates[best_index]

        test_start_time = frame.index[test_start]
        test_end_time = frame.index[test_end - 1]
        selected_returns = candidate_returns[best_index]
        fold_returns = selected_returns.loc[
            (selected_returns.index >= test_start_time)
            & (selected_returns.index <= test_end_time)
        ].copy()
        fold_returns["fold"] = fold_id
        oos_rows.extend(fold_returns.reset_index().to_dict(orient="records"))
        folds.append(
            {
                "fold": fold_id,
                "train_start": str(frame.index[0]),
                "train_end": str(frame.index[train_end - 1]),
                "test_start": str(test_start_time),
                "test_end": str(test_end_time),
                "n_train_bars": train_end,
                "n_test_bars": test_end - test_start,
                "selected_params": {
                    "score_threshold": selected_config.score_threshold,
                    "rebalance_every": selected_config.rebalance_every,
                    "standardize_window": selected_config.standardize_window,
                },
                "training_objective": "raw_bar_sharpe",
                "selected_training_metrics": best_row,
                "candidate_scores": score_rows,
            }
        )

    oos = pd.DataFrame(oos_rows)
    if not oos.empty:
        oos["time"] = pd.to_datetime(oos["time"], utc=True)
        oos = oos.sort_values("time").drop_duplicates("time", keep="first")
        oos["equity"] = (1.0 + oos["net_return"]).cumprod()
        oos["drawdown"] = oos["equity"] / oos["equity"].cummax() - 1.0
        net_metrics = _path_metrics(oos["net_return"])
        gross_metrics = _path_metrics(oos["gross_return"])
    else:
        net_metrics = _path_metrics(pd.Series(dtype=float))
        gross_metrics = _path_metrics(pd.Series(dtype=float))

    return {
        "status": "computed_short_sample" if folds and not oos.empty else "insufficient_data",
        "config": asdict(config),
        "base_strategy_config": base_config.to_dict(),
        "candidate_count": len(candidates),
        "folds": folds,
        "metrics": {
            **net_metrics,
            "gross_total_return": gross_metrics["total_return"],
            "mean_turnover": float(oos["turnover"].mean()) if not oos.empty else None,
        },
        "returns": [
            {
                **row,
                "time": str(row["time"]),
                "fold": int(row["fold"]),
            }
            for row in oos.to_dict(orient="records")
        ],
        "lookahead_status": "pass_by_walk_forward_construction",
        "lookahead_policy": (
            "Each fold selects parameters only on an expanding training prefix, leaves the "
            "configured embargo, freezes the selected configuration, and reports only the "
            "following non-overlapping test returns."
        ),
        "note": (
            "This is repeated short-sample walk-forward evidence, not the unopened locked "
            "six-month OOS acceptance test and not proof of future alpha."
        ),
    }
