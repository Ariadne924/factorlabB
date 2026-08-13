from __future__ import annotations

import json

from evaluation.strategy_walk_forward import (
    StrategyWalkForwardConfig,
    walk_forward_strategy,
)
from evaluation.time_series_strategy import (
    TimeSeriesAllocation,
    TimeSeriesStrategyConfig,
)
from tests.test_time_series_strategy import price_frame


def base_strategy() -> TimeSeriesStrategyConfig:
    return TimeSeriesStrategyConfig(
        allocations=(TimeSeriesAllocation("return_momentum"),),
        interval="1h",
        standardize_window=24,
        fee_rate=0.0002,
        slippage=0.0001,
    )


def run_walk_forward(frame):
    return walk_forward_strategy(
        frame,
        base_strategy(),
        config=StrategyWalkForwardConfig(
            min_train_size=220,
            test_size=80,
            embargo=1,
        ),
        thresholds=(0.0, 0.25),
        rebalance_values=(1, 4),
        standardize_windows=(24,),
    )


def test_strategy_walk_forward_freezes_parameters_before_each_test() -> None:
    frame = price_frame(620)
    result = run_walk_forward(frame)
    assert result["status"] == "computed_short_sample"
    assert result["lookahead_status"] == "pass_by_walk_forward_construction"
    assert len(result["folds"]) >= 4
    assert result["candidate_count"] == 4
    assert result["metrics"]["n_periods"] > 0
    first_fold = result["folds"][0]
    assert first_fold["train_end"] < first_fold["test_start"]
    assert len(first_fold["candidate_scores"]) == 4
    json.dumps(result)


def test_first_fold_selection_does_not_depend_on_first_test_prices() -> None:
    frame = price_frame(620)
    original = run_walk_forward(frame)
    changed = frame.copy()
    first_test_start = frame.index[221]
    first_test_end = frame.index[300]
    changed.loc[first_test_start:first_test_end, "close"] *= 1.5
    changed_result = run_walk_forward(changed)
    assert (
        original["folds"][0]["selected_params"]
        == changed_result["folds"][0]["selected_params"]
    )
    assert (
        original["folds"][0]["selected_training_metrics"]
        == changed_result["folds"][0]["selected_training_metrics"]
    )
