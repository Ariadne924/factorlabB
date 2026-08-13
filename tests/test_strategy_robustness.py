from __future__ import annotations

from dataclasses import replace

from evaluation.strategy_robustness import (
    benchmark_comparison,
    build_strategy_robustness_report,
    monthly_performance,
    parameter_perturbation,
    reprice_cost_grid,
    rolling_performance,
)
from evaluation.time_series_strategy import (
    TimeSeriesAllocation,
    TimeSeriesStrategyConfig,
    run_time_series_strategy,
)
from tests.test_time_series_strategy import price_frame


def strategy_result():
    frame = price_frame(800)
    config = TimeSeriesStrategyConfig(
        allocations=(TimeSeriesAllocation("return_momentum"),),
        interval="1h",
        standardize_window=24,
        fee_rate=0.0002,
        slippage=0.0001,
    )
    return frame, config, run_time_series_strategy(frame, config)


def test_strategy_robustness_reprices_and_builds_time_diagnostics() -> None:
    _, _, result = strategy_result()
    cost_grid = reprice_cost_grid(result, one_way_costs=(0.0, 0.001, 0.003))
    assert cost_grid[0]["total_return"] >= cost_grid[-1]["total_return"]
    assert len(monthly_performance(result)) >= 2
    assert rolling_performance(result, window=24)
    benchmark = benchmark_comparison(result)
    assert benchmark["status"] == "computed"
    report = build_strategy_robustness_report(result, rolling_window=24)
    assert report["status"] == "computed"
    assert report["lookahead_status"] == "pass_by_construction"


def test_parameter_perturbation_is_bounded_and_serializable() -> None:
    frame, config, _ = strategy_result()
    grid = parameter_perturbation(
        frame,
        replace(config, fee_rate=0.0, slippage=0.0),
        thresholds=(0.0, 0.5),
        rebalance_values=(1, 4),
        standardize_windows=(24,),
    )
    assert len(grid) == 4
    assert all(row["status"] == "computed_short_sample" for row in grid)
