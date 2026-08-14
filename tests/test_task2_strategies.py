from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies import create_strategy, list_strategies
from strategies.backtest import (
    BacktestConfig,
    compare_backtest_engines,
    run_event_backtest,
    run_vectorized_backtest,
)
from trading.risk import RiskLimits


def market_frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    trend = np.linspace(100, 130, rows)
    cycle = np.sin(np.arange(rows) / 8) * 3
    return pd.DataFrame(
        {
            "close": trend + cycle,
            "reference_close": np.linspace(50, 60, rows) + np.sin(np.arange(rows) / 9),
            "funding_rate": np.where(np.arange(rows) % 8 == 0, 0.0001, 0.0),
        },
        index=index,
    )


def test_catalog_exposes_all_four_required_strategy_families() -> None:
    categories = {row["category"] for row in list_strategies()}
    assert categories == {"趋势跟踪", "网格交易", "统计套利", "均值回归"}


@pytest.mark.parametrize(
    "name",
    ["trend_following", "grid_trading", "statistical_arbitrage", "mean_reversion"],
)
def test_each_strategy_generates_bounded_causal_target(name: str) -> None:
    frame = market_frame()
    strategy = create_strategy(name)
    target = strategy.generate_target(frame)
    assert target.index.equals(frame.index)
    assert target.abs().max() <= 1
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 2
    revised = strategy.generate_target(changed)
    pd.testing.assert_series_equal(target.iloc[:-1], revised.iloc[:-1])


def test_vectorized_and_event_backtests_align_with_cost_and_funding() -> None:
    frame = market_frame()
    target = create_strategy("trend_following", fast_window=12, slow_window=48).generate_target(
        frame
    )
    config = BacktestConfig(fee_rate=0.0004, slippage_rate=0.0002, leverage=1.5)
    consistency = compare_backtest_engines(frame, target, config)
    assert consistency["status"] == "pass"
    assert consistency["max_abs_net_return_error"] <= 1e-12
    assert run_vectorized_backtest(frame, target, config)["lookahead_status"] == (
        "pass_by_construction"
    )


def test_event_backtest_records_real_risk_triggers() -> None:
    frame = market_frame(80)
    frame.iloc[50, frame.columns.get_loc("close")] *= 0.7
    target = pd.Series(1.0, index=frame.index)
    result = run_event_backtest(
        frame,
        target,
        BacktestConfig(
            risk_limits=RiskLimits(
                stop_loss_pct=0.03,
                max_drawdown_pct=0.05,
                max_position_fraction=1.0,
                max_leverage=2.0,
                circuit_breaker_return=0.10,
            )
        ),
    )
    assert result["risk_events"]
    assert {row["rule"] for row in result["risk_events"]} & {
        "stop_loss",
        "max_drawdown",
        "extreme_market_move",
    }


def test_vectorized_engine_refuses_path_dependent_risk() -> None:
    frame = market_frame(30)
    with pytest.raises(ValueError, match="event engine"):
        run_vectorized_backtest(
            frame,
            pd.Series(0.0, index=frame.index),
            BacktestConfig(risk_limits=RiskLimits()),
        )
