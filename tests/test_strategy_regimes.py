from __future__ import annotations

import json

import pandas as pd

from evaluation.strategy_regimes import (
    MarketRegimeConfig,
    classify_market_regimes,
    market_regime_analysis,
)
from evaluation.time_series_strategy import (
    TimeSeriesAllocation,
    TimeSeriesStrategyConfig,
    run_time_series_strategy,
)
from tests.test_time_series_strategy import price_frame


def test_regime_classification_is_causal_and_lagged() -> None:
    frame = price_frame(500)
    config = MarketRegimeConfig(
        trend_window=24,
        volatility_window=24,
        threshold_history=48,
    )
    regimes = classify_market_regimes(frame, config=config)
    changed = frame.copy()
    changed.loc[changed.index[350]:, "close"] *= 3.0
    changed_regimes = classify_market_regimes(changed, config=config)
    pd.testing.assert_frame_equal(regimes.iloc[:350], changed_regimes.iloc[:350])
    assert set(regimes["trend_regime"]).issubset(
        {"bull", "bear", "sideways", "warmup"}
    )


def test_market_regime_analysis_is_serializable() -> None:
    frame = price_frame(800)
    result = run_time_series_strategy(
        frame,
        TimeSeriesStrategyConfig(
            allocations=(TimeSeriesAllocation("return_momentum"),),
            interval="1h",
            standardize_window=24,
        ),
    )
    report = market_regime_analysis(
        frame,
        result,
        config=MarketRegimeConfig(24, 24, 48),
    )
    assert report["status"] == "computed"
    assert report["coverage"] > 0
    assert report["lookahead_status"] == "pass_by_one_bar_regime_lag"
    assert report["trend_performance"]
    json.dumps(report)
