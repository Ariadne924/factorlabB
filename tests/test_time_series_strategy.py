from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from evaluation.time_series_strategy import (
    TimeSeriesAllocation,
    TimeSeriesStrategyConfig,
    build_ml_strategy_preset,
    run_time_series_strategy,
)


def price_frame(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(rows) / 13) * 0.003 + 0.0002))
    volume = 1000 + np.arange(rows)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_volume": volume * 0.52,
            "taker_buy_quote_volume": volume * close * 0.52,
            "num_trades": np.arange(rows) + 100,
        },
        index=index,
    )


def test_time_series_strategy_is_lagged_costed_and_serializable() -> None:
    frame = price_frame()
    config = TimeSeriesStrategyConfig(
        allocations=(
            TimeSeriesAllocation("return_momentum", weight=2.0),
            TimeSeriesAllocation("rsi", weight=1.0, direction=-1),
        ),
        interval="1h",
        rebalance_every=4,
        standardize_window=24,
        score_threshold=0.2,
        fee_rate=0.0004,
        slippage=0.0002,
    )
    result = run_time_series_strategy(frame, config)
    assert result["status"] == "computed_short_sample"
    assert result["lookahead_status"] == "pass_by_construction"
    assert result["metrics"]["n_periods"] > 0
    assert result["metrics"]["gross_total_return"] != result["metrics"]["total_return"]
    assert result["returns"][0]["position"] == 0.0
    assert set(result["factor_coverage"]) == {"return_momentum", "rsi"}


def test_time_series_strategy_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TimeSeriesStrategyConfig(allocations=(), interval="1h").validate()
    with pytest.raises(ValueError, match="unknown registered"):
        TimeSeriesAllocation("does_not_exist").validate()


def test_ml_recommendations_create_normalized_deduplicated_preset() -> None:
    preset = build_ml_strategy_preset(
        [
            {
                "feature": "return_momentum__w6",
                "direction": 1,
                "recommendation_score": 0.6,
            },
            {
                "feature": "return_momentum__w12",
                "direction": -1,
                "recommendation_score": 0.4,
            },
            {"feature": "rsi", "direction": -1, "recommendation_score": 0.2},
            {"feature": "unknown", "direction": 1, "recommendation_score": 1.0},
        ]
    )
    assert preset["status"] == "ready"
    assert [row["name"] for row in preset["allocations"]] == ["return_momentum", "rsi"]
    assert preset["allocations"][0]["params"] == {"window": 6}
    assert sum(row["weight"] for row in preset["allocations"]) == pytest.approx(1.0)
    assert "return_momentum__w12" in preset["skipped"]
