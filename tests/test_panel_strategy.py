from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from evaluation.panel_strategy import (
    FactorAllocation,
    PanelStrategyConfig,
    combine_panel_factors,
    run_panel_strategy,
)
from factors.panel import build_panel


def strategy_panel() -> pd.DataFrame:
    generator = np.random.default_rng(17)
    timestamps = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
    frames = {}
    for position, symbol in enumerate(["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"]):
        returns = generator.normal(position * 0.0002, 0.01, len(timestamps))
        close = 100 * (position + 1) * np.exp(np.cumsum(returns))
        volume = generator.lognormal(7 + position / 4, 0.4, len(timestamps))
        frames[symbol] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
            },
            index=timestamps,
        )
    return build_panel(frames)


def test_strategy_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PanelStrategyConfig(allocations=(), interval="1h").validate()
    with pytest.raises(ValueError, match="positive"):
        FactorAllocation("alpha101_003", weight=0).validate()
    with pytest.raises(ValueError, match="unique"):
        PanelStrategyConfig(
            allocations=(
                FactorAllocation("alpha101_003"),
                FactorAllocation("alpha101_003"),
            ),
            interval="1h",
        ).validate()


def test_combination_supports_registered_and_custom_expressions() -> None:
    panel = strategy_panel()
    allocations = (
        FactorAllocation("alpha101_003", weight=2.0),
        FactorAllocation("custom_momentum", expression="cs_rank(returns(close,3))"),
    )
    composite, inputs = combine_panel_factors(panel, allocations)
    assert composite.index.equals(panel.index)
    assert inputs.columns.tolist() == ["alpha101_003", "custom_momentum"]
    assert np.isfinite(composite.dropna()).all()


def test_multi_factor_strategy_returns_serializable_diagnostics() -> None:
    panel = strategy_panel()
    config = PanelStrategyConfig(
        allocations=(
            FactorAllocation("alpha101_003", weight=1.0, direction=-1),
            FactorAllocation("ctrend_rsi_14", weight=0.5),
        ),
        interval="1h",
        rebalance_every=4,
        fee_rate=0.0004,
        slippage=0.0002,
    )
    result = run_panel_strategy(panel, config)
    assert result["status"] == "computed_short_sample"
    assert result["metrics"]["n_periods"] > 0
    assert result["metrics"]["mean_turnover"] >= 0
    assert len(result["returns"]) == result["metrics"]["n_periods"]
    assert result["lookahead_status"] == "pass_by_construction"
    assert result["research_note"].startswith("Interactive current-sample")
