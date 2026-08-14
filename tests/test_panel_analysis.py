from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.panel_analysis import (
    build_panel_factor_report,
    check_panel_truncation_invariance,
    cross_sectional_ic_series,
    panel_forward_returns,
    panel_group_returns,
    panel_strategy_returns,
)
from factors.panel import build_panel


def analysis_panel() -> pd.DataFrame:
    generator = np.random.default_rng(5)
    timestamps = pd.date_range("2025-01-01", periods=60, freq="h", tz="UTC")
    frames = {}
    for position, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        returns = generator.normal(0.001 * position, 0.01, len(timestamps))
        close = 100 * np.exp(np.cumsum(returns))
        frames[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": generator.lognormal(5, 0.2, len(close)),
            },
            index=timestamps,
        )
    return build_panel(frames)


def test_panel_forward_returns_do_not_cross_symbol_boundaries() -> None:
    panel = analysis_panel()
    forward = panel_forward_returns(panel["close"])
    for symbol in panel.index.get_level_values("symbol").unique():
        assert pd.isna(forward.xs(symbol, level="symbol").iloc[-1])


def test_cross_sectional_metrics_grouping_and_costs() -> None:
    panel = analysis_panel()
    forward = panel_forward_returns(panel["close"])
    factor = forward.fillna(0) + np.linspace(0, 1e-6, len(forward))
    ic = cross_sectional_ic_series(factor, forward)
    groups, spread = panel_group_returns(factor, forward)
    strategy = panel_strategy_returns(factor, forward)

    assert ic["rank_ic"].mean() > 0.99
    assert groups.iloc[-1]["mean_return"] > groups.iloc[0]["mean_return"]
    assert spread.mean() > 0
    assert strategy["turnover"].ge(0).all()
    assert strategy["net_return"].mean() <= strategy["gross_return"].mean()


def test_panel_report_and_truncation_check() -> None:
    panel = analysis_panel()

    def factor(data: pd.DataFrame) -> pd.Series:
        return data["close"].groupby(level="symbol").pct_change(3)

    values = factor(panel)
    assert check_panel_truncation_invariance(factor, panel)
    report = build_panel_factor_report(
        "test_panel_factor",
        values,
        panel["close"],
        interval="1h",
        metadata={"source": "test"},
        lookahead_status="pass",
    )
    assert report["scope"] == "cross_sectional"
    assert report["lookahead_status"] == "pass"
    assert report["metrics"]["turnover"] is not None
    assert report["research_note"].startswith("Current-sample")
