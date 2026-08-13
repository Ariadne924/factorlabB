from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.panel import (
    PanelValidationError,
    build_panel,
    cs_neutralize,
    cs_rank,
    cs_scale,
    cs_winsorize,
    cs_zscore,
    panel_coverage,
)


def symbol_frame(values: list[float], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(values, index=timestamps)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
        },
        index=timestamps,
    )


def test_build_panel_preserves_gaps_and_reports_coverage() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    panel = build_panel(
        {
            "btcusdt": symbol_frame([1.0, 2.0, 3.0], timestamps),
            "ethusdt": symbol_frame([10.0, 30.0], timestamps[[0, 2]]),
        }
    )

    assert panel.index.names == ["timestamp", "symbol"]
    assert len(panel) == 5
    assert (timestamps[1], "ETHUSDT") not in panel.index
    coverage = panel_coverage(panel)
    assert coverage["available_symbols"].tolist() == [2, 1, 2]
    assert coverage["coverage_ratio"].tolist() == [1.0, 0.5, 1.0]


def test_inner_panel_keeps_only_shared_timestamps() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    panel = build_panel(
        {
            "BTCUSDT": symbol_frame([1.0, 2.0, 3.0], timestamps),
            "ETHUSDT": symbol_frame([10.0, 30.0], timestamps[[0, 2]]),
        },
        join="inner",
    )
    assert panel.index.get_level_values("timestamp").unique().tolist() == [
        timestamps[0],
        timestamps[2],
    ]
    assert panel_coverage(panel)["is_complete"].all()


def test_panel_rejects_naive_or_duplicate_timestamps() -> None:
    naive = pd.date_range("2025-01-01", periods=2, freq="h")
    with pytest.raises(PanelValidationError, match="timezone-aware"):
        build_panel({"BTCUSDT": symbol_frame([1.0, 2.0], naive)})

    duplicated = pd.DatetimeIndex(["2025-01-01T00:00Z", "2025-01-01T00:00Z"])
    with pytest.raises(PanelValidationError, match="duplicate timestamps"):
        build_panel({"BTCUSDT": symbol_frame([1.0, 2.0], duplicated)})


def cross_section(values: list[list[float]]) -> pd.Series:
    timestamps = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [timestamps, symbols], names=["timestamp", "symbol"]
    )
    return pd.Series(np.asarray(values).reshape(-1), index=index, dtype=float)


def test_cross_sectional_rank_scale_zscore_and_winsorize() -> None:
    values = cross_section([[1, 2, 3, 100], [4, 3, 2, 1]])
    ranked = cs_rank(values)
    scaled = cs_scale(values)
    standardized = cs_zscore(values)
    winsorized = cs_winsorize(values, lower=0.25, upper=0.75)

    assert ranked.groupby(level="timestamp").max().eq(1.0).all()
    assert scaled.abs().groupby(level="timestamp").sum().eq(1.0).all()
    assert np.allclose(standardized.groupby(level="timestamp").mean(), 0.0)
    assert winsorized.iloc[3] < 100.0


def test_neutralize_uses_only_same_timestamp_exposures() -> None:
    values = cross_section([[2, 5, 8, 11], [20, 15, 10, 5]])
    exposure = cross_section([[0, 1, 2, 3], [0, 1, 2, 3]]).rename("size")
    residual = cs_neutralize(values, exposure.to_frame())

    for timestamp in residual.index.get_level_values("timestamp").unique():
        group = residual.xs(timestamp, level="timestamp")
        group_exposure = exposure.xs(timestamp, level="timestamp")
        assert abs(float(group.mean())) < 1e-10
        assert abs(float(group.cov(group_exposure))) < 1e-10


def test_future_cross_section_does_not_change_historical_results() -> None:
    original = cross_section([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 6, 8]])
    changed = original.copy()
    final_timestamp = changed.index.get_level_values("timestamp").max()
    changed.loc[final_timestamp] = [1_000, -1_000, 500, -500]

    historical = original.index.get_level_values("timestamp") < final_timestamp
    pd.testing.assert_series_equal(
        cs_zscore(original).loc[historical],
        cs_zscore(changed).loc[historical],
    )
