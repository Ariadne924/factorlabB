from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.threshold_research import run_threshold_study, threshold_positions


def frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=300, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(300) / 20) * 0.002 + 0.0001))
    return pd.DataFrame({"close": close}, index=index)


def test_threshold_positions_are_symmetric_and_bounded() -> None:
    signal = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert threshold_positions(signal, 1.0).tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_threshold_study_reports_surface_plateau_and_regime_drift() -> None:
    data = frame()
    signal = pd.Series(np.sin(np.arange(len(data)) / 12), index=data.index)
    report = run_threshold_study(data, signal, [0.0, 0.25, 0.5, 0.75], regime_window=24)
    assert len(report["threshold_surface"]) == 4
    assert report["stable_interval"]["stable_min"] <= report["stable_interval"][
        "recommended_threshold"
    ]
    assert report["stable_interval"]["recommended_threshold"] <= report[
        "stable_interval"
    ]["stable_max"]
    assert len(report["regime_surface"]) == 12
    assert report["lookahead_status"] == "pass_by_construction"


def test_future_signal_change_does_not_change_past_threshold_results() -> None:
    data = frame()
    signal = pd.Series(np.sin(np.arange(len(data)) / 12), index=data.index)
    changed = signal.copy()
    changed.iloc[-1] = 100
    first = threshold_positions(signal, 0.5)
    second = threshold_positions(changed, 0.5)
    pd.testing.assert_series_equal(first.iloc[:-1], second.iloc[:-1])
