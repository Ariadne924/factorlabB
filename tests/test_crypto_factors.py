from __future__ import annotations

import numpy as np
import pandas as pd

import factors  # noqa: F401
from data.silver import merge_point_in_time_features
from evaluation.forward_check import ForwardCheck
from factors.registry import compute_factor, list_factors


def factor_frame(rows: int = 30) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC", name="open_time_utc")
    close = pd.Series(np.linspace(100, 130, rows), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
            "funding_rate": np.linspace(-0.001, 0.001, rows),
            "open_interest": np.linspace(1000, 1300, rows),
            "basis": np.linspace(-0.01, 0.02, rows),
        },
        index=index,
    )


def test_crypto_factors_registered_and_point_in_time_safe() -> None:
    frame = factor_frame()
    for name in ["funding_rate", "open_interest_change", "basis"]:
        assert name in list_factors()
        result = compute_factor(name, frame)
        assert result.index.equals(frame.index)
        assert ForwardCheck.check_truncation_invariance(
            lambda data, factor_name=name: compute_factor(factor_name, data), frame
        )


def test_point_in_time_merge_never_backfills_future_value() -> None:
    klines = factor_frame(3).drop(columns=["funding_rate", "open_interest", "basis"])
    features = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:30", "2026-01-01 01:30"], utc=True),
            "funding_rate": [0.01, 0.02],
        }
    )
    merged = merge_point_in_time_features(klines, features, feature_columns=["funding_rate"])
    assert pd.isna(merged.iloc[0]["funding_rate"])
    assert merged.iloc[1]["funding_rate"] == 0.01
    assert merged.iloc[2]["funding_rate"] == 0.02
