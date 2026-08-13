from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import factors  # noqa: F401
from factors.qlib_style import QLIB_ALPHA158_URL
from factors.registry import compute_factor, get_factor_metadata, list_factors


class QlibStyleFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
        close = pd.Series(np.linspace(100, 140, len(index)), index=index)
        self.frame = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.linspace(1000, 1800, len(index)),
            },
            index=index,
        )
        self.names = [
            "k_mid",
            "k_length",
            "k_mid2",
            "upper_shadow",
            "upper_shadow2",
            "lower_shadow",
            "lower_shadow2",
            "k_shift",
            "k_shift2",
            "linear_trend_slope",
            "trend_r_squared",
            "trend_residual",
            "price_position_rsv",
            "high_recency",
            "low_recency",
            "high_low_recency_diff",
            "price_volume_correlation",
            "return_volume_correlation",
            "volume_weighted_volatility",
            "volume_change_strength",
        ]

    def test_registry_contains_qlib_style_factors_with_provenance(self) -> None:
        self.assertTrue(set(self.names).issubset(list_factors()))
        for name in self.names:
            metadata = get_factor_metadata(name)
            self.assertEqual(metadata["source"], "Microsoft Qlib Alpha158")
            self.assertEqual(metadata["source_url"], QLIB_ALPHA158_URL)
            self.assertEqual(metadata["scope"], "time_series")

    def test_intrabar_formula_and_trend(self) -> None:
        k_mid2 = compute_factor("k_mid2", self.frame)
        self.assertAlmostEqual(k_mid2.iloc[-1], 0.5 / 3.0)
        r_squared = compute_factor("trend_r_squared", self.frame, {"window": 12})
        self.assertAlmostEqual(r_squared.iloc[-1], 1.0, places=10)
        self.assertAlmostEqual(compute_factor("high_recency", self.frame).iloc[-1], 1.0)
        self.assertAlmostEqual(compute_factor("low_recency", self.frame).iloc[-1], 0.0)

    def test_every_factor_is_past_only(self) -> None:
        cutoff = 59
        prefix = self.frame.iloc[: cutoff + 1]
        for name in self.names:
            full = compute_factor(name, self.frame)
            truncated = compute_factor(name, prefix)
            pd.testing.assert_series_equal(
                full.iloc[: cutoff + 1], truncated, check_names=True, obj=name
            )


if __name__ == "__main__":
    unittest.main()
