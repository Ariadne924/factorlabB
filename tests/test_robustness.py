from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.robustness import (
    benjamini_hochberg,
    block_bootstrap_ic,
    forward_return,
    group_monotonicity,
    sign_consistency,
    window_horizon_robustness,
)


class RobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
        self.close = pd.Series(np.exp(np.linspace(4.0, 4.2, 240)), index=self.index)

    def test_forward_return_uses_future_close_only_as_target(self) -> None:
        result = forward_return(self.close, 3)
        self.assertAlmostEqual(result.iloc[0], self.close.iloc[3] / self.close.iloc[0] - 1)
        self.assertTrue(result.tail(3).isna().all())

    def test_window_horizon_grid_and_sign_consistency(self) -> None:
        factor = self.close.pct_change(3)
        grid = window_horizon_robustness(
            factor,
            self.close,
            lookback_days=(3, 7),
            horizons=(1, 3),
            min_obs=10,
        )
        self.assertEqual(len(grid), 4)
        self.assertEqual(set(grid["status"]), {"computed"})
        self.assertIsNotNone(sign_consistency(grid))

    def test_block_bootstrap_is_deterministic(self) -> None:
        returns = forward_return(self.close, 1)
        factor = returns.shift(1)
        first = block_bootstrap_ic(factor, returns, n_bootstrap=50, block_size=12)
        second = block_bootstrap_ic(factor, returns, n_bootstrap=50, block_size=12)
        self.assertEqual(first, second)
        self.assertEqual(first["n_obs"], 238)
        self.assertGreater(first["p_value"] or 0.0, 0.0)

    def test_fdr_and_monotonicity(self) -> None:
        adjusted = benjamini_hochberg(pd.Series({"a": 0.001, "b": 0.02, "c": 0.9}))
        self.assertTrue(bool(adjusted.loc["a", "reject"]))
        self.assertFalse(bool(adjusted.loc["c", "reject"]))
        groups = pd.DataFrame({"group": [1, 2, 3], "mean_return": [-1.0, 0.0, 1.0]})
        self.assertAlmostEqual(group_monotonicity(groups) or 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
