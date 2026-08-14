from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from evaluation.forward_check import ForwardCheck
from factors.registry import compute_factor, get_factor_metadata, list_factors

DERIVED_FACTORS = [
    "funding_rate_zscore",
    "funding_rate_change",
    "funding_rate_acceleration",
    "funding_price_divergence",
    "open_interest_zscore",
    "open_interest_momentum",
    "open_interest_price_divergence",
    "open_interest_volume_confirmation",
    "basis_zscore",
    "basis_change",
    "basis_funding_spread",
    "taker_imbalance",
    "taker_imbalance_momentum",
    "signed_price_impact",
    "volume_shock_zscore",
    "volatility_term_ratio",
    "momentum_volatility_interaction",
    "momentum_liquidity_interaction",
    "trade_intensity_zscore",
    "average_trade_size_zscore",
]


def derived_frame(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    steps = np.arange(rows, dtype=float)
    close = pd.Series(100 + steps * 0.05 + np.sin(steps / 5), index=index)
    volume = pd.Series(1000 + steps * 2 + 80 * np.sin(steps / 7), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_volume": volume * (0.5 + 0.1 * np.sin(steps / 9)),
            "num_trades": 100 + steps + 10 * np.cos(steps / 8),
            "funding_rate": 0.0001 * np.sin(steps / 10) + steps * 1e-7,
            "open_interest": 10_000 + steps * 10 + 100 * np.cos(steps / 11),
            "basis": 0.001 * np.cos(steps / 12) + steps * 1e-6,
        },
        index=index,
    )


def test_all_derived_factors_are_registered_finite_and_lookahead_safe() -> None:
    frame = derived_frame()
    registered = set(list_factors())
    assert set(DERIVED_FACTORS).issubset(registered)
    assert set(DERIVED_FACTORS).issubset(registered)

    for name in DERIVED_FACTORS:
        values = compute_factor(name, frame)
        assert values.index.equals(frame.index)
        assert np.isfinite(values.dropna()).all(), name
        assert values.notna().sum() > 20, name
        assert ForwardCheck.check_truncation_invariance(
            lambda data, factor_name=name: compute_factor(factor_name, data), frame
        ), name


def test_derived_factor_metadata_exposes_dependencies_and_parameters() -> None:
    metadata = get_factor_metadata("basis_funding_spread")
    assert {"basis", "funding_rate"}.issubset(metadata["data_dependencies"])
    assert metadata["default_params"] == {"window": 24}
    assert metadata["category"] == "数字资产交互"


def test_optional_data_dependency_fails_explicitly() -> None:
    frame = derived_frame().drop(columns=["open_interest"])
    with pytest.raises(ValueError, match="需要 open_interest 列"):
        compute_factor("open_interest_zscore", frame)


def test_short_window_must_precede_long_window() -> None:
    with pytest.raises(ValueError, match="short_window 必须小于 long_window"):
        compute_factor(
            "volatility_term_ratio",
            derived_frame(),
            params={"short_window": 24, "long_window": 6},
        )
