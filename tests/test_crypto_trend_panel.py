from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from factors.crypto_trend_panel import CTREND_TECHNICAL_EXPRESSIONS
from factors.panel import build_panel
from factors.panel_registry import (
    compute_panel_factor,
    get_panel_factor_metadata,
    list_panel_factors,
)


@pytest.fixture(scope="module")
def trend_panel() -> pd.DataFrame:
    generator = np.random.default_rng(84)
    timestamps = pd.date_range("2024-01-01", periods=260, freq="h", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"]):
        changes = generator.normal(0.0001, 0.012 + position * 0.001, len(timestamps))
        close = 100 * (position + 1) * np.exp(np.cumsum(changes))
        open_ = close * (1 + generator.normal(0, 0.004, len(timestamps)))
        spread = generator.uniform(0.003, 0.025, len(timestamps))
        volume = generator.lognormal(8 + position / 5, 0.6, len(timestamps))
        frames[symbol] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1 + spread),
                "low": np.minimum(open_, close) * (1 - spread),
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
            },
            index=timestamps,
        )
    return build_panel(frames)


def test_ctrend_registers_28_inputs_without_claiming_fitted_signal() -> None:
    names = list_panel_factors("crypto_trend_technical")
    assert len(names) == 28
    assert set(names) == set(CTREND_TECHNICAL_EXPRESSIONS)
    metadata = get_panel_factor_metadata("ctrend_rsi_14")
    assert metadata["source_url"].startswith("https://doi.org/")
    assert "not the paper's fitted" in metadata["adaptation_note"]


@pytest.mark.parametrize("name", sorted(CTREND_TECHNICAL_EXPRESSIONS))
def test_ctrend_inputs_are_finite_aligned_and_past_only(
    name: str, trend_panel: pd.DataFrame
) -> None:
    original = compute_panel_factor(name, trend_panel)
    assert original.index.equals(trend_panel.index)
    assert np.isfinite(original.dropna()).all()
    assert original.notna().sum() > 0

    changed = trend_panel.copy()
    final_timestamp = changed.index.get_level_values("timestamp").max()
    changed.loc[(final_timestamp, slice(None)), "close"] *= 5
    changed.loc[(final_timestamp, slice(None)), "quote_volume"] *= 7
    historical = trend_panel.index.get_level_values("timestamp") < final_timestamp
    pd.testing.assert_series_equal(
        original.loc[historical],
        compute_panel_factor(name, changed).loc[historical],
    )
