from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from factors.alpha101_panel import ALPHA101_PANEL_EXPRESSIONS
from factors.panel import build_panel
from factors.panel_registry import (
    compute_panel_factor,
    get_panel_factor_metadata,
    list_panel_factors,
)


@pytest.fixture(scope="module")
def alpha_panel() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    timestamps = pd.date_range("2025-01-01", periods=180, freq="h", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"]):
        returns = generator.normal(0.0002 * (position + 1), 0.01, len(timestamps))
        close = 100 * (position + 1) * np.exp(np.cumsum(returns))
        open_ = close * (1 + generator.normal(0, 0.003, len(timestamps)))
        spread = generator.uniform(0.002, 0.02, len(timestamps))
        volume = generator.lognormal(8 + position / 5, 0.5, len(timestamps))
        frames[symbol] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1 + spread),
                "low": np.minimum(open_, close) * (1 - spread),
                "close": close,
                "volume": volume,
                "quote_volume": volume * close * generator.uniform(0.995, 1.005, len(close)),
            },
            index=timestamps,
        )
    return build_panel(frames)


def test_first_alpha101_batch_is_registered_with_provenance() -> None:
    names = list_panel_factors("alpha101_cross_sectional")
    assert set(names) == set(ALPHA101_PANEL_EXPRESSIONS)
    assert len(names) == 25
    metadata = get_panel_factor_metadata("alpha101_002")
    assert metadata["source"].startswith("101 Formulaic Alphas")
    assert metadata["source_url"] == "https://arxiv.org/abs/1601.00991"
    assert metadata["scope"] == "cross_sectional"
    assert "not validated alpha" in metadata["adaptation_note"]
    assert metadata["source_formula"]


@pytest.mark.parametrize("name", sorted(ALPHA101_PANEL_EXPRESSIONS))
def test_alpha101_panel_factors_are_finite_and_aligned(
    name: str, alpha_panel: pd.DataFrame
) -> None:
    result = compute_panel_factor(name, alpha_panel)
    assert result.index.equals(alpha_panel.index)
    assert result.name == name
    assert np.isfinite(result.dropna()).all()
    assert result.notna().sum() > 0


@pytest.mark.parametrize("name", sorted(ALPHA101_PANEL_EXPRESSIONS))
def test_alpha101_panel_factors_do_not_rewrite_history(
    name: str, alpha_panel: pd.DataFrame
) -> None:
    changed = alpha_panel.copy()
    final_timestamp = changed.index.get_level_values("timestamp").max()
    changed.loc[(final_timestamp, slice(None)), "close"] *= 10
    changed.loc[(final_timestamp, slice(None)), "volume"] *= 20
    historical = alpha_panel.index.get_level_values("timestamp") < final_timestamp

    pd.testing.assert_series_equal(
        compute_panel_factor(name, alpha_panel).loc[historical],
        compute_panel_factor(name, changed).loc[historical],
    )


def test_vwap_dependent_alpha_requires_quote_volume(alpha_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="quote_volume"):
        compute_panel_factor("alpha101_005", alpha_panel.drop(columns="quote_volume"))
