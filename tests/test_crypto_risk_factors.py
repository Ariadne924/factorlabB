from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.crypto_risk_factors import build_ltw_factor_returns
from factors.panel import build_panel


def risk_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    frames = {}
    for position, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        close = 100 * (position + 1) * np.cumprod(
            np.full(len(timestamps), 1.001 + position * 0.001)
        )
        frames[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 10.0,
                "market_cap": 1_000_000.0 * (position + 1),
            },
            index=timestamps,
        )
    return build_panel(frames)


def test_ltw_factors_require_market_cap() -> None:
    with pytest.raises(ValueError, match="market_cap"):
        build_ltw_factor_returns(risk_panel().drop(columns="market_cap"), momentum_lookback=3)


def test_ltw_factors_are_aligned_and_point_in_time() -> None:
    panel = risk_panel()
    factors = build_ltw_factor_returns(panel, momentum_lookback=3)
    assert factors.columns.tolist() == ["cmkt", "csmb", "cmom"]
    assert factors.index.tz is not None
    assert np.isfinite(factors.dropna()).all().all()
    assert factors["cmom"].dropna().mean() > 0
