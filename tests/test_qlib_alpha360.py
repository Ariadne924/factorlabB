from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401
from evaluation.forward_check import ForwardCheck
from factors.qlib_alpha360 import ALPHA360_FIELDS, ALPHA360_LAGS, QLIB_ALPHA360_URL
from factors.registry import compute_factor, get_factor_metadata, list_factors


def alpha360_frame(rows: int = 90) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, rows), index=index)
    volume = pd.Series(np.linspace(1000, 1900, rows), index=index)
    vwap = close - 0.25
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
            "quote_volume": volume * vwap,
        },
        index=index,
    )


def test_alpha360_registers_exactly_360_features_with_provenance() -> None:
    names = [name for name in list_factors() if name.startswith("qlib360_")]
    assert len(names) == len(ALPHA360_FIELDS) * len(ALPHA360_LAGS) == 360
    metadata = get_factor_metadata("qlib360_vwap_59")
    assert metadata["source"] == "Microsoft Qlib Alpha360"
    assert metadata["source_url"] == QLIB_ALPHA360_URL
    assert "quote_volume" in metadata["data_dependencies"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("qlib360_close_0", 1.0),
        ("qlib360_open_0", (130.0 - 0.5) / 130.0),
        ("qlib360_high_0", 131.0 / 130.0),
        ("qlib360_low_0", 129.0 / 130.0),
        ("qlib360_vwap_0", (130.0 - 0.25) / 130.0),
        ("qlib360_volume_0", 1.0),
    ],
)
def test_alpha360_current_bar_formulas(name: str, expected: float) -> None:
    assert compute_factor(name, alpha360_frame()).iloc[-1] == pytest.approx(expected)


def test_alpha360_lags_are_past_only() -> None:
    frame = alpha360_frame()
    for name in ["qlib360_close_59", "qlib360_vwap_12", "qlib360_volume_24"]:
        assert ForwardCheck.check_truncation_invariance(
            lambda data, factor_name=name: compute_factor(factor_name, data), frame
        )


def test_alpha360_vwap_dependency_is_explicit() -> None:
    with pytest.raises(ValueError, match="需要 vwap 或 quote_volume"):
        compute_factor("qlib360_vwap_0", alpha360_frame().drop(columns="quote_volume"))
