from __future__ import annotations

import pandas as pd

from data.validator import DataValidator


def silver_frame() -> pd.DataFrame:
    opens = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 04:00"],
        utc=True,
    )
    close = [100.0, 100.0, 100.0, 160.0]
    return pd.DataFrame(
        {
            "open_time_utc": opens,
            "close_time_utc": opens + pd.Timedelta(minutes=59),
            "open": close,
            "high": [101.0, 101.0, 101.0, 161.0],
            "low": [99.0, 99.0, 99.0, 159.0],
            "close": close,
            "volume": [1.0, 0.0, 1.0, 1.0],
            "quote_volume": [100.0, 0.0, 100.0, 160.0],
            "taker_buy_volume": [0.5, 0.0, 0.5, 0.5],
            "taker_buy_quote_volume": [50.0, 0.0, 50.0, 80.0],
            "num_trades": pd.Series([1, 0, 1, 1], dtype="int64"),
            "symbol": "BTCUSDT",
            "interval": "1h",
        }
    )


def test_all_basic_anomaly_flags_and_summary() -> None:
    flags = DataValidator.build_quality_flags(
        silver_frame(), expected_interval="1h", spike_threshold=0.2, stale_window=3
    )
    assert flags["zero_volume"].sum() == 1
    assert flags["stale_flat_data"].sum() == 1
    assert flags["extreme_price_spike"].sum() == 1
    assert flags["exchange_gap"].sum() == 1
    summary = DataValidator.summarize_quality_flags(flags)
    assert summary["rows"] == 4
    assert summary["flag_counts"]["exchange_gap"] == 1
