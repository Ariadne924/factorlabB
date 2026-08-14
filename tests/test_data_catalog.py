from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.catalog import build_data_catalog, missing_download_ranges
from data.panel_loader import load_silver_panel


def write_silver(
    data_dir: Path,
    symbol: str,
    interval: str,
    timestamps: pd.DatetimeIndex,
) -> None:
    close = pd.Series(range(100, 100 + len(timestamps)), dtype=float)
    frame = pd.DataFrame(
        {
            "open_time_utc": timestamps,
            "close_time_utc": timestamps + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close + 0.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "taker_buy_volume": 6.0,
            "taker_buy_quote_volume": 600.0,
            "num_trades": 10,
            "symbol": symbol,
            "interval": interval,
        }
    )
    path = data_dir / "silver" / "binance" / "perpetual" / symbol / interval / "klines.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_catalog_detects_internal_gaps_and_missing_edges(tmp_path: Path) -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01T01:00Z", "2025-01-01T02:00Z", "2025-01-01T04:00Z"]
    )
    write_silver(tmp_path, "BTCUSDT", "1h", timestamps)
    output = tmp_path / "catalog.json"
    catalog = build_data_catalog(tmp_path, output)

    entry = catalog["entries"][0]
    assert entry["missing_bars"] == 1
    assert entry["coverage_ratio"] == 0.75
    assert output.exists()
    ranges = missing_download_ranges(
        catalog,
        symbol="BTCUSDT",
        interval="1h",
        start="2025-01-01T00:00Z",
        end="2025-01-01T05:00Z",
    )
    assert ranges == [
        {"start": "2025-01-01T00:00:00+00:00", "end": "2025-01-01T00:00:00+00:00"},
        {"start": "2025-01-01T03:00:00+00:00", "end": "2025-01-01T03:00:00+00:00"},
        {"start": "2025-01-01T05:00:00+00:00", "end": "2025-01-01T05:00:00+00:00"},
    ]


def test_catalog_returns_no_range_when_request_is_covered(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    write_silver(tmp_path, "BTCUSDT", "1h", timestamps)
    catalog = build_data_catalog(tmp_path)
    assert missing_download_ranges(
        catalog,
        symbol="BTCUSDT",
        interval="1h",
        start="2025-01-01T01:00Z",
        end="2025-01-01T03:00Z",
    ) == []


def test_catalog_detects_gap_between_separate_coverage_entries() -> None:
    catalog = {
        "entries": [
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T00:00:00+00:00",
                "end": "2025-01-01T01:00:00+00:00",
                "gap_ranges": [],
            },
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T04:00:00+00:00",
                "end": "2025-01-01T05:00:00+00:00",
                "gap_ranges": [],
            },
        ]
    }
    assert missing_download_ranges(
        catalog,
        symbol="BTCUSDT",
        interval="1h",
        start="2025-01-01T00:00Z",
        end="2025-01-01T05:00Z",
    ) == [
        {"start": "2025-01-01T02:00:00+00:00", "end": "2025-01-01T03:00:00+00:00"}
    ]


def test_load_silver_panel_filters_symbols_and_preserves_gaps(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    write_silver(tmp_path, "BTCUSDT", "1h", timestamps)
    write_silver(tmp_path, "ETHUSDT", "1h", timestamps.delete(2))
    panel = load_silver_panel(
        tmp_path,
        interval="1h",
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    assert panel.index.names == ["timestamp", "symbol"]
    assert len(panel) == 7
    assert (timestamps[2], "ETHUSDT") not in panel.index
