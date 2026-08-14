from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_task2_acceptance import run


def _silver(symbol: str, rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 100 + np.linspace(0, 10, rows) + np.sin(np.arange(rows) / 7)
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": symbol,
            "interval": "1h",
            "open_time_utc": index,
            "close_time_utc": index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
            "quote_volume": 1_000.0,
            "num_trades": 100,
            "taker_buy_volume": 5.0,
            "taker_buy_quote_volume": 500.0,
            "return": pd.Series(close, index=index).pct_change().to_numpy(),
            "quality_flags": [[] for _ in range(rows)],
            "is_valid": True,
        }
    )


def test_task2_acceptance_writes_four_strategy_and_threshold_results(tmp_path) -> None:
    data_dir = tmp_path / "data"
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = data_dir / "silver" / "binance" / "futures" / symbol / "1h" / "klines.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _silver(symbol).to_parquet(path, index=False)
    output = tmp_path / "report.json"
    report = run(data_dir=data_dir, output=output)
    assert output.exists()
    assert set(report["strategy_results"]) == {
        "trend_following",
        "grid_trading",
        "statistical_arbitrage",
        "mean_reversion",
    }
    assert all(
        row["consistency"]["status"] == "pass"
        for row in report["strategy_results"].values()
    )
    assert report["threshold_research"]["stable_interval"]
    assert report["oos_status"] == "not_run"
