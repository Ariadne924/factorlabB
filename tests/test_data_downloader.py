from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from data.downloader import DataDownloader
from data.schema import KlineRaw, raw_to_dataframe
from scripts.download_research_data import build_plan, main


class FakeBinanceClient:
    def __init__(self) -> None:
        self.base = datetime(2026, 1, 1, tzinfo=UTC)
        self.kline_calls = 0

    def fetch_futures_klines(self, interval, start_time, end_time, limit):
        del interval, limit
        self.kline_calls += 1
        records = []
        for hour in range(4):
            opened = int(self.base.timestamp() * 1000) + hour * 3_600_000
            if int(start_time.timestamp() * 1000) <= opened <= int(end_time.timestamp() * 1000):
                records.append(
                    KlineRaw.from_binance_row(
                        [
                            opened,
                            "100",
                            "102",
                            "99",
                            "101",
                            "10",
                            opened + 3_599_999,
                            "1000",
                            10,
                            "6",
                            "600",
                            "0",
                        ]
                    )
                )
        records = records[:2]
        return (
            raw_to_dataframe(records, symbol="BTCUSDT", interval="1h")
            if records
            else pd.DataFrame()
        )

    def fetch_funding_rate(self, start_time, end_time, limit):
        del limit
        return self._features(
            start_time,
            end_time,
            [(0, "funding_rate", 0.001), (2, "funding_rate", 0.002)],
        )

    def fetch_open_interest(self, period, start_time, end_time, limit):
        del period, limit
        return self._features(
            start_time,
            end_time,
            [(1, "open_interest", 1000.0), (2, "open_interest", 1100.0)],
        )

    def fetch_historical_basis(self, period, start_time, end_time, limit):
        del period, limit
        return self._features(
            start_time,
            end_time,
            [(0, "basis", 0.01), (1, "basis", 0.02)],
        )

    def _features(self, start, end, records):
        rows = []
        for hour, name, value in records:
            timestamp = pd.Timestamp(self.base) + pd.Timedelta(hours=hour)
            if pd.Timestamp(start) <= timestamp <= pd.Timestamp(end):
                rows.append({"timestamp": timestamp, "symbol": "BTCUSDT", name: value})
        return pd.DataFrame(rows)


def test_bundle_paginates_and_merges_features_without_basis_lookahead(tmp_path) -> None:
    fake = FakeBinanceClient()
    downloader = DataDownloader(data_dir=tmp_path, client_factory=lambda _: fake)
    manifest = downloader.download_bundle(
        symbol="BTCUSDT",
        interval="1h",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T03:00:00Z",
    )
    assert manifest["rows"] == {
        "klines": 4,
        "funding_rate": 2,
        "open_interest": 2,
        "basis": 2,
    }
    assert fake.kline_calls == 2
    silver = pd.read_parquet(manifest["paths"]["silver_klines"])
    assert len(silver) == 4
    assert silver.loc[0, "funding_rate"] == 0.001
    assert pd.isna(silver.loc[0, "open_interest"])
    assert pd.isna(silver.loc[0, "basis"])
    assert silver.loc[1, "basis"] == 0.01


def test_download_plan_is_safe_by_default(capsys) -> None:
    plan = build_plan(
        symbols=["btcusdt", "ethusdt"],
        interval="1h",
        start="2026-01-01",
        end="2026-01-02",
    )
    assert plan["status"] == "dry_run"
    assert plan["estimated_kline_rows_per_symbol"] == 25
    assert main(["--start", "2026-01-01", "--end", "2026-01-02"]) == 0
    assert '"status": "dry_run"' in capsys.readouterr().out
