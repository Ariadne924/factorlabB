from __future__ import annotations

import json

import pandas as pd

from data.health import build_data_health_report


def test_health_separates_history_freshness_and_ml_readiness(tmp_path) -> None:
    catalog = {
        "entries": [
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T00:00:00+00:00",
                "end": "2025-07-01T00:00:00+00:00",
                "rows": 4345,
                "expected_rows": 4345,
                "missing_bars": 0,
                "coverage_ratio": 1.0,
            },
            {
                "symbol": "ETHUSDT",
                "interval": "1m",
                "start": "2025-06-30T00:00:00+00:00",
                "end": "2025-07-01T00:00:00+00:00",
                "rows": 1441,
                "expected_rows": 1441,
                "missing_bars": 0,
                "coverage_ratio": 1.0,
            },
        ],
        "errors": [],
    }
    report = build_data_health_report(
        catalog,
        tmp_path,
        as_of="2025-07-01T04:01:00+00:00",
        core_symbols=("BTCUSDT", "ETHUSDT"),
        core_intervals=("1m", "1h"),
    )
    btc, eth = report["datasets"]
    assert btc["research_ready"] is True
    assert btc["ml_eligible"] is True
    assert btc["fresh"] is False
    assert eth["research_ready"] is False
    assert eth["ml_eligible"] is False
    assert report["summary"]["core_target_datasets"] == 4
    assert report["summary"]["core_covered_datasets"] == 2
    assert len(report["missing_scopes"]) == 2


def test_health_reads_derivative_coverage_and_collection_manifest(tmp_path) -> None:
    feature = tmp_path / "bronze/binance/futures/BTCUSDT/1h/funding_rate.parquet"
    feature.parent.mkdir(parents=True)
    feature.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "completed", "mode": "rest", "errors": []}),
        encoding="utf-8",
    )

    def reader(path, *, columns):
        assert path == feature
        assert columns == ["timestamp"]
        return pd.DataFrame({"timestamp": ["2025-07-01T00:00:00Z"]})

    report = build_data_health_report(
        {"entries": [], "errors": []},
        tmp_path,
        manifest_path=manifest,
        as_of="2025-07-01T12:00:00Z",
        parquet_reader=reader,
    )
    assert report["collection"]["status"] == "completed"
    assert report["summary"]["derivative_datasets"] == 1
    assert report["derivatives"][0]["recent_48h"] is True
