from __future__ import annotations

import json

from visualization.report_health import inspect_report_generation


def test_report_health_finds_missing_references(tmp_path) -> None:
    single = tmp_path / "single_factor"
    single.mkdir()
    valid = {"factor_name": "momentum"}
    (single / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    summary = {
        "results": [
            {
                "report": "single_factor/valid.json",
                "factor_name": "momentum",
                "symbol": "BTCUSDT",
                "interval": "1h",
            },
            {
                "report": "single_factor/missing.json",
                "factor_name": "rsi",
                "symbol": "BTCUSDT",
                "interval": "1h",
            },
        ]
    }
    (tmp_path / "research_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    health = inspect_report_generation(tmp_path)
    assert health["status"] == "partial"
    assert health["valid"] == 1
