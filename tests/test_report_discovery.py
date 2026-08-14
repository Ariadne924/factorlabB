from __future__ import annotations

import json

from visualization.report_discovery import load_factor_reports, load_single_factor_index


def test_report_discovery_repairs_scope_and_hides_unknown_legacy(tmp_path) -> None:
    scoped = {"factor_name": "momentum", "metrics": {"rank_ic": 0.1}}
    legacy = {"factor_name": "basis", "metrics": {"rank_ic": 0.2}}
    (tmp_path / "BTCUSDT_1h_momentum.json").write_text(
        json.dumps(scoped), encoding="utf-8"
    )
    (tmp_path / "basis.json").write_text(json.dumps(legacy), encoding="utf-8")

    reports = load_factor_reports(tmp_path, require_symbol=True)
    assert len(reports) == 1
    assert reports[0]["symbol"] == "BTCUSDT"
    assert reports[0]["interval"] == "1h"
    assert reports[0]["display_frequency"] == "1h"


def test_single_factor_index_uses_summary_without_loading_report_body(tmp_path) -> None:
    report_directory = tmp_path / "single_factor"
    report_directory.mkdir()
    report_path = report_directory / "BTCUSDT_1h_momentum.json"
    report_path.write_text("not parsed by index", encoding="utf-8")
    summary = {
        "results": [
            {
                "report": "single_factor/BTCUSDT_1h_momentum.json",
                "factor_name": "momentum",
                "symbol": "BTCUSDT",
                "interval": "1h",
                "category": "momentum",
                "metrics": {"rank_ic": 0.1},
                "multiple_testing": {"reject_fdr_5pct": False},
            }
        ]
    }
    summary_path = tmp_path / "research_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    reports = load_single_factor_index(summary_path, report_directory)

    assert len(reports) == 1
    assert reports[0]["_file"] == report_path.name
    assert reports[0]["provenance"]["category"] == "momentum"
    assert reports[0]["robustness"]["multiple_testing"]["reject_fdr_5pct"] is False
