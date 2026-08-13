from __future__ import annotations

import json

from visualization.report_discovery import load_factor_reports


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
