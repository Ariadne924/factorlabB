from __future__ import annotations

import json

from scripts.run_panel_research import run


def test_panel_research_is_explicit_when_no_data(tmp_path) -> None:
    summary = run(tmp_path / "data", tmp_path / "reports")
    assert summary["status"] == "insufficient_data"
    assert summary["computed_report_count"] == 0
    assert summary["panel_factor_count"] >= 53
    stored = json.loads(
        (tmp_path / "reports" / "panel_research_summary.json").read_text("utf-8")
    )
    assert stored["research_note"].endswith("validated-alpha claim.")
