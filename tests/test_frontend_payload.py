from __future__ import annotations

import json

from visualization.frontend_payload import build_frontend_payload, write_frontend_payload


def test_frontend_payload_is_versioned_and_filterable(tmp_path) -> None:
    summary = {
        "status": "ok",
        "factor_count": 53,
        "computed_report_count": 2,
        "fdr_5pct_pass_count": 0,
        "research_note": "not validated alpha",
        "data_health": "data_health.json",
        "data_health_summary": {"core_covered_datasets": 25},
        "results": [
            {
                "factor_name": "basis_zscore",
                "symbol": "BTCUSDT",
                "display_frequency": "1h",
                "category": "加密货币特有",
                "source": "project",
            },
            {
                "factor_name": "basis_zscore",
                "symbol": "ETHUSDT",
                "display_frequency": "24h",
                "category": "加密货币特有",
                "source": "project",
            },
        ],
        "cross_frequency": [{"factor_name": "basis_zscore"}],
    }
    payload = build_frontend_payload(summary)
    assert payload["contract_version"] == "1.2"
    assert payload["filters"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["filters"]["frequencies"] == ["1h", "24h"]
    assert payload["overview"]["factor_count"] == 53
    assert payload["panel_factor_results"] == []
    assert payload["data_health"]["path"] == "data_health.json"
    assert payload["data_health"]["summary"]["core_covered_datasets"] == 25

    output = write_frontend_payload(summary, tmp_path / "frontend_payload.json")
    assert json.loads(output.read_text("utf-8"))["disclaimer"] == "not validated alpha"
