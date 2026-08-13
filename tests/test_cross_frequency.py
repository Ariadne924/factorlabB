from __future__ import annotations

from evaluation.cross_frequency import build_cross_frequency_summary, display_frequency


def test_frequency_label_and_cross_frequency_summary() -> None:
    assert display_frequency("1d") == "24h"
    rows = [
        {
            "factor_name": "momentum",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "status": "computed_short_sample",
            "metrics": {"rank_ic": 0.1},
            "lookahead_status": "pass",
            "multiple_testing": {"reject_fdr_5pct": False},
        },
        {
            "factor_name": "momentum",
            "symbol": "ETHUSDT",
            "interval": "1d",
            "status": "computed_short_sample",
            "metrics": {"rank_ic": 0.2},
            "lookahead_status": "pass",
            "multiple_testing": {"reject_fdr_5pct": True},
        },
    ]
    summary = build_cross_frequency_summary(rows)[0]
    assert summary["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert set(summary["frequencies"]) == {"1h", "24h"}
    assert summary["direction_consistency"] == 1.0
    assert summary["fdr_5pct_pass_count"] == 1
