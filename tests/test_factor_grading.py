from __future__ import annotations

from evaluation.factor_grading import build_factor_grade_report


def test_factor_grading_retains_high_low_and_missing_candidates() -> None:
    summary = {
        "results": [
            {
                "factor_name": "strong",
                "symbol": symbol,
                "display_frequency": "1h",
                "status": "computed",
                "sample": {"n_obs": 10_000},
                "metrics": {"rank_ic": 0.08, "icir": 0.8},
                "sign_consistency": 0.9,
                "group_monotonicity": 0.8,
                "multiple_testing": {"reject_fdr_5pct": True},
                "lookahead_status": "pass",
            }
            for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        ]
        + [
            {
                "factor_name": "weak",
                "symbol": "BTCUSDT",
                "display_frequency": "1h",
                "status": "computed_short_sample",
                "sample": {"n_obs": 200},
                "metrics": {"rank_ic": 0.002, "icir": 0.01},
                "sign_consistency": 0.2,
                "group_monotonicity": 0.1,
                "multiple_testing": {"reject_fdr_5pct": False},
                "lookahead_status": "pass",
            }
        ]
    }
    report = build_factor_grade_report(
        summary,
        factor_universe=[("single", "strong"), ("single", "weak"), ("single", "unused")],
        as_of="2026-08-01T00:00:00Z",
    )
    grades = {row["factor_name"]: row for row in report["factors"]}

    assert grades["strong"]["tier"] == "A"
    assert grades["weak"]["tier"] == "C"
    assert grades["unused"]["tier"] == "D"
    assert all(row["retained"] is True for row in report["factors"])
    assert report["summary"]["removed_factor_count"] == 0


def test_factor_grading_records_rotation_without_using_test_returns() -> None:
    previous = {"factors": [{"factor_id": "single:factor", "tier": "C"}]}
    report = build_factor_grade_report(
        {
            "results": [
                {
                    "factor_name": "factor",
                    "symbol": "BTCUSDT",
                    "display_frequency": "1h",
                    "status": "computed",
                    "sample": {"n_obs": 5000},
                    "metrics": {"rank_ic": 0.05, "icir": 0.5},
                    "sign_consistency": 0.8,
                    "group_monotonicity": 0.8,
                    "multiple_testing": {"reject_fdr_5pct": True},
                    "lookahead_status": "pass",
                }
            ]
        },
        factor_universe=[("single", "factor")],
        previous_report=previous,
        ml_reports=[
            {
                "feature_recommendations": [
                    {"feature": "factor__w24", "selection_frequency": 1.0, "sign_consistency": 1.0}
                ]
            }
        ],
        as_of="2026-08-01T00:00:00Z",
        rotation_days=7,
    )
    row = report["factors"][0]
    assert row["rotation_movement"] == "promoted"
    assert row["ml_train_stability"] == 1.0
    assert report["next_rotation_at"] == "2026-08-08T00:00:00+00:00"
