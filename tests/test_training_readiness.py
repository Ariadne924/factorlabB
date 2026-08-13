from __future__ import annotations

from data.training_readiness import (
    TrainingSplitPolicy,
    build_training_readiness_report,
)


def test_training_readiness_separates_ready_and_blocked_scopes(tmp_path) -> None:
    health = {
        "generated_at": "2026-08-01T00:00:00Z",
        "datasets": [
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T23:00:00Z",
                "history_days": 365,
                "coverage_ratio": 1.0,
                "missing_bars": 0,
            },
            {
                "symbol": "ETHUSDT",
                "interval": "1m",
                "start": "2025-12-01T00:00:00Z",
                "end": "2025-12-31T23:59:00Z",
                "history_days": 31,
                "coverage_ratio": 0.99,
                "missing_bars": 0,
            },
        ],
    }
    output = tmp_path / "training_readiness.json"
    report = build_training_readiness_report(health, output=output)

    assert report["summary"] == {
        "scope_count": 2,
        "ready_scope_count": 1,
        "blocked_scope_count": 1,
        "locked_oos_available_count": 1,
    }
    assert report["scopes"][0]["split"]["locked_oos_end"] == "2025-12-31T23:00:00+00:00"
    assert report["scopes"][1]["status"] == "blocked"
    assert "历史仅" in report["scopes"][1]["reasons"][0]
    assert report["locked_oos_completed"] is False
    assert output.exists()


def test_training_policy_validates_total_window() -> None:
    policy = TrainingSplitPolicy(min_train_days=120, validation_days=30, locked_oos_days=30)
    assert policy.required_history_days == 180
