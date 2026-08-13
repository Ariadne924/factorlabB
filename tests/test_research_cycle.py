from __future__ import annotations

import json

import pytest

import scripts.run_research_cycle as cycle


def test_research_cycle_runs_stages_in_order_without_claiming_six_month_oos(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cycle, "build_data_catalog", lambda *_args, **_kwargs: {"entries": []})

    def health(*_args, **_kwargs):
        calls.append("health")
        return {"summary": {"ml_eligible_datasets": 0}, "datasets": []}

    def readiness(*_args, **_kwargs):
        calls.append("readiness")
        return {"summary": {"ready_scope_count": 0}, "scopes": []}

    def research(*_args, **_kwargs):
        calls.append("research")
        return {"factor_count": 466, "lookahead_status": "pass"}

    def grades(*_args, **_kwargs):
        calls.append("grades")
        return {"summary": {"retained_factor_count": 466, "removed_factor_count": 0}}

    monkeypatch.setattr(cycle, "build_data_health_report", health)
    monkeypatch.setattr(cycle, "build_training_readiness_report", readiness)
    monkeypatch.setattr(cycle, "run_all_research", research)
    monkeypatch.setattr(cycle, "grade_factors", grades)

    result = cycle.run(tmp_path / "data", tmp_path / "reports")

    assert calls == ["health", "readiness", "research", "grades"]
    assert result["machine_learning"]["status"] == "insufficient_data"
    assert result["factor_grades"]["removed_factor_count"] == 0
    assert result["oos_6_months_completed"] is False


def test_research_cycle_records_failure_in_training_status(tmp_path, monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise ValueError("broken catalog")

    monkeypatch.setattr(cycle, "build_data_catalog", fail)
    reports_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="broken catalog"):
        cycle.run(tmp_path / "data", reports_dir)

    status = json.loads((reports_dir / "training_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stage"] == "data_health"
    assert status["error"] == "ValueError: broken catalog"
