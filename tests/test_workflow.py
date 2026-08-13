from __future__ import annotations

from visualization.workflow import (
    build_data_tasks,
    build_workflow_steps,
    next_workflow_action,
    workflow_progress,
)


def health_report(*, covered: int = 25, ready: int = 23, gaps: int = 1):
    return {
        "summary": {
            "core_covered_datasets": covered,
            "core_target_datasets": 30,
            "research_ready_datasets": ready,
            "fresh_datasets": 0,
            "gap_datasets": gaps,
            "derivative_datasets": 0,
        },
        "missing_scopes": [
            {"symbol": "ETHUSDT", "interval": "1m"}
        ] if covered < 30 else [],
        "datasets": [
            {"symbol": "BTCUSDT", "interval": "5m", "missing_bars": 12}
        ] if gaps else [],
    }


def test_workflow_skips_optional_factor_review_for_next_action() -> None:
    steps = build_workflow_steps(
        health_report(covered=30, ready=30, gaps=0),
        factor_report_count=0,
        has_strategy_result=False,
        has_robustness_result=False,
        has_walk_forward_result=False,
    )
    assert steps[0]["state"] == "completed"
    assert steps[1]["optional"] is True
    assert next_workflow_action(steps)["key"] == "strategy"
    assert workflow_progress(steps) == 1 / 3


def test_workflow_requires_validation_after_strategy_result() -> None:
    steps = build_workflow_steps(
        health_report(covered=30, ready=30, gaps=0),
        factor_report_count=18,
        has_strategy_result=True,
        has_robustness_result=False,
        has_walk_forward_result=False,
    )
    assert next_workflow_action(steps)["key"] == "validation"
    assert next_workflow_action(steps)["state"] == "ready"
    assert workflow_progress(steps) == 2 / 3


def test_workflow_completion_does_not_depend_on_optional_factor_step() -> None:
    steps = build_workflow_steps(
        health_report(covered=30, ready=30, gaps=0),
        factor_report_count=0,
        has_strategy_result=True,
        has_robustness_result=True,
        has_walk_forward_result=True,
    )
    assert workflow_progress(steps) == 1.0
    assert steps[-1]["state"] == "completed"


def test_data_tasks_prioritise_missing_scopes_and_gaps() -> None:
    tasks = build_data_tasks(health_report())
    assert [row["priority"] for row in tasks[:2]] == ["P0", "P0"]
    assert "ETHUSDT·1m" in tasks[0]["detail"]
    assert "12" in tasks[1]["detail"]
    assert any(row["task"] == "刷新近端行情" for row in tasks)
    assert any(row["task"] == "补充衍生品特征" for row in tasks)
