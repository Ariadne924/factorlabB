"""User-facing workflow state derived from research artifacts and session results."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

WorkflowState = Literal["blocked", "attention", "ready", "completed"]


class WorkflowStep(TypedDict):
    key: str
    label: str
    state: WorkflowState
    summary: str
    action_label: str
    action_page: str
    optional: bool


def _data_step(health: dict[str, Any]) -> WorkflowStep:
    summary = health.get("summary", {})
    covered = int(summary.get("core_covered_datasets") or 0)
    target = int(summary.get("core_target_datasets") or 0)
    ready = int(summary.get("research_ready_datasets") or 0)
    gaps = int(summary.get("gap_datasets") or 0)
    missing = max(0, target - covered)
    if target > 0 and missing == 0 and gaps == 0 and ready >= target:
        state: WorkflowState = "completed"
        message = f"核心 {covered}/{target} 组数据均达到研究门槛。"
    elif ready > 0:
        state = "attention"
        details = []
        if missing:
            details.append(f"缺 {missing} 组口径")
        if gaps:
            details.append(f"{gaps} 组存在缺口")
        message = f"已有 {ready} 组可研究；" + "、".join(details or ["仍需检查新鲜度"])
    else:
        state = "blocked"
        message = "尚无达到研究门槛的数据。"
    return {
        "key": "data",
        "label": "准备数据",
        "state": state,
        "summary": message,
        "action_label": "处理数据任务" if state != "completed" else "查看数据状态",
        "action_page": "数据中心",
        "optional": False,
    }


def build_workflow_steps(
    health: dict[str, Any],
    *,
    factor_report_count: int,
    has_strategy_result: bool,
    has_robustness_result: bool,
    has_walk_forward_result: bool,
) -> list[WorkflowStep]:
    """Build the four research stages without claiming statistical validation."""
    data = _data_step(health)
    research_ready = int(health.get("summary", {}).get("research_ready_datasets") or 0)
    factor_state: WorkflowState = "ready" if factor_report_count else "attention"
    factor: WorkflowStep = {
        "key": "factor",
        "label": "选择候选因子",
        "state": factor_state,
        "summary": (
            f"已有 {factor_report_count} 份报告可参考；本轮暂不扩展因子检验。"
            if factor_report_count
            else "尚无报告；也可以直接用正式 registry 构建策略。"
        ),
        "action_label": "浏览候选因子",
        "action_page": "因子检验",
        "optional": True,
    }
    if has_strategy_result:
        strategy_state: WorkflowState = "completed"
        strategy_summary = "当前会话已有策略回测结果。"
    elif research_ready:
        strategy_state = "ready"
        strategy_summary = "数据已可用，可以选择因子、权重和回测区间。"
    else:
        strategy_state = "blocked"
        strategy_summary = "先完成至少一组可研究数据。"
    strategy: WorkflowStep = {
        "key": "strategy",
        "label": "构建策略",
        "state": strategy_state,
        "summary": strategy_summary,
        "action_label": "继续构建策略" if has_strategy_result else "开始构建策略",
        "action_page": "策略研究",
        "optional": False,
    }
    if has_strategy_result and has_robustness_result and has_walk_forward_result:
        validation_state: WorkflowState = "completed"
        validation_summary = "已生成基础稳健性和策略 Walk-Forward 结果。"
    elif has_strategy_result:
        validation_state = "ready"
        validation_summary = "回测已完成，下一步检查成本、状态和 Walk-Forward。"
    else:
        validation_state = "blocked"
        validation_summary = "需要先生成策略回测结果。"
    validation: WorkflowStep = {
        "key": "validation",
        "label": "验证与比较",
        "state": validation_state,
        "summary": validation_summary,
        "action_label": "查看策略验证",
        "action_page": "策略研究",
        "optional": False,
    }
    return [data, factor, strategy, validation]


def next_workflow_action(steps: list[WorkflowStep]) -> WorkflowStep:
    """Return the first required unfinished step, allowing factor review to be deferred."""
    for step in steps:
        if not step["optional"] and step["state"] != "completed":
            return step
    return steps[-1]


def workflow_progress(steps: list[WorkflowStep]) -> float:
    required = [step for step in steps if not step["optional"]]
    completed = sum(step["state"] == "completed" for step in required)
    return completed / len(required) if required else 1.0


def build_data_tasks(health: dict[str, Any]) -> list[dict[str, str]]:
    """Turn health findings into a short, prioritised user task queue."""
    summary = health.get("summary", {})
    tasks: list[dict[str, str]] = []
    missing_scopes = health.get("missing_scopes", [])
    if missing_scopes:
        labels = [
            f"{row['symbol']}·{'24h' if row['interval'] == '1d' else row['interval']}"
            for row in missing_scopes
        ]
        tasks.append(
            {
                "priority": "P0",
                "state": "需处理",
                "task": "补齐核心研究口径",
                "detail": "、".join(labels),
                "action": "运行可恢复数据采集",
            }
        )
    gap_rows = [row for row in health.get("datasets", []) if row.get("missing_bars", 0)]
    if gap_rows:
        missing_bars = sum(int(row.get("missing_bars", 0)) for row in gap_rows)
        tasks.append(
            {
                "priority": "P0",
                "state": "需处理",
                "task": "修复内部 K 线缺口",
                "detail": f"{len(gap_rows)} 组数据共缺 {missing_bars} 根 K 线",
                "action": "按缺口范围增量补数",
            }
        )
    fresh = int(summary.get("fresh_datasets") or 0)
    ready = int(summary.get("research_ready_datasets") or 0)
    if ready and fresh < ready:
        tasks.append(
            {
                "priority": "P1",
                "state": "可选",
                "task": "刷新近端行情",
                "detail": f"{ready} 组可研究数据中仅 {fresh} 组达到新鲜度门槛",
                "action": "刷新最近 24 小时",
            }
        )
    if not int(summary.get("derivative_datasets") or 0):
        tasks.append(
            {
                "priority": "P1",
                "state": "可选",
                "task": "补充衍生品特征",
                "detail": "Funding Rate / Open Interest / Basis 尚未落盘",
                "action": "执行 REST 补数",
            }
        )
    if not tasks:
        tasks.append(
            {
                "priority": "—",
                "state": "已完成",
                "task": "核心数据任务已完成",
                "detail": "覆盖率、内部缺口和基础衍生品文件均已通过检查",
                "action": "进入策略构建",
            }
        )
    return tasks
