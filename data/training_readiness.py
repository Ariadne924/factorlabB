"""为机器学习研究生成固定、可审计的数据切分计划。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import get_interval_ms

TRAINING_READINESS_VERSION = "1.0"


@dataclass(frozen=True)
class TrainingSplitPolicy:
    min_train_days: int = 120
    validation_days: int = 30
    locked_oos_days: int = 30
    min_coverage_ratio: float = 0.98

    def validate(self) -> None:
        if min(self.min_train_days, self.validation_days, self.locked_oos_days) < 1:
            raise ValueError("训练、验证和锁定 OOS 天数必须为正整数")
        if not 0 < self.min_coverage_ratio <= 1:
            raise ValueError("min_coverage_ratio 必须在 (0, 1] 内")

    @property
    def required_history_days(self) -> int:
        return self.min_train_days + self.validation_days + self.locked_oos_days


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _scope_plan(
    dataset: dict[str, Any], policy: TrainingSplitPolicy
) -> dict[str, Any]:
    interval = str(dataset["interval"])
    step = pd.Timedelta(milliseconds=get_interval_ms(interval))
    start, end = _utc(dataset["start"]), _utc(dataset["end"])
    history_days = float(dataset.get("history_days") or 0.0)
    coverage = float(dataset.get("coverage_ratio") or 0.0)
    missing_bars = int(dataset.get("missing_bars") or 0)
    reasons: list[str] = []
    if coverage < policy.min_coverage_ratio:
        reasons.append(f"覆盖率 {coverage:.2%} 低于 {policy.min_coverage_ratio:.0%}")
    if missing_bars:
        reasons.append(f"存在 {missing_bars} 根内部缺口")
    if history_days < policy.required_history_days:
        reasons.append(
            f"历史仅 {history_days:.1f} 天，固定切分至少需要 {policy.required_history_days} 天"
        )
    ready = not reasons
    split: dict[str, str] | None = None
    if ready:
        locked_oos_start = end - pd.Timedelta(days=policy.locked_oos_days) + step
        validation_end = locked_oos_start - step
        validation_start = validation_end - pd.Timedelta(days=policy.validation_days) + step
        train_end = validation_start - step
        split = {
            "train_start": start.isoformat(),
            "train_end": train_end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "locked_oos_start": locked_oos_start.isoformat(),
            "locked_oos_end": end.isoformat(),
        }
    return {
        "symbol": str(dataset["symbol"]),
        "interval": interval,
        "display_interval": "24h" if interval == "1d" else interval,
        "status": "ready" if ready else "blocked",
        "history_days": history_days,
        "coverage_ratio": coverage,
        "missing_bars": missing_bars,
        "reasons": reasons,
        "split": split,
        "locked_oos_status": "available_not_evaluated" if ready else "unavailable",
    }


def build_training_readiness_report(
    health: dict[str, Any],
    *,
    policy: TrainingSplitPolicy | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """将数据健康结果转换为固定训练/验证/锁定 OOS 计划。"""
    selected_policy = policy or TrainingSplitPolicy()
    selected_policy.validate()
    scopes = [_scope_plan(dict(row), selected_policy) for row in health.get("datasets", [])]
    ready = [row for row in scopes if row["status"] == "ready"]
    report = {
        "version": TRAINING_READINESS_VERSION,
        "generated_at": health.get("generated_at"),
        "policy": {
            **asdict(selected_policy),
            "required_history_days": selected_policy.required_history_days,
        },
        "summary": {
            "scope_count": len(scopes),
            "ready_scope_count": len(ready),
            "blocked_scope_count": len(scopes) - len(ready),
            "locked_oos_available_count": len(ready),
        },
        "scopes": scopes,
        "locked_oos_completed": False,
        "note": (
            "available_not_evaluated 只表示已预留窗口，不表示完成六个月 OOS，"
            "也不表示模型或因子已经验证有效。"
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
