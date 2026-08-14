"""锁定样本外区间的元数据门禁；不读取锁定数据。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_oos_lock(
    path: Path,
    *,
    start: str,
    end: str,
    symbols: tuple[str, ...],
    intervals: tuple[str, ...],
) -> dict[str, Any]:
    start_at, end_at = pd.Timestamp(start), pd.Timestamp(end)
    if start_at.tzinfo is None:
        start_at = start_at.tz_localize("UTC")
    else:
        start_at = start_at.tz_convert("UTC")
    if end_at.tzinfo is None:
        end_at = end_at.tz_localize("UTC")
    else:
        end_at = end_at.tz_convert("UTC")
    if end_at <= start_at:
        raise ValueError("OOS end 必须晚于 start")
    if end_at - start_at < pd.Timedelta(days=180):
        raise ValueError("锁定 OOS 区间必须至少覆盖 180 天")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected_scope = {
            "start": start_at.isoformat(),
            "end": end_at.isoformat(),
            "symbols": sorted({item.upper() for item in symbols}),
            "intervals": sorted({"1d" if item == "24h" else item for item in intervals}),
        }
        if any(existing.get(key) != value for key, value in expected_scope.items()):
            raise ValueError("现有 OOS 锁范围不同，不能覆盖")
        return existing
    payload = {
        "status": "locked_unopened",
        "created_at": datetime.now(UTC).isoformat(),
        "start": start_at.isoformat(),
        "end": end_at.isoformat(),
        "symbols": sorted({item.upper() for item in symbols}),
        "intervals": sorted({"1d" if item == "24h" else item for item in intervals}),
        "opened_at": None,
        "completed": False,
        "note": "开发期不得读取或用于调参；最终提交前只允许执行一次并如实报告。",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
