"""检查研究汇总引用的报告是否完整、可解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_report_generation(reports_dir: Path) -> dict[str, Any]:
    summary_path = reports_dir / "research_summary.json"
    if not summary_path.exists():
        return {"status": "missing", "referenced": 0, "valid": 0, "errors": []}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "corrupt", "referenced": 0, "valid": 0, "errors": [str(exc)]}
    errors: list[dict[str, str]] = []
    valid = 0
    results = summary.get("results", [])
    for row in results:
        relative = str(row.get("report", ""))
        path = reports_dir / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("factor_name") != row.get("factor_name"):
                raise ValueError("factor_name 与汇总不一致")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"report": relative, "error": str(exc)})
        else:
            valid += 1
    actual_files = len(list((reports_dir / "single_factor").glob("*.json")))
    return {
        "status": "ok" if not errors else "partial",
        "generated_scope_count": len(
            {(row.get("symbol"), row.get("interval")) for row in results}
        ),
        "referenced": len(results),
        "valid": valid,
        "errors": errors,
        "unreferenced_old_files": max(0, actual_files - len(results)),
        "note": "未引用文件是历史运行产物，不会混入当前汇总。",
    }
