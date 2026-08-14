"""基于已有研究与 ML 训练折结果轮动因子等级。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import factors  # noqa: E402,F401
from evaluation.factor_grading import (  # noqa: E402
    build_factor_grade_report,
    write_factor_grade_report,
)
from factors.panel_registry import list_panel_factors  # noqa: E402
from factors.registry import list_factors  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(reports_dir: Path, *, rotation_days: int = 7) -> dict[str, Any]:
    output_dir = reports_dir / "factor_grades"
    previous = _load(output_dir / "latest.json")
    ml_reports = [
        _load(path) for path in sorted((reports_dir / "ml_factor").glob("*.json"))
    ]
    universe = [
        *(("single", name) for name in list_factors()),
        *(("panel", name) for name in list_panel_factors()),
    ]
    report = build_factor_grade_report(
        _load(reports_dir / "research_summary.json"),
        factor_universe=universe,
        ml_reports=ml_reports,
        previous_report=previous,
        rotation_days=rotation_days,
    )
    latest, history = write_factor_grade_report(report, output_dir)
    report["paths"] = {
        "latest": latest.relative_to(reports_dir).as_posix(),
        "history": history.relative_to(reports_dir).as_posix(),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="轮动全部保留因子的研究优先级")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--rotation-days", type=int, default=7)
    args = parser.parse_args()
    report = run(args.reports_dir, rotation_days=args.rotation_days)
    print(
        json.dumps(
            {"summary": report["summary"], "paths": report["paths"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
