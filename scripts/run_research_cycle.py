"""从数据资格、研究报告、ML 训练到因子轮动的一键研究周期。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.catalog import build_data_catalog  # noqa: E402
from data.health import build_data_health_report  # noqa: E402
from data.training_readiness import build_training_readiness_report  # noqa: E402
from scripts.grade_factors import run as grade_factors  # noqa: E402
from scripts.run_all_research import run as run_all_research  # noqa: E402
from scripts.run_ml_factor_mining import run as run_ml_factor_mining  # noqa: E402


def _run(
    data_dir: Path,
    reports_dir: Path,
    *,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    run_ml: bool = True,
    rotation_days: int = 7,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_status_path = reports_dir / "training_status.json"
    training_status: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "stage": "data_health",
        "error": None,
    }
    training_status_path.write_text(
        json.dumps(training_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    catalog = build_data_catalog(data_dir, reports_dir / "data_catalog.json")
    health = build_data_health_report(
        catalog,
        data_dir,
        output=reports_dir / "data_health.json",
        manifest_path=reports_dir / "real_data_collection_manifest.json",
    )
    readiness = build_training_readiness_report(
        health, output=reports_dir / "training_readiness.json"
    )
    training_status["stage"] = "factor_research"
    training_status_path.write_text(
        json.dumps(training_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    research = run_all_research(
        data_dir,
        reports_dir,
        symbols=symbols,
        intervals=intervals,
    )
    ml: dict[str, Any] = {"status": "skipped"}
    if run_ml:
        ready_scopes = {
            (str(row["symbol"]), str(row["interval"]))
            for row in readiness["scopes"]
            if row["status"] == "ready"
        }
        selected_symbols = symbols or tuple(sorted({symbol for symbol, _ in ready_scopes}))
        selected_intervals = intervals or tuple(sorted({interval for _, interval in ready_scopes}))
        if ready_scopes:
            training_status["stage"] = "machine_learning"
            training_status["ready_scopes"] = sorted(
                f"{symbol}:{interval}" for symbol, interval in ready_scopes
            )
            training_status_path.write_text(
                json.dumps(training_status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ml = run_ml_factor_mining(
                data_dir,
                reports_dir,
                symbols=selected_symbols,
                intervals=selected_intervals,
                scopes=tuple(sorted(ready_scopes)),
            )
        else:
            ml = {"status": "insufficient_data", "results": []}
    grades = grade_factors(reports_dir, rotation_days=rotation_days)
    result = {
        "status": "ok",
        "data_health": health["summary"],
        "training_readiness": readiness["summary"],
        "research": {
            "factor_count": research.get("factor_count"),
            "lookahead_status": research.get("lookahead_status"),
        },
        "machine_learning": {
            "status": ml.get("status"),
            "result_count": len(ml.get("results", [])),
        },
        "factor_grades": grades["summary"],
        "oos_6_months_completed": False,
    }
    training_status.update(
        {
            "status": "completed",
            "stage": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "ml_status": ml.get("status"),
            "ml_result_count": len(ml.get("results", [])),
        }
    )
    training_status_path.write_text(
        json.dumps(training_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run(
    data_dir: Path,
    reports_dir: Path,
    *,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    run_ml: bool = True,
    rotation_days: int = 7,
) -> dict[str, Any]:
    """运行研究周期，并确保异常不会把训练状态永久留在 running。"""
    try:
        return _run(
            data_dir,
            reports_dir,
            symbols=symbols,
            intervals=intervals,
            run_ml=run_ml,
            rotation_days=rotation_days,
        )
    except Exception as exc:
        reports_dir.mkdir(parents=True, exist_ok=True)
        status_path = reports_dir / "training_status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
        status.update(
            {
                "status": "failed",
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="运行完整研究周期")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--intervals", nargs="+")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--rotation-days", type=int, default=7)
    args = parser.parse_args()
    result = run(
        args.data_dir,
        args.reports_dir,
        symbols=tuple(args.symbols) if args.symbols else None,
        intervals=tuple(args.intervals) if args.intervals else None,
        run_ml=not args.skip_ml,
        rotation_days=args.rotation_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
