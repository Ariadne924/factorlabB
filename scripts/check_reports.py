"""检查当前研究汇总引用的单因子报告是否完整。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.report_health import inspect_report_generation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查研究报告引用、JSON 和因子名一致性")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    health = inspect_report_generation(args.reports_dir)
    print(json.dumps(health, ensure_ascii=False, indent=2))
    return 0 if health["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
