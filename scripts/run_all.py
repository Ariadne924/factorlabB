"""兼容的一键入口：复用正式研究管道，不再保留占位步骤。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_all_research import run as run_research  # noqa: E402


def main(
    *,
    dry_run: bool = False,
    data_dir: Path = PROJECT_ROOT / "data",
    reports_dir: Path = PROJECT_ROOT / "reports",
    research: Any | None = None,
) -> int:
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run", "pipeline": ["Silver", "Gold", "Evaluate", "Report"],
                    "note": "行情下载是显式外部步骤；本入口只消费已有 Silver 数据。",
                }, ensure_ascii=False, indent=2,
            )
        )
        return 0
    manifest = (research or run_research)(data_dir, reports_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行已有 Silver 数据的完整研究管道")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    raise SystemExit(
        main(dry_run=args.dry_run, data_dir=args.data_dir, reports_dir=args.reports_dir)
    )
