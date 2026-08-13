"""Generate the local data-health report without downloading anything."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.catalog import build_data_catalog  # noqa: E402
from data.health import build_data_health_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成本地数据健康与可研究性报告")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    catalog = build_data_catalog(args.data_dir, args.reports_dir / "data_catalog.json")
    report = build_data_health_report(
        catalog,
        args.data_dir,
        output=args.reports_dir / "data_health.json",
        manifest_path=args.reports_dir / "real_data_collection_manifest.json",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
