"""增量刷新现有研究数据，并可重新生成研究报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.catalog import build_data_catalog  # noqa: E402
from data.refresh import refresh_recent_market_data  # noqa: E402
from scripts.run_all_research import run as run_research  # noqa: E402


def main(
    *,
    dry_run: bool = False,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    intervals: tuple[str, ...] = ("1m", "5m", "1h"),
    lookback_hours: int = 24,
    data_dir: Path = PROJECT_ROOT / "data",
    reports_dir: Path = PROJECT_ROOT / "reports",
    refresh: Any | None = None,
    research: Any | None = None,
) -> int:
    plan = {
        "symbols": list(symbols), "intervals": list(intervals),
        "lookback_hours": lookback_hours, "data_dir": str(data_dir),
        "reports_dir": str(reports_dir),
    }
    if dry_run:
        print(json.dumps({"status": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return 0
    refresh_fn = refresh or refresh_recent_market_data
    research_fn = research or run_research
    result = refresh_fn(
        data_dir, symbols=symbols, intervals=intervals, lookback_hours=lookback_hours
    )
    build_data_catalog(data_dir, reports_dir / "data_catalog.json")
    if not result.get("successful"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    manifest = research_fn(data_dir, reports_dir, symbols=symbols, intervals=intervals)
    print(json.dumps({"refresh": result, "research": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增量刷新数据并更新研究报告")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "1h"])
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    raise SystemExit(
        main(
            dry_run=args.dry_run, symbols=tuple(args.symbols), intervals=tuple(args.intervals),
            lookback_hours=args.lookback_hours, data_dir=args.data_dir,
            reports_dir=args.reports_dir,
        )
    )
