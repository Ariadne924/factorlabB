"""Refresh a bounded trailing market-data window for UI/research use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.catalog import build_data_catalog  # noqa: E402
from data.refresh import refresh_recent_market_data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh recent Binance research data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "DOGEUSDT",
        ],
    )
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "1h"])
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args(argv)
    result = refresh_recent_market_data(
        args.data_dir,
        symbols=tuple(args.symbols),
        intervals=tuple(args.intervals),
        lookback_hours=args.lookback_hours,
    )
    build_data_catalog(args.data_dir, args.reports_dir / "data_catalog.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
