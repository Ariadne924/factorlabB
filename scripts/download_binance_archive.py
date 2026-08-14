"""按月下载 Binance 官方公开 K 线归档。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.binance_archive import BinanceArchiveDownloader  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 data.binance.vision 月度 K 线")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--market", choices=("spot", "futures"), default="futures")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = {
        "status": "dry_run", "source": "data.binance.vision", "symbol": args.symbol.upper(),
        "interval": args.interval, "market": args.market, "year": args.year,
        "months": args.months,
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    downloader = BinanceArchiveDownloader(data_dir=args.data_dir)
    results = [
        downloader.download_month(
            symbol=args.symbol, interval=args.interval, year=args.year,
            month=month, market=args.market,
        )
        for month in args.months
    ]
    print(
        json.dumps(
            {**plan, "status": "downloaded", "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
