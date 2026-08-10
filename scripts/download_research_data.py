"""下载 Binance 永续研究数据；默认只展示计划，--execute 才发起请求。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import get_interval_ms  # noqa: E402
from data.downloader import DataDownloader  # noqa: E402


def build_plan(
    *, symbols: list[str], interval: str, start: str, end: str
) -> dict[str, object]:
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    if start_at.tzinfo is None:
        start_at = start_at.tz_localize("UTC")
    if end_at.tzinfo is None:
        end_at = end_at.tz_localize("UTC")
    if start_at >= end_at:
        raise ValueError("start 必须早于 end")
    estimated_bars = (
        int((end_at - start_at).total_seconds() * 1000 // get_interval_ms(interval)) + 1
    )
    return {
        "status": "dry_run",
        "symbols": [symbol.upper() for symbol in symbols],
        "market": "binance_usdt_perpetual",
        "interval": interval,
        "start": start_at.isoformat(),
        "end": end_at.isoformat(),
        "estimated_kline_rows_per_symbol": estimated_bars,
        "datasets": ["futures_klines", "funding_rate", "open_interest", "basis"],
        "limitations": [
            "Open interest and basis from Binance REST cover only about the latest 30 days.",
            "Raw parquet files are git-ignored and must not be uploaded to GitHub.",
            "Use --execute only when the date range and disk location are correct.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    today = datetime.now(UTC).date()
    parser = argparse.ArgumentParser(description="下载 Binance 永续因子研究数据")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default=str(today - timedelta(days=365)))
    parser.add_argument("--end", default=str(today))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(
        symbols=args.symbols, interval=args.interval, start=args.start, end=args.end
    )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    downloader = DataDownloader(data_dir=args.data_dir)
    results = [
        downloader.download_bundle(
            symbol=symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
        for symbol in args.symbols
    ]
    manifest = {**plan, "status": "downloaded", "results": results}
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reports_dir / "data_download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
