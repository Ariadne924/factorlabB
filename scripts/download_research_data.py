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
from data.catalog import build_data_catalog, missing_download_ranges  # noqa: E402
from data.downloader import DataDownloader  # noqa: E402

DEFAULT_RESEARCH_INTERVALS = ("1m", "5m", "15m", "1h", "6h", "1d")
STARTER_RESEARCH_SYMBOLS = ("BTCUSDT", "ETHUSDT")
CORE_RESEARCH_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
)


def normalize_research_interval(interval: str) -> str:
    """接受业务口径 24h，并映射到 Binance 的 1d。"""
    normalized = interval.strip()
    return "1d" if normalized == "24h" else normalized


def build_plan(
    *, symbols: list[str], interval: str, start: str, end: str
) -> dict[str, object]:
    interval = normalize_research_interval(interval)
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


def build_multi_frequency_plan(
    *, symbols: list[str], intervals: list[str], start: str, end: str
) -> dict[str, object]:
    """构建多频率下载计划；本函数不联网也不写行情文件。"""
    normalized = list(dict.fromkeys(normalize_research_interval(item) for item in intervals))
    plans = [
        build_plan(symbols=symbols, interval=interval, start=start, end=end)
        for interval in normalized
    ]
    estimated_total_rows = 0
    for plan in plans:
        estimated_rows = plan["estimated_kline_rows_per_symbol"]
        if not isinstance(estimated_rows, int):
            raise TypeError("estimated_kline_rows_per_symbol must be an integer")
        estimated_total_rows += estimated_rows * len(symbols)
    return {
        "status": "dry_run",
        "symbols": [symbol.upper() for symbol in symbols],
        "intervals": normalized,
        "display_intervals": ["24h" if item == "1d" else item for item in normalized],
        "start": plans[0]["start"] if plans else start,
        "end": plans[0]["end"] if plans else end,
        "plans": plans,
        "estimated_total_kline_rows": estimated_total_rows,
        "limitations": [
            "1m 长区间数据量较大，执行前请确认时间范围与磁盘空间。",
            "Open interest and basis history may be unavailable for some intervals.",
            "Raw parquet files are git-ignored and must not be uploaded to GitHub.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    today = datetime.now(UTC).date()
    parser = argparse.ArgumentParser(description="下载 Binance 永续因子研究数据")
    parser.add_argument("--symbols", nargs="+", help="显式资产列表；传入后覆盖 --universe")
    parser.add_argument(
        "--universe",
        choices=("starter", "core"),
        default="starter",
        help="starter=BTC/ETH；core=12个主流永续。默认 starter",
    )
    parser.add_argument("--interval", help="兼容旧命令的单一频率，例如 1h 或 24h")
    parser.add_argument(
        "--intervals",
        nargs="+",
        help="多频率列表；默认 1m 5m 15m 1h 6h 24h",
    )
    parser.add_argument("--start", default=str(today - timedelta(days=365)))
    parser.add_argument("--end", default=str(today))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    symbols = args.symbols or list(
        CORE_RESEARCH_SYMBOLS if args.universe == "core" else STARTER_RESEARCH_SYMBOLS
    )
    intervals = args.intervals or (
        [args.interval] if args.interval else list(DEFAULT_RESEARCH_INTERVALS)
    )
    plan = build_multi_frequency_plan(
        symbols=symbols, intervals=intervals, start=args.start, end=args.end
    )
    plan["universe"] = "custom" if args.symbols else args.universe
    plan["survivorship_note"] = (
        "预设币池按当前研究范围固定，仅用于工程与候选研究；"
        "不能据此声称完成无幸存者偏差的历史横截面回测。"
    )
    catalog_path = args.reports_dir / "data_catalog.json"
    catalog = build_data_catalog(args.data_dir, catalog_path)
    download_ranges: list[dict[str, str]] = []
    for symbol in symbols:
        for interval in intervals:
            normalized = normalize_research_interval(interval)
            for missing in missing_download_ranges(
                catalog,
                symbol=symbol,
                interval=normalized,
                start=args.start,
                end=args.end,
            ):
                download_ranges.append(
                    {
                        "symbol": symbol.upper(),
                        "interval": normalized,
                        "start": missing["start"],
                        "end": missing["end"],
                    }
                )
    plan["catalog"] = str(catalog_path)
    plan["download_ranges"] = download_ranges
    plan["up_to_date"] = not download_ranges
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    downloader = DataDownloader(data_dir=args.data_dir)
    results = []
    for item in download_ranges:
        results.append(
            downloader.download_bundle(
                symbol=item["symbol"],
                interval=item["interval"],
                start=item["start"],
                end=item["end"],
            )
        )
    refreshed_catalog = build_data_catalog(args.data_dir, catalog_path)
    manifest = {
        **plan,
        "status": "downloaded" if results else "up_to_date",
        "results": results,
        "catalog_entry_count": refreshed_catalog["entry_count"],
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reports_dir / "data_download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
