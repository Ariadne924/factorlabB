"""可恢复的多资产、多频率真实数据采集入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.binance_archive import BinanceArchiveDownloader  # noqa: E402
from data.catalog import build_data_catalog  # noqa: E402
from data.downloader import DataDownloader  # noqa: E402

DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
)
DEFAULT_INTERVALS = ("1m", "5m", "1h", "6h", "1d")
ARCHIVE_INTERVALS = frozenset({"5m", "1h", "6h", "1d"})
DATA_SOURCE_PROBES = {
    "binance_futures_rest": "https://fapi.binance.com/fapi/v1/time",
    "binance_spot_rest": "https://api.binance.com/api/v3/time",
    "binance_public_archive": (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2025-01.zip.CHECKSUM"
    ),
    "okx_public_rest": "https://www.okx.com/api/v5/public/time",
}


class ArchiveLike(Protocol):
    def download_month(
        self, *, symbol: str, interval: str, year: int, month: int, market: str
    ) -> dict[str, object]: ...


class RestLike(Protocol):
    def download_bundle(
        self, *, symbol: str, interval: str, start: datetime, end: datetime
    ) -> dict[str, Any]: ...


def probe_data_sources(
    *, session: requests.Session | None = None, timeout: float = 8.0
) -> dict[str, Any]:
    """分别探测数据源，便于比较 VPN 开/关时的实际连通性。"""
    client = session or requests.Session()
    results: list[dict[str, Any]] = []
    for name, url in DATA_SOURCE_PROBES.items():
        started = time.perf_counter()
        try:
            response = client.get(url, timeout=timeout)
            elapsed = int((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "source": name,
                    "url": url,
                    "ok": response.status_code == 200,
                    "status_code": response.status_code,
                    "latency_ms": elapsed,
                    "error": None,
                }
            )
        except requests.RequestException as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "source": name,
                    "url": url,
                    "ok": False,
                    "status_code": None,
                    "latency_ms": elapsed,
                    "error": str(exc),
                }
            )
    return {
        "status": "ok" if all(item["ok"] for item in results) else "partial_or_failed",
        "results": results,
        "guidance": (
            "优先保证 binance_public_archive 可用；衍生品特征还需要 "
            "binance_futures_rest。VPN 开关以两次探测的实际结果为准。"
        ),
    }


def _utc(value: str | datetime) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _completed_months(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """返回 [start, end] 内已经完整结束的 UTC 自然月。"""
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    result: list[tuple[int, int]] = []
    while cursor < end:
        next_month = (
            datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
            if cursor.month == 12
            else datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
        )
        if next_month <= end:
            result.append((cursor.year, cursor.month))
        cursor = next_month
    return result


def build_collection_plan(
    *,
    symbols: tuple[str, ...],
    intervals: tuple[str, ...],
    start: str | datetime,
    end: str | datetime,
    recent_days: int = 29,
    mode: str = "all",
) -> dict[str, Any]:
    start_at, end_at = _utc(start), _utc(end)
    if start_at >= end_at:
        raise ValueError("start 必须早于 end")
    if not 1 <= recent_days <= 29:
        raise ValueError("recent_days 必须在 1..29，避免 OI/Basis 超出公开保留期")
    if mode not in {"all", "archive", "rest"}:
        raise ValueError("mode 必须是 all/archive/rest")
    normalized_symbols = tuple(
        dict.fromkeys(item.strip().upper() for item in symbols)
    )
    normalized_intervals = tuple(
        dict.fromkeys("1d" if item == "24h" else item.strip() for item in intervals)
    )
    if not normalized_symbols or not normalized_intervals:
        raise ValueError("symbols 和 intervals 不能为空")
    months = _completed_months(start_at, end_at)
    archive_tasks = (
        [
            {"symbol": symbol, "interval": interval, "year": year, "month": month}
            for symbol in normalized_symbols
            for interval in normalized_intervals
            if interval in ARCHIVE_INTERVALS
            for year, month in months
        ]
        if mode in {"all", "archive"}
        else []
    )
    rest_start = max(start_at, end_at - timedelta(days=recent_days))
    rest_tasks = (
        [
            {
                "symbol": symbol,
                "interval": interval,
                "start": rest_start.isoformat(),
                "end": end_at.isoformat(),
            }
            for symbol in normalized_symbols
            for interval in normalized_intervals
        ]
        if mode in {"all", "rest"}
        else []
    )
    return {
        "status": "planned",
        "symbols": list(normalized_symbols),
        "intervals": list(normalized_intervals),
        "start": start_at.isoformat(),
        "end": end_at.isoformat(),
        "recent_days": recent_days,
        "mode": mode,
        "archive_tasks": archive_tasks,
        "rest_tasks": rest_tasks,
        "archive_task_count": len(archive_tasks),
        "rest_task_count": len(rest_tasks),
        "estimated_rows": {
            "1m_recent_per_symbol": recent_days * 24 * 60,
            "5m_year_per_symbol": 365 * 24 * 12,
            "1h_year_per_symbol": 365 * 24,
        },
    }


def _task_key(prefix: str, task: dict[str, Any]) -> str:
    fields = [prefix, task["symbol"], task["interval"]]
    if prefix == "archive":
        fields.extend([str(task["year"]), f"{int(task['month']):02d}"])
    else:
        fields.extend([str(task["start"]), str(task["end"])])
    return "|".join(fields)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def collect_real_data(
    plan: dict[str, Any],
    *,
    data_dir: Path,
    reports_dir: Path,
    archive: ArchiveLike | None = None,
    rest: RestLike | None = None,
    catalog_builder: Any = build_data_catalog,
) -> dict[str, Any]:
    """顺序执行并在每个任务后落盘检查点；失败任务留待下次重试。"""
    manifest_path = reports_dir / "real_data_collection_manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = set(existing.get("completed_keys", []))
    manifest = {
        **plan,
        "status": "running",
        "started_at": existing.get("started_at") or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "completed_keys": sorted(completed),
        "archive_results": existing.get("archive_results", []),
        "rest_results": existing.get("rest_results", []),
        "errors": [],
    }
    archive_client = archive or BinanceArchiveDownloader(data_dir=data_dir)
    rest_client = rest or DataDownloader(data_dir=data_dir)
    pending_archive = [
        task
        for task in plan["archive_tasks"]
        if _task_key("archive", task) not in completed
    ]
    batch_download = getattr(archive_client, "download_months", None)
    if callable(batch_download):
        batches: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for task in pending_archive:
            batches.setdefault((task["symbol"], task["interval"]), []).append(task)
        for tasks in batches.values():
            try:
                batch = batch_download(tasks=tasks, market="futures")
                for result in batch["results"]:
                    task = {
                        field: result[field]
                        for field in ("symbol", "interval", "year", "month")
                    }
                    manifest["archive_results"].append(result)
                    completed.add(_task_key("archive", task))
                for failure in batch["errors"]:
                    manifest["errors"].append(
                        {
                            "key": _task_key("archive", failure["task"]),
                            "error": failure["error"],
                        }
                    )
            except Exception as exc:
                for task in tasks:
                    manifest["errors"].append(
                        {"key": _task_key("archive", task), "error": str(exc)}
                    )
            manifest["completed_keys"] = sorted(completed)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _write_manifest(manifest_path, manifest)
    else:
        for task in pending_archive:
            key = _task_key("archive", task)
            try:
                result = archive_client.download_month(**task, market="futures")
                manifest["archive_results"].append(result)
                completed.add(key)
            except Exception as exc:  # 单月失败留痕并继续其他资产/月份
                manifest["errors"].append({"key": key, "error": str(exc)})
            manifest["completed_keys"] = sorted(completed)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _write_manifest(manifest_path, manifest)
    for task in plan["rest_tasks"]:
        key = _task_key("rest", task)
        if key in completed:
            continue
        try:
            result = rest_client.download_bundle(
                symbol=task["symbol"],
                interval=task["interval"],
                start=_utc(task["start"]),
                end=_utc(task["end"]),
            )
            manifest["rest_results"].append(result)
            completed.add(key)
        except Exception as exc:  # 网络波动后下次运行会从未完成 key 续跑
            manifest["errors"].append({"key": key, "error": str(exc)})
        manifest["completed_keys"] = sorted(completed)
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, manifest)
    catalog = catalog_builder(data_dir, reports_dir / "data_catalog.json")
    expected = plan["archive_task_count"] + plan["rest_task_count"]
    current_keys = {
        _task_key("archive", task) for task in plan["archive_tasks"]
    } | {_task_key("rest", task) for task in plan["rest_tasks"]}
    completed_current = len(completed.intersection(current_keys))
    manifest.update(
        {
            "status": (
                "completed"
                if completed_current == expected
                else "partial" if completed_current else "failed"
            ),
            "completed_task_count": completed_current,
            "expected_task_count": expected,
            "catalog_entry_count": catalog.get("entry_count", 0),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    today = datetime.now(UTC)
    parser = argparse.ArgumentParser(description="断点续传真实研究数据")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--intervals", nargs="+", default=list(DEFAULT_INTERVALS))
    parser.add_argument(
        "--start", default=(today - timedelta(days=365)).date().isoformat()
    )
    # Date-only default keeps resume keys stable across repeated runs on the same day.
    parser.add_argument("--end", default=today.date().isoformat())
    parser.add_argument("--recent-days", type=int, default=29)
    parser.add_argument(
        "--mode", choices=("all", "archive", "rest"), default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe", action="store_true", help="只探测数据源连通性")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过执行前连通性门禁；仅在已确认探测误报时使用",
    )
    args = parser.parse_args(argv)
    if args.probe:
        probe = probe_data_sources()
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return 0 if any(item["ok"] for item in probe["results"]) else 1
    plan = build_collection_plan(
        symbols=tuple(args.symbols),
        intervals=tuple(args.intervals),
        start=args.start,
        end=args.end,
        recent_days=args.recent_days,
        mode=args.mode,
    )
    if not args.execute:
        summary = {
            key: plan[key]
            for key in (
                "status",
                "symbols",
                "intervals",
                "start",
                "end",
                "recent_days",
                "mode",
                "archive_task_count",
                "rest_task_count",
                "estimated_rows",
            )
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.skip_preflight:
        preflight = probe_data_sources()
        availability = {
            item["source"]: item["ok"] for item in preflight["results"]
        }
        required = {
            "archive": ("binance_public_archive",),
            "rest": ("binance_futures_rest",),
            "all": ("binance_public_archive", "binance_futures_rest"),
        }[args.mode]
        unavailable = [source for source in required if not availability.get(source)]
        if unavailable:
            print(
                json.dumps(
                    {
                        "status": "preflight_failed",
                        "mode": args.mode,
                        "unavailable": unavailable,
                        "probe": preflight,
                        "next": (
                            "切换 VPN 状态后先重跑 --probe；若只有单一入口可用，"
                            "改用 --mode archive 或 --mode rest --execute。"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
    result = collect_real_data(
        plan, data_dir=args.data_dir, reports_dir=args.reports_dir
    )
    output = {
        "status": result["status"],
        "completed_task_count": result["completed_task_count"],
        "expected_task_count": result["expected_task_count"],
        "catalog_entry_count": result["catalog_entry_count"],
        "error_count": len(result["errors"]),
        "recent_errors": result["errors"][-10:],
        "manifest": str(args.reports_dir / "real_data_collection_manifest.json"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
