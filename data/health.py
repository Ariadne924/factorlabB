"""Build a user-facing health report from the local data catalog and feature files."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import get_interval_ms

HEALTH_REPORT_VERSION = "1.0"
CORE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT")
CORE_INTERVALS = ("1m", "5m", "1h", "6h", "1d")
DERIVATIVE_FEATURES = ("funding_rate", "open_interest", "basis")


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _dataset_health(entry: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
    interval = str(entry["interval"])
    step = pd.Timedelta(milliseconds=get_interval_ms(interval))
    start, end = _utc(entry["start"]), _utc(entry["end"])
    history_days = max(0.0, float((end - start + step) / pd.Timedelta(days=1)))
    age = max(pd.Timedelta(0), as_of - (end + step))
    age_hours = float(age / pd.Timedelta(hours=1))
    lag_bars = int(math.ceil(age / step)) if age > pd.Timedelta(0) else 0
    coverage = float(entry.get("coverage_ratio", 0.0))
    missing_bars = int(entry.get("missing_bars", 0))
    research_ready = coverage >= 0.98 and history_days >= 14
    ml_eligible = coverage >= 0.98 and history_days >= 120
    fresh = lag_bars <= 2
    if coverage < 0.98 or missing_bars > max(5, int(entry.get("expected_rows", 0) * 0.02)):
        status = "gaps"
    elif history_days < 14:
        status = "limited_history"
    elif not fresh:
        status = "historical_ready_stale"
    else:
        status = "ready"
    return {
        **entry,
        "display_interval": "24h" if interval == "1d" else interval,
        "history_days": history_days,
        "age_hours": age_hours,
        "lag_bars": lag_bars,
        "fresh": fresh,
        "research_ready": research_ready,
        "ml_eligible": ml_eligible,
        "health_status": status,
    }


def _inspect_derivative_files(
    data_dir: Path,
    *,
    as_of: pd.Timestamp,
    parquet_reader: Callable[..., pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("bronze/binance/futures/*/*/*.parquet")):
        feature = path.stem
        if feature not in DERIVATIVE_FEATURES:
            continue
        try:
            frame = parquet_reader(path, columns=["timestamp"])
            timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
            if timestamps.empty:
                raise ValueError("feature file has no valid timestamps")
            latest = timestamps.max()
            age_hours = max(0.0, float((as_of - latest) / pd.Timedelta(hours=1)))
            rows.append(
                {
                    "symbol": path.parts[-3],
                    "interval": path.parts[-2],
                    "feature": feature,
                    "rows": int(len(frame)),
                    "start": timestamps.min().isoformat(),
                    "end": latest.isoformat(),
                    "age_hours": age_hours,
                    "recent_48h": age_hours <= 48,
                    "path": path.relative_to(data_dir).as_posix(),
                }
            )
        except (ImportError, KeyError, OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def build_data_health_report(
    catalog: dict[str, Any],
    data_dir: Path,
    *,
    output: Path | None = None,
    manifest_path: Path | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    core_symbols: Sequence[str] = CORE_SYMBOLS,
    core_intervals: Sequence[str] = CORE_INTERVALS,
    parquet_reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:
    """Summarize coverage, freshness, research readiness and derivative availability."""
    now = _utc(as_of or datetime.now(UTC))
    normalized_symbols = tuple(dict.fromkeys(str(item).upper() for item in core_symbols))
    normalized_intervals = tuple(
        dict.fromkeys("1d" if item == "24h" else str(item) for item in core_intervals)
    )
    datasets = [_dataset_health(dict(entry), now) for entry in catalog.get("entries", [])]
    available = {(str(row["symbol"]).upper(), str(row["interval"])) for row in datasets}
    target = {
        (symbol, interval)
        for symbol in normalized_symbols
        for interval in normalized_intervals
    }
    missing_scopes = [
        {"symbol": symbol, "interval": interval}
        for symbol, interval in sorted(target.difference(available))
    ]
    derivatives, derivative_errors = _inspect_derivative_files(
        data_dir, as_of=now, parquet_reader=parquet_reader
    )
    manifest: dict[str, Any] = {}
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            derivative_errors.append({"path": str(manifest_path), "error": str(exc)})

    target_covered = len(target.intersection(available))
    summary = {
        "catalog_entries": len(datasets),
        "catalog_errors": len(catalog.get("errors", [])),
        "core_target_datasets": len(target),
        "core_covered_datasets": target_covered,
        "core_coverage_ratio": target_covered / len(target) if target else 1.0,
        "symbols": len({row["symbol"] for row in datasets}),
        "intervals": len({row["interval"] for row in datasets}),
        "total_rows": sum(int(row.get("rows", 0)) for row in datasets),
        "missing_bars": sum(int(row.get("missing_bars", 0)) for row in datasets),
        "gap_datasets": sum(row["health_status"] == "gaps" for row in datasets),
        "fresh_datasets": sum(bool(row["fresh"]) for row in datasets),
        "research_ready_datasets": sum(bool(row["research_ready"]) for row in datasets),
        "ml_eligible_datasets": sum(bool(row["ml_eligible"]) for row in datasets),
        "derivative_datasets": len(derivatives),
        "recent_derivative_datasets": sum(bool(row["recent_48h"]) for row in derivatives),
    }
    report = {
        "version": HEALTH_REPORT_VERSION,
        "generated_at": now.isoformat(),
        "summary": summary,
        "collection": {
            "status": manifest.get("status", "not_run"),
            "mode": manifest.get("mode"),
            "completed_task_count": manifest.get("completed_task_count"),
            "expected_task_count": manifest.get("expected_task_count"),
            "error_count": len(manifest.get("errors", [])),
            "updated_at": manifest.get("updated_at"),
        },
        "core_symbols": list(normalized_symbols),
        "core_intervals": list(normalized_intervals),
        "missing_scopes": missing_scopes,
        "datasets": datasets,
        "derivatives": derivatives,
        "errors": [*catalog.get("errors", []), *derivative_errors],
        "interpretation": {
            "research_ready": "至少 14 天且时间覆盖率不低于 98%。",
            "ml_eligible": "至少 120 天且时间覆盖率不低于 98%。",
            "fresh": "最新已完成 K 线距离当前时间不超过两个周期。",
            "warning": "历史可研究不代表实时；ML eligible 也不代表已验证 alpha。",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
