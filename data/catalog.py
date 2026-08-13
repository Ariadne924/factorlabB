"""Silver data coverage catalog and incremental missing-range planning."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import get_interval_ms

CATALOG_VERSION = "1.0"


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat()


def _gap_ranges(timestamps: pd.DatetimeIndex, interval: str) -> list[dict[str, Any]]:
    if len(timestamps) < 2:
        return []
    step = pd.Timedelta(milliseconds=get_interval_ms(interval))
    differences = timestamps.to_series().diff()
    gaps: list[dict[str, Any]] = []
    for position in differences[differences > step].index:
        current = pd.Timestamp(position)
        previous_position = int(timestamps.searchsorted(current)) - 1
        previous = timestamps[previous_position]
        missing_start = previous + step
        missing_end = current - step
        missing_bars = int((current - previous) / step) - 1
        gaps.append(
            {
                "start": _iso(missing_start),
                "end": _iso(missing_end),
                "missing_bars": missing_bars,
            }
        )
    return gaps


def inspect_silver_file(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    """Read only catalog columns from one Silver parquet file."""
    columns = ["open_time_utc", "symbol", "interval"]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError(f"Silver file is empty: {path}")
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(frame["open_time_utc"], utc=True).sort_values().drop_duplicates()
    )
    symbols = frame["symbol"].astype(str).str.upper().unique()
    intervals = frame["interval"].astype(str).unique()
    if len(symbols) != 1 or len(intervals) != 1:
        raise ValueError(f"Silver file must contain exactly one symbol and interval: {path}")
    interval = str(intervals[0])
    step = pd.Timedelta(milliseconds=get_interval_ms(interval))
    expected_rows = int((timestamps[-1] - timestamps[0]) / step) + 1
    relative_path = (
        path.resolve().relative_to(base.resolve()).as_posix()
        if base is not None and path.resolve().is_relative_to(base.resolve())
        else str(path.resolve())
    )
    gaps = _gap_ranges(timestamps, interval)
    return {
        "path": relative_path,
        "symbol": str(symbols[0]),
        "interval": interval,
        "start": _iso(timestamps[0]),
        "end": _iso(timestamps[-1]),
        "rows": int(len(frame)),
        "unique_timestamps": int(len(timestamps)),
        "expected_rows": expected_rows,
        "missing_bars": int(sum(int(gap["missing_bars"]) for gap in gaps)),
        "gap_ranges": gaps,
        "coverage_ratio": float(len(timestamps) / expected_rows),
    }


def build_data_catalog(data_dir: Path, output: Path | None = None) -> dict[str, Any]:
    """Scan Silver files and optionally write a versioned JSON catalog."""
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("silver/**/klines.parquet")):
        try:
            entries.append(inspect_silver_file(path, base=data_dir))
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    catalog = {
        "version": CATALOG_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "entry_count": len(entries),
        "entries": entries,
        "errors": errors,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def missing_download_ranges(
    catalog: dict[str, Any],
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> list[dict[str, str]]:
    """Return only uncovered edge and internal ranges for a request."""
    requested_start = _utc(start)
    requested_end = _utc(end)
    if requested_start > requested_end:
        raise ValueError("start must not be later than end")
    normalized_interval = "1d" if interval == "24h" else interval
    matching = [
        entry
        for entry in catalog.get("entries", [])
        if str(entry.get("symbol", "")).upper() == symbol.upper()
        and str(entry.get("interval")) == normalized_interval
    ]
    if not matching:
        return [{"start": _iso(requested_start), "end": _iso(requested_end)}]

    step = pd.Timedelta(milliseconds=get_interval_ms(normalized_interval))
    covered = sorted(
        (_utc(entry["start"]), _utc(entry["end"])) for entry in matching
    )
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for left, right in covered:
        if not merged or left > merged[-1][1] + step:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = requested_start
    for covered_start, covered_end in merged:
        if covered_end < requested_start or covered_start > requested_end:
            continue
        if cursor < covered_start:
            ranges.append((cursor, min(requested_end, covered_start - step)))
        cursor = max(cursor, covered_end + step)
        if cursor > requested_end:
            break
    if cursor <= requested_end:
        ranges.append((cursor, requested_end))

    for entry in matching:
        for gap in entry.get("gap_ranges", []):
            gap_start = max(requested_start, _utc(gap["start"]))
            gap_end = min(requested_end, _utc(gap["end"]))
            if gap_start <= gap_end:
                ranges.append((gap_start, gap_end))
    unique = sorted({(left, right) for left, right in ranges if left <= right})
    planned: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for left, right in unique:
        if not planned or left > planned[-1][1] + step:
            planned.append((left, right))
        else:
            planned[-1] = (planned[-1][0], max(planned[-1][1], right))
    return [{"start": _iso(left), "end": _iso(right)} for left, right in planned]
