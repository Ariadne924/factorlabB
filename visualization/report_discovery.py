"""Discover factor reports while repairing legacy filename-only scope metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KNOWN_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
)


def infer_report_scope(report: dict[str, Any], path: Path) -> dict[str, Any]:
    """Fill symbol/frequency from ``SYMBOL_INTERVAL_FACTOR.json`` when absent."""
    output = dict(report)
    parts = path.stem.split("_", 2)
    if len(parts) == 3 and parts[1] in KNOWN_INTERVALS:
        symbol, interval, _ = parts
        if symbol.upper().endswith(("USDT", "USD", "BTC", "ETH")):
            output.setdefault("symbol", symbol.upper())
            output.setdefault("interval", interval)
            output.setdefault("display_frequency", "24h" if interval == "1d" else interval)
    return output


def load_factor_reports(
    directory: Path,
    *,
    require_symbol: bool = False,
) -> list[dict[str, Any]]:
    """Load report objects and optionally exclude unscoped legacy artifacts."""
    reports: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "factor_name" not in raw:
            continue
        report = infer_report_scope(raw, path)
        if require_symbol and not report.get("symbol"):
            continue
        report["_file"] = path.name
        reports.append(report)
    return reports
