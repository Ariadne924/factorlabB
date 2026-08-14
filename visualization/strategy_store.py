"""Safe local JSON persistence and comparison for interactive strategy snapshots."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STORE_VERSION = "1.0"
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_-]+")


def _clean_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
        raise ValueError("strategy name must contain 1..80 printable characters")
    return name


def _filename(name: str) -> str:
    slug = _SAFE_LABEL.sub("-", name).strip("-")[:40] or "strategy"
    return f"{slug}-{uuid.uuid4().hex[:10]}.json"


def save_strategy_snapshot(
    directory: Path,
    *,
    name: str,
    symbol: str,
    interval: str,
    result: dict[str, Any],
    robustness: dict[str, Any] | None = None,
) -> Path:
    """Persist a new immutable snapshot without accepting a caller-controlled path."""
    clean_name = _clean_name(name)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / _filename(clean_name)
    payload = {
        "version": STORE_VERSION,
        "id": output.stem,
        "name": clean_name,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": str(symbol).upper(),
        "interval": "1d" if interval == "24h" else str(interval),
        "result": result,
        "robustness": robustness or {},
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    return output


def load_strategy_snapshots(directory: Path) -> list[dict[str, Any]]:
    """Load valid snapshots; malformed unrelated JSON files are skipped."""
    snapshots: list[dict[str, Any]] = []
    if not directory.exists():
        return snapshots
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or payload.get("version") != STORE_VERSION:
            continue
        payload["_file"] = path.name
        snapshots.append(payload)
    return sorted(snapshots, key=lambda row: str(row.get("created_at", "")), reverse=True)


def compare_strategy_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact table without copying full return paths into the comparison."""
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        result = snapshot.get("result", {})
        metrics = result.get("metrics", {})
        stability = snapshot.get("robustness", {}).get("stability_summary", {})
        rows.append(
            {
                "id": snapshot.get("id"),
                "name": snapshot.get("name"),
                "symbol": snapshot.get("symbol"),
                "interval": snapshot.get("interval"),
                "created_at": snapshot.get("created_at"),
                "total_return": metrics.get("total_return"),
                "gross_total_return": metrics.get("gross_total_return"),
                "bar_sharpe": metrics.get("bar_sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "mean_turnover": metrics.get("mean_turnover"),
                "hit_rate": metrics.get("hit_rate"),
                "positive_month_ratio": stability.get("positive_month_ratio"),
                "lookahead_status": result.get("lookahead_status"),
            }
        )
    return rows
