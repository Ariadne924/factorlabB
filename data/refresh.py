"""Bounded near-real-time REST refresh for the research UI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import requests

from data.downloader import DataDownloader


class BundleDownloader(Protocol):
    def download_bundle(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]: ...


def refresh_recent_market_data(
    data_dir: Path,
    *,
    symbols: tuple[str, ...],
    intervals: tuple[str, ...],
    lookback_hours: int = 24,
    downloader: BundleDownloader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refresh a small trailing window and merge it idempotently into Bronze/Silver.

    This is deliberately a bounded REST refresh, not a tick-level streaming store.
    Re-requesting a short overlap lets the existing downloader repair the latest
    incomplete bar while its merge keys prevent duplicate rows.
    """
    normalized_symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    normalized_intervals = tuple(
        dict.fromkeys("1d" if interval == "24h" else interval for interval in intervals)
    )
    if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
        raise ValueError("at least one non-empty symbol is required")
    if not normalized_intervals or any(not interval for interval in normalized_intervals):
        raise ValueError("at least one interval is required")
    if lookback_hours < 1 or lookback_hours > 24 * 30:
        raise ValueError("lookback_hours must be between 1 and 720")

    effective_now = now or datetime.now(UTC)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=UTC)
    else:
        effective_now = effective_now.astimezone(UTC)
    start = effective_now - timedelta(hours=lookback_hours)
    client = downloader or DataDownloader(data_dir=data_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for symbol in normalized_symbols:
        for interval in normalized_intervals:
            try:
                results.append(
                    client.download_bundle(
                        symbol=symbol,
                        interval=interval,
                        start=start,
                        end=effective_now,
                    )
                )
            except (OSError, RuntimeError, requests.RequestException, ValueError) as exc:
                errors.append(
                    {"symbol": symbol, "interval": interval, "error": str(exc)}
                )
    return {
        "status": "ok" if results and not errors else "partial" if results else "failed",
        "requested_at": effective_now.isoformat(),
        "lookback_hours": lookback_hours,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "mode": "bounded_rest_refresh",
    }
