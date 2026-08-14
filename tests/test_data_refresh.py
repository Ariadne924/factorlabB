from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from data.refresh import refresh_recent_market_data


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def download_bundle(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        call: dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
        }
        self.calls.append(call)
        if symbol == "BADUSDT":
            raise ValueError("unavailable")
        return {"symbol": symbol, "interval": interval}


def test_recent_refresh_is_bounded_normalized_and_partial(tmp_path) -> None:
    fake = FakeDownloader()
    result = refresh_recent_market_data(
        tmp_path,
        symbols=("btcusdt", "BADUSDT"),
        intervals=("1h", "24h"),
        lookback_hours=12,
        downloader=fake,
        now=datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert result["status"] == "partial"
    assert result["successful"] == 2
    assert result["failed"] == 2
    assert {call["interval"] for call in fake.calls} == {"1h", "1d"}
    assert fake.calls[0]["symbol"] == "BTCUSDT"


def test_recent_refresh_validates_scope(tmp_path) -> None:
    with pytest.raises(ValueError, match="lookback_hours"):
        refresh_recent_market_data(
            tmp_path,
            symbols=("BTCUSDT",),
            intervals=("1h",),
            lookback_hours=0,
        )
