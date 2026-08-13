from __future__ import annotations

from datetime import UTC, datetime, timedelta

from data.market_alerts import MarketAlertConfig, evaluate_market_alerts


def test_market_alerts_detect_move_spread_funding_and_staleness() -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    snapshot = {
        "symbols": {
            "BTCUSDT": {
                "updated_at": (now - timedelta(seconds=20)).isoformat(),
                "current_kline": {"open": 100, "close": 102},
                "book": {"spread_bps": 8},
                "derivatives": {"funding_rate": 0.001},
            }
        }
    }
    alerts = evaluate_market_alerts(snapshot, config=MarketAlertConfig(), now=now)
    assert {row["type"] for row in alerts} == {
        "data_stale",
        "kline_move",
        "wide_spread",
        "extreme_funding",
    }
    assert all(row["id"].startswith("BTCUSDT:") for row in alerts)
