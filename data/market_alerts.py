"""研究级实时市场异动检测；只生成提醒，不触发交易。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class MarketAlertConfig:
    kline_move_pct: float = 0.8
    spread_bps: float = 5.0
    funding_abs: float = 0.0005
    stale_seconds: float = 10.0


def evaluate_market_alerts(
    snapshot: dict[str, Any],
    *,
    config: MarketAlertConfig | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """根据当前快照生成可解释、无未来数据的活动提醒。"""
    selected = config or MarketAlertConfig()
    current = now or datetime.now(UTC)
    alerts: list[dict[str, Any]] = []
    for symbol, state in snapshot.get("symbols", {}).items():
        updated_at = state.get("updated_at")
        if updated_at:
            updated = datetime.fromisoformat(str(updated_at)).astimezone(UTC)
            age = (current - updated).total_seconds()
            if age > selected.stale_seconds:
                alerts.append(_alert(symbol, "data_stale", "warning", age, "秒未更新"))
        kline = state.get("current_kline", {})
        open_price, close_price = kline.get("open"), kline.get("close")
        if open_price and close_price:
            move_pct = (float(close_price) / float(open_price) - 1) * 100
            if abs(move_pct) >= selected.kline_move_pct:
                alerts.append(_alert(symbol, "kline_move", "high", move_pct, "分钟涨跌幅%"))
        spread = state.get("book", {}).get("spread_bps")
        if spread is not None and float(spread) >= selected.spread_bps:
            alerts.append(_alert(symbol, "wide_spread", "warning", float(spread), "点差bps"))
        funding = state.get("derivatives", {}).get("funding_rate")
        if funding is not None and abs(float(funding)) >= selected.funding_abs:
            alerts.append(_alert(symbol, "extreme_funding", "warning", float(funding), "资金费率"))
    return sorted(alerts, key=lambda row: (row["severity"] != "high", row["symbol"]))


def _alert(
    symbol: str, kind: str, severity: str, value: float, unit: str
) -> dict[str, Any]:
    return {
        "id": f"{symbol}:{kind}",
        "symbol": symbol,
        "type": kind,
        "severity": severity,
        "value": value,
        "message": f"{symbol} {unit}: {value:.4f}",
    }
