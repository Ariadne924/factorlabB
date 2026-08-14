"""Transparent vectorized/event backtests and their consistency gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading.risk import RiskEngine, RiskLimits, RiskState


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    leverage: float = 1.0
    funding_column: str = "funding_rate"
    risk_limits: RiskLimits | None = None

    def validate(self) -> None:
        if self.initial_equity <= 0 or self.leverage <= 0:
            raise ValueError("initial_equity and leverage must be positive")
        if min(self.fee_rate, self.slippage_rate) < 0:
            raise ValueError("cost assumptions cannot be negative")


def _validate_inputs(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    if "close" not in frame:
        raise ValueError("backtest input requires close")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("backtest input requires a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("backtest index must be unique and increasing")
    close = pd.to_numeric(frame["close"], errors="coerce")
    aligned_target = pd.to_numeric(target.reindex(frame.index), errors="coerce").fillna(0.0)
    if (aligned_target.abs() > 1.0 + 1e-12).any():
        raise ValueError("target exposure must stay in [-1, 1]")
    return close, aligned_target.clip(-1.0, 1.0)


def _funding(frame: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    if config.funding_column not in frame:
        return pd.Series(0.0, index=frame.index)
    return (
        pd.to_numeric(frame[config.funding_column], errors="coerce")
        .fillna(0.0)
        .clip(-0.01, 0.01)
    )


def _result(
    frame: pd.DataFrame,
    *,
    position: pd.Series,
    gross: pd.Series,
    costs: pd.Series,
    funding_cost: pd.Series,
    net: pd.Series,
    config: BacktestConfig,
    engine: str,
    risk_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    equity = config.initial_equity * (1.0 + net).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    turnover = position.diff().abs().fillna(position.abs())
    active = position.ne(0)
    std = net.std(ddof=1)
    return {
        "status": "computed_short_sample" if len(net) else "insufficient_data",
        "engine": engine,
        "config": asdict(config),
        "metrics": {
            "n_periods": int(len(net)),
            "gross_total_return": float((1.0 + gross).prod() - 1.0),
            "total_return": float(equity.iloc[-1] / config.initial_equity - 1.0),
            "max_drawdown": float(drawdown.min()),
            "mean_turnover": float(turnover.mean()),
            "total_fees_slippage": float(costs.sum()),
            "total_funding": float(funding_cost.sum()),
            "bar_sharpe": float(net.mean() / std) if std > 0 else None,
            "hit_rate": float(gross[active].gt(0).mean()) if active.any() else None,
        },
        "risk_events": risk_events or [],
        "lookahead_status": "pass_by_construction",
        "lookahead_policy": "target at t is first executed for the t-to-t+1 return",
        "returns": [
            {
                "time": str(index),
                "close": float(frame.loc[index, "close"]),
                "position": float(position.loc[index]),
                "gross_return": float(gross.loc[index]),
                "cost": float(costs.loc[index]),
                "funding_cost": float(funding_cost.loc[index]),
                "net_return": float(net.loc[index]),
                "equity": float(equity.loc[index]),
                "drawdown": float(drawdown.loc[index]),
            }
            for index in frame.index
        ],
        "research_note": (
            "Research backtest with explicit one-bar execution lag. It is not evidence of "
            "live fill quality, six-month OOS completion, or validated alpha."
        ),
    }


def run_vectorized_backtest(
    frame: pd.DataFrame,
    target: pd.Series,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    selected = config or BacktestConfig()
    selected.validate()
    if selected.risk_limits is not None:
        raise ValueError("path-dependent risk rules require the event engine")
    close, aligned_target = _validate_inputs(frame, target)
    position = aligned_target.shift(1).fillna(0.0).mul(selected.leverage)
    market_return = close.pct_change(fill_method=None).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    gross = position * market_return
    costs = turnover * (selected.fee_rate + selected.slippage_rate)
    funding_cost = position * _funding(frame, selected)
    net = gross - costs - funding_cost
    return _result(
        frame,
        position=position,
        gross=gross,
        costs=costs,
        funding_cost=funding_cost,
        net=net,
        config=selected,
        engine="vectorized",
    )


def run_event_backtest(
    frame: pd.DataFrame,
    target: pd.Series,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    selected = config or BacktestConfig()
    selected.validate()
    close, aligned_target = _validate_inputs(frame, target)
    market_return = close.pct_change(fill_method=None).fillna(0.0)
    funding = _funding(frame, selected)
    risk_engine = RiskEngine(selected.risk_limits) if selected.risk_limits else None
    positions: list[float] = []
    gross_values: list[float] = []
    cost_values: list[float] = []
    funding_values: list[float] = []
    net_values: list[float] = []
    risk_events: list[dict[str, Any]] = []
    equity = selected.initial_equity
    peak_equity = equity
    previous_position = 0.0
    entry_price: float | None = None
    previous_trigger_rules: set[str] = set()

    for index, timestamp in enumerate(frame.index):
        requested = float(aligned_target.iloc[index - 1]) if index else 0.0
        normalized_position = requested
        if risk_engine is not None:
            previous_market_return = float(market_return.iloc[index - 1]) if index else 0.0
            decision = risk_engine.evaluate(
                RiskState(
                    equity=equity,
                    peak_equity=peak_equity,
                    requested_position_fraction=requested,
                    requested_leverage=selected.leverage,
                    current_position=previous_position,
                    entry_price=entry_price,
                    current_price=float(close.iloc[index - 1]) if index else float(close.iloc[0]),
                    market_return=previous_market_return,
                )
            )
            normalized_position = decision.target_position_fraction
            leverage = decision.leverage
            current_trigger_rules = {trigger.rule for trigger in decision.triggers}
            for trigger in decision.triggers:
                if trigger.rule not in previous_trigger_rules:
                    risk_events.append({"time": str(timestamp), **asdict(trigger)})
            previous_trigger_rules = current_trigger_rules
        else:
            leverage = selected.leverage
        position = normalized_position * leverage
        if position == 0:
            entry_price = None
        elif previous_position == 0 or np.sign(position) != np.sign(previous_position):
            entry_price = float(close.iloc[index - 1]) if index else float(close.iloc[0])
        turnover = abs(position - previous_position)
        gross = position * float(market_return.iloc[index])
        cost = turnover * (selected.fee_rate + selected.slippage_rate)
        current_funding = position * float(funding.iloc[index])
        net = gross - cost - current_funding
        equity *= 1.0 + net
        peak_equity = max(peak_equity, equity)
        positions.append(position)
        gross_values.append(gross)
        cost_values.append(cost)
        funding_values.append(current_funding)
        net_values.append(net)
        previous_position = position

    index = frame.index
    return _result(
        frame,
        position=pd.Series(positions, index=index),
        gross=pd.Series(gross_values, index=index),
        costs=pd.Series(cost_values, index=index),
        funding_cost=pd.Series(funding_values, index=index),
        net=pd.Series(net_values, index=index),
        config=selected,
        engine="event",
        risk_events=risk_events,
    )


def compare_backtest_engines(
    frame: pd.DataFrame,
    target: pd.Series,
    config: BacktestConfig | None = None,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    selected = config or BacktestConfig()
    if selected.risk_limits is not None:
        raise ValueError("consistency gate compares engines before path-dependent risk")
    vectorized = run_vectorized_backtest(frame, target, selected)
    event = run_event_backtest(frame, target, selected)
    left = pd.DataFrame(vectorized["returns"])["net_return"]
    right = pd.DataFrame(event["returns"])["net_return"]
    maximum_error = float((left - right).abs().max()) if len(left) else 0.0
    return {
        "status": "pass" if maximum_error <= tolerance else "fail",
        "max_abs_net_return_error": maximum_error,
        "tolerance": tolerance,
        "vectorized_total_return": vectorized["metrics"]["total_return"],
        "event_total_return": event["metrics"]["total_return"],
        "difference_policy": (
            "The consistency gate disables path-dependent risk; risk-triggered execution "
            "is intentionally evaluated only by the event engine."
        ),
    }
