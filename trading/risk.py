"""Executable risk rules shared by research backtests and testnet execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RiskLimits:
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    max_position_fraction: float = 0.25
    max_leverage: float = 2.0
    circuit_breaker_return: float = 0.10
    max_spread_bps: float = 100.0
    max_stale_seconds: float = 30.0

    def validate(self) -> None:
        fractions = (
            self.stop_loss_pct,
            self.max_drawdown_pct,
            self.max_position_fraction,
            self.circuit_breaker_return,
        )
        if any(value <= 0 or value > 1 for value in fractions):
            raise ValueError("risk percentages must be in (0, 1]")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.max_spread_bps <= 0 or self.max_stale_seconds <= 0:
            raise ValueError("spread and stale limits must be positive")


@dataclass(frozen=True)
class RiskState:
    equity: float
    peak_equity: float
    requested_position_fraction: float
    requested_leverage: float
    current_position: float = 0.0
    entry_price: float | None = None
    current_price: float | None = None
    market_return: float = 0.0
    spread_bps: float | None = None
    stale_seconds: float | None = None


@dataclass(frozen=True)
class RiskTrigger:
    rule: str
    action: str
    observed: float
    limit: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    target_position_fraction: float
    leverage: float
    triggers: tuple[RiskTrigger, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.limits.validate()

    def evaluate(self, state: RiskState) -> RiskDecision:
        if state.equity <= 0 or state.peak_equity <= 0:
            raise ValueError("equity and peak_equity must be positive")
        triggers: list[RiskTrigger] = []
        target = max(
            -self.limits.max_position_fraction,
            min(self.limits.max_position_fraction, state.requested_position_fraction),
        )
        if target != state.requested_position_fraction:
            triggers.append(
                RiskTrigger(
                    "position_limit",
                    "clip",
                    abs(state.requested_position_fraction),
                    self.limits.max_position_fraction,
                )
            )
        leverage = min(state.requested_leverage, self.limits.max_leverage)
        if leverage != state.requested_leverage:
            triggers.append(
                RiskTrigger(
                    "leverage_limit",
                    "clip",
                    state.requested_leverage,
                    self.limits.max_leverage,
                )
            )

        drawdown = max(0.0, 1.0 - state.equity / state.peak_equity)
        if drawdown >= self.limits.max_drawdown_pct:
            triggers.append(
                RiskTrigger(
                    "max_drawdown",
                    "flatten",
                    drawdown,
                    self.limits.max_drawdown_pct,
                )
            )

        if (
            state.current_position != 0
            and state.entry_price is not None
            and state.current_price is not None
            and state.entry_price > 0
        ):
            signed_return = (
                (state.current_price / state.entry_price - 1.0)
                * (1.0 if state.current_position > 0 else -1.0)
            )
            if signed_return <= -self.limits.stop_loss_pct:
                triggers.append(
                    RiskTrigger(
                        "stop_loss",
                        "flatten",
                        -signed_return,
                        self.limits.stop_loss_pct,
                    )
                )

        circuit_observations = (
            (
                "extreme_market_move",
                abs(state.market_return),
                self.limits.circuit_breaker_return,
            ),
            (
                "wide_spread",
                state.spread_bps,
                self.limits.max_spread_bps,
            ),
            (
                "stale_market_data",
                state.stale_seconds,
                self.limits.max_stale_seconds,
            ),
        )
        for rule, observed, limit in circuit_observations:
            if observed is not None and observed >= limit:
                triggers.append(RiskTrigger(rule, "flatten", float(observed), limit))

        flatten = any(trigger.action == "flatten" for trigger in triggers)
        return RiskDecision(
            allowed=not flatten,
            target_position_fraction=0.0 if flatten else target,
            leverage=leverage,
            triggers=tuple(triggers),
        )

