"""Configurable multi-factor panel combination and lightweight backtesting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from evaluation.panel_analysis import panel_forward_returns, panel_positions
from factors.expression_engine import compile_panel_expression
from factors.panel import SYMBOL_LEVEL, TIME_LEVEL, cs_rank, cs_scale, cs_zscore, validate_panel
from factors.panel_registry import compute_panel_factor, list_panel_factors


@dataclass(frozen=True)
class FactorAllocation:
    name: str
    weight: float = 1.0
    direction: Literal[-1, 1] = 1
    expression: str | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("factor allocation name cannot be empty")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"factor weight must be positive and finite: {self.name}")
        if self.direction not in {-1, 1}:
            raise ValueError(f"factor direction must be -1 or 1: {self.name}")
        if self.expression is None and self.name not in list_panel_factors():
            raise ValueError(f"unknown registered panel factor: {self.name}")


@dataclass(frozen=True)
class PanelStrategyConfig:
    allocations: tuple[FactorAllocation, ...]
    interval: str
    horizon: int = 1
    rebalance_every: int = 1
    position_mode: Literal["long_short", "long_only"] = "long_short"
    fee_rate: float = 0.001
    slippage: float = 0.0005

    def validate(self) -> None:
        if not self.allocations:
            raise ValueError("at least one factor allocation is required")
        names = [allocation.name for allocation in self.allocations]
        if len(names) != len(set(names)):
            raise ValueError("factor allocation names must be unique")
        for allocation in self.allocations:
            allocation.validate()
        if self.horizon < 1 or self.rebalance_every < 1:
            raise ValueError("horizon and rebalance_every must be positive")
        if self.position_mode not in {"long_short", "long_only"}:
            raise ValueError("unsupported position mode")
        if self.fee_rate < 0 or self.slippage < 0:
            raise ValueError("cost assumptions cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_panel_factors(
    panel: pd.DataFrame,
    allocations: tuple[FactorAllocation, ...],
) -> tuple[pd.Series, pd.DataFrame]:
    """Cross-sectionally standardize inputs and combine available weighted values."""
    validate_panel(panel)
    if not allocations:
        raise ValueError("at least one factor allocation is required")
    standardized: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for allocation in allocations:
        allocation.validate()
        raw = (
            compile_panel_expression(allocation.expression).evaluate(panel)
            if allocation.expression is not None
            else compute_panel_factor(allocation.name, panel)
        )
        standardized[allocation.name] = cs_zscore(raw) * allocation.direction
        weights[allocation.name] = allocation.weight
    matrix = pd.DataFrame(standardized, index=panel.index)
    weighted = matrix.mul(pd.Series(weights))
    available_weight = matrix.notna().mul(pd.Series(weights)).sum(axis=1)
    composite = weighted.sum(axis=1, min_count=1).div(available_weight.where(available_weight > 0))
    return composite.rename("composite_score"), matrix


def _strategy_positions(
    composite: pd.Series,
    *,
    mode: Literal["long_short", "long_only"],
    rebalance_every: int,
) -> pd.Series:
    if mode == "long_short":
        target = panel_positions(composite)
    else:
        target = cs_scale(cs_rank(composite)).rename("position")
    wide = target.unstack(SYMBOL_LEVEL)
    rebalance_mask = np.arange(len(wide)) % rebalance_every == 0
    held = wide.copy()
    held.iloc[~rebalance_mask] = np.nan
    held = held.ffill().fillna(0.0)
    stacked = held.stack(future_stack=True)
    if not isinstance(stacked, pd.Series):
        raise TypeError("strategy positions must be one-dimensional")
    return stacked.reorder_levels([TIME_LEVEL, SYMBOL_LEVEL]).sort_index()


def run_panel_strategy(
    panel: pd.DataFrame,
    config: PanelStrategyConfig,
) -> dict[str, Any]:
    """Run a transparent score-to-position backtest on an existing panel."""
    config.validate()
    composite, inputs = combine_panel_factors(panel, config.allocations)
    positions = _strategy_positions(
        composite,
        mode=config.position_mode,
        rebalance_every=config.rebalance_every,
    )
    forward = panel_forward_returns(panel["close"], config.horizon)
    aligned = pd.concat(
        [positions.rename("position"), forward.rename("forward_return")], axis=1
    ).dropna()
    gross = (aligned["position"] * aligned["forward_return"]).groupby(
        level=TIME_LEVEL
    ).sum()
    position_wide = positions.unstack(SYMBOL_LEVEL).fillna(0.0)
    turnover = position_wide.diff().abs().sum(axis=1).mul(0.5).reindex(gross.index).fillna(0.0)
    net = gross - turnover * (config.fee_rate + config.slippage)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    standard_deviation = net.std(ddof=1)
    bar_sharpe = net.mean() / standard_deviation if standard_deviation > 0 else float("nan")
    return {
        "status": "computed_short_sample" if len(net) else "insufficient_data",
        "config": config.to_dict(),
        "metrics": {
            "n_periods": int(len(net)),
            "total_return": float(equity.iloc[-1] - 1) if len(equity) else None,
            "mean_bar_return": float(net.mean()) if len(net) else None,
            "bar_sharpe": float(bar_sharpe) if np.isfinite(bar_sharpe) else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
            "mean_turnover": float(turnover.mean()) if len(turnover) else None,
        },
        "factor_coverage": {
            column: float(inputs[column].notna().mean()) for column in inputs.columns
        },
        "lookahead_status": "pass_by_construction",
        "lookahead_policy": (
            "same-time cross-sectional signals at t are evaluated only against "
            "explicit t-to-t+h forward returns"
        ),
        "returns": [
            {
                "time": str(index),
                "gross_return": float(gross.loc[index]),
                "net_return": float(net.loc[index]),
                "turnover": float(turnover.loc[index]),
                "equity": float(equity.loc[index]),
                "drawdown": float(drawdown.loc[index]),
            }
            for index in net.index
        ],
        "research_note": (
            "Interactive current-sample backtest; no guarantee of execution quality, "
            "six-month OOS validation, or future performance."
        ),
    }
