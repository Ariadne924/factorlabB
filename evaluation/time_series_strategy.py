"""Causal single-asset multi-factor combination and lightweight backtesting."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from factors.registry import compute_factor, get_factor_metadata, list_factors


@dataclass(frozen=True)
class TimeSeriesAllocation:
    name: str
    weight: float = 1.0
    direction: Literal[-1, 1] = 1
    params: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.name not in list_factors():
            raise ValueError(f"unknown registered time-series factor: {self.name}")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"factor weight must be positive and finite: {self.name}")
        if self.direction not in {-1, 1}:
            raise ValueError(f"factor direction must be -1 or 1: {self.name}")


@dataclass(frozen=True)
class TimeSeriesStrategyConfig:
    allocations: tuple[TimeSeriesAllocation, ...]
    interval: str
    rebalance_every: int = 1
    position_mode: Literal["long_short", "long_only"] = "long_short"
    score_threshold: float = 0.0
    standardize_window: int = 168
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
        if self.rebalance_every < 1 or self.standardize_window < 5:
            raise ValueError("rebalance_every must be positive and standardize_window at least 5")
        if self.position_mode not in {"long_short", "long_only"}:
            raise ValueError("unsupported position mode")
        if not np.isfinite(self.score_threshold) or self.score_threshold < 0:
            raise ValueError("score_threshold must be non-negative and finite")
        if self.fee_rate < 0 or self.slippage < 0:
            raise ValueError("cost assumptions cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _causal_zscore(values: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    minimum = min(window, max(5, window // 4))
    mean = numeric.rolling(window, min_periods=minimum).mean()
    std = numeric.rolling(window, min_periods=minimum).std(ddof=0)
    return numeric.sub(mean).div(std.where(std > 0))


def combine_time_series_factors(
    frame: pd.DataFrame,
    allocations: tuple[TimeSeriesAllocation, ...],
    *,
    standardize_window: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Causally standardize factors and combine only values available at each row."""
    if not allocations:
        raise ValueError("at least one factor allocation is required")
    standardized: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for allocation in allocations:
        allocation.validate()
        raw = compute_factor(allocation.name, frame, params=allocation.params)
        standardized[allocation.name] = (
            _causal_zscore(raw, standardize_window) * allocation.direction
        )
        weights[allocation.name] = allocation.weight
    matrix = pd.DataFrame(standardized, index=frame.index)
    weight_series = pd.Series(weights, dtype=float)
    available_weight = matrix.notna().mul(weight_series).sum(axis=1)
    composite = matrix.mul(weight_series).sum(axis=1, min_count=1).div(
        available_weight.where(available_weight > 0)
    )
    return composite.rename("composite_score"), matrix


def _target_positions(
    composite: pd.Series,
    *,
    mode: Literal["long_short", "long_only"],
    threshold: float,
    rebalance_every: int,
) -> pd.Series:
    if mode == "long_short":
        target = pd.Series(
            np.select(
                [composite > threshold, composite < -threshold],
                [1.0, -1.0],
                default=0.0,
            ),
            index=composite.index,
            name="target_position",
        )
    else:
        target = composite.gt(threshold).astype(float).rename("target_position")
    rebalance = np.arange(len(target)) % rebalance_every == 0
    held = target.where(rebalance).ffill().fillna(0.0)
    return held.rename("target_position")


def run_time_series_strategy(
    frame: pd.DataFrame,
    config: TimeSeriesStrategyConfig,
) -> dict[str, Any]:
    """Run a one-bar-lagged strategy so a signal never trades its own source bar."""
    config.validate()
    if "close" not in frame.columns:
        raise ValueError("strategy input requires close")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("strategy input requires a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("strategy input index must be unique and increasing")
    composite, inputs = combine_time_series_factors(
        frame,
        config.allocations,
        standardize_window=config.standardize_window,
    )
    target = _target_positions(
        composite,
        mode=config.position_mode,
        threshold=config.score_threshold,
        rebalance_every=config.rebalance_every,
    )
    position = target.shift(1).fillna(0.0).rename("position")
    bar_return = frame["close"].pct_change(fill_method=None).rename("bar_return")
    aligned = pd.concat([composite, target, position, bar_return], axis=1).dropna()
    turnover = aligned["position"].diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(aligned["position"].iloc[0])
    gross = aligned["position"] * aligned["bar_return"]
    costs = turnover * (config.fee_rate + config.slippage)
    net = gross - costs
    gross_equity = (1.0 + gross).cumprod()
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    standard_deviation = net.std(ddof=1)
    bar_sharpe = net.mean() / standard_deviation if standard_deviation > 0 else np.nan
    active = aligned["position"].ne(0)
    hit_rate = gross.loc[active].gt(0).mean() if active.any() else np.nan
    return {
        "status": "computed_short_sample" if len(net) else "insufficient_data",
        "config": config.to_dict(),
        "metrics": {
            "n_periods": int(len(net)),
            "gross_total_return": float(gross_equity.iloc[-1] - 1) if len(net) else None,
            "total_return": float(equity.iloc[-1] - 1) if len(net) else None,
            "cost_drag": (
                float(gross_equity.iloc[-1] - equity.iloc[-1]) if len(net) else None
            ),
            "mean_bar_return": float(net.mean()) if len(net) else None,
            "bar_sharpe": float(bar_sharpe) if np.isfinite(bar_sharpe) else None,
            "max_drawdown": float(drawdown.min()) if len(net) else None,
            "mean_turnover": float(turnover.mean()) if len(net) else None,
            "hit_rate": float(hit_rate) if np.isfinite(hit_rate) else None,
            "active_exposure": float(active.mean()) if len(net) else None,
        },
        "factor_coverage": {
            column: float(inputs[column].notna().mean()) for column in inputs.columns
        },
        "lookahead_status": "pass_by_construction",
        "lookahead_policy": (
            "factor and composite are formed at t; target position is shifted one full bar "
            "and first affects the t-to-t+1 return recorded at t+1"
        ),
        "returns": [
            {
                "time": str(index),
                "score": float(aligned.loc[index, "composite_score"]),
                "target_position": float(aligned.loc[index, "target_position"]),
                "position": float(aligned.loc[index, "position"]),
                "bar_return": float(aligned.loc[index, "bar_return"]),
                "gross_return": float(gross.loc[index]),
                "cost": float(costs.loc[index]),
                "net_return": float(net.loc[index]),
                "turnover": float(turnover.loc[index]),
                "equity": float(equity.loc[index]),
                "drawdown": float(drawdown.loc[index]),
            }
            for index in net.index
        ],
        "research_note": (
            "Interactive current-sample backtest with one-bar signal lag; it is not proof "
            "of execution quality, six-month OOS validation, or future performance."
        ),
    }


def _parse_candidate_feature(name: str) -> tuple[str, dict[str, int] | None]:
    window = re.fullmatch(r"(.+)__w(\d+)", name)
    if window and window.group(1) in list_factors():
        return window.group(1), {"window": int(window.group(2))}
    pair = re.fullmatch(r"(.+)__s(\d+)_l(\d+)", name)
    if pair and pair.group(1) in list_factors():
        base = pair.group(1)
        defaults = get_factor_metadata(base).get("default_params", {})
        return pair.group(1), {
            **defaults,
            "short_window": int(pair.group(2)),
            "long_window": int(pair.group(3)),
        }
    return name, None


def build_ml_strategy_preset(
    recommendations: list[dict[str, Any]],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Convert train-fold-only ML recommendations into a Strategy Builder preset."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    allocations: list[dict[str, Any]] = []
    skipped: list[str] = []
    selected_names: set[str] = set()
    for row in recommendations:
        candidate = str(row.get("feature", ""))
        name, params = _parse_candidate_feature(candidate)
        if name not in list_factors() or name in selected_names:
            skipped.append(candidate)
            continue
        score = float(row.get("recommendation_score") or 0.0)
        weight = score if np.isfinite(score) and score > 0 else 1.0
        direction = 1 if int(row.get("direction", 1)) >= 0 else -1
        allocations.append(
            {
                "name": name,
                "weight": weight,
                "direction": direction,
                "params": params,
                "source_feature": candidate,
            }
        )
        selected_names.add(name)
        if len(allocations) >= top_n:
            break
    total = sum(float(item["weight"]) for item in allocations)
    if total > 0:
        for item in allocations:
            item["weight"] = float(item["weight"]) / total
    return {
        "status": "ready" if allocations else "insufficient_recommendations",
        "allocations": allocations,
        "skipped": skipped,
        "source": "ml_train_fold_recommendations",
        "warning": "Preset weights are candidates, not validated optimal portfolio weights.",
    }
