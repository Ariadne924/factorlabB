"""Four transparent Task 2 strategy families with causal signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategyMetadata


def _validate_windows(short: int, long: int) -> None:
    if short < 2 or long <= short:
        raise ValueError("windows must satisfy 2 <= short < long")


@dataclass
class TrendFollowingStrategy(BaseStrategy):
    fast_window: int = 24
    slow_window: int = 168

    metadata = StrategyMetadata(
        name="trend_following",
        category="趋势跟踪",
        description="快慢均线方向策略",
        required_columns=("close",),
    )

    def generate_target(self, frame: pd.DataFrame) -> pd.Series:
        self.validate_frame(frame)
        _validate_windows(self.fast_window, self.slow_window)
        close = pd.to_numeric(frame["close"], errors="coerce")
        fast = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
        slow = close.rolling(self.slow_window, min_periods=self.slow_window).mean()
        target = pd.Series(np.sign(fast - slow), index=frame.index, dtype=float)
        return target.where(fast.notna() & slow.notna(), 0.0).rename("target_position")


@dataclass
class MeanReversionStrategy(BaseStrategy):
    window: int = 72
    entry_z: float = 1.5

    metadata = StrategyMetadata(
        name="mean_reversion",
        category="均值回归",
        description="价格偏离滚动均值后的反向策略",
        required_columns=("close",),
    )

    def generate_target(self, frame: pd.DataFrame) -> pd.Series:
        self.validate_frame(frame)
        if self.window < 5 or self.entry_z <= 0:
            raise ValueError("window must be at least 5 and entry_z positive")
        close = pd.to_numeric(frame["close"], errors="coerce")
        mean = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std(ddof=0)
        zscore = close.sub(mean).div(std.where(std > 0))
        target = pd.Series(
            np.select([zscore >= self.entry_z, zscore <= -self.entry_z], [-1.0, 1.0]),
            index=frame.index,
        )
        return target.where(zscore.notna(), 0.0).rename("target_position")


@dataclass
class GridTradingStrategy(BaseStrategy):
    anchor_window: int = 72
    grid_step: float = 0.01
    levels: int = 4

    metadata = StrategyMetadata(
        name="grid_trading",
        category="网格交易",
        description="围绕滚动锚点分层调整库存",
        required_columns=("close",),
    )

    def generate_target(self, frame: pd.DataFrame) -> pd.Series:
        self.validate_frame(frame)
        if self.anchor_window < 5 or self.grid_step <= 0 or self.levels < 1:
            raise ValueError("invalid grid parameters")
        close = pd.to_numeric(frame["close"], errors="coerce")
        anchor = close.rolling(
            self.anchor_window, min_periods=self.anchor_window
        ).mean()
        deviation = close.div(anchor).sub(1.0)
        inventory_level = -np.floor(deviation / self.grid_step)
        target = inventory_level.clip(-self.levels, self.levels).div(self.levels)
        return target.where(anchor.notna(), 0.0).rename("target_position")


@dataclass
class StatisticalArbitrageStrategy(BaseStrategy):
    hedge_window: int = 168
    z_window: int = 72
    entry_z: float = 1.5

    metadata = StrategyMetadata(
        name="statistical_arbitrage",
        category="统计套利",
        description="目标资产与参照资产的滚动对冲价差策略",
        required_columns=("close", "reference_close"),
    )

    def generate_target(self, frame: pd.DataFrame) -> pd.Series:
        self.validate_frame(frame)
        if min(self.hedge_window, self.z_window) < 5 or self.entry_z <= 0:
            raise ValueError("invalid statistical-arbitrage parameters")
        left = np.log(pd.to_numeric(frame["close"], errors="coerce").where(lambda x: x > 0))
        right = np.log(
            pd.to_numeric(frame["reference_close"], errors="coerce").where(lambda x: x > 0)
        )
        covariance = left.rolling(
            self.hedge_window, min_periods=self.hedge_window
        ).cov(right)
        variance = right.rolling(
            self.hedge_window, min_periods=self.hedge_window
        ).var(ddof=0)
        hedge = covariance.div(variance.where(variance > 0))
        spread = left - hedge * right
        spread_mean = spread.rolling(self.z_window, min_periods=self.z_window).mean()
        spread_std = spread.rolling(self.z_window, min_periods=self.z_window).std(ddof=0)
        zscore = spread.sub(spread_mean).div(spread_std.where(spread_std > 0))
        target = pd.Series(
            np.select([zscore >= self.entry_z, zscore <= -self.entry_z], [-1.0, 1.0]),
            index=frame.index,
        )
        return target.where(zscore.notna(), 0.0).rename("target_position")


_STRATEGIES: dict[str, type[BaseStrategy]] = {
    strategy.metadata.name: strategy
    for strategy in (
        TrendFollowingStrategy,
        GridTradingStrategy,
        StatisticalArbitrageStrategy,
        MeanReversionStrategy,
    )
}


def list_strategies() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "category": strategy.metadata.category,
            "description": strategy.metadata.description,
            "required_columns": list(strategy.metadata.required_columns),
        }
        for name, strategy in _STRATEGIES.items()
    ]


def create_strategy(name: str, **params: Any) -> BaseStrategy:
    try:
        strategy_type = _STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {name}") from exc
    return strategy_type(**params)

