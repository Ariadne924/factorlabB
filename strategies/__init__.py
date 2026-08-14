"""Task 2 strategy templates with one causal target-position contract."""

from strategies.catalog import (
    GridTradingStrategy,
    MeanReversionStrategy,
    StatisticalArbitrageStrategy,
    TrendFollowingStrategy,
    create_strategy,
    list_strategies,
)

__all__ = [
    "GridTradingStrategy",
    "MeanReversionStrategy",
    "StatisticalArbitrageStrategy",
    "TrendFollowingStrategy",
    "create_strategy",
    "list_strategies",
]

