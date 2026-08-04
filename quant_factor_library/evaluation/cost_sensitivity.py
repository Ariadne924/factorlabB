"""
成本敏感性分析模块

评估交易成本（手续费、滑点）对因子收益的影响。
"""

from __future__ import annotations

import pandas as pd


def compute_breakeven_cost(
    factor_values: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """计算因子策略的盈亏平衡成本

    Args:
        factor_values: 因子值序列
        forward_returns: 前向收益率序列

    Returns:
        策略可承受的最大单边交易成本
    """
    # TODO: 实现盈亏平衡成本计算
    raise NotImplementedError


def cost_adjusted_return(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
) -> pd.Series:
    """计算扣除交易成本后的净收益

    Args:
        factor_values: 因子值序列
        forward_returns: 前向收益率序列
        fee_rate: 手续费率
        slippage: 滑点

    Returns:
        扣除成本后的净收益序列
    """
    # TODO: 实现成本调整逻辑
    raise NotImplementedError
