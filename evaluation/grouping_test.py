"""
分组测试模块

将因子值排序分组，计算各组的多空收益差，
评估因子的区分度。
"""

from __future__ import annotations

import pandas as pd


def grouping_backtest(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_groups: int = 5,
) -> pd.DataFrame:
    """因子分组回测

    按因子值从低到高分为 n_groups 组，
    计算各组等权平均收益，以及多空组收益差。

    Args:
        factor_values: 因子值序列
        forward_returns: 前向收益率序列
        n_groups: 分组数量（默认 5 组）

    Returns:
        各组收益统计表
    """
    if n_groups < 2:
        raise ValueError("n_groups 必须至少为 2")
    data = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("forward_return")], axis=1
    ).dropna()
    if len(data) < n_groups:
        return pd.DataFrame(columns=["group", "mean_return", "count"])
    ranked = data["factor"].rank(method="first")
    data["group"] = pd.qcut(ranked, q=n_groups, labels=False) + 1
    result = (
        data.groupby("group", observed=True)["forward_return"]
        .agg(mean_return="mean", count="count")
        .reset_index()
    )
    spread = float(result.iloc[-1]["mean_return"] - result.iloc[0]["mean_return"])
    result["top_bottom_spread"] = spread
    return result
