"""
稳定性分析模块

评估因子 IC 和分组收益在时间序列上的稳定性。
"""

from __future__ import annotations

import pandas as pd


def compute_ic_decay(factor_values: pd.Series, returns: pd.Series, max_lag: int = 20) -> pd.Series:
    """计算 IC 衰减

    分析因子对未来不同持有期收益的预测能力衰减情况。

    Args:
        factor_values: 因子值序列
        returns: 各期收益率
        max_lag: 最大滞后期数

    Returns:
        IC 衰减序列
    """
    if max_lag < 1:
        raise ValueError("max_lag 必须大于 0")
    values = {}
    for lag in range(1, max_lag + 1):
        target = returns.shift(-lag)
        aligned = pd.concat([factor_values, target], axis=1).dropna()
        values[lag] = (
            aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank())
            if len(aligned) >= 2
            else float("nan")
        )
    return pd.Series(values, name="rank_ic", dtype="float64").rename_axis("lag")


def turnover_analysis(factor_values: pd.DataFrame) -> pd.DataFrame:
    """因子换手率分析

    计算因子值在各期之间的变化程度。

    Args:
        factor_values: 多因子值 DataFrame

    Returns:
        换手率统计表
    """
    if factor_values.empty:
        return pd.DataFrame(columns=["mean_turnover", "median_turnover"])
    ranked = factor_values.rank(pct=True)
    changes = ranked.diff().abs()
    return pd.DataFrame(
        {
            "mean_turnover": changes.mean(),
            "median_turnover": changes.median(),
        }
    )


def compute_turnover(factor_values: pd.Series) -> float:
    """单序列的归一化秩变化率，用于同一资产的稳定性描述。"""
    clean = factor_values.dropna()
    if len(clean) < 2:
        return float("nan")
    rank = clean.rank(pct=True)
    return float(rank.diff().abs().mean())
