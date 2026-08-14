"""
IC 分析模块

计算因子的 Rank IC 和 Pearson IC，评估因子的预测能力。
"""

from __future__ import annotations

import pandas as pd


def _aligned(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.concat([a.rename("factor"), b.rename("forward_return")], axis=1).dropna()


def compute_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """计算 Pearson IC；有效样本少于 2 时返回 NaN。"""
    data = _aligned(factor_values, forward_returns)
    return (
        float(data["factor"].corr(data["forward_return"], method="pearson"))
        if len(data) >= 2
        else float("nan")
    )


def compute_rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """计算 Rank IC（Spearman 秩相关系数）

    Args:
        factor_values: 因子值序列
        forward_returns: 前向收益率序列

    Returns:
        Rank IC 值
    """
    data = _aligned(factor_values, forward_returns)
    return (
        float(data["factor"].rank().corr(data["forward_return"].rank()))
        if len(data) >= 2
        else float("nan")
    )


def compute_ic_summary(factor_values: pd.DataFrame, forward_returns: pd.Series) -> pd.DataFrame:
    """计算 IC 统计摘要（IC 均值、ICIR、IC 胜率等）

    Args:
        factor_values: 多因子值 DataFrame，每列一个因子
        forward_returns: 前向收益率序列

    Returns:
        IC 统计摘要表
    """
    records: list[dict[str, float | int | str]] = []
    for name in factor_values.columns:
        data = _aligned(factor_values[name], forward_returns)
        rank_ic = compute_rank_ic(data["factor"], data["forward_return"])
        pearson_ic = compute_ic(data["factor"], data["forward_return"])
        window = min(20, len(data))
        rolling = (
            data["factor"]
            .rolling(window, min_periods=max(3, window // 2))
            .corr(data["forward_return"])
        )
        ic_std = rolling.std(ddof=1)
        icir = rolling.mean() / ic_std if pd.notna(ic_std) and ic_std > 0 else float("nan")
        records.append(
            {
                "factor": str(name),
                "n_obs": int(len(data)),
                "ic": pearson_ic,
                "rank_ic": rank_ic,
                "icir": float(icir),
            }
        )
    return pd.DataFrame.from_records(records).set_index("factor")


def rolling_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    window: int = 20,
    method: str = "spearman",
) -> pd.Series:
    """时间序列滚动相关；用于短样本诊断，不等同于跨资产截面 IC。"""
    if window < 3:
        raise ValueError("window 必须至少为 3")
    data = _aligned(factor_values, forward_returns)
    if method == "spearman":
        left = data["factor"].rank()
        right = data["forward_return"].rank()
    elif method == "pearson":
        left, right = data["factor"], data["forward_return"]
    else:
        raise ValueError("method 仅支持 pearson/spearman")
    return left.rolling(window, min_periods=window).corr(right).rename("rolling_ic")
