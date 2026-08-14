"""
因子相关性分析模块

计算因子间的截面相关性，辅助因子筛选和组合。
"""

from __future__ import annotations

import pandas as pd


def correlation_matrix(factor_values: pd.DataFrame) -> pd.DataFrame:
    """计算因子截面相关性矩阵

    Args:
        factor_values: 多因子值 DataFrame，每列一个因子

    Returns:
        相关系数矩阵
    """
    return factor_values.rank().corr(method="pearson")


def find_redundant_factors(
    factor_values: pd.DataFrame,
    threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    """查找高度相关的冗余因子对

    Args:
        factor_values: 多因子值 DataFrame
        threshold: 相关性阈值，超过此值视为冗余

    Returns:
        [(因子A, 因子B, 相关系数), ...] 列表
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 0..1")
    matrix = correlation_matrix(factor_values)
    names = list(matrix.columns)
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((str(left), str(right), float(value)))
    return pairs
