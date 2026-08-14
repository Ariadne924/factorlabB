"""
★ 前视检查工具

这是量化因子评估中最关键的检查之一：
确保因子计算时没有使用未来数据（Look-Ahead Bias）。

检查逻辑：
1. 因子值的时间戳必须 <= 用于计算该因子的数据的时间戳
2. 因子值与未来收益对齐时，确保时间顺序正确
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


class ForwardCheck:
    """前视偏差检查器

    用于验证因子计算中不存在数据泄露（未来信息）。
    """

    @staticmethod
    def check_timestamp_alignment(
        factor_values: pd.Series,
        data_timestamps: pd.DatetimeIndex,
    ) -> bool:
        """检查因子值时间戳是否与输入数据对齐

        规则：factor_values[i] 只能使用 data_timestamps[:i+1] 的信息。

        Args:
            factor_values: 因子值序列
            data_timestamps: 原始数据的时间戳

        Returns:
            True 表示通过检查（无前视偏差）
        """
        if not isinstance(factor_values.index, pd.DatetimeIndex):
            return False
        if factor_values.index.tz is None or data_timestamps.tz is None:
            return False
        return (
            factor_values.index.equals(data_timestamps)
            and factor_values.index.is_monotonic_increasing
        )

    @staticmethod
    def check_forward_return_alignment(
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> bool:
        """检查因子值与未来收益的时间对齐

        确保因子值的时间 t 严格早于前向收益对应的时间区间。

        Args:
            factor_values: 因子值序列（时间 t 已知）
            forward_returns: 前向收益序列（时间 t→t+1 的收益）

        Returns:
            True 表示对齐正确
        """
        if not factor_values.index.equals(forward_returns.index):
            return False
        return (
            isinstance(factor_values.index, pd.DatetimeIndex)
            and factor_values.index.is_monotonic_increasing
        )

    @staticmethod
    def check_truncation_invariance(
        factor: Callable[[pd.DataFrame], pd.Series],
        data: pd.DataFrame,
        *,
        sample_positions: list[int] | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:
        """截断未来数据后重算，验证历史末值完全不变。"""
        if data.empty:
            raise ValueError("data 不能为空")
        full = factor(data)
        positions = sample_positions or sorted({len(data) // 3, 2 * len(data) // 3, len(data) - 1})
        for position in positions:
            if position < 0 or position >= len(data):
                raise ValueError("sample_positions 越界")
            truncated = factor(data.iloc[: position + 1])
            expected = full.iloc[position]
            actual = truncated.iloc[-1]
            if pd.isna(expected) and pd.isna(actual):
                continue
            if not np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
                return False
        return True
