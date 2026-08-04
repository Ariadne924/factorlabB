"""
数据校验工具

对下载的原始数据进行检查，包括：
- 完整性检查（是否有缺失 K 线）
- 异常值检查（价格/成交量异常）
- 时间戳连续性检查
"""

from __future__ import annotations

import pandas as pd


class DataValidator:
    """数据校验器

    对下载后的数据进行多维度校验，确保数据质量。
    """

    @staticmethod
    def check_missing_klines(df: pd.DataFrame, expected_interval: str) -> pd.DataFrame:
        """检查 K 线数据是否存在缺失

        Args:
            df: K 线 DataFrame
            expected_interval: 期望的 K 线周期

        Returns:
            缺失时间段信息
        """
        # TODO: 实现缺失 K 线检测逻辑
        raise NotImplementedError

    @staticmethod
    def check_outliers(df: pd.DataFrame, column: str, n_std: float = 5.0) -> pd.Series:
        """检查异常值（基于标准差法）

        Args:
            df: 数据 DataFrame
            column: 待检查的列名
            n_std: 标准差倍数阈值

        Returns:
            异常值的布尔掩码
        """
        # TODO: 实现异常值检测逻辑
        raise NotImplementedError

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> bool:
        """检查 DataFrame 是否包含所有必需列"""
        return all(col in df.columns for col in required_columns)
