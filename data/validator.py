"""
数据校验工具

对下载的原始数据进行检查，包括：
- 完整性检查（是否有缺失 K 线）
- 异常值检查（价格/成交量异常）
- 时间戳连续性检查
"""

from __future__ import annotations

import pandas as pd

from config.constants import get_interval_ms
from data.columns import SILVER_COLUMNS


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
        DataValidator.validate_klines(df)
        interval_ms = get_interval_ms(expected_interval)
        timestamps = pd.DatetimeIndex(df["open_time_utc"]).sort_values()
        expected = pd.date_range(
            start=timestamps[0],
            end=timestamps[-1],
            freq=pd.Timedelta(milliseconds=interval_ms),
            tz="UTC",
        )
        missing = expected.difference(timestamps)
        return pd.DataFrame({"open_time_utc": missing})

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
        if column not in df.columns:
            raise ValueError(f"待检查列不存在: {column!r}")
        if n_std <= 0:
            raise ValueError("n_std 必须大于 0")
        values = pd.to_numeric(df[column], errors="coerce")
        std = values.std()
        if pd.isna(std) or std == 0:
            return pd.Series(False, index=df.index, dtype=bool)
        return (values - values.mean()).abs() > n_std * std

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> bool:
        """检查 DataFrame 是否包含所有必需列"""
        return all(col in df.columns for col in required_columns)

    @staticmethod
    def validate_klines(df: pd.DataFrame) -> None:
        """严格校验 Silver K 线的数据契约。"""
        missing = [column for column in SILVER_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Silver K 线缺少必须列: {missing}")
        if df.empty:
            raise ValueError("Silver K 线不能为空")

        for column in ("open_time_utc", "close_time_utc"):
            if not isinstance(df[column].dtype, pd.DatetimeTZDtype):
                raise ValueError(f"{column} 必须是带时区的 datetime64 类型")

        timestamps = pd.DatetimeIndex(df["open_time_utc"])
        close_timestamps = pd.DatetimeIndex(df["close_time_utc"])
        if timestamps.tz is None or str(timestamps.tz) != "UTC":
            raise ValueError("open_time_utc 必须是 UTC 时区时间戳")
        if close_timestamps.tz is None or str(close_timestamps.tz) != "UTC":
            raise ValueError("close_time_utc 必须是 UTC 时区时间戳")
        if (close_timestamps < timestamps).any():
            raise ValueError("close_time_utc 不能早于 open_time_utc")
        if timestamps.has_duplicates:
            raise ValueError("open_time_utc 不能包含重复时间戳")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("open_time_utc 必须按时间升序排列")

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "num_trades",
        ]
        non_numeric = [
            column
            for column in numeric_columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric:
            raise ValueError(f"Silver K 线数值列类型不合法: {non_numeric}")
        if not pd.api.types.is_integer_dtype(df["num_trades"]):
            raise ValueError("num_trades 必须是整数类型")
        if df[numeric_columns].isna().any().any():
            raise ValueError("Silver K 线数值列不能包含缺失值")

        price_columns = ["open", "high", "low", "close"]
        if (df[price_columns] <= 0).any().any():
            raise ValueError("OHLC 价格必须大于 0")
        if (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        ).any():
            raise ValueError("OHLC 价格关系不合法")

        nonnegative_columns = [
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "num_trades",
        ]
        if (df[nonnegative_columns] < 0).any().any():
            raise ValueError("成交量、成交额和成交笔数不能为负数")
        for column in ("symbol", "interval"):
            if not df[column].map(
                lambda value: isinstance(value, str) and bool(value)
            ).all():
                raise ValueError(f"{column} 必须是非空字符串")
