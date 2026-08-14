"""
Silver 层 — 标准化数据

职责：将 Bronze 原始数据清洗为标准格式。
  - 时间戳：毫秒 int → datetime64[ms, UTC]
  - 列名：统一 snake_case
  - 数据类型：float → float64，int → int64
  - 校验：完整性、连续性、OHLCV 逻辑正确性

数据格式：parquet（压缩：zstd）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DEFAULT_DATA_DIR
from data.paths import normalize_path_segment, safe_data_path
from data.validator import DataValidator
from utils.io_utils import write_parquet_safe


def silver_klines_path(
    *,
    exchange: str,
    market: str = "spot",
    symbol: str,
    interval: str,
) -> Path:
    """生成 Silver 层 K 线数据路径"""
    return safe_data_path(
        DEFAULT_DATA_DIR,
        "silver",
        normalize_path_segment(exchange, field_name="exchange", case="lower"),
        normalize_path_segment(market, field_name="market", case="lower"),
        normalize_path_segment(symbol, field_name="symbol", case="upper"),
        normalize_path_segment(interval, field_name="interval", case="lower"),
        "klines.parquet",
    )


def build_silver_from_bronze(bronze_path: Path, silver_path: Path) -> pd.DataFrame:
    """Bronze → Silver 转换并写入

    Args:
        bronze_path: Bronze 层 parquet 文件路径
        silver_path: Silver 层输出路径

    Returns:
        转换后的 Silver DataFrame
    """
    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze 文件不存在: {bronze_path}")

    # 延迟导入以避免在没有 pydantic 的环境中提前触发依赖
    from data.schema import bronze_to_silver

    bronze_df = pd.read_parquet(bronze_path)
    silver_df = bronze_to_silver(bronze_df)
    DataValidator.validate_klines(silver_df)

    # 写入 Silver 层
    write_parquet_safe(silver_df, silver_path)

    return silver_df


def read_silver_klines(path: Path) -> pd.DataFrame:
    """读取 Silver 层 parquet 文件"""
    if not path.exists():
        raise FileNotFoundError(f"Silver 文件不存在: {path}")
    return pd.read_parquet(path)


def silver_to_factor_input(silver_df: pd.DataFrame) -> pd.DataFrame:
    """将 Silver DataFrame 转为因子计算的输入格式

    因子签名是 DataFrame → Series，Silver 层数据可直接作为因子输入。
    此函数负责：
      1. 确保索引为 open_time_utc（因子计算中用时间对齐）
      2. 可选：添加额外列（如从其他数据源 join 的资金费率）

    Args:
        silver_df: Silver 层标准化 DataFrame

    Returns:
        因子计算输入 DataFrame（索引为 UTC 时间戳）
    """
    df = silver_df.copy()
    DataValidator.validate_klines(df)
    df = df.set_index("open_time_utc")
    df = df.sort_index()
    return df


def merge_point_in_time_features(
    klines: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_time_column: str = "timestamp",
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """向后合并扩展数据，确保时间 t 不会读到 t 之后发布的值。"""
    if not isinstance(klines.index, pd.DatetimeIndex) or klines.index.tz is None:
        raise ValueError("klines 必须使用带时区 DatetimeIndex")
    required = [feature_time_column, *feature_columns]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"扩展数据缺少列: {missing}")
    original_index = klines.index
    normalized_index = pd.DatetimeIndex(
        pd.to_datetime(original_index, utc=True)
    ).astype("datetime64[ns, UTC]")
    right = features[required].copy()
    right[feature_time_column] = pd.to_datetime(
        right[feature_time_column], utc=True
    ).astype("datetime64[ns, UTC]")
    right = right.sort_values(feature_time_column)
    left_name = klines.index.name or "open_time_utc"
    normalized_klines = klines.copy()
    normalized_klines.index = normalized_index
    left = normalized_klines.reset_index(names=left_name)
    left[left_name] = pd.to_datetime(left[left_name], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    left = left.sort_values(left_name)
    merged = pd.merge_asof(
        left,
        right,
        left_on=left_name,
        right_on=feature_time_column,
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    ).drop(columns=[feature_time_column])
    result = merged.set_index(left_name).reindex(normalized_index)
    result.index = original_index
    return result
