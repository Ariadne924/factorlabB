"""
Bronze 层 — 原始数据存储

职责：存储交易所 API 原始数据，不做任何清洗或转换。
数据格式：parquet（压缩：zstd）
目录结构：data/bronze/{exchange}/{market}/{symbol}/{interval}/klines.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DEFAULT_DATA_DIR
from data.paths import normalize_path_segment, safe_data_path


def bronze_klines_path(
    *,
    exchange: str,
    market: str = "spot",
    symbol: str,
    interval: str,
    file_format: str = "parquet",
) -> Path:
    """生成 Bronze 层 K 线数据路径

    Args:
        exchange: 交易所名称（binance/okx）
        market: 市场类型（spot/futures）
        symbol: 交易对（'BTCUSDT'）
        interval: K 线周期（'1h'）
        file_format: 文件格式（parquet/json）

    Returns:
        文件路径
    """
    norm_exchange = normalize_path_segment(
        exchange, field_name="exchange", case="lower"
    )
    norm_market = normalize_path_segment(market, field_name="market", case="lower")
    norm_symbol = normalize_path_segment(symbol, field_name="symbol", case="upper")
    norm_interval = normalize_path_segment(
        interval, field_name="interval", case="lower"
    )
    norm_fmt = normalize_path_segment(
        file_format.lstrip("."), field_name="file_format", case="lower"
    )

    if norm_fmt not in {"parquet", "json"}:
        raise ValueError(f"不支持的文件格式: '{file_format}'，仅支持 parquet/json")

    return safe_data_path(
        DEFAULT_DATA_DIR,
        "bronze",
        norm_exchange,
        norm_market,
        norm_symbol,
        norm_interval,
        f"klines.{norm_fmt}",
    )


def write_bronze_klines(df: pd.DataFrame, path: Path) -> None:
    """写入 Bronze 层 parquet 文件

    Args:
        df: 原始 K 线 DataFrame（来自 KlineRaw 转换）
        path: 输出文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)


def read_bronze_klines(path: Path) -> pd.DataFrame:
    """读取 Bronze 层 parquet 文件"""
    if not path.exists():
        raise FileNotFoundError(f"Bronze 文件不存在: {path}")
    return pd.read_parquet(path)
