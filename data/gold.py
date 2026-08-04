"""
Gold 层 — 因子产出

职责：存储因子计算结果，是评估和可视化的直接数据源。
每个因子一个 parquet 文件，以因子名命名。
目录结构：data/gold/{symbol}/{interval}/{factor_name}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DEFAULT_DATA_DIR
from data.paths import normalize_path_segment, safe_data_path


def gold_factor_path(
    *,
    symbol: str,
    interval: str,
    factor_name: str,
) -> Path:
    """生成 Gold 层因子文件路径

    Args:
        symbol: 交易对（'BTCUSDT'）
        interval: K 线周期（'1h'）
        factor_name: 因子名称

    Returns:
        文件路径
    """
    normalized_symbol = normalize_path_segment(
        symbol, field_name="symbol", case="upper"
    )
    normalized_interval = normalize_path_segment(
        interval, field_name="interval", case="lower"
    )
    normalized_factor = normalize_path_segment(
        factor_name, field_name="factor_name"
    )
    return safe_data_path(
        DEFAULT_DATA_DIR,
        "gold",
        normalized_symbol,
        normalized_interval,
        f"{normalized_factor}.parquet",
    )


def write_gold_factor(
    factor_series: pd.Series,
    *,
    symbol: str,
    interval: str,
    factor_name: str,
) -> Path:
    """将因子值写入 Gold 层

    Args:
        factor_series: 因子值序列（索引为时间戳）
        symbol: 交易对
        interval: K 线周期
        factor_name: 因子名称

    Returns:
        写入的文件路径
    """
    output_path = gold_factor_path(
        symbol=symbol,
        interval=interval,
        factor_name=factor_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = factor_series.to_frame(name=factor_name)
    df.to_parquet(output_path, compression="zstd", index=True)

    return output_path


def read_gold_factor(
    *,
    symbol: str,
    interval: str,
    factor_name: str,
) -> pd.Series | None:
    """从 Gold 层读取因子值

    Returns:
        因子值 Series，文件不存在时返回 None
    """
    input_path = gold_factor_path(
        symbol=symbol,
        interval=interval,
        factor_name=factor_name,
    )
    if not input_path.exists():
        return None

    df = pd.read_parquet(input_path)
    return df.iloc[:, 0]  # 第一列为因子值


def list_gold_factors(
    *,
    symbol: str,
    interval: str,
) -> list[str]:
    """列出某交易对/周期下所有已计算的因子名称"""
    base = safe_data_path(
        DEFAULT_DATA_DIR,
        "gold",
        normalize_path_segment(symbol, field_name="symbol", case="upper"),
        normalize_path_segment(interval, field_name="interval", case="lower"),
    )
    if not base.exists():
        return []
    return sorted([p.stem for p in base.glob("*.parquet")])
