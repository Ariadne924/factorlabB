"""
IO 工具模块

统一的文件读写接口，处理路径规范化、编码管理等。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_safe(filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
    """安全读取 CSV 文件（统一编码和解析设置）

    Args:
        filepath: 文件路径
        **kwargs: 传递给 pd.read_csv 的额外参数

    Returns:
        DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 默认参数
    defaults = {
        "encoding": "utf-8",
        "parse_dates": True,
    }
    defaults.update(kwargs)

    return pd.read_csv(path, **defaults)


def write_csv_safe(df: pd.DataFrame, filepath: str | Path, **kwargs: Any) -> None:
    """安全写入 CSV 文件（统一编码）

    Args:
        df: 要写入的 DataFrame
        filepath: 目标文件路径
        **kwargs: 传递给 df.to_csv 的额外参数
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    defaults = {
        "encoding": "utf-8",
        "index": False,
    }
    defaults.update(kwargs)

    df.to_csv(path, **defaults)


def read_json_safe(filepath: str | Path) -> dict[str, Any]:
    """安全读取 JSON 文件"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_safe(data: dict[str, Any], filepath: str | Path) -> None:
    """安全写入 JSON 文件"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
