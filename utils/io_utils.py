"""
IO 工具模块

统一的文件读写接口，处理路径规范化、编码管理等。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from utils.file_lock import FileLock


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
    """原子写入 JSON 文件，避免读方看到半个文档。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _replace_parquet(df: pd.DataFrame, path: Path, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    df.to_parquet(temporary, compression="zstd", index=index)
    temporary.replace(path)


def write_parquet_safe(
    df: pd.DataFrame,
    filepath: str | Path,
    *,
    index: bool = False,
    timeout: float = 15.0,
) -> None:
    """Lock and atomically replace one parquet dataset."""
    path = Path(filepath)
    with FileLock(path.with_suffix(f"{path.suffix}.lock"), timeout=timeout):
        _replace_parquet(df, path, index=index)


def merge_parquet_safe(
    filepath: str | Path,
    frame: pd.DataFrame,
    *,
    key: str,
    index: bool = False,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Lock the full read/merge/replace transaction to prevent lost updates."""
    path = Path(filepath)
    with FileLock(path.with_suffix(f"{path.suffix}.lock"), timeout=timeout):
        combined = frame.copy()
        if path.exists():
            combined = pd.concat([pd.read_parquet(path), combined], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(subset=[key], keep="last").sort_values(key)
        combined = combined.reset_index(drop=True)
        _replace_parquet(combined, path, index=index)
        return combined
