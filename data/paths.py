"""
数据存储路径安全工具。

所有来自配置、API 或因子注册表的动态路径片段都必须先经过本模块校验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_INVALID_PATH_CHARS = frozenset('/\\\0<>:"|?*')


def normalize_path_segment(
    value: str,
    *,
    field_name: str,
    case: Literal["lower", "upper", "preserve"] = "preserve",
) -> str:
    """校验并规范化单个存储路径片段。"""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} 不能为空或包含首尾空白")
    if value in {".", ".."}:
        raise ValueError(f"{field_name} 不能是路径导航片段: {value!r}")
    if any(char in _INVALID_PATH_CHARS or ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} 包含非法路径字符: {value!r}")

    if case == "lower":
        return value.lower()
    if case == "upper":
        return value.upper()
    return value


def safe_data_path(base: Path, *segments: str) -> Path:
    """构造位于 base 内部的路径，并执行最终包含关系校验。"""
    resolved_base = base.resolve()
    candidate = resolved_base.joinpath(*segments)
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"数据路径越过允许的根目录: {candidate}") from exc
    return candidate
