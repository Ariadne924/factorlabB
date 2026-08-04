"""
因子注册机制

提供全局因子注册表 + 装饰器注册方式。
所有因子在注册后可由评估框架和可视化模块统一发现和调用。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from factors.base import FactorFunction

# ============================================================================
# 全局因子注册表
# ============================================================================
FACTOR_REGISTRY: dict[str, dict[str, Any]] = {}
"""
因子注册表结构：

{
    "factor_name": {
        "func": callable,          # 因子计算函数
        "category": str,           # 因子分类（动量/波动率/...）
        "description": str,        # 因子描述
        "params": dict,            # 默认参数
    },
    ...
}
"""


# ============================================================================
# 注册装饰器
# ============================================================================
def register_factor(
    name: Optional[str] = None,
    category: str = "未分类",
    description: str = "",
    params: Optional[dict[str, Any]] = None,
) -> Callable:
    """因子注册装饰器

    用法示例：

        @register_factor(
            name="rsi_14",
            category="动量",
            description="14 周期 RSI 指标",
            params={"window": 14},
        )
        def rsi_14(open, high, low, close, volume):
            ...

    Args:
        name: 因子名称（唯一标识），None 则使用函数名
        category: 因子分类标签
        description: 因子描述文本
        params: 默认参数字典

    Returns:
        装饰后的函数（不改变原函数行为）
    """

    def decorator(func: FactorFunction) -> FactorFunction:
        factor_name = name if name is not None else func.__name__

        if factor_name in FACTOR_REGISTRY:
            raise ValueError(f"因子 '{factor_name}' 已经注册，请使用不同的名称")

        FACTOR_REGISTRY[factor_name] = {
            "func": func,
            "category": category,
            "description": description or (func.__doc__ or "").strip(),
            "params": params or {},
        }

        return func

    return decorator


# ============================================================================
# 查询工具
# ============================================================================
def get_factor(name: str) -> dict[str, Any]:
    """按名称获取已注册的因子信息"""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"因子 '{name}' 未注册。已注册因子：{list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name]


def list_factors(category: Optional[str] = None) -> list[str]:
    """列出所有已注册的因子名称，可按分类筛选"""
    if category is None:
        return list(FACTOR_REGISTRY.keys())
    return [
        name
        for name, info in FACTOR_REGISTRY.items()
        if info.get("category") == category
    ]


def compute_factor(name: str, open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """按名称计算因子值

    Args:
        name: 因子名称
        open/high/low/close/volume: 价格和成交量序列

    Returns:
        因子值序列
    """
    factor_info = get_factor(name)
    return factor_info["func"](open, high, low, close, volume)
