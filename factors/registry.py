"""
因子注册机制（v2 — 参数化因子工厂）

核心变更：
  旧版：FACTOR_REGISTRY[name] = {"func": f, ...}  ← 注册裸函数
  新版：FACTOR_REGISTRY[name] = {"factory": f, ...} ← 注册工厂函数

工厂函数的签名：
  (params: dict | None) -> FactorFunction

这解决了 RSI_7、RSI_14、RSI_21 必须作为三个独立因子注册的问题。
现在只需注册一个 "RSI" 因子族，使用时传不同 params。

用法示例：

  @register_factor(
      name="RSI",
      category="动量",
      description="相对强弱指标",
      default_params={"window": 14},
  )
  def make_rsi(params: dict[str, Any] | None = None) -> FactorFunction:
      window = (params or {}).get("window", 14)
      def rsi(df: pd.DataFrame) -> pd.Series:
          delta = df["close"].diff()
          gain = delta.clip(lower=0)
          loss = (-delta).clip(lower=0)
          avg_gain = gain.rolling(window).mean()
          avg_loss = loss.rolling(window).mean()
          rs = avg_gain / avg_loss
          return 100.0 - (100.0 / (1.0 + rs))
      return rsi

  # 计算时：
  factor_func = build_factor("RSI", params={"window": 7})
  result = factor_func(df)
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pandas as pd

from factors.base import FactorFunction, validate_factor_input

# ══════════════════════════════════════════════════════════════════════
# 类型别名
# ══════════════════════════════════════════════════════════════════════

# 工厂函数：接收参数字典，返回一个可调用的因子函数
FactorFactory = Callable[[dict[str, Any] | None], FactorFunction]

# ══════════════════════════════════════════════════════════════════════
# 全局因子注册表
# ══════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY: dict[str, dict[str, Any]] = {}
"""
因子注册表（v2 结构）：

{
    "RSI": {
        "factory": make_rsi,        # 工厂函数: (params) -> FactorFunction
        "category": "动量",
        "description": "相对强弱指标",
        "default_params": {"window": 14},
    },
    ...
}
"""

# ══════════════════════════════════════════════════════════════════════
# 注册装饰器
# ══════════════════════════════════════════════════════════════════════


def register_factor(
    name: str | None = None,
    *,
    category: str = "未分类",
    description: str = "",
    default_params: dict[str, Any] | None = None,
) -> Callable[[FactorFactory], FactorFactory]:
    """因子注册装饰器（v2 — 注册工厂函数）

    装饰的是一个**工厂函数**，签名: (params: dict | None) -> FactorFunction
    不是装饰因子计算函数本身。

    Args:
        name: 因子族名称（唯一标识），None 则使用工厂函数名
        category: 因子分类标签
        description: 因子描述文本
        default_params: 默认参数，build_factor 不传 params 时使用

    Returns:
        装饰器

    Raises:
        ValueError: 因子名已存在
    """

    def decorator(factory: FactorFactory) -> FactorFactory:
        factor_name = name if name is not None else factory.__name__

        if factor_name in FACTOR_REGISTRY:
            raise ValueError(f"因子 '{factor_name}' 已经注册，请使用不同的名称")

        FACTOR_REGISTRY[factor_name] = {
            "factory": factory,
            "category": category,
            "description": description or (factory.__doc__ or "").strip(),
            "default_params": deepcopy(default_params) if default_params is not None else {},
        }

        return factory

    return decorator


# ══════════════════════════════════════════════════════════════════════
# 查询与构建
# ══════════════════════════════════════════════════════════════════════


def get_factor_metadata(name: str) -> dict[str, Any]:
    """获取因子族的元信息（不执行计算）"""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"因子 '{name}' 未注册。已注册因子: {list(FACTOR_REGISTRY.keys())}")
    return deepcopy(FACTOR_REGISTRY[name])


def build_factor(name: str, params: dict[str, Any] | None = None) -> FactorFunction:
    """构建因子计算函数

    从注册表中找到因子族的工厂函数，传入参数，返回一个可调用的因子函数。

    Args:
        name: 因子族名称（如 "RSI"）
        params: 参数字典（如 {"window": 7}），None 则使用 default_params

    Returns:
        因子计算函数: (df: pd.DataFrame) -> pd.Series

    用法:
        >>> rsi_7 = build_factor("RSI", params={"window": 7})
        >>> result = rsi_7(df)
    """
    meta = FACTOR_REGISTRY.get(name)
    if meta is None:
        raise KeyError(f"因子 '{name}' 未注册。已注册因子: {list(FACTOR_REGISTRY.keys())}")
    factory: FactorFactory = meta["factory"]
    effective_params = deepcopy(
        params if params is not None else meta.get("default_params", {})
    )
    return factory(effective_params)


def compute_factor(
    name: str,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """一步式计算因子值（构建 + 计算）

    Args:
        name: 因子族名称
        df: 因子输入 DataFrame（必须含 OHLCV 列）
        params: 参数字典，None 则使用 default_params

    Returns:
        因子值 Series
    """
    validate_factor_input(df)
    factor_func = build_factor(name, params)
    result = factor_func(df)
    if not isinstance(result, pd.Series):
        raise TypeError(f"因子 '{name}' 必须返回 pd.Series")
    if not result.index.equals(df.index):
        raise ValueError(f"因子 '{name}' 的结果索引必须与输入 DataFrame 对齐")
    return result


def list_factors(category: str | None = None) -> list[str]:
    """列出所有已注册的因子族名称，可按分类筛选"""
    if category is None:
        return list(FACTOR_REGISTRY.keys())
    return [
        name
        for name, meta in FACTOR_REGISTRY.items()
        if meta.get("category") == category
    ]
