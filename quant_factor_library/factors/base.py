"""
因子基类与统一签名模块

本模块定义所有因子的统一函数签名和抽象基类。
任何新增因子必须遵循此签名约定，确保后续评估框架可以
无差别地消费所有因子。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd


# ============================================================================
# 因子函数统一签名（Protocol 定义）
# ============================================================================
@runtime_checkable
class FactorFunction(Protocol):
    """因子函数协议：所有因子必须遵循此签名

    输入：open、high、low、close、volume 五个价格/成交量序列
    输出：一个与输入等长的 pd.Series，代表因子的时序值
    """

    def __call__(
        self,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series: ...


# ============================================================================
# 因子模板函数（供开发者参考）
# ============================================================================
def factor_template(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """【模板】所有自定义因子必须遵循此签名

    参数说明：
        open:   开盘价序列，索引为 datetime64[ns] (UTC)
        high:   最高价序列
        low:    最低价序列
        close:  收盘价序列
        volume: 成交量序列

    返回值：
        pd.Series：因子值序列，索引与输入对齐，
                   NaN 表示该位置无法计算（如窗口不足）

    示例（均线偏离因子）：
        >>> ma = close.rolling(20).mean()
        >>> return (close - ma) / ma
    """
    # TODO: 替换为实际因子计算逻辑
    raise NotImplementedError("请实现具体的因子计算逻辑")


# ============================================================================
# 因子抽象基类（复杂因子可用此方式封装）
# ============================================================================
class BaseFactor(ABC):
    """因子抽象基类

    对于需要内部状态或复杂参数的因子，可继承此类。
    简单因子可直接用函数实现（遵循 FactorFunction 协议即可）。
    """

    def __init__(self) -> None:
        self._name: str = self.__class__.__name__

    @property
    def name(self) -> str:
        """因子名称"""
        return self._name

    @abstractmethod
    def compute(
        self,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """计算因子值

        子类必须实现此方法，签名与 FactorFunction 协议一致。
        """
        raise NotImplementedError
