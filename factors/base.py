"""
因子基类与统一签名模块（工程化改造版）

设计变更（v2）：
  1. 因子签名为 DataFrame → Series，不再限制为 5 个 OHLCV 参数
     输入 DataFrame 必须包含 open/high/low/close/volume 列，
     可额外包含 funding_rate、oi、chain_data 等任意列。
     这解决了旧版无法支持资金费率因子、多周期因子的根本问题。

  2. BaseFactor 类支持参数化：__init__ 接收 params，compute 只接收 DataFrame。
     简单因子仍可用函数实现（遵循 FactorFunction 协议即可）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd

from data.columns import REQUIRED_FACTOR_COLUMNS

# ══════════════════════════════════════════════════════════════════════
# 因子函数签名（Protocol）
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class FactorFunction(Protocol):
    """因子函数协议（v2）

    输入：一个 pd.DataFrame，**必须**包含 open/high/low/close/volume 列，
          可选包含 funding_rate、taker_buy_volume、open_interest 等扩展列。
          索引必须是带 UTC 时区的 DatetimeIndex。
    输出：一个 pd.Series（索引对齐，NaN 表示无法计算）

    设计理由：
      旧签名 (open, high, low, close, volume) 限制了只能用 OHLCV 数据，
      资金费率、链上数据、订单簿等额外信息无处可放。
      DataFrame 输入让因子自己决定用哪些列。
    """

    def __call__(self, df: pd.DataFrame) -> pd.Series: ...


# ══════════════════════════════════════════════════════════════════════
# 必须列定义（从 data.columns 导入，避免重复定义）
# ══════════════════════════════════════════════════════════════════════

# 重新导出供外部使用
REQUIRED_COLUMNS = REQUIRED_FACTOR_COLUMNS


def validate_factor_input(df: pd.DataFrame) -> None:
    """校验因子输入 DataFrame 是否满足最小列要求

    Args:
        df: 因子输入 DataFrame

    Raises:
        ValueError: 缺少必须列
    """
    missing = [col for col in REQUIRED_FACTOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"因子输入 DataFrame 缺少必须列: {missing}。"
            f"必须包含: {list(REQUIRED_FACTOR_COLUMNS)}"
        )


# ══════════════════════════════════════════════════════════════════════
# 因子模板（供开发者参考）
# ══════════════════════════════════════════════════════════════════════


def factor_template(df: pd.DataFrame) -> pd.Series:
    """【模板】所有因子必须遵循此签名

    Args:
        df: 必须包含 open/high/low/close/volume 列，
            可包含任意额外列。索引必须带 UTC 时区。

    Returns:
        pd.Series：因子值序列，索引与输入对齐

    示例（均线偏离因子）：
        >>> ma = df["close"].rolling(20).mean()
        >>> return (df["close"] - ma) / ma

    示例（资金费率动量因子，用到额外列）：
        >>> fr = df["funding_rate"]
        >>> return fr.rolling(8).mean() - fr.rolling(24).mean()
    """
    validate_factor_input(df)
    raise NotImplementedError("请实现具体的因子计算逻辑")


# ══════════════════════════════════════════════════════════════════════
# 因子抽象基类（参数化因子）
# ══════════════════════════════════════════════════════════════════════


class BaseFactor(ABC):
    """因子抽象基类（v2 — 支持参数化）

    对于需要内部状态或可变参数的因子，继承此类。
    简单因子可直接用函数实现（遵循 FactorFunction 协议即可）。

    用法示例：
        class RSI(BaseFactor):
            def __init__(self, window: int = 14):
                super().__init__()
                self.window = window

            def compute(self, df: pd.DataFrame) -> pd.Series:
                validate_factor_input(df)
                delta = df["close"].diff()
                gain = delta.clip(lower=0)
                loss = (-delta).clip(lower=0)
                avg_gain = gain.rolling(self.window).mean()
                avg_loss = loss.rolling(self.window).mean()
                rs = avg_gain / avg_loss
                return 100.0 - (100.0 / (1.0 + rs))
    """

    def __init__(self) -> None:
        self._name: str = self.__class__.__name__

    @property
    def name(self) -> str:
        """因子类名"""
        return self._name

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值

        Args:
            df: 因子输入 DataFrame（必须含 OHLCV 列）

        Returns:
            因子值 Series
        """
        raise NotImplementedError
