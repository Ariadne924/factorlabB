"""
核心数据 Schema 模块（工程化改造版）

设计原则（参考 crypto-factor-lab）：
  1. Pydantic 只负责 API 入口校验（边界防御），通过 "extra: forbid" 检测交易所 API 变更
  2. 校验通过后立即转为 DataFrame，管道内部只操作 DataFrame
  3. 列名常量和类型定义在 data.columns 模块中（零依赖）
  4. 数据校验逻辑在 data.validator 模块中

所有 datetime 字段必须带 UTC 时区，naive datetime 视为 Bug 并当场拒绝。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from data.columns import SILVER_COLUMNS  # noqa: F401 — 重新导出供外部使用
from data.paths import normalize_path_segment


# ══════════════════════════════════════════════════════════════════════
# K 线 Schema（Bronze 入口校验）
# ══════════════════════════════════════════════════════════════════════
class KlineRaw(BaseModel):
    """单根 K 线的原始数据格式（Binance API）

    本模型用于 API 响应的一级校验，通过后转为 DataFrame 存入 Bronze 层。
    "extra: forbid" 确保交易所新增字段时报错而非静默丢弃。

    Binance GET /api/v3/klines 返回固定 12 元素数组：
    [
      0: open_time (int, ms),
      1: open (str),
      2: high (str),
      3: low (str),
      4: close (str),
      5: volume (str),
      6: close_time (int, ms),
      7: quote_asset_volume (str),
      8: number_of_trades (int),
      9: taker_buy_base_volume (str),
      10: taker_buy_quote_volume (str),
      11: ignore (str),
    ]
    """

    open_time: int = Field(..., ge=0, description="K 线起始时间（Unix 毫秒）")
    open: float = Field(..., gt=0, description="开盘价")
    high: float = Field(..., gt=0, description="最高价")
    low: float = Field(..., gt=0, description="最低价")
    close: float = Field(..., gt=0, description="收盘价")
    volume: float = Field(..., ge=0, description="成交量（基础资产）")
    close_time: int = Field(..., ge=0, description="K 线结束时间（Unix 毫秒）")
    quote_asset_volume: float = Field(..., ge=0, description="成交额（计价资产）")
    number_of_trades: int = Field(..., ge=0, description="成交笔数")
    taker_buy_base_volume: float = Field(..., ge=0, description="主动买入量（基础资产）")
    taker_buy_quote_volume: float = Field(..., ge=0, description="主动买入额（计价资产）")

    @model_validator(mode="after")
    def _validate_kline_relationships(self) -> KlineRaw:
        """校验时间和 OHLC 关系。"""
        if self.close_time < self.open_time:
            raise ValueError("close_time 不能早于 open_time")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high 必须不小于 open、close 和 low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low 必须不大于 open、close 和 high")
        return self

    @classmethod
    def from_binance_row(cls, row: list[Any]) -> KlineRaw:
        """从 Binance API 原始行构建 KlineRaw

        Binance 返回固定 12 元素数组。恰好检测行长度以发现 API 结构变更。

        Raises:
            ValueError: 行长度不是 12
        """
        if len(row) != 12:
            msg = (
                f"Binance kline 预期恰好 12 个字段，实际收到 {len(row)}。"
                f"Binance API 返回值结构可能已变更，请检查。"
            )
            raise ValueError(msg)

        return cls(
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=int(row[6]),
            quote_asset_volume=float(row[7]),
            number_of_trades=int(row[8]),
            taker_buy_base_volume=float(row[9]),
            taker_buy_quote_volume=float(row[10]),
        )

    model_config = {"extra": "forbid"}


# ══════════════════════════════════════════════════════════════════════
# 数据层转换函数（核心管道）
# ══════════════════════════════════════════════════════════════════════

def raw_to_dataframe(
    raw_klines: list[KlineRaw],
    *,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    """将校验后的 KlineRaw 列表转为 Bronze DataFrame

    Args:
        raw_klines: 已通过 Pydantic 校验的原始 K 线列表
        symbol: 交易对（如 'BTCUSDT'）
        interval: K 线周期（如 '1h'）

    Returns:
        Bronze 层 DataFrame，可直接存 parquet
    """
    if not raw_klines:
        raise ValueError("raw_klines 不能为空")

    records = [r.model_dump() for r in raw_klines]
    df = pd.DataFrame(records)
    df["symbol"] = normalize_path_segment(
        symbol, field_name="symbol", case="upper"
    )
    df["interval"] = normalize_path_segment(
        interval, field_name="interval", case="lower"
    )
    return df


def bronze_to_silver(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Bronze → Silver 转换

    1. open_time (ms int) → open_time_utc (datetime64[ms, UTC])
    2. 列重命名为 snake_case 规范名称
    3. 对照 SILVER_COLUMNS 检查，缺失列直接抛错（不静默产生残缺数据）

    Args:
        bronze_df: Bronze 层 DataFrame（来自 raw_to_dataframe）

    Returns:
        Silver 层标准化 DataFrame

    Raises:
        ValueError: 转换后缺少 SILVER_COLUMNS 中的必须列
    """
    df = bronze_df.copy()
    required_bronze_columns = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "number_of_trades",
        "symbol",
        "interval",
    }
    missing_bronze = sorted(required_bronze_columns.difference(df.columns))
    if missing_bronze:
        raise ValueError(f"Bronze K 线缺少必须列: {missing_bronze}")
    if df.empty:
        raise ValueError("Bronze K 线不能为空")

    # 毫秒时间戳 → UTC datetime64（实际精度为 ms）
    df["open_time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # 列重命名
    df = df.rename(
        columns={
            "quote_asset_volume": "quote_volume",
            "taker_buy_base_volume": "taker_buy_volume",
            "taker_buy_quote_volume": "taker_buy_quote_volume",
            "number_of_trades": "num_trades",
        }
    )

    # 对照 SILVER_COLUMNS 检查缺失列
    missing = [c for c in SILVER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Silver 转换后缺少必须列: {missing}。"
            f"Bronze 数据可能不完整，请检查原始数据源。"
        )

    silver_df = df[SILVER_COLUMNS].copy()
    float_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    silver_df[float_columns] = silver_df[float_columns].astype("float64")
    silver_df["num_trades"] = silver_df["num_trades"].astype("int64")
    silver_df["symbol"] = silver_df["symbol"].map(
        lambda value: normalize_path_segment(
            value, field_name="symbol", case="upper"
        )
    )
    silver_df["interval"] = silver_df["interval"].map(
        lambda value: normalize_path_segment(
            value, field_name="interval", case="lower"
        )
    )

    from data.validator import DataValidator

    DataValidator.validate_klines(silver_df)
    return silver_df


# ══════════════════════════════════════════════════════════════════════
# 逐笔成交 Schema（Bronze 入口校验）
# ══════════════════════════════════════════════════════════════════════

class TradeRaw(BaseModel):
    """逐笔成交原始数据格式（API 边界校验）

    各交易所的成交数据字段差异较大，此处定义最小公共字段。
    具体交易所实现可通过 from_xxx 类方法处理字段映射。
    """

    timestamp: datetime = Field(..., description="成交时间（UTC）")
    price: float = Field(..., gt=0, description="成交价格")
    quantity: float = Field(..., gt=0, description="成交数量")
    side: Literal["buy", "sell"] = Field(..., description="成交方向：buy / sell")
    symbol: str = Field(..., min_length=1, description="交易对 ID")

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """确保时间戳带 UTC 时区——naive datetime 直接拒绝"""
        if v.tzinfo is None:
            raise ValueError(
                "TradeRaw.timestamp 必须带时区信息（UTC），不能是 naive datetime"
            )
        if v.tzinfo.utcoffset(v) != UTC.utcoffset(v):
            return v.astimezone(UTC)
        return v

    model_config = {"extra": "forbid"}


# ══════════════════════════════════════════════════════════════════════
# 资金费率 Schema（Bronze 入口校验）
# ══════════════════════════════════════════════════════════════════════

class FundingRateRaw(BaseModel):
    """资金费率原始数据格式（API 边界校验）

    资金费率是永续合约特有的机制，用于锚定现货价格。
    一般为每 8 小时结算一次。
    """

    timestamp: datetime = Field(..., description="资金费率结算时间（UTC）")
    funding_rate: float = Field(..., description="资金费率，如 0.0001 表示 0.01%")
    symbol: str = Field(..., min_length=1, description="交易对 ID")

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """确保时间戳带 UTC 时区——naive datetime 直接拒绝"""
        if v.tzinfo is None:
            raise ValueError(
                "FundingRateRaw.timestamp 必须带时区信息（UTC），不能是 naive datetime"
            )
        if v.tzinfo.utcoffset(v) != UTC.utcoffset(v):
            return v.astimezone(UTC)
        return v

    model_config = {"extra": "forbid"}
