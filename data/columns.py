"""
数据列定义（零依赖模块）

Bronze/Silver/Gold 三层共用的列名常量和类型映射。
不依赖 pydantic/numpy/pandas，可被所有模块安全导入。
"""

from __future__ import annotations

from typing import Final

# ══════════════════════════════════════════════════════════════════════
# Silver 层标准化列定义
# ══════════════════════════════════════════════════════════════════════

SILVER_COLUMNS: Final[list[str]] = [
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "num_trades",
    "symbol",
    "interval",
]

# Silver 列的类型映射（用于写入后的类型校验）
SILVER_DTYPES: Final[dict[str, str]] = {
    "open_time_utc": "datetime64[ms, UTC]",
    "close_time_utc": "datetime64[ms, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "num_trades": "int64",
    "symbol": "object",
    "interval": "object",
}

# ══════════════════════════════════════════════════════════════════════
# 因子输入必须列
# ══════════════════════════════════════════════════════════════════════

REQUIRED_FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "open", "high", "low", "close", "volume",
)
"""因子输入 DataFrame 必须包含的最小列集"""
