"""Stable schemas shared by backtests and Binance testnet forward validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TradingEnvironment(StrEnum):
    BACKTEST = "backtest"
    TESTNET = "testnet"


class MarketType(StrEnum):
    SPOT = "spot"
    USD_M_FUTURES = "usdm_futures"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    market: MarketType = MarketType.USD_M_FUTURES
    environment: TradingEnvironment = TradingEnvironment.TESTNET
    price: float | None = None
    reduce_only: bool = False

    def validate(self) -> None:
        if not self.client_order_id or len(self.client_order_id) > 36:
            raise ValueError("client_order_id must contain 1..36 characters")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id cannot be empty")
        if not self.symbol.isalnum() or self.symbol != self.symbol.upper():
            raise ValueError("symbol must be an uppercase exchange symbol")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            raise ValueError("limit order requires a positive price")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("market order cannot include a price")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class FillRecord:
    exchange_trade_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    fee: float = 0.0
    fee_asset: str | None = None
    executed_at: str | None = None

    def validate(self) -> None:
        if not self.exchange_trade_id or not self.client_order_id:
            raise ValueError("fill identifiers cannot be empty")
        if min(self.price, self.quantity) <= 0 or self.fee < 0:
            raise ValueError("fill price/quantity must be positive and fee non-negative")


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    market: MarketType
    quantity: float
    entry_price: float | None
    mark_price: float | None
    leverage: float
    unrealized_pnl: float
    observed_at: str

    def validate(self) -> None:
        if not self.symbol or self.leverage < 0:
            raise ValueError("position symbol cannot be empty and leverage cannot be negative")

