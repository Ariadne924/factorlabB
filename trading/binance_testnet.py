"""Safe Binance Spot/USD-M testnet adapter; it never points at production trading URLs."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from trading.ledger import TradingLedger
from trading.models import MarketType, OrderIntent, OrderStatus, OrderType

SPOT_TESTNET_URL = "https://testnet.binance.vision"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class BinanceTestnetConfig:
    market: MarketType = MarketType.USD_M_FUTURES
    api_key: str | None = field(default=None, repr=False)
    api_secret: str | None = field(default=None, repr=False)
    dry_run: bool = True
    recv_window_ms: int = 5_000
    timeout: float = 10.0
    max_retries: int = 3
    backoff_base: float = 0.5

    @classmethod
    def from_env(
        cls,
        *,
        market: MarketType = MarketType.USD_M_FUTURES,
        dry_run: bool = True,
    ) -> BinanceTestnetConfig:
        return cls(
            market=market,
            api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
            api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
            dry_run=dry_run,
        )

    @property
    def base_url(self) -> str:
        return SPOT_TESTNET_URL if self.market == MarketType.SPOT else FUTURES_TESTNET_URL

    def validate(self, *, require_credentials: bool = False) -> None:
        if not 1_000 <= self.recv_window_ms <= 60_000:
            raise ValueError("recv_window_ms must be in 1000..60000")
        if self.timeout <= 0 or self.max_retries < 0 or self.backoff_base < 0:
            raise ValueError("invalid request retry configuration")
        if require_credentials and (not self.api_key or not self.api_secret):
            raise RuntimeError(
                "Binance testnet credentials are missing; set BINANCE_TESTNET_API_KEY "
                "and BINANCE_TESTNET_API_SECRET"
            )


def make_client_order_id(
    strategy_id: str,
    *,
    symbol: str,
    signal_time: str,
    side: str,
) -> str:
    """Create a deterministic key so a restarted signal cannot submit twice."""
    prefix = _SAFE_ID.sub("-", strategy_id).strip("-")[:10] or "strategy"
    payload = f"{strategy_id}|{symbol.upper()}|{signal_time}|{side.upper()}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"cfl-{prefix}-{digest}"[:36]


class BinanceTestnetClient:
    def __init__(
        self,
        config: BinanceTestnetConfig | None = None,
        *,
        session: requests.Session | None = None,
        ledger: TradingLedger | None = None,
        clock_ms: Callable[[], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or BinanceTestnetConfig.from_env()
        self.config.validate()
        self.session = session or requests.Session()
        self.ledger = ledger
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._sleep = sleep
        self._time_offset_ms = 0

    @property
    def _paths(self) -> dict[str, str]:
        if self.config.market == MarketType.SPOT:
            return {"time": "/api/v3/time", "account": "/api/v3/account", "order": "/api/v3/order"}
        return {
            "time": "/fapi/v1/time",
            "account": "/fapi/v2/account",
            "order": "/fapi/v1/order",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        request_params = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            self.config.validate(require_credentials=True)
            request_params["timestamp"] = self._clock_ms() + self._time_offset_ms
            request_params["recvWindow"] = self.config.recv_window_ms
            query = urlencode(request_params, doseq=True)
            assert self.config.api_secret is not None
            request_params["signature"] = hmac.new(
                self.config.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            assert self.config.api_key is not None
            headers["X-MBX-APIKEY"] = self.config.api_key
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    f"{self.config.base_url}{path}",
                    params=request_params,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            except requests.RequestException:
                if attempt >= self.config.max_retries:
                    raise
                self._sleep(self.config.backoff_base * 2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.config.max_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                self._sleep(
                    float(retry_after)
                    if retry_after
                    else self.config.backoff_base * 2**attempt
                )
                continue
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except ValueError:
                    detail = "no JSON error body"
                raise requests.HTTPError(
                    f"Binance testnet {response.status_code} {path}: {detail}",
                    response=response,
                )
            return response.json()
        raise RuntimeError("Binance testnet retries exhausted")

    def sync_time(self) -> int:
        payload = self._request("GET", self._paths["time"])
        self._time_offset_ms = int(payload["serverTime"]) - self._clock_ms()
        return self._time_offset_ms

    def account(self) -> dict[str, Any]:
        payload = self._request("GET", self._paths["account"], signed=True)
        if not isinstance(payload, dict):
            raise TypeError("Binance account response must be an object")
        return payload

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        intent.validate()
        if intent.environment.value != "testnet":
            raise ValueError("BinanceTestnetClient accepts testnet intents only")
        if intent.market != self.config.market:
            raise ValueError("order market does not match client configuration")
        if self.ledger is not None and not self.ledger.create_order(intent):
            existing = self.ledger.get_order(intent.client_order_id)
            return {"status": "IDEMPOTENT_REPLAY", "order": existing}
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "type": intent.order_type.value,
            "quantity": format(intent.quantity, ".16g"),
            "newClientOrderId": intent.client_order_id,
        }
        if intent.order_type == OrderType.LIMIT:
            params.update({"price": format(intent.price, ".16g"), "timeInForce": "GTC"})
        if self.config.market == MarketType.USD_M_FUTURES:
            params["reduceOnly"] = "true" if intent.reduce_only else "false"
        if self.config.dry_run:
            return {
                "status": "DRY_RUN",
                "endpoint": f"{self.config.base_url}{self._paths['order']}",
                "params": params,
            }
        try:
            payload = self._request("POST", self._paths["order"], params=params, signed=True)
        except Exception as exc:
            if self.ledger is not None:
                self.ledger.append_event(
                    "order_submission_uncertain",
                    {"client_order_id": intent.client_order_id, "error": str(exc)},
                )
            raise
        if not isinstance(payload, dict):
            raise TypeError("Binance order response must be an object")
        if self.ledger is not None:
            self.ledger.update_order(
                intent.client_order_id,
                _status(payload.get("status")),
                exchange_order_id=str(payload.get("orderId", "")) or None,
                raw_response=payload,
            )
        return payload

    def query_order(self, client_order_id: str, *, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            self._paths["order"],
            params={"symbol": symbol.upper(), "origClientOrderId": client_order_id},
            signed=True,
        )
        if not isinstance(payload, dict):
            raise TypeError("Binance order response must be an object")
        if self.ledger is not None:
            self.ledger.update_order(
                client_order_id,
                _status(payload.get("status")),
                exchange_order_id=str(payload.get("orderId", "")) or None,
                raw_response=payload,
            )
        return payload

    def cancel_order(self, client_order_id: str, *, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "DELETE",
            self._paths["order"],
            params={"symbol": symbol.upper(), "origClientOrderId": client_order_id},
            signed=True,
        )
        if not isinstance(payload, dict):
            raise TypeError("Binance cancel response must be an object")
        if self.ledger is not None:
            self.ledger.update_order(client_order_id, OrderStatus.CANCELED, raw_response=payload)
        return payload


def _status(value: Any) -> OrderStatus:
    normalized = str(value or "SUBMITTED").upper()
    aliases = {"NEW": OrderStatus.SUBMITTED, "PENDING_NEW": OrderStatus.SUBMITTED}
    if normalized in aliases:
        return aliases[normalized]
    try:
        return OrderStatus(normalized)
    except ValueError:
        return OrderStatus.SUBMITTED
