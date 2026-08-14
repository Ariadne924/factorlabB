from __future__ import annotations

import hashlib
import hmac
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest
import requests

from trading.binance_testnet import (
    FUTURES_TESTNET_URL,
    BinanceTestnetClient,
    BinanceTestnetConfig,
    make_client_order_id,
)
from trading.ledger import TradingLedger
from trading.models import OrderIntent, OrderSide, OrderType


def response(status: int, payload: dict[str, object]) -> Mock:
    result = Mock(status_code=status, headers={})
    result.json.return_value = payload
    result.raise_for_status.side_effect = requests.HTTPError(str(status)) if status >= 400 else None
    return result


def order() -> OrderIntent:
    return OrderIntent(
        client_order_id="cfl-trend-123",
        strategy_id="trend-v1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001,
    )


def test_client_order_id_is_deterministic_and_bounded() -> None:
    first = make_client_order_id(
        "trend / very long strategy", symbol="BTCUSDT", signal_time="2026-08-14T00:00Z", side="BUY"
    )
    second = make_client_order_id(
        "trend / very long strategy", symbol="BTCUSDT", signal_time="2026-08-14T00:00Z", side="BUY"
    )
    assert first == second
    assert len(first) <= 36


def test_dry_run_never_calls_network_and_ledger_blocks_duplicate(tmp_path) -> None:
    session = Mock()
    ledger = TradingLedger(tmp_path / "trading.sqlite3")
    client = BinanceTestnetClient(
        BinanceTestnetConfig(dry_run=True), session=session, ledger=ledger
    )
    first = client.place_order(order())
    second = client.place_order(order())
    assert first["status"] == "DRY_RUN"
    assert first["endpoint"].startswith(FUTURES_TESTNET_URL)
    assert second["status"] == "IDEMPOTENT_REPLAY"
    session.request.assert_not_called()


def test_signed_testnet_order_uses_time_offset_and_expected_signature(tmp_path) -> None:
    session = Mock()
    session.request.side_effect = [
        response(200, {"serverTime": 2_000}),
        response(200, {"orderId": 99, "status": "NEW"}),
    ]
    config = BinanceTestnetConfig(
        api_key="test-key", api_secret="test-secret", dry_run=False, max_retries=0
    )
    ledger = TradingLedger(tmp_path / "trading.sqlite3")
    client = BinanceTestnetClient(
        config, session=session, ledger=ledger, clock_ms=lambda: 1_000
    )
    assert client.sync_time() == 1_000
    payload = client.place_order(order())
    assert payload["orderId"] == 99
    call = session.request.call_args_list[1]
    params = call.kwargs["params"]
    unsigned = {key: value for key, value in params.items() if key != "signature"}
    expected = hmac.new(
        b"test-secret", urlencode(unsigned).encode(), hashlib.sha256
    ).hexdigest()
    assert params["signature"] == expected
    assert params["timestamp"] == 2_000
    assert call.kwargs["headers"] == {"X-MBX-APIKEY": "test-key"}
    assert ledger.get_order(order().client_order_id)["status"] == "SUBMITTED"  # type: ignore[index]


def test_live_testnet_mode_requires_credentials() -> None:
    client = BinanceTestnetClient(BinanceTestnetConfig(dry_run=False, max_retries=0))
    with pytest.raises(RuntimeError, match="credentials"):
        client.place_order(order())


def test_client_configuration_repr_never_leaks_secret() -> None:
    config = BinanceTestnetConfig(api_key="public", api_secret="super-secret")
    assert "public" not in repr(config)
    assert "super-secret" not in repr(config)
