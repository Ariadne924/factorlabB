from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from data.binance_client import BinanceClient
from data.schema import OrderBookRaw, TradeRaw


class FakeResponse:
    def __init__(
        self, payload: object, status_code: int = 200, headers: dict[str, str] | None = None
    ):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_trade_schema_and_mock_endpoint() -> None:
    session = Mock()
    session.get.return_value = FakeResponse(
        [{"a": 7, "p": "100.5", "q": "2", "T": 1_700_000_000_000, "m": False}]
    )
    client = BinanceClient("btcusdt", session=session, sleep=lambda _: None)
    result = client.fetch_trades(limit=10)
    assert list(result.columns) == [
        "timestamp",
        "symbol",
        "trade_id",
        "price",
        "quantity",
        "quote_quantity",
        "side",
    ]
    assert result.loc[0, "symbol"] == "BTCUSDT"
    assert result.loc[0, "side"] == "buy"
    assert result.loc[0, "quote_quantity"] == pytest.approx(201.0)
    assert isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)
    assert session.get.call_args.kwargs["params"]["limit"] == 10


def test_depth_schema_and_mock_endpoint() -> None:
    session = Mock()
    session.get.return_value = FakeResponse(
        {"lastUpdateId": 42, "bids": [["99", "2"], ["98", "3"]], "asks": [["101", "1"]]}
    )
    result = BinanceClient("BTCUSDT", session=session, sleep=lambda _: None).fetch_depth(limit=100)
    assert set(result["side"]) == {"bid", "ask"}
    assert result["last_update_id"].unique().tolist() == [42]
    assert len(result) == 3


def test_crossed_order_book_rejected() -> None:
    with pytest.raises(ValueError, match="盘口交叉"):
        OrderBookRaw.from_binance_payload(
            {"lastUpdateId": 1, "bids": [["101", "1"]], "asks": [["100", "1"]]},
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        )


def test_rest_retry_uses_backoff_and_recovers() -> None:
    session = Mock()
    session.get.side_effect = [FakeResponse({}, 500), FakeResponse({})]
    sleeps: list[float] = []
    client = BinanceClient(
        "BTCUSDT", session=session, max_retries=1, backoff_base=0.25, sleep=sleeps.append
    )
    assert client.ping() is True
    assert session.get.call_count == 2
    assert sleeps == [0.25]


def test_trade_schema_rejects_missing_direction() -> None:
    with pytest.raises(ValueError):
        TradeRaw.from_binance_payload(
            {"a": 1, "p": "1", "q": "1", "T": 1_700_000_000_000}, symbol="BTCUSDT"
        )
