from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from config.constants import KlineInterval
from data.okx_client import OKXClient


class FakeResponse:
    status_code = 200

    def __init__(self, data: list[object]):
        self.data = data

    def json(self) -> dict[str, object]:
        return {"code": "0", "msg": "", "data": self.data}

    def raise_for_status(self) -> None:
        return None


def test_okx_kline_maps_to_common_bronze_schema() -> None:
    session = Mock()
    session.get.return_value = FakeResponse(
        [["1700000000000", "100", "102", "99", "101", "2", "2", "202", "1"]]
    )
    result = OKXClient("BTCUSDT", session=session).fetch_klines(KlineInterval.HOUR_1)
    assert result.loc[0, "symbol"] == "BTCUSDT"
    assert result.loc[0, "quote_asset_volume"] == 202.0
    assert session.get.call_args.kwargs["params"]["instId"] == "BTC-USDT"


def test_okx_trades_depth_and_funding_use_unified_schemas() -> None:
    session = Mock()
    session.get.side_effect = [
        FakeResponse(
            [{"tradeId": "7", "px": "100", "sz": "2", "side": "buy", "ts": "1700000000000"}]
        ),
        FakeResponse(
            [{
                "asks": [["101", "1", "0", "1"]],
                "bids": [["99", "2", "0", "1"]],
                "ts": "1700000000000",
                "seqId": "9",
            }]
        ),
        FakeResponse([{"fundingRate": "0.0001", "fundingTime": "1700000000000"}]),
    ]
    client = OKXClient("BTC-USDT", session=session)
    trades = client.fetch_trades()
    depth = client.fetch_depth()
    funding = client.fetch_funding_rate()
    assert trades.loc[0, "side"] == "buy"
    assert isinstance(trades["timestamp"].dtype, pd.DatetimeTZDtype)
    assert set(depth["side"]) == {"bid", "ask"}
    assert funding.loc[0, "symbol"] == "BTCUSDT"
    assert session.get.call_args.kwargs["params"]["instId"] == "BTC-USDT-SWAP"


def test_okx_retry_recovers_after_server_error() -> None:
    session = Mock()
    failing = FakeResponse([])
    failing.status_code = 500
    recovered = FakeResponse([{"ts": "1700000000000"}])
    session.get.side_effect = [failing, recovered]
    sleeps: list[float] = []
    client = OKXClient(
        "BTCUSDT", session=session, max_retries=1, backoff_base=0.25, sleep=sleeps.append
    )
    assert client.ping() is True
    assert sleeps == [0.25]
