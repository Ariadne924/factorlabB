from __future__ import annotations

import unittest

import pandas as pd
from pydantic import ValidationError

from data.schema import KlineRaw, TradeRaw, bronze_to_silver, raw_to_dataframe
from data.silver import silver_to_factor_input


def valid_binance_row() -> list[object]:
    return [
        0,
        "1.0",
        "2.0",
        "0.5",
        "1.5",
        "10.0",
        59_999,
        "15.0",
        3,
        "5.0",
        "7.0",
        "0",
    ]


class SchemaTests(unittest.TestCase):
    def test_binance_row_requires_exact_shape(self) -> None:
        with self.assertRaises(ValueError):
            KlineRaw.from_binance_row(valid_binance_row()[:-1])

    def test_invalid_ohlc_is_rejected(self) -> None:
        row = valid_binance_row()
        row[2] = "0.25"
        with self.assertRaises(ValidationError):
            KlineRaw.from_binance_row(row)

    def test_trade_rejects_naive_datetime(self) -> None:
        with self.assertRaises(ValidationError):
            TradeRaw(
                timestamp="2026-01-01T00:00:00",
                price=1,
                quantity=1,
                side="buy",
                symbol="BTCUSDT",
            )

    def test_silver_contract_and_factor_index(self) -> None:
        raw = KlineRaw.from_binance_row(valid_binance_row())
        bronze = raw_to_dataframe([raw], symbol="btcusdt", interval="1H")
        silver = bronze_to_silver(bronze)
        self.assertEqual(silver.loc[0, "symbol"], "BTCUSDT")
        self.assertEqual(str(silver["open_time_utc"].dt.tz), "UTC")

        factor_input = silver_to_factor_input(silver)
        self.assertIsInstance(factor_input.index, pd.DatetimeIndex)
        self.assertEqual(str(factor_input.index.tz), "UTC")

    def test_missing_time_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            silver_to_factor_input(pd.DataFrame({"open": [1.0]}))


if __name__ == "__main__":
    unittest.main()
