from __future__ import annotations

import unittest

import pandas as pd

from utils.time_utils import to_utc_timestamp


class TimeUtilsTests(unittest.TestCase):
    def test_numeric_timestamp_requires_unit(self) -> None:
        with self.assertRaises(ValueError):
            to_utc_timestamp(946684800000)

    def test_historical_millisecond_timestamp(self) -> None:
        result = to_utc_timestamp(946684800000, numeric_unit="ms")
        self.assertEqual(result, pd.Timestamp("2000-01-01T00:00:00Z"))

    def test_timezone_is_converted_to_utc(self) -> None:
        result = to_utc_timestamp("2026-01-01T08:00:00+08:00")
        self.assertEqual(result, pd.Timestamp("2026-01-01T00:00:00Z"))


if __name__ == "__main__":
    unittest.main()
