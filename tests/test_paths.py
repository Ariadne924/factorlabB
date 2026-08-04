from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

from data.bronze import bronze_klines_path, read_bronze_klines, write_bronze_klines
from data.gold import gold_factor_path


class StoragePathTests(unittest.TestCase):
    def test_path_segments_are_normalized(self) -> None:
        path = bronze_klines_path(
            exchange="BINANCE",
            market="SPOT",
            symbol="btcusdt",
            interval="1H",
        )
        self.assertEqual(
            path.parts[-6:],
            ("bronze", "binance", "spot", "BTCUSDT", "1h", "klines.parquet"),
        )

    def test_traversal_segments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gold_factor_path(
                symbol="BTCUSDT",
                interval="1h",
                factor_name="../../escape",
            )
        with self.assertRaises(ValueError):
            bronze_klines_path(
                exchange="../../escape",
                symbol="BTCUSDT",
                interval="1h",
            )

    @unittest.skipUnless(find_spec("pyarrow"), "需要 pyarrow")
    def test_parquet_round_trip(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.parquet"
            write_bronze_klines(frame, path)
            pd.testing.assert_frame_equal(read_bronze_klines(path), frame)


if __name__ == "__main__":
    unittest.main()
