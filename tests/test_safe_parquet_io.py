from __future__ import annotations

import pandas as pd

from utils.io_utils import merge_parquet_safe, write_parquet_safe


def test_atomic_parquet_write_and_merge_prevent_lost_rows(tmp_path) -> None:
    path = tmp_path / "dataset.parquet"
    write_parquet_safe(pd.DataFrame({"time": [1, 2], "value": [10, 20]}), path)
    merged = merge_parquet_safe(
        path,
        pd.DataFrame({"time": [2, 3], "value": [200, 30]}),
        key="time",
    )
    assert merged.to_dict("records") == [
        {"time": 1, "value": 10},
        {"time": 2, "value": 200},
        {"time": 3, "value": 30},
    ]
    pd.testing.assert_frame_equal(pd.read_parquet(path), merged)
