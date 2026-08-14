from __future__ import annotations

import json

import pytest

from evaluation.oos_lock import create_oos_lock


def test_oos_lock_requires_six_months_and_records_unopened_state(tmp_path) -> None:
    path = tmp_path / "oos_lock.json"
    result = create_oos_lock(
        path, start="2025-01-01", end="2025-07-01",
        symbols=("btcusdt",), intervals=("1h", "24h"),
    )
    assert result["status"] == "locked_unopened"
    assert result["completed"] is False
    assert json.loads(path.read_text("utf-8"))["intervals"] == ["1d", "1h"]
    assert create_oos_lock(
        path, start="2025-01-01", end="2025-07-01",
        symbols=("BTCUSDT",), intervals=("1h", "24h"),
    )["created_at"] == result["created_at"]


def test_oos_lock_rejects_short_window(tmp_path) -> None:
    with pytest.raises(ValueError, match="180"):
        create_oos_lock(
            tmp_path / "lock.json", start="2025-01-01", end="2025-03-01",
            symbols=("BTCUSDT",), intervals=("1h",),
        )
