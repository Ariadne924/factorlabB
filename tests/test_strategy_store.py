from __future__ import annotations

import json

import pytest

from visualization.strategy_store import (
    compare_strategy_snapshots,
    load_strategy_snapshots,
    save_strategy_snapshot,
)


def test_strategy_store_saves_loads_and_compares_without_path_injection(tmp_path) -> None:
    result = {
        "metrics": {
            "total_return": 0.12,
            "gross_total_return": 0.15,
            "bar_sharpe": 0.2,
            "max_drawdown": -0.1,
            "mean_turnover": 0.03,
            "hit_rate": 0.55,
        },
        "lookahead_status": "pass_by_construction",
        "returns": [],
    }
    output = save_strategy_snapshot(
        tmp_path,
        name="BTC momentum / baseline",
        symbol="btcusdt",
        interval="1h",
        result=result,
        robustness={"stability_summary": {"positive_month_ratio": 0.75}},
    )
    assert output.parent == tmp_path
    snapshots = load_strategy_snapshots(tmp_path)
    assert len(snapshots) == 1
    comparison = compare_strategy_snapshots(snapshots)
    assert comparison[0]["symbol"] == "BTCUSDT"
    assert comparison[0]["positive_month_ratio"] == 0.75


def test_strategy_store_rejects_bad_names_and_ignores_bad_json(tmp_path) -> None:
    with pytest.raises(ValueError, match="1..80"):
        save_strategy_snapshot(
            tmp_path,
            name="",
            symbol="BTCUSDT",
            interval="1h",
            result={},
        )
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps({"version": "other"}), encoding="utf-8")
    assert load_strategy_snapshots(tmp_path) == []
