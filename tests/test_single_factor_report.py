from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.run_all_research import run
from visualization.single_factor_report import build_single_factor_report_data, write_report_data


def test_report_contains_required_sections(tmp_path) -> None:
    index = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    factor = pd.Series(np.arange(40, dtype=float), index=index)
    returns = factor.shift(-1) / 1000
    report = build_single_factor_report_data(
        "probe",
        factor,
        returns,
        frequency="1h",
        rolling_window=10,
        lookahead_status="pass",
        close_prices=pd.Series(np.linspace(100, 110, 40), index=index),
        lookback_days=(1,),
        horizons=(1, 3),
        bootstrap_samples=20,
        metadata={"source": "unit-test", "scope": "time_series"},
    )
    assert set(report["metrics"]) == {"ic", "rank_ic", "icir", "turnover"}
    assert report["lookahead_status"] == "pass"
    assert report["ic_decay"]
    assert report["group_returns"]
    assert report["rolling_ic"]
    assert len(report["robustness"]["window_horizon"]) == 2
    assert report["robustness"]["bootstrap_rank_ic"]["n_bootstrap"] == 20
    assert report["provenance"]["source"] == "unit-test"
    path = write_report_data(report, tmp_path / "probe.json")
    assert json.loads(path.read_text(encoding="utf-8"))["factor_name"] == "probe"


def test_research_entry_is_honest_without_data(tmp_path) -> None:
    manifest = run(tmp_path / "data", tmp_path / "reports")
    assert manifest["status"] == "insufficient_data"
    assert manifest["oos_6_months_completed"] is False
    generated = list((tmp_path / "reports" / "single_factor").glob("*.json"))
    assert len(generated) == 33
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["metrics"]["ic"] is None for path in generated
    )
