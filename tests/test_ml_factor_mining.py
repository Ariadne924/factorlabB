from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.ml_factor_mining import (
    WalkForwardConfig,
    aggregate_feature_recommendations,
    backtest_oos_predictions,
    walk_forward_models,
    walk_forward_ridge,
)
from scripts.run_ml_factor_mining import build_candidate_matrix, run


def test_walk_forward_predictions_are_oos_and_future_target_invariant() -> None:
    index = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    feature = pd.Series(np.linspace(-1, 1, len(index)), index=index)
    features = pd.DataFrame({"signal": feature, "noise": np.sin(np.arange(len(index)))})
    target = feature.shift(1).fillna(0) * 0.01
    config = WalkForwardConfig(
        min_train_size=20, test_size=10, horizon=1, embargo=1, alpha=1.0
    )

    prediction, folds = walk_forward_ridge(features, target, config=config)
    changed_target = target.copy()
    changed_target.iloc[40:] = 999.0
    changed_prediction, _ = walk_forward_ridge(features, changed_target, config=config)

    assert prediction.iloc[:22].isna().all()
    assert prediction.iloc[22:32].notna().all()
    pd.testing.assert_series_equal(
        prediction.iloc[22:42], changed_prediction.iloc[22:42], check_names=False
    )
    assert folds[0]["train_end"] < folds[0]["test_start"]
    assert folds[0]["feature_count"] == 2


def test_multi_model_walk_forward_is_oos_and_uses_fixed_ensemble() -> None:
    index = pd.date_range("2026-01-01", periods=70, freq="h", tz="UTC")
    feature = pd.Series(np.sin(np.arange(len(index)) / 5), index=index)
    features = pd.DataFrame({"signal": feature, "trend": np.linspace(-1, 1, len(index))})
    target = feature.shift(1).fillna(0) * 0.01
    config = WalkForwardConfig(min_train_size=20, test_size=10, horizon=1, embargo=1)

    predictions, folds = walk_forward_models(features, target, config=config)

    assert set(predictions) == {
        "ridge",
        "elastic_net",
        "robust_ridge",
        "tree_stumps",
        "ensemble",
    }
    expected = pd.concat(
        [
            predictions["ridge"],
            predictions["elastic_net"],
            predictions["robust_ridge"],
            predictions["tree_stumps"],
        ],
        axis=1,
    ).mean(axis=1)
    pd.testing.assert_series_equal(
        predictions["ensemble"].dropna(), expected.dropna(), check_names=False
    )
    assert folds[0]["ensemble_rule"] == "equal_weight_no_test_period_model_selection"
    assert set(folds[0]["model_weights"]) == {
        "ridge",
        "elastic_net",
        "robust_ridge",
        "tree_stumps",
    }


def test_candidate_matrix_generates_controlled_parameter_variants() -> None:
    index = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.linspace(1000, 1200, len(index)),
        },
        index=index,
    )
    candidates, skipped = build_candidate_matrix(
        frame, parameter_windows=(6, 12, 24, 48)
    )
    assert "return_momentum" in candidates
    assert "return_momentum__w6" in candidates
    assert "volatility_term_ratio__s12_l24" in candidates
    assert "funding_rate_zscore" in skipped

    with pytest.raises(ValueError, match="严格递增"):
        build_candidate_matrix(frame, parameter_windows=(24, 12))


def test_ml_recommendations_and_oos_sign_backtest() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    predictions = pd.Series([np.nan, 0.1, -0.2, 0.3, -0.1, 0.2], index=index)
    target = pd.Series([0.0, 0.02, -0.01, -0.02, -0.03, 0.01], index=index)
    folds = [
        {
            "weights": {"momentum": 0.4, "volume": -0.2},
            "screening_scores": {"momentum": 0.2, "volume": 0.1},
        },
        {
            "weights": {"momentum": 0.3},
            "screening_scores": {"momentum": 0.25},
        },
    ]
    recommendations = aggregate_feature_recommendations(folds, top_n=2)
    assert recommendations[0]["feature"] == "momentum"
    assert recommendations[0]["selection_frequency"] == 1.0
    assert recommendations[0]["direction"] == 1

    result = backtest_oos_predictions(predictions, target, fee_rate=0, slippage=0)
    assert result["status"] == "computed_strict_oos"
    assert result["metrics"]["n_periods"] == 5
    assert result["metrics"]["directional_accuracy"] == 0.8
    assert result["lookahead_status"] == "pass_by_walk_forward_construction"


def test_ml_manifest_is_versioned_and_honest_without_data(tmp_path) -> None:
    manifest = run(tmp_path / "data", tmp_path / "reports")
    assert manifest["contract_version"] == "2.0"
    assert manifest["status"] == "insufficient_data"
    assert manifest["oos_6_months_completed"] is False
    assert manifest["results"] == []


def test_ml_scope_filter_does_not_expand_to_symbol_interval_cartesian_product(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "data" / "silver" / "first" / "klines.parquet"
    second = tmp_path / "data" / "silver" / "second" / "klines.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    def fake_read(path):
        symbol, interval = (
            ("BTCUSDT", "6h") if "first" in str(path) else ("ETHUSDT", "1h")
        )
        return pd.DataFrame({"symbol": [symbol], "interval": [interval]})

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    monkeypatch.setattr(
        "scripts.run_ml_factor_mining.DataValidator.validate_klines", lambda _frame: None
    )
    monkeypatch.setattr(
        "scripts.run_ml_factor_mining.silver_to_factor_input",
        lambda _frame: pytest.fail("non-ready cross scope must not be trained"),
    )

    manifest = run(
        tmp_path / "data",
        tmp_path / "reports",
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1h", "6h"),
        scopes=(("BTCUSDT", "1h"), ("ETHUSDT", "6h")),
    )

    assert manifest["status"] == "insufficient_data"
