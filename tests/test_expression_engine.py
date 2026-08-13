from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.forward_check import ForwardCheck
from factors.expression_engine import (
    ExpressionSyntaxError,
    ExpressionValidationError,
    compile_expression,
    compile_panel_expression,
    register_expression_factor,
)
from factors.panel import build_panel
from factors.registry import FACTOR_REGISTRY, compute_factor, get_factor_metadata


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10.0, 30.0, len(index)),
        },
        index=index,
    )


def test_compile_normalizes_hashes_and_dependencies() -> None:
    left = compile_expression("delta(close, 1) / delay(close, 1)")
    right = compile_expression("delta( close,1)/delay(close,1)")

    assert left.canonical == "(delta(close,1)/delay(close,1))"
    assert left.digest == right.digest
    assert left.dependencies == ("close",)
    assert left.complexity > 0


def test_evaluate_past_only_formula_and_protected_division(frame: pd.DataFrame) -> None:
    compiled = compile_expression(
        "zscore(returns(close, 1), 12) + protected_div(delta(volume, 2), volume)"
    )
    result = compiled.evaluate(frame)

    assert result.index.equals(frame.index)
    assert np.isfinite(result.dropna()).all()
    assert ForwardCheck.check_truncation_invariance(compiled.evaluate, frame)


@pytest.mark.parametrize(
    "expression",
    [
        "delay(close, -1)",
        "close.shift(-1)",
        "__import__('os').system('echo unsafe')",
        "[value for value in close]",
        "rolling_mean(close, 2.5)",
        "rolling_mean(close, 100001)",
        "close ** 1000",
        "signed_power(close, 1000)",
        "unknown_function(close)",
    ],
)
def test_rejects_future_or_unsafe_syntax(expression: str) -> None:
    with pytest.raises((ExpressionSyntaxError, ExpressionValidationError)):
        compile_expression(expression)


def test_missing_dependency_is_explicit(frame: pd.DataFrame) -> None:
    compiled = compile_expression("rolling_mean(funding_rate, 8)")
    with pytest.raises(ExpressionValidationError, match="funding_rate"):
        compiled.evaluate(frame)


def test_controlled_condition_functions(frame: pd.DataFrame) -> None:
    compiled = compile_expression(
        "where(logical_and(gt(close,open),ge(volume,10)),delta(close,1),-1)"
    )
    result = compiled.evaluate(frame)
    assert result.index.equals(frame.index)
    assert result.iloc[0] == -1
    assert np.isfinite(result).all()


def test_rolling_operators_have_expected_last_values(frame: pd.DataFrame) -> None:
    expression = compile_expression(
        "rolling_rank(close, 5) + argmax(high, 5) + argmin(low, 5)"
    )
    result = expression.evaluate(frame)
    assert result.iloc[-1] == pytest.approx(1.0 + 5.0 + 1.0)


def test_expression_factor_integrates_with_registry(frame: pd.DataFrame) -> None:
    name = "__test_expression_factor__"
    FACTOR_REGISTRY.pop(name, None)
    try:
        compiled = register_expression_factor(
            name,
            "decay_linear(returns(close, 1), 5)",
            description="test formula",
        )
        result = compute_factor(name, frame)
        metadata = get_factor_metadata(name)

        assert result.index.equals(frame.index)
        assert metadata["expression_hash"] == compiled.digest
        assert metadata["canonical_expression"] == compiled.canonical
        assert metadata["lookahead_policy"] == "past_only_static_validation"
        assert metadata["data_dependencies"] == ["close"]
    finally:
        FACTOR_REGISTRY.pop(name, None)


def test_panel_expression_separates_symbol_history_and_cross_section(
    frame: pd.DataFrame,
) -> None:
    eth = frame.copy()
    eth["close"] = eth["close"] * 1.5
    eth["volume"] = eth["volume"].iloc[::-1].to_numpy()
    panel = build_panel({"BTCUSDT": frame, "ETHUSDT": eth})
    compiled = compile_panel_expression(
        "cs_rank(returns(close, 1)) + cs_zscore(rolling_mean(volume, 4))"
    )
    result = compiled.evaluate(panel)

    assert result.index.equals(panel.index)
    assert compiled.metadata()["scope"] == "cross_sectional"
    assert set(compiled.dependencies) == {"close", "volume"}
    assert np.isfinite(result.dropna()).all()


def test_single_symbol_compiler_rejects_cross_sectional_operator() -> None:
    with pytest.raises(ExpressionValidationError, match="unsupported function"):
        compile_expression("cs_rank(returns(close, 1))")


def test_panel_expression_is_invariant_to_later_cross_section(frame: pd.DataFrame) -> None:
    panel = build_panel({"BTCUSDT": frame, "ETHUSDT": frame * 1.1})
    changed = panel.copy()
    final_timestamp = changed.index.get_level_values("timestamp").max()
    changed.loc[final_timestamp, "close"] *= 100
    formula = compile_panel_expression("cs_rank(returns(close, 2))")
    historical = panel.index.get_level_values("timestamp") < final_timestamp

    pd.testing.assert_series_equal(
        formula.evaluate(panel).loc[historical],
        formula.evaluate(changed).loc[historical],
    )
