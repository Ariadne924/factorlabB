"""Signal-threshold surfaces with plateau and market-regime diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.backtest import BacktestConfig, run_vectorized_backtest


def threshold_positions(signal: pd.Series, threshold: float) -> pd.Series:
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    numeric = pd.to_numeric(signal, errors="coerce")
    return pd.Series(
        np.select([numeric > threshold, numeric < -threshold], [1.0, -1.0]),
        index=signal.index,
        name="target_position",
    ).where(numeric.notna(), 0.0)


def _regimes(close: pd.Series, window: int) -> pd.Series:
    if window < 5:
        raise ValueError("regime_window must be at least 5")
    trend = close.pct_change(window, fill_method=None)
    volatility = close.pct_change(fill_method=None).rolling(window).std(ddof=0)
    band = volatility * np.sqrt(window) * 0.5
    return pd.Series(
        np.select([trend > band, trend < -band], ["up", "down"], default="sideways"),
        index=close.index,
    ).where(trend.notna(), "warmup")


def _plateau(rows: list[dict[str, Any]], *, tolerance_ratio: float) -> dict[str, Any]:
    scores = np.array([float(row["robustness_score"]) for row in rows])
    best_index = int(np.nanargmax(scores))
    best_score = float(scores[best_index])
    tolerance = max(abs(best_score) * tolerance_ratio, 0.001)
    accepted = scores >= best_score - tolerance
    left = best_index
    right = best_index
    while left > 0 and accepted[left - 1]:
        left -= 1
    while right + 1 < len(rows) and accepted[right + 1]:
        right += 1
    return {
        "recommended_threshold": rows[best_index]["threshold"],
        "stable_min": rows[left]["threshold"],
        "stable_max": rows[right]["threshold"],
        "grid_points": right - left + 1,
        "plateau_status": "stable_plateau" if right - left + 1 >= 2 else "isolated_peak",
        "best_score": best_score,
        "selection_policy": (
            "Prefer the contiguous near-best plateau, not an isolated maximum; "
            "score = total_return - 0.5*abs(max_drawdown) - 0.1*mean_turnover."
        ),
        "warning": (
            None
            if right - left + 1 >= 2
            else (
                "Only one grid point is near-best; treat it as possible overfitting, "
                "not a recommendation."
            )
        ),
    }


def run_threshold_study(
    frame: pd.DataFrame,
    signal: pd.Series,
    thresholds: list[float],
    *,
    config: BacktestConfig | None = None,
    regime_window: int = 72,
    plateau_tolerance_ratio: float = 0.10,
) -> dict[str, Any]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    ordered = sorted(set(float(value) for value in thresholds))
    if ordered[0] < 0 or plateau_tolerance_ratio <= 0:
        raise ValueError("thresholds must be non-negative and plateau tolerance positive")
    selected = config or BacktestConfig()
    if selected.risk_limits is not None:
        raise ValueError("threshold surface uses the pre-risk consistency engine")
    aligned_signal = signal.reindex(frame.index)
    regimes = _regimes(pd.to_numeric(frame["close"], errors="coerce"), regime_window)
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for threshold in ordered:
        target = threshold_positions(aligned_signal, threshold)
        result = run_vectorized_backtest(frame, target, selected)
        metrics = result["metrics"]
        score = (
            float(metrics["total_return"])
            - 0.5 * abs(float(metrics["max_drawdown"]))
            - 0.1 * float(metrics["mean_turnover"])
        )
        rows.append({"threshold": threshold, **metrics, "robustness_score": score})
        returns = pd.DataFrame(result["returns"])
        returns.index = frame.index
        returns["regime"] = regimes
        for regime in ("up", "down", "sideways"):
            sample = returns.loc[returns["regime"] == regime, "net_return"]
            regime_rows.append(
                {
                    "threshold": threshold,
                    "regime": regime,
                    "n_periods": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0) if len(sample) else None,
                    "hit_rate": float(sample.gt(0).mean()) if len(sample) else None,
                }
            )
    stable = _plateau(rows, tolerance_ratio=plateau_tolerance_ratio)
    regime_recommendations: dict[str, float | None] = {}
    for regime in ("up", "down", "sideways"):
        candidates = [
            row
            for row in regime_rows
            if row["regime"] == regime and row["total_return"] is not None
        ]
        regime_recommendations[regime] = (
            float(max(candidates, key=lambda row: float(row["total_return"]))["threshold"])
            if candidates
            else None
        )
    return {
        "status": "computed_short_sample",
        "threshold_surface": rows,
        "stable_interval": stable,
        "regime_surface": regime_rows,
        "regime_recommendations": regime_recommendations,
        "regime_drift": len(
            {value for value in regime_recommendations.values() if value is not None}
        )
        > 1,
        "lookahead_status": "pass_by_construction",
        "research_note": (
            "Current-sample threshold diagnostics. The plateau is a development candidate, "
            "not a validated optimum; locked OOS must remain unopened until final evaluation."
        ),
    }
