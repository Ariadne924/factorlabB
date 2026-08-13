"""无泄漏的 walk-forward 机器学习复合因子。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """以 K 线根数表达的时间序列训练配置。"""

    min_train_size: int
    test_size: int
    horizon: int = 1
    embargo: int = 1
    alpha: float = 10.0
    min_feature_coverage: float = 0.8
    max_features: int = 80
    max_pairwise_correlation: float = 0.95

    def validate(self) -> None:
        if self.min_train_size < 20:
            raise ValueError("min_train_size 必须至少为 20")
        if self.test_size < 1 or self.horizon < 1 or self.embargo < 0:
            raise ValueError("test_size/horizon 必须为正，embargo 不得为负")
        if self.alpha < 0:
            raise ValueError("alpha 不得为负")
        if not 0 < self.min_feature_coverage <= 1:
            raise ValueError("min_feature_coverage 必须在 (0, 1] 内")
        if self.max_features < 1:
            raise ValueError("max_features 必须大于 0")
        if not 0 < self.max_pairwise_correlation <= 1:
            raise ValueError("max_pairwise_correlation 必须在 (0, 1] 内")


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float) * alpha
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def _screen_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_features: int,
    max_pairwise_correlation: float,
) -> tuple[list[str], dict[str, float]]:
    """仅使用当前训练折进行监督排序和相关性去冗余。"""
    scores = x_train.corrwith(y_train).abs().replace([np.inf, -np.inf], np.nan).dropna()
    ordered = scores.sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    for name in ordered:
        if len(kept) >= max_features:
            break
        if kept:
            correlations = x_train[kept].corrwith(x_train[name]).abs()
            if bool((correlations >= max_pairwise_correlation).any()):
                continue
        kept.append(name)
    return kept, {name: float(scores[name]) for name in kept}


def walk_forward_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    config: WalkForwardConfig,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """生成严格样本外预测；每折训练标签在测试期开始前已完全实现。"""
    config.validate()
    if not features.index.equals(target.index):
        target = target.reindex(features.index)
    numeric = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    target = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    predictions = pd.Series(np.nan, index=features.index, name="ml_ridge_composite")
    folds: list[dict[str, Any]] = []
    n_rows = len(features)
    first_test = config.min_train_size + config.horizon + config.embargo

    for test_start in range(first_test, n_rows, config.test_size):
        test_end = min(test_start + config.test_size, n_rows)
        train_end = test_start - config.horizon - config.embargo
        train_x = numeric.iloc[:train_end]
        train_y = target.iloc[:train_end]
        valid_target = train_y.notna()
        train_x = train_x.loc[valid_target]
        train_y = train_y.loc[valid_target]
        coverage = train_x.notna().mean()
        columns = coverage[coverage >= config.min_feature_coverage].index.tolist()
        if len(train_y) < config.min_train_size or not columns:
            continue

        selected_train = train_x[columns]
        means = selected_train.mean()
        stds = selected_train.std(ddof=0)
        columns = stds[stds > 0].index.tolist()
        if not columns:
            continue
        means = means[columns]
        stds = stds[columns]
        x_train = selected_train[columns].fillna(means).sub(means).div(stds)
        columns, screening_scores = _screen_features(
            x_train,
            train_y,
            max_features=config.max_features,
            max_pairwise_correlation=config.max_pairwise_correlation,
        )
        if not columns:
            continue
        means = means[columns]
        stds = stds[columns]
        x_train = x_train[columns]
        target_mean = float(train_y.mean())
        weights = _fit_ridge(
            x_train.to_numpy(), train_y.to_numpy() - target_mean, config.alpha
        )
        test_x = numeric.iloc[test_start:test_end][columns]
        x_test = test_x.fillna(means).sub(means).div(stds)
        predictions.iloc[test_start:test_end] = x_test.to_numpy() @ weights + target_mean
        folds.append(
            {
                "train_start": str(train_x.index.min()),
                "train_end": str(train_x.index.max()),
                "test_start": str(test_x.index.min()),
                "test_end": str(test_x.index.max()),
                "n_train": int(len(train_y)),
                "n_test": int(len(test_x)),
                "feature_count": len(columns),
                "intercept": target_mean,
                "weights": {
                    name: float(value) for name, value in zip(columns, weights, strict=True)
                },
                "screening_scores": screening_scores,
            }
        )
    return predictions, folds


def aggregate_feature_recommendations(
    folds: list[dict[str, Any]],
    *,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Summarize train-fold feature selection without inspecting test returns."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    aggregate: dict[str, dict[str, list[float] | int]] = {}
    for fold in folds:
        weights = fold.get("weights", {})
        screening = fold.get("screening_scores", {})
        for name, raw_weight in weights.items():
            bucket = aggregate.setdefault(
                str(name),
                {"weights": [], "screening_scores": [], "selected_folds": 0},
            )
            weight_values = bucket["weights"]
            score_values = bucket["screening_scores"]
            if not isinstance(weight_values, list) or not isinstance(score_values, list):
                raise TypeError("invalid recommendation accumulator")
            weight_values.append(float(raw_weight))
            if name in screening:
                score_values.append(float(screening[name]))
            bucket["selected_folds"] = int(bucket["selected_folds"]) + 1

    total_folds = max(1, len(folds))
    rows: list[dict[str, Any]] = []
    for name, bucket in aggregate.items():
        weights = np.asarray(bucket["weights"], dtype=float)
        screening_scores = np.asarray(bucket["screening_scores"], dtype=float)
        selected_folds = int(bucket["selected_folds"])
        mean_weight = float(weights.mean())
        sign_consistency = float(abs(np.sign(weights).mean()))
        selection_frequency = selected_folds / total_folds
        rows.append(
            {
                "feature": name,
                "direction": 1 if mean_weight >= 0 else -1,
                "mean_weight": mean_weight,
                "mean_absolute_weight": float(np.abs(weights).mean()),
                "selection_frequency": float(selection_frequency),
                "sign_consistency": sign_consistency,
                "mean_train_screening_score": (
                    float(screening_scores.mean()) if len(screening_scores) else None
                ),
                "selected_folds": selected_folds,
                "total_folds": len(folds),
                "recommendation_score": float(
                    selection_frequency * sign_consistency * np.abs(weights).mean()
                ),
                "status": "ml_train_fold_candidate",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["recommendation_score"]),
            float(row["selection_frequency"]),
        ),
        reverse=True,
    )[:top_n]


def backtest_oos_predictions(
    predictions: pd.Series,
    target: pd.Series,
    *,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
) -> dict[str, Any]:
    """Turn strict OOS predicted-return signs into a transparent costed strategy."""
    if fee_rate < 0 or slippage < 0:
        raise ValueError("cost assumptions cannot be negative")
    aligned = pd.concat(
        [predictions.rename("prediction"), target.rename("forward_return")], axis=1
    ).dropna()
    if aligned.empty:
        return {
            "status": "insufficient_data",
            "metrics": {},
            "returns": [],
        }
    position = np.sign(aligned["prediction"]).astype(float)
    turnover = position.diff().abs().div(2.0)
    turnover.iloc[0] = abs(position.iloc[0]) / 2.0
    gross = position * aligned["forward_return"]
    net = gross - turnover * (fee_rate + slippage)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    standard_deviation = net.std(ddof=1)
    bar_sharpe = net.mean() / standard_deviation if standard_deviation > 0 else np.nan
    return {
        "status": "computed_strict_oos",
        "metrics": {
            "n_periods": int(len(net)),
            "total_return": float(equity.iloc[-1] - 1.0),
            "mean_bar_return": float(net.mean()),
            "bar_sharpe": float(bar_sharpe) if np.isfinite(bar_sharpe) else None,
            "max_drawdown": float(drawdown.min()),
            "mean_turnover": float(turnover.mean()),
            "directional_accuracy": float(
                (np.sign(aligned["prediction"]) == np.sign(aligned["forward_return"])).mean()
            ),
        },
        "cost_assumptions": {
            "fee_rate": fee_rate,
            "slippage": slippage,
            "units": "one-way",
        },
        "returns": [
            {
                "time": str(index),
                "prediction": float(aligned.loc[index, "prediction"]),
                "position": float(position.loc[index]),
                "gross_return": float(gross.loc[index]),
                "net_return": float(net.loc[index]),
                "turnover": float(turnover.loc[index]),
                "equity": float(equity.loc[index]),
                "drawdown": float(drawdown.loc[index]),
            }
            for index in aligned.index
        ],
        "lookahead_status": "pass_by_walk_forward_construction",
        "research_note": (
            "Positions use only strict walk-forward OOS predictions. This is a simple "
            "sign strategy, not proof of deployable alpha or execution quality."
        ),
    }
