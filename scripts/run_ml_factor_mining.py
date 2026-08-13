"""使用已有候选因子生成严格 walk-forward 的机器学习复合因子。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import factors  # noqa: E402,F401 — 注册因子
from config.constants import get_interval_ms  # noqa: E402
from data.silver import silver_to_factor_input  # noqa: E402
from data.validator import DataValidator  # noqa: E402
from evaluation.ml_factor_mining import (  # noqa: E402
    WalkForwardConfig,
    aggregate_feature_recommendations,
    backtest_oos_predictions,
    walk_forward_ridge,
)
from evaluation.time_series_strategy import build_ml_strategy_preset  # noqa: E402
from evaluation.robustness import forward_return  # noqa: E402
from factors.registry import compute_factor, get_factor_metadata, list_factors  # noqa: E402
from visualization.single_factor_report import (  # noqa: E402
    build_single_factor_report_data,
    write_report_data,
)


def _parameter_variants(
    factor_name: str, windows: tuple[int, ...]
) -> list[tuple[str, dict[str, int] | None]]:
    """根据 registry 参数形状生成受约束的候选变体。"""
    defaults = get_factor_metadata(factor_name).get("default_params", {})
    variants: list[tuple[str, dict[str, int] | None]] = [(factor_name, None)]
    if "window" in defaults:
        variants.extend(
            (f"{factor_name}__w{window}", {**defaults, "window": window})
            for window in windows
            if window != defaults["window"]
        )
    if {"short_window", "long_window"}.issubset(defaults):
        pairs = zip(windows[:-1], windows[1:], strict=True)
        variants.extend(
            (
                f"{factor_name}__s{short}_l{long}",
                {**defaults, "short_window": short, "long_window": long},
            )
            for short, long in pairs
            if (short, long) != (defaults["short_window"], defaults["long_window"])
        )
    return variants


def build_candidate_matrix(
    frame: pd.DataFrame, *, parameter_windows: tuple[int, ...] = (6, 12, 24, 48)
) -> tuple[pd.DataFrame, dict[str, str]]:
    """计算默认因子及参数变体；数据依赖不满足时显式记录。"""
    if len(parameter_windows) < 2 or any(value < 1 for value in parameter_windows):
        raise ValueError("parameter_windows 至少包含两个正整数")
    if tuple(sorted(set(parameter_windows))) != parameter_windows:
        raise ValueError("parameter_windows 必须严格递增且不得重复")
    columns: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    for factor_name in list_factors():
        for candidate_name, params in _parameter_variants(factor_name, parameter_windows):
            try:
                values = compute_factor(factor_name, frame, params=params)
                if values.notna().any():
                    columns[candidate_name] = values
                else:
                    skipped[candidate_name] = "全部为缺失值"
            except (KeyError, ValueError) as exc:
                skipped[candidate_name] = str(exc)
    return pd.DataFrame(columns, index=frame.index), skipped


def _bars_for_days(interval: str, days: int) -> int:
    bars_per_day = max(1, round(86_400_000 / get_interval_ms(interval)))
    return bars_per_day * days


def run(
    data_dir: Path,
    reports_dir: Path,
    *,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    horizon: int = 1,
    min_train_days: int = 90,
    test_days: int = 14,
    embargo_bars: int = 1,
    alpha: float = 10.0,
    parameter_windows: tuple[int, ...] = (6, 12, 24, 48),
    max_features: int = 80,
    max_pairwise_correlation: float = 0.95,
) -> dict[str, Any]:
    output_dir = reports_dir / "ml_factor"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol_filter = {item.upper() for item in symbols or ()}
    interval_filter = {"1d" if item == "24h" else item for item in intervals or ()}
    results: list[dict[str, Any]] = []

    for silver_path in sorted(data_dir.glob("silver/**/klines.parquet")):
        silver = pd.read_parquet(silver_path)
        DataValidator.validate_klines(silver)
        symbol = str(silver["symbol"].iloc[0])
        interval = str(silver["interval"].iloc[0])
        if symbol_filter and symbol not in symbol_filter:
            continue
        if interval_filter and interval not in interval_filter:
            continue
        frame = silver_to_factor_input(silver)
        candidates, skipped = build_candidate_matrix(
            frame, parameter_windows=parameter_windows
        )
        config = WalkForwardConfig(
            min_train_size=_bars_for_days(interval, min_train_days),
            test_size=_bars_for_days(interval, test_days),
            horizon=horizon,
            embargo=embargo_bars,
            alpha=alpha,
            max_features=max_features,
            max_pairwise_correlation=max_pairwise_correlation,
        )
        target = forward_return(frame["close"], horizon)
        prediction, folds = walk_forward_ridge(candidates, target, config=config)
        recommendations = aggregate_feature_recommendations(folds)
        strategy_preset = build_ml_strategy_preset(recommendations)
        strategy_backtest = backtest_oos_predictions(prediction, target)
        report = build_single_factor_report_data(
            "ml_ridge_composite",
            prediction,
            target,
            frequency=interval,
            lookahead_status="pass_by_construction",
            close_prices=frame["close"],
            metadata={
                "category": "机器学习复合因子",
                "description": "由已有候选因子经 expanding walk-forward Ridge 组合而成。",
                "source": "project",
                "scope": "strict_out_of_sample_time_series",
                "data_dependencies": list(candidates.columns),
                "default_params": {
                    "horizon": horizon,
                    "min_train_days": min_train_days,
                    "test_days": test_days,
                    "embargo_bars": embargo_bars,
                    "alpha": alpha,
                    "parameter_windows": list(parameter_windows),
                    "max_features": max_features,
                    "max_pairwise_correlation": max_pairwise_correlation,
                },
            },
        )
        report.update(
            {
                "symbol": symbol,
                "interval": interval,
                "folds": folds,
                "feature_recommendations": recommendations,
                "strategy_preset": strategy_preset,
                "strategy_backtest": strategy_backtest,
                "candidate_feature_count": len(candidates.columns),
                "skipped_features": skipped,
                "leakage_controls": {
                    "split": "expanding_walk_forward",
                    "shuffle": False,
                    "train_only_standardization": True,
                    "train_only_feature_screening": True,
                    "train_only_correlation_pruning": True,
                    "target_horizon_bars": horizon,
                    "embargo_bars": embargo_bars,
                    "rule": "训练标签的收益终点必须早于测试期开始。",
                },
                "research_note": (
                    "该报告只评估严格样本外的模型预测；仍需跨资产、跨频率和锁定参数验证，"
                    "不得直接描述为已验证 alpha。"
                ),
            }
        )
        output = output_dir / f"{symbol}_{interval}_ml_ridge_composite.json"
        write_report_data(report, output)
        results.append(
            {
                "symbol": symbol,
                "interval": interval,
                "status": report["status"],
                "report": output.relative_to(reports_dir).as_posix(),
                "fold_count": len(folds),
                "candidate_feature_count": len(candidates.columns),
                "metrics": report["metrics"],
            }
        )

    manifest: dict[str, Any] = {
        "contract_version": "1.0",
        "status": "ok" if results else "insufficient_data",
        "model": "ridge",
        "filters": {
            "symbols": sorted({str(row["symbol"]) for row in results}),
            "frequencies": sorted({str(row["interval"]) for row in results}),
        },
        "results": results,
        "oos_6_months_completed": False,
        "note": "机器学习结果必须以 walk-forward OOS 为准，禁止随机切分时间序列。",
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ml_factor_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="运行无泄漏机器学习复合因子研究")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--intervals", nargs="+")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--parameter-windows", nargs="+", type=int, default=[6, 12, 24, 48])
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--max-pairwise-correlation", type=float, default=0.95)
    args = parser.parse_args()
    manifest = run(
        args.data_dir,
        args.reports_dir,
        symbols=tuple(args.symbols) if args.symbols else None,
        intervals=tuple(args.intervals) if args.intervals else None,
        horizon=args.horizon,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        embargo_bars=args.embargo_bars,
        alpha=args.alpha,
        parameter_windows=tuple(args.parameter_windows),
        max_features=args.max_features,
        max_pairwise_correlation=args.max_pairwise_correlation,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
