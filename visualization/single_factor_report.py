"""单因子报告数据构建与轻量 HTML/JSON 输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.cost_sensitivity import compute_breakeven_cost, cost_adjusted_return
from evaluation.grouping_test import grouping_backtest
from evaluation.ic_analysis import compute_ic, compute_rank_ic, rolling_ic
from evaluation.robustness import (
    block_bootstrap_ic,
    group_monotonicity,
    sign_consistency,
    window_horizon_robustness,
)
from evaluation.stability import compute_ic_decay, compute_turnover


def _number(value: Any) -> float | None:
    return None if pd.isna(value) or not np.isfinite(value) else float(value)


def _provenance(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        key: metadata.get(key)
        for key in (
            "category",
            "description",
            "source",
            "source_url",
            "scope",
            "data_dependencies",
            "default_params",
        )
    }


def insufficient_report(
    factor_name: str, reason: str, *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """构造无数据报告；所有实证指标保持为空。"""
    return {
        "factor_name": factor_name,
        "status": "insufficient_data",
        "reason": reason,
        "sample": {"start": None, "end": None, "frequency": None, "n_obs": 0},
        "metrics": {"ic": None, "rank_ic": None, "icir": None, "turnover": None},
        "provenance": _provenance(metadata),
        "robustness": {
            "window_horizon": [],
            "bootstrap_rank_ic": {},
            "sign_consistency": None,
            "group_monotonicity": None,
            "multiple_testing": {"p_value": None, "q_value": None, "reject_fdr_5pct": None},
        },
        "ic_decay": [],
        "group_returns": [],
        "rolling_ic": [],
        "cost_sensitivity": [],
        "breakeven_cost": None,
        "lookahead_status": "not_run",
        "cost_assumptions": {"fee_rate": 0.001, "slippage": 0.0005, "units": "one-way"},
        "research_note": "无足够真实样本；不得据此宣称已验证 alpha 或完成 6 个月 OOS。",
    }


def build_single_factor_report_data(
    factor_name: str,
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    frequency: str | None = None,
    rolling_window: int = 20,
    max_lag: int = 10,
    n_groups: int = 5,
    lookahead_status: str = "not_run",
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    close_prices: pd.Series | None = None,
    lookback_days: tuple[int, ...] = (30, 60, 90, 180),
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    bootstrap_samples: int = 500,
    bootstrap_block_size: int = 24,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aligned = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("forward_return")], axis=1
    ).dropna()
    if len(aligned) < max(5, n_groups):
        return insufficient_report(
            factor_name, f"有效因子/收益配对仅 {len(aligned)} 条", metadata=metadata
        )

    ic = compute_ic(aligned["factor"], aligned["forward_return"])
    rank_ic_value = compute_rank_ic(aligned["factor"], aligned["forward_return"])
    window = min(rolling_window, len(aligned))
    rolling = rolling_ic(
        aligned["factor"], aligned["forward_return"], window=max(3, window)
    ).dropna()
    rolling_std = rolling.std(ddof=1)
    icir = (
        rolling.mean() / rolling_std if pd.notna(rolling_std) and rolling_std > 0 else float("nan")
    )
    decay = compute_ic_decay(
        aligned["factor"], aligned["forward_return"], max_lag=min(max_lag, len(aligned) - 1)
    )
    groups = grouping_backtest(
        aligned["factor"], aligned["forward_return"], n_groups=min(n_groups, len(aligned))
    )
    robustness_grid = (
        window_horizon_robustness(
            factor_values,
            close_prices,
            lookback_days=lookback_days,
            horizons=horizons,
            min_obs=max(20, n_groups * 2),
        )
        if close_prices is not None
        else pd.DataFrame()
    )
    bootstrap = block_bootstrap_ic(
        aligned["factor"],
        aligned["forward_return"],
        n_bootstrap=bootstrap_samples,
        block_size=bootstrap_block_size,
    )
    cost_grid = []
    for total_cost in (0.0, 0.0005, 0.001, 0.0015, 0.002):
        # Split evenly so the helper's fee + slippage equals the displayed one-way cost.
        net = cost_adjusted_return(
            aligned["factor"], aligned["forward_return"],
            fee_rate=total_cost / 2, slippage=total_cost / 2,
        )
        cost_grid.append(
            {
                "one_way_cost": total_cost,
                "mean_net_return": _number(net.mean()),
                "cumulative_net_return": _number(net.sum()),
            }
        )
    sample_index = aligned.index
    start = (
        sample_index.min().isoformat()
        if isinstance(sample_index, pd.DatetimeIndex)
        else str(sample_index.min())
    )
    end = (
        sample_index.max().isoformat()
        if isinstance(sample_index, pd.DatetimeIndex)
        else str(sample_index.max())
    )
    return {
        "factor_name": factor_name,
        "status": "computed_short_sample",
        "sample": {"start": start, "end": end, "frequency": frequency, "n_obs": len(aligned)},
        "metrics": {
            "ic": _number(ic),
            "rank_ic": _number(rank_ic_value),
            "icir": _number(icir),
            "turnover": _number(compute_turnover(aligned["factor"])),
        },
        "provenance": _provenance(metadata),
        "robustness": {
            "window_horizon": [
                {
                    "lookback_days": int(row["lookback_days"]),
                    "horizon": int(row["horizon"]),
                    "n_obs": int(row["n_obs"]),
                    "ic": _number(row["ic"]),
                    "rank_ic": _number(row["rank_ic"]),
                    "status": str(row["status"]),
                }
                for _, row in robustness_grid.iterrows()
            ],
            "bootstrap_rank_ic": {
                key: (_number(value) if isinstance(value, float) else value)
                for key, value in bootstrap.items()
            },
            "sign_consistency": _number(sign_consistency(robustness_grid)),
            "group_monotonicity": _number(group_monotonicity(groups)),
            "multiple_testing": {
                "p_value": _number(bootstrap.get("p_value")),
                "q_value": None,
                "reject_fdr_5pct": None,
            },
        },
        "ic_decay": [{"lag": int(lag), "rank_ic": _number(value)} for lag, value in decay.items()],
        "group_returns": [
            {
                "group": int(row["group"]),
                "mean_return": _number(row["mean_return"]),
                "count": int(row["count"]),
                "top_bottom_spread": _number(row["top_bottom_spread"]),
            }
            for _, row in groups.iterrows()
        ],
        "rolling_ic": [
            {"time": str(index), "value": _number(value)} for index, value in rolling.items()
        ],
        "cost_sensitivity": cost_grid,
        "breakeven_cost": _number(
            compute_breakeven_cost(aligned["factor"], aligned["forward_return"])
        ),
        "lookahead_status": lookahead_status,
        "cost_assumptions": {"fee_rate": fee_rate, "slippage": slippage, "units": "one-way"},
        "research_note": (
            "指标仅描述当前样本；短样本 IC 不代表已验证 alpha，"
            "亦不表示完成 6 个月 OOS。"
        ),
    }


def write_report_data(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        escaped = (
            json.dumps(report, ensure_ascii=False, indent=2)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
        )
        path.write_text(
            f"<!doctype html><meta charset='utf-8'><title>{report['factor_name']}</title>"
            f"<h1>{report['factor_name']}</h1><pre>{escaped}</pre>",
            encoding="utf-8",
        )
    return path


def generate_single_factor_report(
    factor_name: str,
    factor_values: pd.Series,
    forward_returns: pd.Series,
    output_path: str = "",
) -> None:
    """兼容旧入口；构建报告并在给定路径写入 HTML 或 JSON。"""
    report = build_single_factor_report_data(factor_name, factor_values, forward_returns)
    if output_path:
        write_report_data(report, output_path)
