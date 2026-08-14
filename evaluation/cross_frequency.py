"""跨资产、跨频率的因子结果汇总。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

FREQUENCY_LABELS = {"1m": "1m", "1h": "1h", "6h": "6h", "1d": "24h", "24h": "24h"}


def display_frequency(interval: str | None) -> str:
    """将交易所周期转换为研究报告中的展示口径。"""
    if interval is None:
        return "unknown"
    return FREQUENCY_LABELS.get(str(interval), str(interval))


def _finite(values: list[object]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    return numeric[np.isfinite(numeric)].to_numpy(dtype=float)


def _direction_consistency(values: np.ndarray) -> float | None:
    non_zero = values[values != 0]
    if non_zero.size == 0:
        return None
    positive_share = float(np.mean(non_zero > 0))
    return max(positive_share, 1.0 - positive_share)


def build_cross_frequency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按因子汇总不同资产和频率的 RankIC，不制造缺失组合的结果。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("factor_name"))].append(row)

    output: list[dict[str, Any]] = []
    for factor_name, factor_rows in sorted(grouped.items()):
        computed = [row for row in factor_rows if row.get("status") != "insufficient_data"]
        rank_ics = _finite([row.get("metrics", {}).get("rank_ic") for row in computed])
        symbols = sorted({str(row.get("symbol")) for row in computed if row.get("symbol")})
        frequencies = sorted(
            {
                display_frequency(str(row.get("interval")))
                for row in computed
                if row.get("interval")
            }
        )
        output.append(
            {
                "factor_name": factor_name,
                "computed_reports": len(computed),
                "symbols": symbols,
                "frequencies": frequencies,
                "mean_rank_ic": float(rank_ics.mean()) if rank_ics.size else None,
                "median_rank_ic": float(np.median(rank_ics)) if rank_ics.size else None,
                "direction_consistency": _direction_consistency(rank_ics),
                "lookahead_pass_rate": (
                    float(
                        np.mean(
                            [row.get("lookahead_status") == "pass" for row in computed]
                        )
                    )
                    if computed
                    else None
                ),
                "fdr_5pct_pass_count": sum(
                    row.get("multiple_testing", {}).get("reject_fdr_5pct") is True
                    for row in computed
                ),
                "note": "跨频率汇总是稳定性诊断，不等同于独立样本外验证。",
            }
        )
    return output
