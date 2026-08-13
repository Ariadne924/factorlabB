"""不删除候选因子的证据分级与定期轮动。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

FACTOR_GRADING_VERSION = "1.0"
TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
TIER_LABELS = {
    "A": "当前适配度高",
    "B": "当前适配度中等",
    "C": "当前适配度较低",
    "D": "数据不足",
}


def _finite(values: Sequence[object]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                output.append(number)
    return output


def _base_feature(name: str) -> str:
    return name.split("__", maxsplit=1)[0]


def _normalized_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{**row, "scope_type": "single"} for row in summary.get("results", [])]
    for row in summary.get("panel_results", []):
        rows.append(
            {
                **row,
                "factor_name": row.get("factor_name") or row.get("factor"),
                "display_frequency": row.get("display_frequency") or row.get("interval"),
                "scope_type": "panel",
            }
        )
    return rows


def _ml_stability(ml_reports: Sequence[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for report in ml_reports:
        for row in report.get("feature_recommendations", []):
            name = _base_feature(str(row.get("feature") or ""))
            if not name:
                continue
            selection = float(row.get("selection_frequency") or 0.0)
            consistency = float(row.get("sign_consistency") or 0.0)
            values[name].append(max(0.0, min(1.0, selection * consistency)))
    return {name: float(np.mean(scores)) for name, scores in values.items()}


def _grade_factor(
    factor_name: str,
    scope_type: str,
    rows: Sequence[dict[str, Any]],
    *,
    ml_stability: float | None,
) -> dict[str, Any]:
    computed = [row for row in rows if str(row.get("status", "")).startswith("computed")]
    lookahead_pass = bool(computed) and all(
        str(row.get("lookahead_status", "")).startswith("pass") for row in computed
    )
    rank_ics = _finite([row.get("metrics", {}).get("rank_ic") for row in computed])
    icirs = _finite([row.get("metrics", {}).get("icir") for row in computed])
    sign_consistency = _finite([row.get("sign_consistency") for row in computed])
    monotonicity = _finite([row.get("group_monotonicity") for row in computed])
    sample_sizes = _finite([row.get("sample", {}).get("n_obs") for row in computed])
    fdr_flags = [
        row.get("multiple_testing", {}).get("reject_fdr_5pct") is True
        for row in computed
    ]
    scopes = {
        (str(row.get("symbol") or "panel"), str(row.get("display_frequency") or "unknown"))
        for row in computed
    }
    if not computed or not rank_ics or not lookahead_pass:
        score = 0.0
        tier = "D"
    else:
        rank_score = min(abs(median(rank_ics)) / 0.05, 1.0)
        icir_score = min(abs(median(icirs)) / 0.5, 1.0) if icirs else 0.0
        sign_score = median(sign_consistency) if sign_consistency else 0.0
        monotonicity_score = (
            min(abs(median(monotonicity)), 1.0) if monotonicity else 0.0
        )
        fdr_score = sum(fdr_flags) / len(fdr_flags) if fdr_flags else 0.0
        sample_score = min(max(sample_sizes, default=0.0) / 5000.0, 1.0)
        scope_score = min(len(scopes) / 3.0, 1.0)
        evidence_score = 100.0 * (
            0.25 * rank_score
            + 0.15 * icir_score
            + 0.20 * sign_score
            + 0.10 * monotonicity_score
            + 0.10 * fdr_score
            + 0.10 * sample_score
            + 0.10 * scope_score
        )
        score = (
            evidence_score
            if ml_stability is None
            else 0.85 * evidence_score + 15 * ml_stability
        )
        if score >= 70 and len(scopes) >= 2:
            tier = "A"
        elif score >= 50:
            tier = "B"
        else:
            tier = "C"
    return {
        "factor_id": f"{scope_type}:{factor_name}",
        "factor_name": factor_name,
        "scope_type": scope_type,
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "score": round(score, 4),
        "report_count": len(rows),
        "computed_report_count": len(computed),
        "scope_count": len(scopes),
        "median_rank_ic": median(rank_ics) if rank_ics else None,
        "median_icir": median(icirs) if icirs else None,
        "median_sign_consistency": median(sign_consistency) if sign_consistency else None,
        "fdr_pass_count": sum(fdr_flags),
        "max_sample_size": int(max(sample_sizes, default=0)),
        "ml_train_stability": ml_stability,
        "lookahead_status": "pass" if lookahead_pass else "insufficient_or_failed",
        "retained": True,
        "interpretation": "评级用于研究优先级轮动，不会删除或停用该因子。",
    }


def build_factor_grade_report(
    research_summary: dict[str, Any],
    *,
    factor_universe: Sequence[tuple[str, str]],
    ml_reports: Sequence[dict[str, Any]] = (),
    previous_report: dict[str, Any] | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    rotation_days: int = 7,
) -> dict[str, Any]:
    """根据当前研究证据为全部因子分级；低分因子仍完整保留。"""
    if rotation_days < 1:
        raise ValueError("rotation_days 必须为正整数")
    timestamp = pd.Timestamp(as_of or datetime.now(UTC))
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    rows = _normalized_rows(research_summary)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("factor_name") or "")
        if name:
            grouped[(str(row["scope_type"]), name)].append(row)
    ml_scores = _ml_stability(ml_reports)
    previous = {
        str(row.get("factor_id")): str(row.get("tier"))
        for row in (previous_report or {}).get("factors", [])
    }
    factors: list[dict[str, Any]] = []
    for scope_type, factor_name in sorted(set(factor_universe)):
        grade = _grade_factor(
            factor_name,
            scope_type,
            grouped.get((scope_type, factor_name), []),
            ml_stability=ml_scores.get(factor_name) if scope_type == "single" else None,
        )
        old_tier = previous.get(str(grade["factor_id"]))
        if old_tier is None:
            movement = "new"
        elif TIER_ORDER[str(grade["tier"])] < TIER_ORDER[old_tier]:
            movement = "promoted"
        elif TIER_ORDER[str(grade["tier"])] > TIER_ORDER[old_tier]:
            movement = "demoted"
        else:
            movement = "unchanged"
        grade["previous_tier"] = old_tier
        grade["rotation_movement"] = movement
        factors.append(grade)
    factors.sort(
        key=lambda row: (
            TIER_ORDER[str(row["tier"])],
            -float(row["score"]),
            str(row["factor_id"]),
        )
    )
    tier_counts = {tier: sum(row["tier"] == tier for row in factors) for tier in TIER_ORDER}
    return {
        "version": FACTOR_GRADING_VERSION,
        "generated_at": timestamp.isoformat(),
        "next_rotation_at": (timestamp + pd.Timedelta(days=rotation_days)).isoformat(),
        "rotation_days": rotation_days,
        "summary": {
            "factor_count": len(factors),
            "retained_factor_count": sum(bool(row["retained"]) for row in factors),
            "removed_factor_count": 0,
            "tier_counts": tier_counts,
            "promoted_count": sum(row["rotation_movement"] == "promoted" for row in factors),
            "demoted_count": sum(row["rotation_movement"] == "demoted" for row in factors),
        },
        "factors": factors,
        "note": (
            "评级只表示当前样本和训练折下的研究优先级。所有因子均保留；"
            "高评级不等于已验证 alpha，低评级也不等于永久无效。"
        ),
    }


def write_factor_grade_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.json"
    stamp = pd.Timestamp(report["generated_at"]).strftime("%Y%m%dT%H%M%SZ")
    history = output_dir / "history" / f"{stamp}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    latest.write_text(payload, encoding="utf-8")
    history.write_text(payload, encoding="utf-8")
    return latest, history
