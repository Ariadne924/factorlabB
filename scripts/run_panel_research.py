"""Offline cross-sectional research entry point for registered panel factors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import factors  # noqa: E402,F401
from config.constants import get_interval_ms  # noqa: E402
from data.catalog import build_data_catalog  # noqa: E402
from data.panel_loader import load_silver_panel  # noqa: E402
from evaluation.crypto_risk_factors import build_ltw_factor_returns  # noqa: E402
from evaluation.panel_analysis import (  # noqa: E402
    build_panel_factor_report,
    check_panel_truncation_invariance,
)
from evaluation.robustness import benjamini_hochberg  # noqa: E402
from factors.panel import PanelValidationError, panel_coverage  # noqa: E402
from factors.panel_registry import (  # noqa: E402
    compute_panel_factor,
    get_panel_factor_metadata,
    list_panel_factors,
)
from visualization.single_factor_report import write_report_data  # noqa: E402


def _intervals(catalog: dict[str, Any], selected: tuple[str, ...] | None) -> list[str]:
    available = sorted({str(entry["interval"]) for entry in catalog.get("entries", [])})
    if selected is None:
        return available
    requested = {"1d" if value == "24h" else value for value in selected}
    return [interval for interval in available if interval in requested]


def run(
    data_dir: Path,
    reports_dir: Path,
    *,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    factor_names: tuple[str, ...] | None = None,
    min_assets: int = 3,
    horizon: int = 1,
) -> dict[str, Any]:
    """Evaluate real panels only; insufficient universes remain explicit."""
    report_dir = reports_dir / "panel_factor"
    report_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = reports_dir / "data_catalog.json"
    catalog = build_data_catalog(data_dir, catalog_path)
    selected_factors = list(factor_names or tuple(list_panel_factors()))
    unknown = sorted(set(selected_factors) - set(list_panel_factors()))
    if unknown:
        raise ValueError(f"unknown panel factors: {unknown}")

    reports: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    risk_factor_outputs: list[dict[str, Any]] = []
    for interval in _intervals(catalog, intervals):
        try:
            panel = load_silver_panel(
                data_dir,
                interval=interval,
                symbols=symbols,
                join="outer",
            )
        except PanelValidationError as exc:
            issues.append({"interval": interval, "status": "insufficient_data", "reason": str(exc)})
            continue
        available_symbols = panel.index.get_level_values("symbol").nunique()
        if available_symbols < min_assets:
            issues.append(
                {
                    "interval": interval,
                    "status": "insufficient_data",
                    "reason": f"requires {min_assets} symbols, found {available_symbols}",
                }
            )
            continue
        close = panel["close"]
        coverage = panel_coverage(panel)
        if "market_cap" in panel.columns:
            bars_per_week = max(
                1,
                int(pd.Timedelta(days=7).total_seconds() * 1000 / get_interval_ms(interval)),
            )
            risk_factors = build_ltw_factor_returns(
                panel,
                momentum_lookback=3 * bars_per_week,
            )
            risk_path = reports_dir / f"ltw_risk_factors_{interval}.json"
            risk_payload = {
                "interval": interval,
                "momentum_lookback_bars": 3 * bars_per_week,
                "rows": [
                    {
                        "time": str(index),
                        "cmkt": None if pd.isna(row["cmkt"]) else float(row["cmkt"]),
                        "csmb": None if pd.isna(row["csmb"]) else float(row["csmb"]),
                        "cmom": None if pd.isna(row["cmom"]) else float(row["cmom"]),
                    }
                    for index, row in risk_factors.iterrows()
                ],
                "note": "Point-in-time market_cap required; bar mapping is explicit.",
            }
            risk_path.write_text(
                json.dumps(risk_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            risk_factor_outputs.append(
                {"interval": interval, "status": "computed", "path": risk_path.name}
            )
        else:
            risk_factor_outputs.append(
                {
                    "interval": interval,
                    "status": "insufficient_data",
                    "reason": "point-in-time market_cap is unavailable",
                }
            )
        for factor_name in selected_factors:
            metadata = get_panel_factor_metadata(factor_name)
            output = report_dir / f"{interval}_{factor_name}.json"
            try:
                values = compute_panel_factor(factor_name, panel)

                def compute_selected(
                    data: pd.DataFrame, name: str = factor_name
                ) -> pd.Series:
                    return compute_panel_factor(name, data)

                lookahead = check_panel_truncation_invariance(compute_selected, panel)
                report = build_panel_factor_report(
                    factor_name,
                    values,
                    close,
                    interval=interval,
                    metadata=metadata,
                    lookahead_status="pass" if lookahead else "fail",
                    horizon=horizon,
                    min_assets=min_assets,
                )
            except (ValueError, KeyError) as exc:
                report = {
                    "factor_name": factor_name,
                    "scope": "cross_sectional",
                    "status": "insufficient_data",
                    "reason": str(exc),
                    "interval": interval,
                    "metrics": {
                        "ic": None,
                        "rank_ic": None,
                        "icir": None,
                        "turnover": None,
                    },
                    "provenance": metadata,
                    "lookahead_status": "not_run",
                    "research_note": "No valid panel result; no alpha conclusion is available.",
                }
            report["universe"] = sorted(
                panel.index.get_level_values("symbol").unique().astype(str)
            )
            report["coverage"] = {
                "mean": float(coverage["coverage_ratio"].mean()),
                "minimum": float(coverage["coverage_ratio"].min()),
            }
            write_report_data(report, output)
            reports.append({"path": output.relative_to(reports_dir).as_posix(), **report})

    p_values = pd.Series(
        {
            str(index): report.get("metrics", {}).get("rank_ic_p_value")
            for index, report in enumerate(reports)
        },
        dtype=float,
    )
    adjusted = benjamini_hochberg(p_values)
    for index, report in enumerate(reports):
        key = str(index)
        q_value = adjusted.loc[key, "q_value"] if key in adjusted.index else float("nan")
        report["multiple_testing"] = {
            "q_value": None if pd.isna(q_value) else float(q_value),
            "reject_fdr_5pct": (
                bool(adjusted.loc[key, "reject"]) if pd.notna(q_value) else None
            ),
        }
        path = reports_dir / report["path"]
        stored = {key_: value for key_, value in report.items() if key_ != "path"}
        write_report_data(stored, path)

    summary = {
        "status": "ok" if reports else "insufficient_data",
        "panel_factor_count": len(selected_factors),
        "computed_report_count": sum(
            report.get("status") != "insufficient_data" for report in reports
        ),
        "fdr_5pct_pass_count": sum(
            report.get("multiple_testing", {}).get("reject_fdr_5pct") is True
            for report in reports
        ),
        "intervals": _intervals(catalog, intervals),
        "symbols": sorted({symbol.upper() for symbol in symbols or ()}),
        "issues": issues,
        "risk_factors": risk_factor_outputs,
        "results": reports,
        "catalog": catalog_path.relative_to(reports_dir).as_posix(),
        "research_note": (
            "Cross-sectional current-sample diagnostics only; no six-month OOS or "
            "validated-alpha claim."
        ),
    }
    summary_path = reports_dir / "panel_research_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline cross-sectional factor research")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--intervals", nargs="+")
    parser.add_argument("--factors", nargs="+")
    parser.add_argument("--min-assets", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    summary = run(
        args.data_dir,
        args.reports_dir,
        symbols=tuple(args.symbols) if args.symbols else None,
        intervals=tuple(args.intervals) if args.intervals else None,
        factor_names=tuple(args.factors) if args.factors else None,
        min_assets=args.min_assets,
        horizon=args.horizon,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
