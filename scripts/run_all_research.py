"""离线一键研究入口；只消费已有 Silver 数据，不自动下载或伪造样本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import factors  # noqa: E402,F401 — 注册正式因子
from data.gold import write_gold_factor  # noqa: E402
from data.silver import silver_to_factor_input  # noqa: E402
from data.validator import DataValidator  # noqa: E402
from evaluation.forward_check import ForwardCheck  # noqa: E402
from evaluation.robustness import benjamini_hochberg, forward_return, trailing_window  # noqa: E402
from factors.registry import compute_factor, get_factor_metadata, list_factors  # noqa: E402
from visualization.single_factor_report import (  # noqa: E402
    build_single_factor_report_data,
    insufficient_report,
    write_report_data,
)


def _portable_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def run(
    data_dir: Path,
    reports_dir: Path,
    *,
    lookback_days: int = 180,
    robustness_windows: tuple[int, ...] = (30, 60, 90, 180),
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    bootstrap_samples: int = 500,
) -> dict[str, object]:
    report_dir = reports_dir / "single_factor"
    report_dir.mkdir(parents=True, exist_ok=True)
    silver_files = sorted(data_dir.glob("silver/**/klines.parquet"))
    report_paths: list[str] = []
    manifest: dict[str, object] = {
        "status": "ok" if silver_files else "insufficient_data",
        "silver_files": [_portable_path(path, PROJECT_ROOT) for path in silver_files],
        "factor_count": len(list_factors()),
        "research_settings": {
            "lookback_days": lookback_days,
            "robustness_windows": list(robustness_windows),
            "horizons": list(horizons),
            "bootstrap_samples": bootstrap_samples,
        },
        "reports": report_paths,
        "oos_6_months_completed": False,
        "note": "不生成合成样本；短样本指标不得表述为已验证 alpha。",
    }
    if not silver_files:
        for factor_name in list_factors():
            output = report_dir / f"{factor_name}.json"
            write_report_data(
                insufficient_report(
                    factor_name,
                    "未找到 Silver K 线数据",
                    metadata=get_factor_metadata(factor_name),
                ),
                output,
            )
            report_paths.append(str(output.relative_to(reports_dir)))
    completed_reports: list[tuple[Path, dict[str, Any]]] = []
    for silver_path in silver_files:
        silver = pd.read_parquet(silver_path)
        DataValidator.validate_klines(silver)
        interval = str(silver["interval"].iloc[0])
        symbol = str(silver["symbol"].iloc[0])
        flags = DataValidator.build_quality_flags(silver, expected_interval=interval)
        quality = DataValidator.summarize_quality_flags(flags)
        quality_path = reports_dir / f"data_quality_{symbol}_{interval}.json"
        quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
        frame = silver_to_factor_input(silver)
        factor_input = trailing_window(frame["close"], lookback_days).index
        report_frame = frame.loc[factor_input]
        forward_returns = forward_return(frame["close"], 1).reindex(report_frame.index)
        for factor_name in list_factors():
            output = report_dir / f"{symbol}_{interval}_{factor_name}.json"
            report: dict[str, Any]
            try:
                values = compute_factor(factor_name, frame)
                report_values = values.reindex(report_frame.index)

                def compute_selected(data: pd.DataFrame, name: str = factor_name) -> pd.Series:
                    return compute_factor(name, data)

                lookahead = ForwardCheck.check_truncation_invariance(compute_selected, frame)
                report = build_single_factor_report_data(
                    factor_name,
                    report_values,
                    forward_returns,
                    frequency=interval,
                    lookahead_status="pass" if lookahead else "fail",
                    close_prices=report_frame["close"],
                    lookback_days=robustness_windows,
                    horizons=horizons,
                    bootstrap_samples=bootstrap_samples,
                    metadata=get_factor_metadata(factor_name),
                )
                if report["status"] != "insufficient_data":
                    write_gold_factor(
                        values, symbol=symbol, interval=interval, factor_name=factor_name
                    )
            except ValueError as exc:
                report = insufficient_report(
                    factor_name, str(exc), metadata=get_factor_metadata(factor_name)
                )
            completed_reports.append((output, report))
            report_paths.append(str(output.relative_to(reports_dir)))
    p_values = pd.Series(
        {
            str(path): report.get("robustness", {})
            .get("multiple_testing", {})
            .get("p_value")
            for path, report in completed_reports
        },
        dtype="float64",
    )
    adjusted = benjamini_hochberg(p_values)
    summary_rows: list[dict[str, Any]] = []
    for output, report in completed_reports:
        key = str(output)
        testing = report["robustness"]["multiple_testing"]
        q_value = adjusted.loc[key, "q_value"]
        testing["q_value"] = None if pd.isna(q_value) else float(q_value)
        testing["reject_fdr_5pct"] = (
            bool(adjusted.loc[key, "reject"]) if pd.notna(q_value) else None
        )
        write_report_data(report, output)
        summary_rows.append(
            {
                "report": output.relative_to(reports_dir).as_posix(),
                "factor_name": report["factor_name"],
                "status": report["status"],
                "sample": report["sample"],
                "metrics": report["metrics"],
                "sign_consistency": report["robustness"]["sign_consistency"],
                "group_monotonicity": report["robustness"]["group_monotonicity"],
                "multiple_testing": testing,
                "lookahead_status": report["lookahead_status"],
            }
        )
    summary = {
        "status": manifest["status"],
        "factor_count": manifest["factor_count"],
        "computed_report_count": len(summary_rows),
        "fdr_5pct_pass_count": sum(
            row["multiple_testing"]["reject_fdr_5pct"] is True for row in summary_rows
        ),
        "results": summary_rows,
        "research_note": (
            "候选结果仅适用于当前样本；FDR 未通过的因子不得描述为显著，"
            "任何结果均不代表完成 6 个月 OOS。"
        ),
    }
    summary_path = reports_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["summary_report"] = str(summary_path.relative_to(reports_dir))
    manifest_path = reports_dir / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="运行全部已注册因子的离线研究")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--robustness-windows", type=int, nargs="+", default=[30, 60, 90, 180])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    manifest = run(
        args.data_dir,
        args.reports_dir,
        lookback_days=args.lookback_days,
        robustness_windows=tuple(args.robustness_windows),
        horizons=tuple(args.horizons),
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
