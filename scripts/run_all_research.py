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
from data.health import build_data_health_report  # noqa: E402
from data.silver import silver_to_factor_input  # noqa: E402
from data.training_readiness import build_training_readiness_report  # noqa: E402
from data.validator import DataValidator  # noqa: E402
from evaluation.correlation import correlation_matrix  # noqa: E402
from evaluation.cross_frequency import (  # noqa: E402
    build_cross_frequency_summary,
    display_frequency,
)
from evaluation.forward_check import ForwardCheck  # noqa: E402
from evaluation.research_runtime import (  # noqa: E402
    ResearchProgress,
    cached_report_paths,
    load_research_cache,
    research_code_fingerprint,
    research_run_signature,
    scope_signature,
    write_json_atomic,
)
from evaluation.robustness import benjamini_hochberg, forward_return, trailing_window  # noqa: E402
from factors.registry import compute_factor, get_factor_metadata, list_factors  # noqa: E402
from scripts.run_panel_research import run as run_panel_research  # noqa: E402
from visualization.frontend_payload import write_frontend_payload  # noqa: E402
from visualization.report_health import inspect_report_generation  # noqa: E402
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


def summarize_lookahead(rows: list[dict[str, Any]]) -> dict[str, int | str]:
    """区分明确失败与因数据依赖不足而未运行的检查。"""
    passed = sum(row["status"] == "pass" for row in rows)
    failed = sum(row["status"] == "fail" for row in rows)
    not_run = sum(row["status"] == "not_run" for row in rows)
    if failed:
        status = "fail"
    elif passed and not_run:
        status = "partial"
    elif passed:
        status = "pass"
    else:
        status = "not_run"
    return {
        "status": status,
        "checked_count": len(rows),
        "passed_count": passed,
        "failed_count": failed,
        "not_run_count": not_run,
    }


def _run(
    data_dir: Path,
    reports_dir: Path,
    *,
    lookback_days: int = 180,
    robustness_windows: tuple[int, ...] = (30, 60, 90, 180),
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    bootstrap_samples: int = 500,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    use_cache: bool = True,
) -> dict[str, object]:
    report_dir = reports_dir / "single_factor"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_silver_files = sorted(data_dir.glob("silver/**/klines.parquet"))
    silver_files = list(all_silver_files)
    symbol_filter = {value.upper() for value in symbols or ()}
    interval_filter = {"1d" if value == "24h" else value for value in intervals or ()}
    if symbol_filter or interval_filter:
        selected_files: list[Path] = []
        for path in silver_files:
            parts = set(path.parts)
            if symbol_filter and not symbol_filter.intersection(parts):
                continue
            if interval_filter and not interval_filter.intersection(parts):
                continue
            selected_files.append(path)
        silver_files = selected_files
    factor_names = list_factors()
    report_paths: list[str] = []
    cache_path = reports_dir / "research_cache.json"
    cache = load_research_cache(cache_path)
    status_path = reports_dir / "research_status.json"
    progress = ResearchProgress(
        status_path,
        total_scopes=len(silver_files),
        factors_per_scope=len(factor_names),
    )
    settings: dict[str, Any] = {
        "lookback_days": lookback_days,
        "robustness_windows": list(robustness_windows),
        "horizons": list(horizons),
        "bootstrap_samples": bootstrap_samples,
        "factors": factor_names,
    }
    code_fingerprint = research_code_fingerprint(PROJECT_ROOT)
    scope_signatures = {
        f"{path.parts[-3]}_{path.parts[-2]}": scope_signature(
            path,
            settings=settings,
            code_fingerprint=code_fingerprint,
        )
        for path in silver_files
    }
    run_signature = research_run_signature(
        scope_signatures,
        context={
            "settings": settings,
            "symbols": sorted(symbol_filter),
            "intervals": sorted(interval_filter),
        },
    )
    manifest: dict[str, object] = {
        "status": "ok" if silver_files else "insufficient_data",
        "silver_files": [_portable_path(path, PROJECT_ROOT) for path in silver_files],
        "factor_count": len(factor_names),
        "research_settings": {
            "lookback_days": lookback_days,
            "robustness_windows": list(robustness_windows),
            "horizons": list(horizons),
            "bootstrap_samples": bootstrap_samples,
            "symbols": sorted(symbol_filter),
            "intervals": sorted(interval_filter),
        },
        "reports": report_paths,
        "oos_6_months_completed": False,
        "note": "不生成合成样本；短样本指标不得表述为已验证 alpha。",
    }
    last_run = cache.get("last_run", {})
    snapshot_paths = [
        reports_dir / "research_manifest.json",
        reports_dir / "research_summary.json",
        reports_dir / "report_health.json",
    ]
    cache_reports_exist = all(
        cached_report_paths(
            cache,
            scope=scope,
            signature=signature,
            reports_dir=reports_dir,
        )
        is not None
        for scope, signature in scope_signatures.items()
    )
    if (
        use_cache
        and silver_files
        and last_run.get("signature") == run_signature
        and all(path.is_file() for path in snapshot_paths)
        and cache_reports_exist
    ):
        manifest = json.loads(snapshot_paths[0].read_text(encoding="utf-8"))
        manifest["research_cache"] = {
            "enabled": True,
            "snapshot_hit": True,
            "cache_hit_scopes": len(silver_files),
            "cache_hit_tasks": len(silver_files) * len(factor_names),
        }
        snapshot_paths[0].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for _ in silver_files:
            progress.complete_scope(cache_hit=True, task_count=len(factor_names))
        progress.complete()
        return manifest
    if not silver_files:
        for factor_name in factor_names:
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
        lookahead_path = reports_dir / "lookahead_report.json"
        lookahead_path.write_text(
            json.dumps(
                {"status": "not_run", "checked_count": 0, "failed_count": 0, "results": []},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    completed_reports: list[tuple[Path, dict[str, Any]]] = []
    cached_correlation_reports: list[dict[str, Any]] = []
    computed_correlation_reports: list[dict[str, Any]] = []
    for silver_path in silver_files:
        scope_parts = silver_path.parts
        scope_name = f"{scope_parts[-3]}_{scope_parts[-2]}"
        progress.start_scope(scope_name)
        signature = scope_signatures[scope_name]
        cached_paths = (
            cached_report_paths(
                cache,
                scope=scope_name,
                signature=signature,
                reports_dir=reports_dir,
            )
            if use_cache
            else None
        )
        if cached_paths is not None:
            try:
                cached_reports = [
                    json.loads(path.read_text(encoding="utf-8")) for path in cached_paths
                ]
            except (OSError, json.JSONDecodeError):
                cached_paths = None
            else:
                completed_reports.extend(zip(cached_paths, cached_reports, strict=True))
                report_paths.extend(
                    path.relative_to(reports_dir).as_posix() for path in cached_paths
                )
                cached_correlation = cache["scopes"][scope_name].get("correlation_report")
                if cached_correlation and (reports_dir / cached_correlation).is_file():
                    cached_correlation_reports.append(
                        {"scope": scope_name, "path": cached_correlation}
                    )
                progress.complete_scope(cache_hit=True, task_count=len(factor_names))
                continue
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
        factor_columns: dict[str, pd.Series] = {}
        forward_returns = forward_return(frame["close"], 1).reindex(report_frame.index)
        scope_report_paths: list[str] = []
        for factor_name in factor_names:
            output = report_dir / f"{symbol}_{interval}_{factor_name}.json"
            report: dict[str, Any]
            try:
                values = compute_factor(factor_name, frame)
                report_values = values.reindex(report_frame.index)
                factor_columns[factor_name] = report_values

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
                report["symbol"] = symbol
                report["interval"] = interval
                report["display_frequency"] = display_frequency(interval)
                if report["status"] != "insufficient_data":
                    write_gold_factor(
                        values, symbol=symbol, interval=interval, factor_name=factor_name
                    )
            except ValueError as exc:
                report = insufficient_report(
                    factor_name, str(exc), metadata=get_factor_metadata(factor_name)
                )
                report["symbol"] = symbol
                report["interval"] = interval
                report["display_frequency"] = display_frequency(interval)
            completed_reports.append((output, report))
            relative_output = output.relative_to(reports_dir).as_posix()
            report_paths.append(relative_output)
            scope_report_paths.append(relative_output)
            progress.advance(factor=factor_name)
        if factor_columns:
            matrix = correlation_matrix(pd.DataFrame(factor_columns))
            correlation_path = reports_dir / f"factor_correlation_{scope_name}.json"
            correlation_path.write_text(
                matrix.to_json(orient="split", force_ascii=False), encoding="utf-8"
            )
            correlation_relative = correlation_path.relative_to(reports_dir).as_posix()
            computed_correlation_reports.append(
                {"scope": scope_name, "path": correlation_relative}
            )
        cache["scopes"][scope_name] = {
            "signature": signature,
            "reports": scope_report_paths,
            "correlation_report": (
                correlation_relative if factor_columns else None
            ),
        }
        write_json_atomic(cache_path, cache)
        progress.complete_scope(cache_hit=False)
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
                "symbol": report.get("symbol"),
                "interval": report.get("interval"),
                "display_frequency": report.get("display_frequency"),
                "category": report.get("provenance", {}).get("category"),
                "source": report.get("provenance", {}).get("source"),
                "status": report["status"],
                "sample": report["sample"],
                "metrics": report["metrics"],
                "sign_consistency": report["robustness"]["sign_consistency"],
                "group_monotonicity": report["robustness"]["group_monotonicity"],
                "multiple_testing": testing,
                "lookahead_status": report["lookahead_status"],
            }
        )
    cross_frequency = build_cross_frequency_summary(summary_rows)
    correlation_reports = cached_correlation_reports + computed_correlation_reports
    lookahead_rows = [
        {
            "factor_name": row["factor_name"],
            "symbol": row.get("symbol"),
            "interval": row.get("interval"),
            "status": row["lookahead_status"],
            "report": row["report"],
        }
        for row in summary_rows
    ]
    lookahead_report = {
        **summarize_lookahead(lookahead_rows),
        "results": lookahead_rows,
    }
    lookahead_path = reports_dir / "lookahead_report.json"
    lookahead_path.write_text(
        json.dumps(lookahead_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": manifest["status"],
        "factor_count": manifest["factor_count"],
        "computed_report_count": len(summary_rows),
        "fdr_5pct_pass_count": sum(
            row["multiple_testing"]["reject_fdr_5pct"] is True for row in summary_rows
        ),
        "results": summary_rows,
        "cross_frequency": cross_frequency,
        "factor_correlation_reports": correlation_reports,
        "lookahead_gate": {
            "status": lookahead_report["status"],
            "report": lookahead_path.relative_to(reports_dir).as_posix(),
        },
        "research_note": (
            "候选结果仅适用于当前样本；FDR 未通过的因子不得描述为显著，"
            "任何结果均不代表完成 6 个月 OOS。"
        ),
    }
    summary_path = reports_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    frontend_path = write_frontend_payload(summary, reports_dir / "frontend_payload.json")
    manifest["summary_report"] = str(summary_path.relative_to(reports_dir))
    manifest["frontend_payload"] = str(frontend_path.relative_to(reports_dir))
    manifest["lookahead_report"] = str(lookahead_path.relative_to(reports_dir))
    manifest["lookahead_status"] = lookahead_report["status"]
    panel_summary = run_panel_research(
        data_dir,
        reports_dir,
        symbols=symbols,
        intervals=intervals,
    )
    manifest["panel_research"] = {
        "status": panel_summary["status"],
        "panel_factor_count": panel_summary["panel_factor_count"],
        "computed_report_count": panel_summary["computed_report_count"],
        "summary": "panel_research_summary.json",
    }
    summary["panel_results"] = panel_summary["results"]
    summary["data_catalog"] = panel_summary["catalog"]
    catalog_path = reports_dir / "data_catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        health = build_data_health_report(
            catalog,
            data_dir,
            output=reports_dir / "data_health.json",
            manifest_path=reports_dir / "real_data_collection_manifest.json",
        )
        summary["data_health"] = "data_health.json"
        summary["data_health_summary"] = health["summary"]
        readiness = build_training_readiness_report(
            health, output=reports_dir / "training_readiness.json"
        )
        summary["training_readiness"] = "training_readiness.json"
        summary["training_readiness_summary"] = readiness["summary"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_frontend_payload(summary, reports_dir / "frontend_payload.json")
    report_health = inspect_report_generation(reports_dir)
    report_health_path = reports_dir / "report_health.json"
    report_health_path.write_text(
        json.dumps(report_health, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["report_health"] = report_health
    manifest["research_cache"] = {
        "enabled": use_cache,
        "snapshot_hit": False,
        "cache_hit_scopes": progress.state["cache_hit_scopes"],
        "cache_hit_tasks": progress.state["cache_hit_tasks"],
    }
    cache["last_run"] = {"signature": run_signature}
    write_json_atomic(cache_path, cache)
    manifest_path = reports_dir / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.complete()
    return manifest


def run(
    data_dir: Path,
    reports_dir: Path,
    *,
    lookback_days: int = 180,
    robustness_windows: tuple[int, ...] = (30, 60, 90, 180),
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    bootstrap_samples: int = 500,
    symbols: tuple[str, ...] | None = None,
    intervals: tuple[str, ...] | None = None,
    use_cache: bool = True,
) -> dict[str, object]:
    """运行离线研究；失败时保留可诊断状态。"""
    try:
        return _run(
            data_dir,
            reports_dir,
            lookback_days=lookback_days,
            robustness_windows=robustness_windows,
            horizons=horizons,
            bootstrap_samples=bootstrap_samples,
            symbols=symbols,
            intervals=intervals,
            use_cache=use_cache,
        )
    except Exception as exc:
        status_path = reports_dir / "research_status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            status.update(
                {
                    "status": "failed",
                    "failed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            write_json_atomic(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="运行全部已注册因子的离线研究")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--robustness-windows", type=int, nargs="+", default=[30, 60, 90, 180])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--symbols", nargs="+", help="只研究指定资产")
    parser.add_argument("--intervals", nargs="+", help="只研究指定频率，24h 等价于 1d")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有研究缓存并重新计算")
    args = parser.parse_args()
    manifest = run(
        args.data_dir,
        args.reports_dir,
        lookback_days=args.lookback_days,
        robustness_windows=tuple(args.robustness_windows),
        horizons=tuple(args.horizons),
        bootstrap_samples=args.bootstrap_samples,
        symbols=tuple(args.symbols) if args.symbols else None,
        intervals=tuple(args.intervals) if args.intervals else None,
        use_cache=not args.no_cache,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
