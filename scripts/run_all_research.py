"""离线一键研究入口；只消费已有 Silver 数据，不自动下载或伪造样本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import factors  # noqa: E402,F401 — 注册正式因子
from data.gold import write_gold_factor  # noqa: E402
from data.silver import silver_to_factor_input  # noqa: E402
from data.validator import DataValidator  # noqa: E402
from evaluation.forward_check import ForwardCheck  # noqa: E402
from factors.registry import compute_factor, list_factors  # noqa: E402
from visualization.single_factor_report import (  # noqa: E402
    build_single_factor_report_data,
    insufficient_report,
    write_report_data,
)


def run(data_dir: Path, reports_dir: Path) -> dict[str, object]:
    report_dir = reports_dir / "single_factor"
    report_dir.mkdir(parents=True, exist_ok=True)
    silver_files = sorted(data_dir.glob("silver/**/klines.parquet"))
    report_paths: list[str] = []
    manifest: dict[str, object] = {
        "status": "ok" if silver_files else "insufficient_data",
        "silver_files": [str(path) for path in silver_files],
        "factor_count": len(list_factors()),
        "reports": report_paths,
        "oos_6_months_completed": False,
        "note": "不生成合成样本；短样本指标不得表述为已验证 alpha。",
    }
    if not silver_files:
        for factor_name in list_factors():
            output = report_dir / f"{factor_name}.json"
            write_report_data(insufficient_report(factor_name, "未找到 Silver K 线数据"), output)
            report_paths.append(str(output.relative_to(reports_dir)))
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
        forward_returns = frame["close"].shift(-1).div(frame["close"]).sub(1)
        for factor_name in list_factors():
            output = report_dir / f"{symbol}_{interval}_{factor_name}.json"
            try:
                values = compute_factor(factor_name, frame)

                def compute_selected(data: pd.DataFrame, name: str = factor_name) -> pd.Series:
                    return compute_factor(name, data)

                lookahead = ForwardCheck.check_truncation_invariance(compute_selected, frame)
                report = build_single_factor_report_data(
                    factor_name,
                    values,
                    forward_returns,
                    frequency=interval,
                    lookahead_status="pass" if lookahead else "fail",
                )
                if report["status"] != "insufficient_data":
                    write_gold_factor(
                        values, symbol=symbol, interval=interval, factor_name=factor_name
                    )
            except ValueError as exc:
                report = insufficient_report(factor_name, str(exc))
            write_report_data(report, output)
            report_paths.append(str(output.relative_to(reports_dir)))
    manifest_path = reports_dir / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="运行全部已注册因子的离线研究")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    manifest = run(args.data_dir, args.reports_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
