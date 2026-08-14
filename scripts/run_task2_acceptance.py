"""Run the Task 2 G3 foundation gate on existing local Silver data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.silver import silver_to_factor_input  # noqa: E402
from strategies import create_strategy  # noqa: E402
from strategies.backtest import (  # noqa: E402
    BacktestConfig,
    compare_backtest_engines,
    run_event_backtest,
)
from strategies.threshold_research import run_threshold_study  # noqa: E402
from trading.risk import RiskLimits  # noqa: E402
from utils.io_utils import write_json_safe  # noqa: E402


def _find_scope(data_dir: Path, symbol: str, interval: str) -> Path:
    matches = [
        path
        for path in data_dir.glob("silver/**/klines.parquet")
        if symbol.upper() in path.parts and interval in path.parts
    ]
    if not matches:
        raise FileNotFoundError(f"missing Silver scope: {symbol} {interval}")
    return sorted(matches)[0]


def _load_scope(data_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    return silver_to_factor_input(pd.read_parquet(_find_scope(data_dir, symbol, interval)))


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    risk_events = list(result.get("risk_events", []))
    counts: dict[str, int] = {}
    for event in risk_events:
        rule = str(event.get("rule", "unknown"))
        counts[rule] = counts.get(rule, 0) + 1
    compact = {
        key: value for key, value in result.items() if key not in {"returns", "risk_events"}
    }
    compact["risk_event_count"] = len(risk_events)
    compact["risk_event_counts"] = counts
    compact["risk_events"] = risk_events[:100]
    return compact


def run(
    *,
    data_dir: Path,
    output: Path,
    symbol: str = "BTCUSDT",
    reference_symbol: str = "ETHUSDT",
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    frame = _load_scope(data_dir, symbol, interval)
    reference = _load_scope(data_dir, reference_symbol, interval)
    if start:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    frame["reference_close"] = reference["close"].reindex(frame.index)
    frame = frame.dropna(subset=["close", "reference_close"])
    if len(frame) < 240:
        raise ValueError("Task 2 acceptance requires at least 240 aligned bars")

    strategy_specs: dict[str, dict[str, Any]] = {
        "trend_following": {"fast_window": 24, "slow_window": 168},
        "grid_trading": {"anchor_window": 72, "grid_step": 0.01, "levels": 4},
        "statistical_arbitrage": {"hedge_window": 168, "z_window": 72, "entry_z": 1.5},
        "mean_reversion": {"window": 72, "entry_z": 1.5},
    }
    base_config = BacktestConfig()
    risk_config = BacktestConfig(
        risk_limits=RiskLimits(
            stop_loss_pct=0.03,
            max_drawdown_pct=0.08,
            max_position_fraction=1.0,
            max_leverage=2.0,
            circuit_breaker_return=0.05,
        )
    )
    results: dict[str, Any] = {}
    all_risk_rules: set[str] = set()
    for name, params in strategy_specs.items():
        strategy = create_strategy(name, **params)
        target = strategy.generate_target(frame)
        consistency = compare_backtest_engines(frame, target, base_config)
        event = run_event_backtest(frame, target, risk_config)
        all_risk_rules.update(str(row["rule"]) for row in event["risk_events"])
        results[name] = {
            "parameters": params,
            "consistency": consistency,
            "backtest": _compact_result(event),
        }

    close = pd.to_numeric(frame["close"], errors="coerce")
    rolling_mean = close.rolling(72, min_periods=72).mean()
    rolling_std = close.rolling(72, min_periods=72).std(ddof=0)
    mean_reversion_signal = rolling_mean.sub(close).div(rolling_std.where(rolling_std > 0))
    threshold_report = run_threshold_study(
        frame,
        mean_reversion_signal,
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        config=base_config,
    )
    consistency_pass = all(
        row["consistency"]["status"] == "pass" for row in results.values()
    )
    report = {
        "version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "g3_foundation_pass" if consistency_pass and all_risk_rules else "partial"
        ),
        "gates": {
            "G3": "pass" if consistency_pass else "fail",
            "G4": "partial",
            "G4_reason": (
                "Locked six-month OOS is not run; not every risk rule triggered in natural history."
            ),
        },
        "sample": {
            "symbol": symbol,
            "reference_symbol": reference_symbol,
            "interval": interval,
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "rows": len(frame),
            "funding_data_available": base_config.funding_column in frame.columns,
        },
        "strategy_results": results,
        "risk_trigger_summary": {
            "triggered_rules": sorted(all_risk_rules),
            "required_rules": [
                "stop_loss",
                "max_drawdown",
                "position_limit",
                "leverage_limit",
                "extreme_market_move",
            ],
            "note": (
                "Rules absent from this natural-history run remain covered by deterministic "
                "stress tests and must not be claimed as live-triggered."
            ),
        },
        "threshold_research": threshold_report,
        "oos_status": "not_run",
        "research_note": (
            "This is a Task 2 foundation gate on current local history, not the final locked "
            "six-month OOS result and not evidence of validated alpha."
        ),
    }
    write_json_safe(report, output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Task 2 策略库基础验收")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "task2" / "strategy_acceptance.json",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--reference-symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args(argv)
    report = run(
        data_dir=args.data_dir,
        output=args.output,
        symbol=args.symbol,
        reference_symbol=args.reference_symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
    print(json.dumps({key: report[key] for key in ("status", "sample", "oos_status")}, indent=2))
    print(f"report: {args.output.resolve()}")
    return 0 if report["status"] in {"g3_foundation_pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
