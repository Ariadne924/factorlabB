"""Streamlit 多资产、多频率 Factor Explorer。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import factors  # noqa: E402,F401
from config.constants import get_interval_ms  # noqa: E402
from data.catalog import build_data_catalog  # noqa: E402
from data.health import build_data_health_report  # noqa: E402
from data.panel_loader import load_silver_panel  # noqa: E402
from data.refresh import refresh_recent_market_data  # noqa: E402
from data.silver import silver_to_factor_input  # noqa: E402
from evaluation.panel_strategy import (  # noqa: E402
    FactorAllocation,
    PanelStrategyConfig,
    run_panel_strategy,
)
from evaluation.strategy_regimes import (  # noqa: E402
    MarketRegimeConfig,
    market_regime_analysis,
)
from evaluation.strategy_robustness import (  # noqa: E402
    build_strategy_robustness_report,
    parameter_perturbation,
)
from evaluation.strategy_walk_forward import (  # noqa: E402
    StrategyWalkForwardConfig,
    walk_forward_strategy,
)
from evaluation.time_series_strategy import (  # noqa: E402
    TimeSeriesAllocation,
    TimeSeriesStrategyConfig,
    build_ml_strategy_preset,
    run_time_series_strategy,
)
from factors.panel_registry import list_panel_factors  # noqa: E402
from factors.registry import list_factors  # noqa: E402
from scripts.run_ml_factor_mining import run as run_ml_factor_research  # noqa: E402
from visualization.report_discovery import load_factor_reports  # noqa: E402
from visualization.strategy_store import (  # noqa: E402
    compare_strategy_snapshots,
    load_strategy_snapshots,
    save_strategy_snapshot,
)
from visualization.workflow import (  # noqa: E402
    WorkflowStep,
    build_data_tasks,
    build_workflow_steps,
    next_workflow_action,
    workflow_progress,
)

REPORT_ROOT = PROJECT_ROOT / "reports"
STRATEGY_STORE = REPORT_ROOT / "strategy_snapshots"
REFRESH_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
)
REFRESH_INTERVALS = ("1m", "5m", "15m", "1h", "6h", "24h")


@st.cache_data(ttl=30)
def load_reports(directory: Path) -> list[dict[str, Any]]:
    return load_factor_reports(
        directory,
        require_symbol=directory.name in {"single_factor", "ml_factor"},
    )


@st.cache_data(ttl=30)
def load_summary() -> dict[str, Any]:
    path = REPORT_ROOT / "research_summary.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


@st.cache_data(ttl=30)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


@st.cache_data(ttl=30)
def load_data_health(catalog_generated_at: str | None) -> dict[str, Any]:
    """Cache the local health scan while still invalidating after a catalog rebuild."""
    del catalog_generated_at
    catalog = load_json(REPORT_ROOT / "data_catalog.json")
    return build_data_health_report(
        catalog,
        PROJECT_ROOT / "data",
        manifest_path=REPORT_ROOT / "real_data_collection_manifest.json",
    )


def scope(report: dict[str, Any]) -> tuple[str, str]:
    sample = report.get("sample", {})
    symbol = str(report.get("symbol") or "unknown")
    interval = str(report.get("display_frequency") or sample.get("frequency") or "unknown")
    interval = "24h" if interval == "1d" else interval
    return symbol, interval


def metric_value(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    return "N/A" if pd.isna(value) else f"{float(value):.4f}"


def navigate_to(page_name: str) -> None:
    st.session_state["page_selector"] = page_name


def render_workflow_overview(steps: list[WorkflowStep]) -> None:
    """Render a compact task-oriented home page instead of a feature directory."""
    state_labels = {
        "blocked": "阻塞",
        "attention": "需处理",
        "ready": "可开始",
        "completed": "已完成",
    }
    progress = workflow_progress(steps)
    st.progress(progress, text=f"核心研究流程完成度 {progress:.0%}")
    next_step = next_workflow_action(steps)
    if next_step["state"] == "blocked":
        st.warning(f"下一步：{next_step['label']}。{next_step['summary']}")
    else:
        st.info(f"下一步：{next_step['label']}。{next_step['summary']}")
    columns = st.columns(len(steps))
    for index, (column, step) in enumerate(zip(columns, steps, strict=True), start=1):
        with column.container(border=True):
            suffix = " · 可后置" if step["optional"] else ""
            st.caption(f"步骤 {index}{suffix}")
            st.subheader(step["label"])
            st.write(f"**{state_labels[step['state']]}**")
            st.write(step["summary"])
            st.button(
                step["action_label"],
                key=f"workflow_{step['key']}",
                type="primary" if step["key"] == next_step["key"] else "secondary",
                use_container_width=True,
                on_click=navigate_to,
                args=(step["action_page"],),
            )


def apply_ml_preset(report: dict[str, Any]) -> None:
    preset = report.get("strategy_preset") or build_ml_strategy_preset(
        report.get("feature_recommendations", [])
    )
    preset["symbol"], preset["interval"] = scope(report)
    preset["interval"] = "1d" if preset["interval"] == "24h" else preset["interval"]
    st.session_state["time_series_preset"] = preset
    st.session_state["ts_selected_factors"] = [
        str(row["name"]) for row in preset.get("allocations", [])
    ]
    for row in preset.get("allocations", []):
        name = str(row["name"])
        st.session_state[f"ts_weight_{name}"] = float(row["weight"])
        st.session_state[f"ts_direction_{name}"] = (
            "正向" if int(row["direction"]) == 1 else "反向"
        )
    st.session_state["page_selector"] = "时序策略"


def render_strategy_result(result: dict[str, Any], *, key_prefix: str) -> None:
    metrics = result.get("metrics", {})
    metric_columns = st.columns(6)
    for column, key, label in zip(
        metric_columns,
        [
            "total_return",
            "gross_total_return",
            "bar_sharpe",
            "max_drawdown",
            "mean_turnover",
            "hit_rate",
        ],
        ["净收益", "成本前收益", "Bar Sharpe", "最大回撤", "换手率", "胜率"],
        strict=True,
    ):
        column.metric(label, metric_value(metrics.get(key)))
    strategy_frame = pd.DataFrame(result.get("returns", []))
    if not strategy_frame.empty:
        strategy_frame["time"] = pd.to_datetime(strategy_frame["time"], utc=True)
        chart_columns = [
            column for column in ("equity", "drawdown") if column in strategy_frame.columns
        ]
        st.line_chart(strategy_frame.set_index("time")[chart_columns])
        with st.expander("逐期结果"):
            st.dataframe(strategy_frame, use_container_width=True, hide_index=True)
    st.caption(result.get("research_note", ""))
    st.caption(f"前视检查：{result.get('lookahead_status', 'not_run')}")
    with st.expander("配置与因子覆盖"):
        st.json(
            {
                "config": result.get("config", {}),
                "factor_coverage": result.get("factor_coverage", {}),
                "lookahead_policy": result.get("lookahead_policy"),
                "result_key": key_prefix,
            }
        )


def render_time_series_robustness(result: dict[str, Any]) -> None:
    st.subheader("策略稳健性实验室")
    n_periods = int(result.get("metrics", {}).get("n_periods") or 0)
    if n_periods < 5 or not result.get("returns"):
        st.info("有效回测记录不足 5 根，暂不生成稳健性诊断。")
        return
    default_window = min(168, max(5, n_periods // 4))
    rolling_window = int(
        st.number_input(
            "滚动统计窗口（K 线根数）",
            min_value=5,
            max_value=max(5, n_periods),
            value=default_window,
            key="robustness_rolling_window",
        )
    )
    robustness = build_strategy_robustness_report(
        result,
        rolling_window=rolling_window,
    )
    st.session_state["time_series_robustness"] = robustness
    overview = robustness["stability_summary"]
    summary_columns = st.columns(3)
    summary_columns[0].metric("覆盖月份", overview["month_count"])
    summary_columns[1].metric(
        "正收益月份占比", metric_value(overview["positive_month_ratio"])
    )
    summary_columns[2].metric(
        "所有成本情景为正", "是" if overview["all_cost_scenarios_positive"] else "否"
    )
    cost_tab, month_tab, rolling_tab, benchmark_tab, perturbation_tab = st.tabs(
        ["成本敏感性", "分月表现", "滚动表现", "Buy & Hold", "参数扰动"]
    )
    with cost_tab:
        cost_frame = pd.DataFrame(robustness["cost_grid"])
        st.dataframe(cost_frame, use_container_width=True, hide_index=True)
        st.line_chart(cost_frame.set_index("one_way_cost")["total_return"])
    with month_tab:
        monthly = pd.DataFrame(robustness["monthly_performance"])
        if monthly.empty:
            st.info("样本不足以生成分月结果。")
        else:
            st.dataframe(monthly, use_container_width=True, hide_index=True)
            st.bar_chart(monthly.set_index("month")[["gross_return", "net_return"]])
    with rolling_tab:
        rolling = pd.DataFrame(robustness["rolling_performance"])
        if rolling.empty:
            st.info("样本不足以生成滚动结果。")
        else:
            rolling["time"] = pd.to_datetime(rolling["time"], utc=True)
            st.line_chart(
                rolling.set_index("time")[["rolling_return", "rolling_bar_sharpe"]]
            )
    with benchmark_tab:
        benchmark = robustness["benchmark"]
        if benchmark.get("status") != "computed":
            st.info(str(benchmark.get("reason", "基准不可用")))
        else:
            comparison = pd.DataFrame(
                [
                    {"portfolio": "strategy", **benchmark["strategy"]},
                    {"portfolio": "buy_and_hold", **benchmark["buy_and_hold"]},
                ]
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.metric("相对 Buy & Hold 总收益", metric_value(benchmark["excess_total_return"]))
    with perturbation_tab:
        context = st.session_state.get("time_series_strategy_context")
        if not context:
            st.info("重新运行一次时序策略后即可进行参数扰动。")
        else:
            st.caption("采用小型邻域网格，不自动寻找最优参数。")
            base_config: TimeSeriesStrategyConfig = context["config"]
            perturbation_columns = st.columns(3)
            threshold_spread = float(
                perturbation_columns[0].number_input(
                    "阈值扰动幅度", min_value=0.0, value=0.25
                )
            )
            rebalance_multiplier = int(
                perturbation_columns[1].number_input(
                    "调仓倍数", min_value=2, value=2
                )
            )
            window_spread = float(
                perturbation_columns[2].number_input(
                    "窗口扰动比例", min_value=0.1, max_value=0.8, value=0.25
                )
            )
            if st.button("运行参数扰动", key="run_parameter_perturbation"):
                thresholds = sorted(
                    {
                        max(0.0, base_config.score_threshold - threshold_spread),
                        base_config.score_threshold,
                        base_config.score_threshold + threshold_spread,
                    }
                )
                rebalances = sorted(
                    {
                        max(1, base_config.rebalance_every // rebalance_multiplier),
                        base_config.rebalance_every,
                        base_config.rebalance_every * rebalance_multiplier,
                    }
                )
                windows = sorted(
                    {
                        max(5, round(base_config.standardize_window * (1 - window_spread))),
                        base_config.standardize_window,
                        round(base_config.standardize_window * (1 + window_spread)),
                    }
                )
                with st.spinner("正在运行参数邻域检验……"):
                    st.session_state["parameter_perturbation"] = parameter_perturbation(
                        context["frame"],
                        base_config,
                        thresholds=thresholds,
                        rebalance_values=rebalances,
                        standardize_windows=windows,
                    )
            perturbations = pd.DataFrame(
                st.session_state.get("parameter_perturbation", [])
            )
            if not perturbations.empty:
                st.dataframe(perturbations, use_container_width=True, hide_index=True)
    render_advanced_strategy_diagnostics(result, robustness)
    st.session_state["time_series_robustness"] = robustness
    st.caption(robustness["note"])


def render_advanced_strategy_diagnostics(
    result: dict[str, Any],
    robustness: dict[str, Any],
) -> None:
    """Render causal regime attribution and explicit strategy walk-forward tests."""
    context = st.session_state.get("time_series_strategy_context")
    if not context:
        return
    frame: pd.DataFrame = context["frame"]
    base_config: TimeSeriesStrategyConfig = context["config"]
    n_rows = len(frame)
    st.subheader("市场状态与 Walk-Forward")
    regime_tab, walk_forward_tab = st.tabs(["市场状态诊断", "策略 Walk-Forward"])

    with regime_tab:
        st.caption(
            "只用历史滚动窗口定义牛市、熊市、震荡和高低波动状态；状态整体延后一根 K 线。"
        )
        defaults = (
            min(72, max(5, n_rows // 10)),
            min(72, max(5, n_rows // 10)),
            min(168, max(5, n_rows // 4)),
        )
        regime_columns = st.columns(4)
        trend_window = int(
            regime_columns[0].number_input(
                "趋势窗口（bars）",
                min_value=5,
                value=defaults[0],
                key="regime_trend_window",
            )
        )
        volatility_window = int(
            regime_columns[1].number_input(
                "波动窗口（bars）",
                min_value=5,
                value=defaults[1],
                key="regime_volatility_window",
            )
        )
        history_window = int(
            regime_columns[2].number_input(
                "状态历史窗口（bars）",
                min_value=5,
                value=defaults[2],
                key="regime_history_window",
            )
        )
        trend_band = float(
            regime_columns[3].number_input(
                "震荡带宽（波动倍数）",
                min_value=0.0,
                value=0.5,
                step=0.1,
                key="regime_trend_band",
            )
        )
        try:
            regime_report = market_regime_analysis(
                frame,
                result,
                config=MarketRegimeConfig(
                    trend_window=trend_window,
                    volatility_window=volatility_window,
                    threshold_history=history_window,
                    trend_band=trend_band,
                ),
            )
        except ValueError as exc:
            st.info(str(exc))
        else:
            robustness["market_regimes"] = regime_report
            st.metric("可归因样本覆盖率", metric_value(regime_report["coverage"]))
            trend_frame = pd.DataFrame(regime_report["trend_performance"])
            volatility_frame = pd.DataFrame(
                regime_report["volatility_performance"]
            )
            combined_frame = pd.DataFrame(regime_report["combined_performance"])
            table_columns = st.columns(2)
            with table_columns[0]:
                st.markdown("**趋势状态**")
                st.dataframe(trend_frame, use_container_width=True, hide_index=True)
            with table_columns[1]:
                st.markdown("**波动状态**")
                st.dataframe(
                    volatility_frame,
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("趋势 × 波动组合状态"):
                st.dataframe(combined_frame, use_container_width=True, hide_index=True)
            st.caption(regime_report["note"])

    with walk_forward_tab:
        st.caption(
            "每一折只在扩展训练集选择参数，留出 embargo 后冻结参数，并仅汇总随后测试段。"
        )
        if n_rows < 50:
            st.info("至少需要 50 根 K 线才能运行策略 Walk-Forward。")
            return
        bars_per_day = max(
            1,
            round(
                86_400_000
                / get_interval_ms(
                    "1d" if str(context["interval"]) == "24h" else str(context["interval"])
                )
            ),
        )
        default_train = min(max(20, n_rows // 2), n_rows - 2)
        default_test = max(1, min(n_rows // 6, n_rows - default_train - 1))
        fold_columns = st.columns(4)
        min_train_size = int(
            fold_columns[0].number_input(
                "最小训练集（bars）",
                min_value=20,
                max_value=max(20, n_rows - 2),
                value=default_train,
                key="strategy_wf_train_size",
            )
        )
        test_size = int(
            fold_columns[1].number_input(
                "每折测试集（bars）",
                min_value=1,
                max_value=max(1, n_rows - 21),
                value=min(default_test, max(1, n_rows - min_train_size - 1)),
                key="strategy_wf_test_size",
            )
        )
        embargo = int(
            fold_columns[2].number_input(
                "Embargo（bars）",
                min_value=0,
                max_value=max(1, n_rows - 21),
                value=1,
                key="strategy_wf_embargo",
            )
        )
        threshold_spread = float(
            fold_columns[3].number_input(
                "阈值搜索半径",
                min_value=0.0,
                value=0.25,
                step=0.05,
                key="strategy_wf_threshold_spread",
            )
        )
        st.caption(
            f"当前数据约 {n_rows / bars_per_day:.1f} 天；训练窗口约 "
            f"{min_train_size / bars_per_day:.1f} 天，测试窗口约 "
            f"{test_size / bars_per_day:.1f} 天。默认只跑 12 个邻域候选。"
        )
        if st.button("运行策略 Walk-Forward", key="run_strategy_walk_forward"):
            thresholds = sorted(
                {
                    max(0.0, base_config.score_threshold - threshold_spread),
                    base_config.score_threshold,
                    base_config.score_threshold + threshold_spread,
                }
            )
            rebalances = sorted(
                {
                    base_config.rebalance_every,
                    base_config.rebalance_every * 2,
                }
            )
            windows = sorted(
                {
                    max(5, round(base_config.standardize_window * 0.75)),
                    max(5, round(base_config.standardize_window * 1.25)),
                }
            )
            try:
                with st.spinner("正在按时间折训练、冻结参数并测试……"):
                    st.session_state["strategy_walk_forward"] = (
                        walk_forward_strategy(
                            frame,
                            base_config,
                            config=StrategyWalkForwardConfig(
                                min_train_size=min_train_size,
                                test_size=test_size,
                                embargo=embargo,
                            ),
                            thresholds=thresholds,
                            rebalance_values=rebalances,
                            standardize_windows=windows,
                        )
                    )
            except ValueError as exc:
                st.error(str(exc))
        walk_forward_report = st.session_state.get("strategy_walk_forward")
        if walk_forward_report:
            robustness["strategy_walk_forward"] = walk_forward_report
            metrics = walk_forward_report["metrics"]
            metric_columns = st.columns(5)
            for column, key, label in zip(
                metric_columns,
                [
                    "total_return",
                    "gross_total_return",
                    "bar_sharpe",
                    "max_drawdown",
                    "mean_turnover",
                ],
                ["OOS 净收益", "OOS 毛收益", "OOS Bar Sharpe", "OOS 最大回撤", "OOS 换手"],
                strict=True,
            ):
                column.metric(label, metric_value(metrics.get(key)))
            fold_frame = pd.DataFrame(
                [
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "candidate_scores",
                                "selected_params",
                                "selected_training_metrics",
                            }
                        },
                        **row["selected_params"],
                        "train_bar_sharpe": row["selected_training_metrics"][
                            "train_bar_sharpe"
                        ],
                        "train_total_return": row["selected_training_metrics"][
                            "train_total_return"
                        ],
                    }
                    for row in walk_forward_report["folds"]
                ]
            )
            st.dataframe(fold_frame, use_container_width=True, hide_index=True)
            oos_frame = pd.DataFrame(walk_forward_report["returns"])
            if not oos_frame.empty:
                oos_frame["time"] = pd.to_datetime(oos_frame["time"], utc=True)
                st.line_chart(oos_frame.set_index("time")["equity"])
            st.caption(walk_forward_report["note"])


def render_strategy_store(
    result: dict[str, Any],
    robustness: dict[str, Any],
    *,
    symbol: str,
    interval: str,
) -> None:
    st.subheader("保存与比较")
    name_column, save_column = st.columns([3, 1])
    snapshot_name = name_column.text_input(
        "策略名称",
        value=f"{symbol}-{interval}-strategy",
        key="strategy_snapshot_name",
    )
    if save_column.button("保存当前快照", use_container_width=True):
        try:
            path = save_strategy_snapshot(
                STRATEGY_STORE,
                name=snapshot_name,
                symbol=symbol,
                interval=interval,
                result=result,
                robustness=robustness,
            )
        except (OSError, TypeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"已保存：{path.name}")
    snapshots = load_strategy_snapshots(STRATEGY_STORE)
    if snapshots:
        comparison = pd.DataFrame(compare_strategy_snapshots(snapshots))
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.caption("还没有已保存策略；保存后可跨配置比较。")


def render_saved_strategy_comparison() -> None:
    st.header("Saved Strategy Comparison")
    st.caption("只比较用户显式保存的本地快照；不会把当前交互结果自动落盘。")
    snapshots = load_strategy_snapshots(STRATEGY_STORE)
    if not snapshots:
        st.info("还没有策略快照。请先在时序策略页面运行并保存。")
        return
    labels = [
        f"{row.get('name')} · {row.get('symbol')} · {row.get('interval')}"
        for row in snapshots
    ]
    selected = st.multiselect(
        "选择策略",
        range(len(labels)),
        default=list(range(min(3, len(labels)))),
        format_func=labels.__getitem__,
    )
    chosen = [snapshots[index] for index in selected]
    if not chosen:
        st.warning("至少选择一个策略快照。")
        return
    comparison = pd.DataFrame(compare_strategy_snapshots(chosen))
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    metrics = comparison.set_index("name")[
        ["total_return", "bar_sharpe", "max_drawdown", "mean_turnover"]
    ]
    return_tab, risk_tab = st.tabs(["收益与 Sharpe", "回撤与换手"])
    with return_tab:
        st.bar_chart(metrics[["total_return", "bar_sharpe"]])
    with risk_tab:
        st.bar_chart(metrics[["max_drawdown", "mean_turnover"]])
    st.caption("不同币种、频率或区间的结果不可直接视为同口径排名。")


def render_single_report(selected: dict[str, Any]) -> None:
    symbol, frequency = scope(selected)
    sample = selected.get("sample", {})
    st.caption(selected.get("research_note", ""))
    st.write(
        f"资产：{symbol} · 频率：{frequency} · "
        f"样本：{sample.get('start')} → {sample.get('end')} · N={sample.get('n_obs')}"
    )
    metrics = selected.get("metrics", {})
    columns = st.columns(5)
    for column, key, label in zip(
        columns,
        ["ic", "rank_ic", "icir", "turnover"],
        ["IC", "RankIC", "ICIR", "Turnover"],
        strict=False,
    ):
        column.metric(label, metric_value(metrics.get(key)))
    columns[-1].metric("Lookahead", str(selected.get("lookahead_status", "not_run")))

    overview_tab, robustness_tab, provenance_tab = st.tabs(
        ["单因子报告", "稳健性", "定义与数据"]
    )
    decay = pd.DataFrame(selected.get("ic_decay", []))
    groups = pd.DataFrame(selected.get("group_returns", []))
    rolling = pd.DataFrame(selected.get("rolling_ic", []))
    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("IC decay")
            st.line_chart(
                decay.set_index("lag")["rank_ic"]
                if not decay.empty
                else pd.Series(dtype=float)
            )
            st.subheader("Rolling IC")
            if not rolling.empty:
                rolling["time"] = pd.to_datetime(rolling["time"], utc=True, errors="coerce")
                st.line_chart(rolling.set_index("time")["value"])
        with right:
            st.subheader("Group returns")
            st.bar_chart(
                groups.set_index("group")["mean_return"]
                if not groups.empty
                else pd.Series(dtype=float)
            )
            st.subheader("Cost assumptions")
            st.json(selected.get("cost_assumptions", {}))

    with robustness_tab:
        robustness = selected.get("robustness", {})
        grid = pd.DataFrame(robustness.get("window_horizon", []))
        robust_metrics = st.columns(3)
        robust_metrics[0].metric(
            "Sign consistency", metric_value(robustness.get("sign_consistency"))
        )
        robust_metrics[1].metric(
            "Group monotonicity", metric_value(robustness.get("group_monotonicity"))
        )
        robust_metrics[2].metric(
            "FDR 5%",
            str(robustness.get("multiple_testing", {}).get("reject_fdr_5pct", "not_run")),
        )
        if grid.empty:
            st.info("没有足够数据生成时间窗 × 预测周期稳健性网格。")
        else:
            st.subheader("RankIC：lookback window × horizon")
            st.dataframe(
                grid.pivot(index="lookback_days", columns="horizon", values="rank_ic"),
                use_container_width=True,
            )
        st.subheader("Block-bootstrap RankIC")
        st.json(robustness.get("bootstrap_rank_ic", {}))

    with provenance_tab:
        provenance = selected.get("provenance", {})
        st.write(provenance.get("description") or "暂无定义说明。")
        st.json(provenance)


def render_panel_report(selected: dict[str, Any]) -> None:
    st.caption(selected.get("research_note", ""))
    sample = selected.get("sample", {})
    st.write(
        f"频率：{selected.get('interval')} · "
        f"区间：{sample.get('start')} → {sample.get('end')} · "
        f"时点：{sample.get('n_periods')} · 中位币种数：{sample.get('median_assets')}"
    )
    metrics = selected.get("metrics", {})
    columns = st.columns(5)
    for column, key, label in zip(
        columns,
        ["ic", "rank_ic", "icir", "turnover", "mean_net_return"],
        ["IC", "RankIC", "ICIR", "Turnover", "Mean net"],
        strict=False,
    ):
        column.metric(label, metric_value(metrics.get(key)))
    decay = pd.DataFrame(selected.get("ic_decay", []))
    groups = pd.DataFrame(selected.get("group_returns", []))
    rolling = pd.DataFrame(selected.get("rolling_ic", []))
    left, right = st.columns(2)
    with left:
        st.subheader("IC decay")
        if not decay.empty:
            st.line_chart(decay.set_index("horizon")[["ic", "rank_ic"]])
        st.subheader("Rolling RankIC")
        if not rolling.empty:
            rolling["time"] = pd.to_datetime(rolling["time"], utc=True)
            st.line_chart(rolling.set_index("time")["value"])
    with right:
        st.subheader("Cross-sectional groups")
        if not groups.empty:
            st.bar_chart(groups.set_index("group")["mean_return"])
        st.subheader("Cost sensitivity")
        costs = pd.DataFrame(selected.get("cost_sensitivity", []))
        if not costs.empty:
            st.dataframe(costs, use_container_width=True, hide_index=True)
    st.json(selected.get("provenance", {}))


@st.fragment(run_every="30s")
def render_data_center() -> None:
    """Poll local coverage and offer an explicit bounded REST refresh."""
    st.caption("本页每 30 秒重新读取本地状态；所有联网操作仍需用户主动点击。")
    with st.expander("数据操作：刷新、扫描与高级范围", expanded=False):
        refresh_symbols = st.multiselect(
            "刷新币种",
            REFRESH_SYMBOLS,
            default=REFRESH_SYMBOLS[:3],
        )
        refresh_intervals = st.multiselect(
            "刷新频率",
            REFRESH_INTERVALS,
            default=("1m", "5m", "1h"),
        )
        lookback_hours = int(
            st.number_input(
                "向前覆盖小时数",
                min_value=1,
                max_value=720,
                value=24,
                help="重复区间会按时间键幂等合并，不会制造重复 K 线。",
            )
        )
        refresh_column, scan_column = st.columns(2)
        refresh_clicked = refresh_column.button(
            "联网刷新最新行情",
            type="primary",
            use_container_width=True,
        )
        scan_clicked = scan_column.button("仅扫描本地数据", use_container_width=True)
        if refresh_clicked:
            try:
                with st.spinner("正在拉取并合并最新数据……"):
                    result = refresh_recent_market_data(
                        PROJECT_ROOT / "data",
                        symbols=tuple(refresh_symbols),
                        intervals=tuple(refresh_intervals),
                        lookback_hours=lookback_hours,
                    )
                    build_data_catalog(
                        PROJECT_ROOT / "data",
                        REPORT_ROOT / "data_catalog.json",
                    )
                    st.cache_data.clear()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                if result["successful"]:
                    st.success(
                        f"成功刷新 {result['successful']} 组，"
                        f"失败 {result['failed']} 组。"
                    )
                else:
                    st.error("本次没有成功刷新任何数据，请查看错误详情。")
                if result["errors"]:
                    st.dataframe(pd.DataFrame(result["errors"]), hide_index=True)
        elif scan_clicked:
            build_data_catalog(
                PROJECT_ROOT / "data",
                REPORT_ROOT / "data_catalog.json",
            )
            st.cache_data.clear()
            st.success("本地数据目录已重新扫描。")

    catalog_path = REPORT_ROOT / "data_catalog.json"
    catalog = (
        json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog_path.exists()
        else {}
    )
    entries = pd.DataFrame(catalog.get("entries", []))
    if entries.empty:
        st.info("尚无 Silver 数据。可在上方刷新，或先运行下载脚本。")
        return
    health = load_data_health(str(catalog.get("generated_at") or "missing"))
    summary = health["summary"]
    overview = st.columns(6)
    overview[0].metric(
        "核心覆盖",
        f"{summary['core_covered_datasets']}/{summary['core_target_datasets']}",
    )
    overview[1].metric("研究可用", summary["research_ready_datasets"])
    overview[2].metric("ML 候选", summary["ml_eligible_datasets"])
    overview[3].metric("行情新鲜", summary["fresh_datasets"])
    overview[4].metric("缺失 K 线", summary["missing_bars"])
    overview[5].metric("衍生数据集", summary["derivative_datasets"])

    collection = health["collection"]
    if collection["status"] in {"partial", "failed", "running"}:
        st.warning(
            f"采集状态：{collection['status']} · "
            f"{collection.get('completed_task_count') or 0}/"
            f"{collection.get('expected_task_count') or 0} · "
            f"错误 {collection['error_count']}"
        )
    elif collection["status"] == "completed":
        st.success(f"最近一次 {collection.get('mode')} 采集已完成。")

    st.subheader("数据任务队列")
    data_tasks = pd.DataFrame(build_data_tasks(health))
    st.dataframe(data_tasks, use_container_width=True, hide_index=True)
    if bool(data_tasks["priority"].eq("P0").any()):
        st.warning("先完成 P0 数据任务；已有可研究口径仍可用于策略原型，但不能代表全口径完成。")
        st.code("python scripts/collect_real_data.py --execute", language="bash")
    else:
        st.info("核心数据没有 P0 阻塞项，可以进入时序策略或截面回测。")

    dataset_health = pd.DataFrame(health["datasets"])
    status_labels = {
        "ready": "可研究且新鲜",
        "historical_ready_stale": "历史可研究，待刷新",
        "limited_history": "历史不足",
        "gaps": "存在缺口",
    }
    dataset_health["status"] = dataset_health["health_status"].map(status_labels)
    dataset_health["coverage_pct"] = dataset_health["coverage_ratio"] * 100
    dataset_health["history_days"] = dataset_health["history_days"].round(1)
    dataset_health["age_hours"] = dataset_health["age_hours"].round(1)
    with st.expander("筛选数据集", expanded=False):
        filter_columns = st.columns(3)
        visible_symbols = filter_columns[0].multiselect(
            "币种",
            sorted(dataset_health["symbol"].unique()),
            default=sorted(dataset_health["symbol"].unique()),
            key="data_health_symbols",
        )
        visible_intervals = filter_columns[1].multiselect(
            "频率",
            sorted(dataset_health["display_interval"].unique()),
            default=sorted(dataset_health["display_interval"].unique()),
            key="data_health_intervals",
        )
        visible_statuses = filter_columns[2].multiselect(
            "状态",
            sorted(dataset_health["status"].dropna().unique()),
            default=sorted(dataset_health["status"].dropna().unique()),
            key="data_health_statuses",
        )
    visible_health = dataset_health.loc[
        dataset_health["symbol"].isin(visible_symbols)
        & dataset_health["display_interval"].isin(visible_intervals)
        & dataset_health["status"].isin(visible_statuses)
    ]
    coverage_tab, derivative_tab, gap_tab = st.tabs(
        ["覆盖与新鲜度", "Funding / OI / Basis", "缺口与口径说明"]
    )
    with coverage_tab:
        st.dataframe(
            visible_health[
                [
                    "symbol",
                    "display_interval",
                    "status",
                    "history_days",
                    "age_hours",
                    "rows",
                    "coverage_pct",
                    "missing_bars",
                    "ml_eligible",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with derivative_tab:
        derivatives = pd.DataFrame(health["derivatives"])
        if derivatives.empty:
            st.info("尚无 Funding Rate、Open Interest 或 Basis 文件；先执行 REST 补数。")
        else:
            derivatives["age_hours"] = derivatives["age_hours"].round(1)
            st.dataframe(derivatives, use_container_width=True, hide_index=True)
            st.caption("OI/Basis 受 Binance 约 30 天公开保留窗口限制；不会用未来值回填。")
    with gap_tab:
        gaps = dataset_health.loc[dataset_health["missing_bars"] > 0]
        if gaps.empty:
            st.success("当前目录未检测到内部时间缺口。")
        else:
            st.dataframe(
                gaps[["symbol", "display_interval", "missing_bars", "coverage_pct"]],
                use_container_width=True,
                hide_index=True,
            )
        if health["missing_scopes"]:
            st.write("核心研究口径尚未落盘：")
            st.dataframe(pd.DataFrame(health["missing_scopes"]), hide_index=True)
        st.caption(health["interpretation"]["warning"])
    st.caption(
        f"目录生成时间：{catalog.get('generated_at', 'unknown')}。"
        "刷新采用短窗口 REST 合并；它不是逐笔级实时行情。"
    )


st.set_page_config(page_title="Crypto Factor Lab", layout="wide")
st.title("Crypto Factor Lab")
st.caption("从数据准备 → 策略构建 → 稳健性验证；研究结果不等同于已验证 Alpha。")

page = st.sidebar.radio(
    "功能入口",
    [
        "开始使用",
        "数据中心",
        "时序策略",
        "多因子回测",
        "策略比较",
        "因子研究",
        "机器学习",
        "截面因子",
        "跨口径稳健性",
    ],
    key="page_selector",
)
st.sidebar.caption("推荐路径：开始使用 → 数据中心 → 策略构建 → 验证与比较")
st.sidebar.divider()
st.sidebar.caption("因子研究与机器学习保留为研究工具，本轮不继续扩展检验范围。")

header_catalog = load_json(REPORT_ROOT / "data_catalog.json")
header_entries = list(header_catalog.get("entries", []))
reports = load_reports(REPORT_ROOT / "single_factor")
header_health = load_data_health(str(header_catalog.get("generated_at") or "missing"))
workflow_steps = build_workflow_steps(
    header_health,
    factor_report_count=len(reports),
    has_strategy_result=bool(st.session_state.get("time_series_strategy_result")),
    has_robustness_result=bool(st.session_state.get("time_series_robustness")),
    has_walk_forward_result=bool(st.session_state.get("strategy_walk_forward")),
)
current_next_step = next_workflow_action(workflow_steps)
status_columns = st.columns(4)
header_summary = header_health.get("summary", {})
status_columns[0].metric(
    "核心流程",
    f"{workflow_progress(workflow_steps):.0%}",
    f"下一步：{current_next_step['label']}",
)
status_columns[1].metric(
    "核心数据覆盖",
    f"{header_summary.get('core_covered_datasets', 0)}/"
    f"{header_summary.get('core_target_datasets', 0)}",
    f"{header_summary.get('research_ready_datasets', 0)} 组可研究",
)
status_columns[2].metric(
    "可用策略入口",
    "单资产 + 多资产",
    f"{len(list_factors())} 单资产因子 · {len(list_panel_factors())} 截面因子",
)
status_columns[3].metric(
    "最近目录刷新",
    str(header_catalog.get("generated_at") or "尚未生成")[:19],
)

symbols = sorted({scope(item)[0] for item in reports})
frequencies = sorted({scope(item)[1] for item in reports})
categories = sorted(
    {str(item.get("provenance", {}).get("category") or "未分类") for item in reports}
)
if page == "因子研究":
    st.sidebar.subheader("研究筛选")
    factor_query = st.sidebar.text_input("搜索因子名称").strip().lower()
    selected_symbols = st.sidebar.multiselect("资产", symbols, default=symbols)
    selected_frequencies = st.sidebar.multiselect("频率", frequencies, default=frequencies)
    selected_categories = st.sidebar.multiselect("类别", categories, default=categories)
else:
    factor_query = ""
    selected_symbols = symbols
    selected_frequencies = frequencies
    selected_categories = categories
filtered = [
    item
    for item in reports
    if scope(item)[0] in selected_symbols
    and scope(item)[1] in selected_frequencies
    and str(item.get("provenance", {}).get("category") or "未分类") in selected_categories
    and factor_query in str(item.get("factor_name", "")).lower()
]

if page == "开始使用":
    st.header("研究工作台")
    st.write("页面会根据本地数据和当前会话结果，自动给出下一步操作。")
    render_workflow_overview(workflow_steps)

    st.subheader("快速入口")
    quick_columns = st.columns(3)
    with quick_columns[0].container(border=True):
        st.write("**单币种择时**")
        st.caption("选择多个因子、方向和权重，完成含成本时序回测。")
        st.button(
            "打开时序策略",
            key="quick_time_series",
            use_container_width=True,
            on_click=navigate_to,
            args=("时序策略",),
        )
    with quick_columns[1].container(border=True):
        st.write("**多币种选币**")
        st.caption("使用同频率多资产面板构建截面多因子组合。")
        st.button(
            "打开多因子回测",
            key="quick_panel_strategy",
            use_container_width=True,
            on_click=navigate_to,
            args=("多因子回测",),
        )
    with quick_columns[2].container(border=True):
        st.write("**已有结果比较**")
        st.caption("比较用户明确保存的策略快照，不自动混入临时结果。")
        st.button(
            "打开策略比较",
            key="quick_strategy_compare",
            use_container_width=True,
            on_click=navigate_to,
            args=("策略比较",),
        )

    p0_tasks = [row for row in build_data_tasks(header_health) if row["priority"] == "P0"]
    if p0_tasks:
        st.warning(
            f"数据中心仍有 {len(p0_tasks)} 项 P0 任务。"
            "已有可研究口径可以做原型，但全口径尚未收口。"
        )
    st.info(
        "本轮已将因子检验后置，重点完成数据任务可见性、策略入口和验证路径；"
        "机器学习推荐仍是候选，不会被展示成已验证 Alpha。"
    )
    with st.expander("状态口径说明", expanded=False):
        st.markdown(
            "- **阻塞**：前置数据或回测结果不存在。\n"
            "- **需处理**：已有部分能力可用，但数据覆盖或缺口尚未收口。\n"
            "- **可开始**：前置条件满足，可以进入当前步骤。\n"
            "- **已完成**：只表示流程产物已生成，不表示策略已通过六个月 OOS。"
        )

elif page == "因子研究":
    st.header("Factor Explorer")
    st.info("本轮暂停扩展因子检验；这里保留已有报告的浏览、筛选和追溯能力。")
    if not filtered:
        st.warning("当前筛选条件下没有报告。")
    else:
        summary_rows = []
        for item in filtered:
            symbol, frequency = scope(item)
            summary_rows.append(
                {
                    "symbol": symbol,
                    "frequency": frequency,
                    "factor": item.get("factor_name"),
                    "category": item.get("provenance", {}).get("category"),
                    "status": item.get("status"),
                    **item.get("metrics", {}),
                    "lookahead": item.get("lookahead_status"),
                    "fdr_5pct": item.get("robustness", {})
                    .get("multiple_testing", {})
                    .get("reject_fdr_5pct"),
                }
            )
        overview = st.columns(4)
        overview[0].metric("Reports", len(filtered))
        overview[1].metric(
            "Computed", sum(item.get("status") != "insufficient_data" for item in filtered)
        )
        overview[2].metric(
            "Lookahead pass", sum(item.get("lookahead_status") == "pass" for item in filtered)
        )
        overview[3].metric("FDR 5%", sum(row["fdr_5pct"] is True for row in summary_rows))
        with st.expander("因子比较表", expanded=False):
            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
            )
        labels = [
            f"{scope(item)[0]} · {scope(item)[1]} · {item.get('factor_name')}"
            for item in filtered
        ]
        choice = st.selectbox("选择报告", range(len(labels)), format_func=labels.__getitem__)
        render_single_report(filtered[choice])

elif page == "截面因子":
    st.header("Panel Factors")
    panel_reports = load_reports(REPORT_ROOT / "panel_factor")
    if not panel_reports:
        st.info("尚无截面报告。至少准备 3 个同频率币种后运行研究入口。")
    else:
        panel_rows = [
            {
                "factor": item.get("factor_name"),
                "interval": item.get("interval"),
                "status": item.get("status"),
                **item.get("metrics", {}),
                "lookahead": item.get("lookahead_status"),
            }
            for item in panel_reports
        ]
        st.dataframe(pd.DataFrame(panel_rows), use_container_width=True, hide_index=True)
        labels = [
            f"{item.get('interval')} · {item.get('factor_name')}" for item in panel_reports
        ]
        choice = st.selectbox(
            "Panel factor report",
            range(len(labels)),
            format_func=labels.__getitem__,
        )
        render_panel_report(panel_reports[choice])

elif page == "跨口径稳健性":
    st.header("跨资产 / 跨频率")
    cross_rows = load_summary().get("cross_frequency", [])
    cross_frame = pd.DataFrame(cross_rows)
    if cross_frame.empty:
        st.info("重新运行研究脚本后，将生成跨资产/跨频率稳定性汇总。")
    else:
        st.subheader("跨口径稳定性概览")
        st.caption("方向一致性只用于筛选稳定候选，不代表完成独立 OOS。")
        st.dataframe(cross_frame, use_container_width=True, hide_index=True)
        chart = cross_frame.dropna(subset=["median_rank_ic"]).set_index("factor_name")
        if not chart.empty:
            st.bar_chart(chart["median_rank_ic"])

elif page == "时序策略":
    st.header("Single-Asset Strategy Builder")
    st.caption("1 选择数据 → 2 组合因子 → 3 设置区间 → 4 运行回测 → 5 查看验证。")
    catalog = load_json(REPORT_ROOT / "data_catalog.json")
    health = load_data_health(str(catalog.get("generated_at") or "missing"))
    ready_scopes = sorted(
        (str(entry["symbol"]), str(entry["interval"]))
        for entry in health["datasets"]
        if entry["research_ready"]
    )
    if not ready_scopes:
        st.info("没有达到研究门槛的数据集；请先在 Data Center 补数。")
    else:
        preset = st.session_state.get("time_series_preset", {})
        preset_scope = (str(preset.get("symbol")), str(preset.get("interval")))
        default_scope = ready_scopes.index(preset_scope) if preset_scope in ready_scopes else 0
        scope_labels = [
            f"{symbol} · {'24h' if interval == '1d' else interval}"
            for symbol, interval in ready_scopes
        ]
        selected_scope = st.selectbox(
            "数据口径",
            range(len(scope_labels)),
            index=default_scope,
            format_func=scope_labels.__getitem__,
            help="只列出至少 14 天历史且覆盖率不低于 98% 的数据集。",
        )
        ts_symbol, ts_interval = ready_scopes[selected_scope]
        matching_entry = next(
            entry
            for entry in health["datasets"]
            if (str(entry["symbol"]), str(entry["interval"]))
            == (ts_symbol, ts_interval)
        )
        scope_columns = st.columns(4)
        scope_columns[0].metric("资产", ts_symbol)
        scope_columns[1].metric("频率", "24h" if ts_interval == "1d" else ts_interval)
        scope_columns[2].metric("历史天数", f"{matching_entry['history_days']:.0f}")
        scope_columns[3].metric("覆盖率", f"{matching_entry['coverage_ratio']:.2%}")
        preset_allocations = {
            str(row["name"]): row for row in preset.get("allocations", [])
        }
        factor_options = sorted(
            name for name in list_factors() if not name.startswith("qlib360_")
        )
        default_names = [name for name in preset_allocations if name in factor_options]
        if not default_names:
            default_names = [
                name
                for name in ("return_momentum", "rsi", "realized_volatility")
                if name in factor_options
            ]
        selected_factors = st.multiselect(
            "因子",
            factor_options,
            default=default_names,
            key="ts_selected_factors",
            help="先使用 2–5 个具有不同经济含义的因子；因子越多不代表策略越稳健。",
        )
        ts_allocations = []
        with st.expander("配置因子权重与方向", expanded=bool(selected_factors)):
            if not selected_factors:
                st.info("至少选择一个因子后才能运行策略。")
            for factor_name in selected_factors:
                seed = preset_allocations.get(factor_name, {})
                weight_column, direction_column = st.columns(2)
                weight = weight_column.number_input(
                    f"{factor_name} 权重",
                    min_value=0.01,
                    value=float(seed.get("weight", 1.0)),
                    key=f"ts_weight_{factor_name}",
                )
                default_direction = 0 if int(seed.get("direction", 1)) == 1 else 1
                direction_label = direction_column.selectbox(
                    f"{factor_name} 方向",
                    ["正向", "反向"],
                    index=default_direction,
                    key=f"ts_direction_{factor_name}",
                )
                ts_direction: Literal[-1, 1] = 1 if direction_label == "正向" else -1
                ts_allocations.append(
                    TimeSeriesAllocation(
                        factor_name,
                        weight=float(weight),
                        direction=ts_direction,
                        params=seed.get("params"),
                    )
                )
        period_choice = st.selectbox(
            "回测区间",
            ["近30天", "近90天", "近180天", "全部", "自定义"],
            key="ts_period",
        )
        ts_start: str | None = None
        ts_end: str | None = None
        if period_choice == "自定义":
            ts_start = st.text_input("开始时间（UTC，可选）", key="ts_start") or None
            ts_end = st.text_input("结束时间（UTC，可选）", key="ts_end") or None
        elif period_choice != "全部":
            days = {"近30天": 30, "近90天": 90, "近180天": 180}[period_choice]
            latest = pd.Timestamp(matching_entry["end"])
            ts_start = (latest - pd.Timedelta(days=days)).isoformat()
            ts_end = latest.isoformat()
            st.caption(f"UTC：{ts_start} → {ts_end}")
        with st.expander("高级执行参数", expanded=False):
            settings = st.columns(3)
            ts_mode = settings[0].selectbox(
                "持仓模式",
                ["long_short", "long_only"],
                help="long_short 可做多、做空或空仓；long_only 只做多或空仓。",
            )
            ts_rebalance = int(
                settings[1].number_input(
                    "每 N 根调仓", min_value=1, value=1, help="数值越大，交易越不频繁。"
                )
            )
            ts_window = int(
                settings[2].number_input(
                    "标准化窗口", min_value=5, value=168, help="仅使用过去窗口计算 z-score。"
                )
            )
            cost_settings = st.columns(3)
            ts_threshold = float(
                cost_settings[0].number_input(
                    "开仓阈值", min_value=0.0, value=0.25, help="提高阈值会减少弱信号交易。"
                )
            )
            ts_fee = float(
                cost_settings[1].number_input(
                    "单边手续费", min_value=0.0, value=0.001, format="%.5f"
                )
            )
            ts_slippage = float(
                cost_settings[2].number_input(
                    "单边滑点", min_value=0.0, value=0.0005, format="%.5f"
                )
            )
        if preset.get("status") == "ready":
            st.info("当前因子和权重来自 ML 训练折推荐；它们仍只是候选预设。")
        if st.button("运行单资产多因子回测", type="primary"):
            try:
                with st.status("正在运行策略研究流程……", expanded=True) as status:
                    status.write("读取所选 Silver 数据并裁剪研究区间。")
                    silver_path = next(
                        path
                        for path in (PROJECT_ROOT / "data").glob(
                            "silver/**/klines.parquet"
                        )
                        if ts_symbol in path.parts and ts_interval in path.parts
                    )
                    silver = pd.read_parquet(silver_path)
                    frame = silver_to_factor_input(silver)
                    if ts_start:
                        frame = frame.loc[frame.index >= pd.Timestamp(ts_start)]
                    if ts_end:
                        frame = frame.loc[frame.index <= pd.Timestamp(ts_end)]
                    status.write(f"生成 {len(ts_allocations)} 个因子输入并应用一根 K 线执行滞后。")
                    ts_position_mode: Literal["long_short", "long_only"] = (
                        "long_short" if ts_mode == "long_short" else "long_only"
                    )
                    ts_config = TimeSeriesStrategyConfig(
                        allocations=tuple(ts_allocations),
                        interval=ts_interval,
                        rebalance_every=ts_rebalance,
                        position_mode=ts_position_mode,
                        score_threshold=ts_threshold,
                        standardize_window=ts_window,
                        fee_rate=ts_fee,
                        slippage=ts_slippage,
                    )
                    st.session_state["time_series_strategy_context"] = {
                        "frame": frame,
                        "config": ts_config,
                        "symbol": ts_symbol,
                        "interval": ts_interval,
                    }
                    st.session_state.pop("parameter_perturbation", None)
                    st.session_state.pop("strategy_walk_forward", None)
                    st.session_state["time_series_strategy_result"] = (
                        run_time_series_strategy(frame, ts_config)
                    )
                    status.update(
                        label="策略回测已完成；继续向下查看稳健性与 Walk-Forward。",
                        state="complete",
                        expanded=False,
                    )
            except (KeyError, OSError, StopIteration, ValueError) as exc:
                st.error(str(exc))
        time_series_result = st.session_state.get("time_series_strategy_result")
        if time_series_result:
            render_strategy_result(time_series_result, key_prefix="time_series")
            render_time_series_robustness(time_series_result)
            if time_series_result.get("returns"):
                render_strategy_store(
                    time_series_result,
                    st.session_state.get("time_series_robustness", {}),
                    symbol=ts_symbol,
                    interval=ts_interval,
                )

elif page == "策略比较":
    render_saved_strategy_comparison()

elif page == "多因子回测":
    st.header("Cross-Sectional Strategy Builder")
    st.caption("1 选择同频率币种 → 2 选择截面因子 → 3 设置持仓与成本 → 4 回测。")
    catalog = load_json(REPORT_ROOT / "data_catalog.json")
    entries = list(catalog.get("entries", []))
    strategy_health = load_data_health(str(catalog.get("generated_at") or "missing"))
    research_ready_scopes = {
        (str(entry["symbol"]), str(entry["interval"]))
        for entry in strategy_health["datasets"]
        if entry["research_ready"]
    }
    interval_options = sorted({interval for _, interval in research_ready_scopes})
    if not interval_options:
        st.info("没有达到研究门槛的数据集：至少需要 14 天历史且覆盖率不低于 98%。")
    else:
        strategy_interval = st.selectbox(
            "数据频率",
            interval_options,
            format_func=lambda value: "24h" if value == "1d" else value,
        )
        available_symbols = sorted(
            {
                symbol
                for symbol, interval in research_ready_scopes
                if interval == strategy_interval
            }
        )
        st.caption(
            "这里只展示通过基础覆盖门槛的数据；最终是否有效仍以成本后稳健性和 OOS 为准。"
        )
        strategy_symbols = st.multiselect(
            "币种",
            available_symbols,
            default=available_symbols,
        )
        factor_options = sorted(list_panel_factors())
        default_factors = factor_options[: min(2, len(factor_options))]
        selected_factors = st.multiselect("因子", factor_options, default=default_factors)
        allocations = []
        for factor_name in selected_factors:
            weight_column, direction_column = st.columns(2)
            weight = weight_column.number_input(
                f"{factor_name} 权重", min_value=0.01, value=1.0, key=f"weight_{factor_name}"
            )
            direction_label = direction_column.selectbox(
                f"{factor_name} 方向",
                ["正向", "反向"],
                key=f"direction_{factor_name}",
            )
            factor_direction: Literal[-1, 1] = 1 if direction_label == "正向" else -1
            allocations.append(
                FactorAllocation(
                    factor_name,
                    weight=float(weight),
                    direction=factor_direction,
                )
            )
        custom_expression = st.text_input("自定义安全表达式（可选）")
        if custom_expression.strip():
            custom_weight_column, custom_direction_column = st.columns(2)
            custom_weight = custom_weight_column.number_input(
                "自定义表达式权重",
                min_value=0.01,
                value=1.0,
                key="weight_custom_expression",
            )
            custom_direction_label = custom_direction_column.selectbox(
                "自定义表达式方向",
                ["正向", "反向"],
                key="direction_custom_expression",
            )
            custom_direction: Literal[-1, 1] = (
                1 if custom_direction_label == "正向" else -1
            )
            allocations.append(
                FactorAllocation(
                    "custom_expression",
                    weight=float(custom_weight),
                    direction=custom_direction,
                    expression=custom_expression.strip(),
                )
            )
        period_choice = st.selectbox(
            "回测区间",
            ["近30天", "近7天", "近90天", "全部", "自定义"],
        )
        start_text: str | None = None
        end_text: str | None = None
        if period_choice == "自定义":
            start_text = st.text_input("开始时间（UTC，可选）") or None
            end_text = st.text_input("结束时间（UTC，可选）") or None
        elif period_choice != "全部":
            period_days = {"近7天": 7, "近30天": 30, "近90天": 90}[period_choice]
            selected_entries = [
                entry
                for entry in entries
                if str(entry.get("interval")) == strategy_interval
                and str(entry.get("symbol")) in strategy_symbols
            ]
            if selected_entries:
                latest = max(pd.Timestamp(entry["end"]) for entry in selected_entries)
                start_text = (latest - pd.Timedelta(days=period_days)).isoformat()
                end_text = latest.isoformat()
                st.caption(f"UTC：{start_text} → {end_text}")
        mode = st.selectbox("持仓模式", ["long_short", "long_only"])
        horizon = st.number_input("未来收益周期（K线根数）", min_value=1, value=1)
        rebalance = st.number_input("每 N 根 K 线调仓", min_value=1, value=1)
        fee = st.number_input("单边手续费", min_value=0.0, value=0.001, format="%.5f")
        slippage = st.number_input("单边滑点", min_value=0.0, value=0.0005, format="%.5f")
        if st.button("运行多因子回测", type="primary"):
            try:
                if len(strategy_symbols) < 3:
                    raise ValueError("截面回测至少选择 3 个同频率币种")
                panel = load_silver_panel(
                    PROJECT_ROOT / "data",
                    interval=strategy_interval,
                    symbols=tuple(strategy_symbols),
                    start=start_text or None,
                    end=end_text or None,
                )
                position_mode: Literal["long_short", "long_only"] = (
                    "long_short" if mode == "long_short" else "long_only"
                )
                config = PanelStrategyConfig(
                    allocations=tuple(allocations),
                    interval=strategy_interval,
                    horizon=int(horizon),
                    rebalance_every=int(rebalance),
                    position_mode=position_mode,
                    fee_rate=float(fee),
                    slippage=float(slippage),
                )
                st.session_state["strategy_result"] = run_panel_strategy(panel, config)
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))
        strategy_result = st.session_state.get("strategy_result")
        if strategy_result:
            metrics = strategy_result["metrics"]
            metric_columns = st.columns(5)
            for column, key, label in zip(
                metric_columns,
                ["total_return", "mean_bar_return", "bar_sharpe", "max_drawdown", "mean_turnover"],
                ["Total return", "Mean/bar", "Bar Sharpe", "Max drawdown", "Turnover"],
                strict=False,
            ):
                column.metric(label, metric_value(metrics.get(key)))
            strategy_frame = pd.DataFrame(strategy_result.get("returns", []))
            if not strategy_frame.empty:
                strategy_frame["time"] = pd.to_datetime(strategy_frame["time"], utc=True)
                st.line_chart(strategy_frame.set_index("time")[["equity", "drawdown"]])
            st.caption(strategy_result["research_note"])
            st.caption("信号在 t 时点形成，并只对应 t→t+h 的未来收益；行情缺口不会回填。")
            st.json(strategy_result["config"])

elif page == "数据中心":
    st.header("数据中心")
    st.caption("先处理 P0 缺口，再决定是否刷新近端行情或补充衍生品特征。")
    render_data_center()

elif page == "机器学习":
    st.header("ML Factor Mining")
    st.caption(
        "训练折内自动筛选、去相关和标准化；测试折只用于严格样本外评估。"
    )
    catalog = load_json(REPORT_ROOT / "data_catalog.json")
    ml_entries = list(catalog.get("entries", []))
    if not ml_entries:
        st.warning("尚无可识别的 Silver 数据，请先到数据中心扫描或刷新。")
    else:
        ml_health = load_data_health(str(catalog.get("generated_at") or "missing"))
        eligible_scopes = {
            (str(entry["symbol"]), str(entry["interval"]))
            for entry in ml_health["datasets"]
            if entry["ml_eligible"]
        }
        available_scopes = sorted(
            {
                (str(entry.get("symbol")), str(entry.get("interval")))
                for entry in ml_entries
                if (str(entry.get("symbol")), str(entry.get("interval")))
                in eligible_scopes
            }
        )
        if not available_scopes:
            st.warning(
                "当前没有达到 ML 门槛的数据集：需要至少 120 天历史且覆盖率不低于 98%。"
            )
        else:
            scope_labels = [
                f"{symbol} · {'24h' if interval == '1d' else interval}"
                for symbol, interval in available_scopes
            ]
            selected_scope_index = st.selectbox(
                "训练数据",
                range(len(scope_labels)),
                format_func=scope_labels.__getitem__,
            )
            ml_symbol, ml_interval = available_scopes[selected_scope_index]
            selected_health = next(
                entry
                for entry in ml_health["datasets"]
                if (entry["symbol"], entry["interval"]) == (ml_symbol, ml_interval)
            )
            readiness = st.columns(3)
            readiness[0].metric("历史天数", f"{selected_health['history_days']:.0f}")
            readiness[1].metric("覆盖率", f"{selected_health['coverage_ratio']:.2%}")
            readiness[2].metric(
                "行情状态", "新鲜" if selected_health["fresh"] else "历史样本"
            )
            setting_columns = st.columns(4)
            ml_horizon = int(
                setting_columns[0].number_input("预测周期", min_value=1, value=1)
            )
            ml_train_days = int(
                setting_columns[1].number_input("最少训练天数", min_value=30, value=90)
            )
            ml_test_days = int(
                setting_columns[2].number_input("每折测试天数", min_value=1, value=14)
            )
            ml_max_features = int(
                setting_columns[3].number_input("最多入选特征", min_value=5, value=40)
            )
            if st.button("自动筛选并执行 OOS 回测", type="primary"):
                try:
                    with st.status(
                        "正在执行机器学习研究流程……", expanded=True
                    ) as status:
                        status.write("生成候选特征，并仅在训练折内筛选和去相关。")
                        run_ml_factor_research(
                            PROJECT_ROOT / "data",
                            REPORT_ROOT,
                            symbols=(ml_symbol,),
                            intervals=(ml_interval,),
                            horizon=ml_horizon,
                            min_train_days=ml_train_days,
                            test_days=ml_test_days,
                            max_features=ml_max_features,
                        )
                        status.write("汇总严格 OOS 预测、成本后回测和推荐稳定性。")
                        status.update(
                            label="机器学习候选与 OOS 报告已生成。",
                            state="complete",
                            expanded=False,
                        )
                    load_reports.clear()
                    st.success("机器学习筛选与严格样本外回测已完成。")
                except (KeyError, OSError, RuntimeError, ValueError) as exc:
                    st.error(str(exc))

    ml_reports = load_reports(REPORT_ROOT / "ml_factor")
    if not ml_reports:
        st.info("尚无 ML 报告，可在上方选择数据后运行。")
    else:
        labels = [
            f"{scope(item)[0]} · {scope(item)[1]} · Ridge composite"
            for item in ml_reports
        ]
        choice = st.selectbox("ML 报告", range(len(labels)), format_func=labels.__getitem__)
        selected_ml = ml_reports[choice]
        render_single_report(selected_ml)
        st.subheader("模型推荐特征")
        recommendations = pd.DataFrame(
            selected_ml.get("feature_recommendations", [])
        )
        if recommendations.empty:
            st.info("旧报告没有推荐字段，请重新运行上方 ML 流程。")
        else:
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
            st.caption(
                "推荐分数仅来自训练折的入选频率、权重和方向稳定性，"
                "不会读取测试折收益。"
            )
            if st.button("用前 5 个推荐因子创建时序策略"):
                preset = build_ml_strategy_preset(
                    selected_ml.get("feature_recommendations", []), top_n=5
                )
                if preset["status"] != "ready":
                    st.warning("当前推荐无法映射到正式因子 registry。")
                else:
                    apply_ml_preset(selected_ml)
                    st.rerun()

        st.subheader("严格样本外策略回测")
        strategy_backtest = selected_ml.get("strategy_backtest", {})
        strategy_metrics = strategy_backtest.get("metrics", {})
        if not strategy_metrics:
            st.info("旧报告没有策略回测字段，请重新运行上方 ML 流程。")
        else:
            metric_columns = st.columns(5)
            for column, key, label in zip(
                metric_columns,
                [
                    "total_return",
                    "bar_sharpe",
                    "max_drawdown",
                    "mean_turnover",
                    "directional_accuracy",
                ],
                ["Total return", "Bar Sharpe", "Max drawdown", "Turnover", "Hit rate"],
                strict=True,
            ):
                column.metric(label, metric_value(strategy_metrics.get(key)))
            strategy_frame = pd.DataFrame(strategy_backtest.get("returns", []))
            if not strategy_frame.empty:
                strategy_frame["time"] = pd.to_datetime(strategy_frame["time"], utc=True)
                st.line_chart(strategy_frame.set_index("time")[["equity", "drawdown"]])
            st.caption(strategy_backtest.get("research_note", ""))
        st.subheader("Walk-forward folds")
        folds = pd.DataFrame(selected_ml.get("folds", []))
        if not folds.empty:
            st.dataframe(
                folds.drop(columns=["weights"], errors="ignore"),
                use_container_width=True,
            )
        st.subheader("Leakage controls")
        st.json(selected_ml.get("leakage_controls", {}))
