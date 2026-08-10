"""Streamlit Factor Explorer / Single Factor Report。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "single_factor"


@st.cache_data
def load_reports() -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for path in sorted(REPORT_DIR.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_file"] = path.name
        reports.append(report)
    return reports


st.set_page_config(page_title="Factor Explorer", layout="wide")
st.title("Factor Explorer / Single Factor Report")
reports = load_reports()
if not reports:
    st.info("尚无报告。先运行：python scripts/run_all_research.py")
    st.stop()

categories = sorted(
    {str(item.get("provenance", {}).get("category") or "未分类") for item in reports}
)
sources = sorted({str(item.get("provenance", {}).get("source") or "project") for item in reports})
status_options = sorted({str(item.get("status")) for item in reports})
category = st.sidebar.selectbox("Category", ["全部", *categories])
source = st.sidebar.selectbox("Source", ["全部", *sources])
status = st.sidebar.selectbox("Status", ["全部", *status_options])
filtered = [
    item
    for item in reports
    if (category == "全部" or item.get("provenance", {}).get("category") == category)
    and (source == "全部" or item.get("provenance", {}).get("source") == source)
    and (status == "全部" or item.get("status") == status)
]
if not filtered:
    st.warning("当前筛选条件下没有报告。")
    st.stop()

summary_rows = [
    {
        "factor": item.get("factor_name"),
        "category": item.get("provenance", {}).get("category"),
        "source": item.get("provenance", {}).get("source"),
        "status": item.get("status"),
        **item.get("metrics", {}),
        "sign_consistency": item.get("robustness", {}).get("sign_consistency"),
        "fdr_reject": item.get("robustness", {})
        .get("multiple_testing", {})
        .get("reject_fdr_5pct"),
    }
    for item in filtered
]
st.subheader("Research overview")
overview_columns = st.columns(4)
overview_columns[0].metric("Reports", len(filtered))
overview_columns[1].metric(
    "Computed", sum(item.get("status") != "insufficient_data" for item in filtered)
)
overview_columns[2].metric(
    "Lookahead pass", sum(item.get("lookahead_status") == "pass" for item in filtered)
)
overview_columns[3].metric(
    "FDR 5%", sum(row.get("fdr_reject") is True for row in summary_rows)
)
with st.expander("Factor comparison", expanded=False):
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

labels = [f"{item.get('factor_name')} · {item.get('_file')}" for item in filtered]
selected = filtered[st.selectbox("Factor", range(len(labels)), format_func=labels.__getitem__)]
st.caption(selected.get("research_note", ""))

sample = selected.get("sample", {})
st.write(
    f"样本：{sample.get('start')} → {sample.get('end')} · "
    f"频率：{sample.get('frequency')} · N={sample.get('n_obs')}"
)
metrics = selected.get("metrics", {})
columns = st.columns(5)
for column, key, label in zip(
    columns,
    ["ic", "rank_ic", "icir", "turnover"],
    ["IC", "RankIC", "ICIR", "Turnover"],
    strict=False,
):
    value = metrics.get(key)
    column.metric(label, "N/A" if value is None else f"{value:.4f}")
columns[-1].metric("Lookahead", str(selected.get("lookahead_status", "not_run")))

overview_tab, robustness_tab, provenance_tab = st.tabs(
    ["Single Factor Report", "Robustness", "Definition & Data"]
)
decay = pd.DataFrame(selected.get("ic_decay", []))
groups = pd.DataFrame(selected.get("group_returns", []))
rolling = pd.DataFrame(selected.get("rolling_ic", []))
with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("IC decay")
        st.line_chart(
            decay.set_index("lag")["rank_ic"] if not decay.empty else pd.Series(dtype=float)
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
    consistency = robustness.get("sign_consistency")
    monotonicity = robustness.get("group_monotonicity")
    fdr = robustness.get("multiple_testing", {})
    robust_metrics[0].metric(
        "Sign consistency", "N/A" if consistency is None else f"{consistency:.1%}"
    )
    robust_metrics[1].metric(
        "Group monotonicity", "N/A" if monotonicity is None else f"{monotonicity:.3f}"
    )
    robust_metrics[2].metric("FDR 5%", str(fdr.get("reject_fdr_5pct", "not_run")))
    if grid.empty:
        st.info("没有足够数据生成时间窗 × 预测周期稳健性网格。")
    else:
        st.subheader("RankIC: lookback window × horizon")
        st.dataframe(
            grid.pivot(index="lookback_days", columns="horizon", values="rank_ic"),
            width="stretch",
        )
        st.caption("行是回看自然日，列是未来 K 线周期；空值表示样本不足。")
    st.subheader("Block-bootstrap RankIC")
    st.json(robustness.get("bootstrap_rank_ic", {}))

with provenance_tab:
    provenance = selected.get("provenance", {})
    st.write(provenance.get("description") or "暂无定义说明。")
    source_url = provenance.get("source_url")
    if source_url:
        st.link_button(str(provenance.get("source") or "Source"), str(source_url))
    st.json(provenance)
