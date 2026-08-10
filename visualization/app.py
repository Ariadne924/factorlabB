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

labels = [f"{item.get('factor_name')} · {item.get('_file')}" for item in reports]
selected = reports[st.selectbox("Factor", range(len(labels)), format_func=labels.__getitem__)]
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

left, right = st.columns(2)
decay = pd.DataFrame(selected.get("ic_decay", []))
groups = pd.DataFrame(selected.get("group_returns", []))
rolling = pd.DataFrame(selected.get("rolling_ic", []))
with left:
    st.subheader("IC decay")
    st.line_chart(decay.set_index("lag")["rank_ic"] if not decay.empty else pd.Series(dtype=float))
    st.subheader("Rolling IC")
    if not rolling.empty:
        rolling["time"] = pd.to_datetime(rolling["time"], utc=True, errors="coerce")
        st.line_chart(rolling.set_index("time")["value"])
with right:
    st.subheader("Group returns")
    st.bar_chart(
        groups.set_index("group")["mean_return"] if not groups.empty else pd.Series(dtype=float)
    )
    st.subheader("Cost assumptions")
    st.json(selected.get("cost_assumptions", {}))
