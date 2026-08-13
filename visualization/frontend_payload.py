"""将研究摘要转换为稳定、版本化的前端数据契约。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FRONTEND_CONTRACT_VERSION = "1.2"


def build_frontend_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """构建不依赖 Streamlit 的前端只读数据结构。"""
    results = list(summary.get("results", []))
    scopes = sorted(
        {
            (str(row.get("symbol")), str(row.get("display_frequency")))
            for row in results
            if row.get("symbol") and row.get("display_frequency")
        }
    )
    categories = sorted(
        {
            str(row.get("category"))
            for row in results
            if row.get("category") is not None
        }
    )
    sources = sorted(
        {str(row.get("source")) for row in results if row.get("source") is not None}
    )
    return {
        "contract_version": FRONTEND_CONTRACT_VERSION,
        "status": summary.get("status"),
        "disclaimer": summary.get("research_note"),
        "overview": {
            "factor_count": summary.get("factor_count", 0),
            "computed_report_count": summary.get("computed_report_count", 0),
            "fdr_5pct_pass_count": summary.get("fdr_5pct_pass_count", 0),
        },
        "filters": {
            "symbols": sorted({symbol for symbol, _ in scopes}),
            "frequencies": sorted({frequency for _, frequency in scopes}),
            "categories": categories,
            "sources": sources,
        },
        "factor_results": results,
        "panel_factor_results": list(summary.get("panel_results", [])),
        "data_catalog": summary.get("data_catalog"),
        "data_health": {
            "path": summary.get("data_health"),
            "summary": summary.get("data_health_summary", {}),
        },
        "cross_frequency": list(summary.get("cross_frequency", [])),
    }


def write_frontend_payload(summary: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_frontend_payload(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
