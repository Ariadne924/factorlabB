"""前端导航配置。"""

from __future__ import annotations

from typing import Any, Protocol


class NavigationState(Protocol):
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def pop(self, key: str) -> Any: ...

PRIMARY_PAGES = (
    "开始使用",
    "实时行情",
    "数据中心",
    "策略研究",
    "因子检验",
    "机器学习",
)

STRATEGY_RESEARCH_VIEWS = (
    "策略模板",
    "单币种择时",
    "多币种选币",
    "结果比较",
)

LEGACY_PAGE_REDIRECTS = {
    "时序策略": ("策略研究", "单币种择时"),
    "多因子回测": ("策略研究", "多币种选币"),
    "策略比较": ("策略研究", "结果比较"),
    "因子研究": ("因子检验", None),
    "截面因子": ("因子检验", None),
    "跨口径稳健性": ("因子检验", None),
}

FACTOR_VALIDATION_TABS = (
    "单资产因子",
    "截面因子",
    "跨资产 / 跨频率",
)


def queue_navigation(
    state: NavigationState,
    page_name: str,
    strategy_view: str | None = None,
) -> None:
    """暂存跳转目标，避免在导航 widget 创建后直接修改它的 key。"""
    state["_pending_page_selector"] = page_name
    if strategy_view is not None:
        state["_pending_strategy_research_view"] = strategy_view


def apply_pending_navigation(state: NavigationState) -> None:
    """必须在创建 page_selector/strategy widget 之前应用待跳转目标。"""
    if "_pending_page_selector" in state:
        state["page_selector"] = state.pop("_pending_page_selector")
    if "_pending_strategy_research_view" in state:
        state["strategy_research_view"] = state.pop(
            "_pending_strategy_research_view"
        )
