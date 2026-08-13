from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from visualization.navigation import (
    FACTOR_VALIDATION_TABS,
    LEGACY_PAGE_REDIRECTS,
    PRIMARY_PAGES,
    STRATEGY_RESEARCH_VIEWS,
    apply_pending_navigation,
    queue_navigation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_factor_validation_has_one_primary_entry() -> None:
    assert "因子检验" in PRIMARY_PAGES
    assert "因子研究" not in PRIMARY_PAGES
    assert "截面因子" not in PRIMARY_PAGES
    assert "跨口径稳健性" not in PRIMARY_PAGES


def test_realtime_market_has_a_primary_entry() -> None:
    assert "实时行情" in PRIMARY_PAGES


def test_navigation_is_queued_then_applied_before_widget_creation() -> None:
    state: dict[str, object] = {"page_selector": "机器学习"}

    queue_navigation(state, "策略研究", "单币种择时")

    assert state["page_selector"] == "机器学习"
    assert state["_pending_page_selector"] == "策略研究"
    apply_pending_navigation(state)
    assert state["page_selector"] == "策略研究"
    assert state["strategy_research_view"] == "单币种择时"
    assert "_pending_page_selector" not in state


def test_strategy_research_has_one_primary_entry() -> None:
    assert "策略研究" in PRIMARY_PAGES
    assert "时序策略" not in PRIMARY_PAGES
    assert "多因子回测" not in PRIMARY_PAGES
    assert "策略比较" not in PRIMARY_PAGES
    assert STRATEGY_RESEARCH_VIEWS == (
        "单币种择时",
        "多币种选币",
        "结果比较",
    )
    assert LEGACY_PAGE_REDIRECTS["时序策略"] == ("策略研究", "单币种择时")
    assert LEGACY_PAGE_REDIRECTS["多因子回测"] == ("策略研究", "多币种选币")
    assert LEGACY_PAGE_REDIRECTS["策略比较"] == ("策略研究", "结果比较")


def test_factor_validation_keeps_all_three_research_views() -> None:
    assert FACTOR_VALIDATION_TABS == (
        "单资产因子",
        "截面因子",
        "跨资产 / 跨频率",
    )


def test_app_exposes_one_factor_validation_entry_with_three_tabs() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "visualization" / "app.py").run(timeout=60)
    assert not app.exception
    assert tuple(app.sidebar.radio[0].options) == PRIMARY_PAGES

    app.sidebar.radio[0].set_value("因子检验").run(timeout=60)

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert all(tab_labels.count(label) == 1 for label in FACTOR_VALIDATION_TABS)
    assert [tab_labels.index(label) for label in FACTOR_VALIDATION_TABS] == sorted(
        tab_labels.index(label) for label in FACTOR_VALIDATION_TABS
    )


def test_app_exposes_one_strategy_entry_with_three_views() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "visualization" / "app.py").run(timeout=60)
    app.sidebar.radio[0].set_value("策略研究").run(timeout=60)

    assert not app.exception
    strategy_control = next(control for control in app.radio if control.label == "研究模式")
    assert tuple(strategy_control.options) == STRATEGY_RESEARCH_VIEWS

    expected_subheaders = {
        "单币种择时": "单币种择时",
        "多币种选币": "多币种选币",
        "结果比较": "结果比较",
    }
    for view, subheader in expected_subheaders.items():
        strategy_control.set_value(view).run(timeout=60)
        assert not app.exception
        assert subheader in [item.value for item in app.subheader]
        strategy_control = next(
            control for control in app.radio if control.label == "研究模式"
        )


def test_app_exposes_realtime_market_entry() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "visualization" / "app.py").run(timeout=60)
    app.sidebar.radio[0].set_value("实时行情").run(timeout=60)

    assert not app.exception
    assert "实时行情" in [item.value for item in app.header]
    assert "采集状态" in [item.label for item in app.metric]
