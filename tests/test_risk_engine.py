from __future__ import annotations

import pytest

from trading.risk import RiskEngine, RiskLimits, RiskState


def state(**changes: float | None) -> RiskState:
    values: dict[str, float | None] = {
        "equity": 100.0,
        "peak_equity": 100.0,
        "requested_position_fraction": 0.1,
        "requested_leverage": 1.0,
        "current_position": 0.0,
        "entry_price": None,
        "current_price": None,
        "market_return": 0.0,
        "spread_bps": 1.0,
        "stale_seconds": 0.0,
    }
    values.update(changes)
    return RiskState(**values)  # type: ignore[arg-type]


def test_risk_engine_clips_position_and_leverage() -> None:
    decision = RiskEngine().evaluate(
        state(requested_position_fraction=0.8, requested_leverage=9.0)
    )
    assert decision.allowed is True
    assert decision.target_position_fraction == 0.25
    assert decision.leverage == 2.0
    assert {trigger.rule for trigger in decision.triggers} == {
        "position_limit",
        "leverage_limit",
    }


@pytest.mark.parametrize(
    ("changes", "rule"),
    [
        ({"equity": 79.0}, "max_drawdown"),
        (
            {"current_position": 1.0, "entry_price": 100.0, "current_price": 94.0},
            "stop_loss",
        ),
        ({"market_return": -0.11}, "extreme_market_move"),
        ({"spread_bps": 101.0}, "wide_spread"),
        ({"stale_seconds": 31.0}, "stale_market_data"),
    ],
)
def test_each_required_risk_rule_can_really_trigger(
    changes: dict[str, float], rule: str
) -> None:
    decision = RiskEngine().evaluate(state(**changes))
    assert decision.allowed is False
    assert decision.target_position_fraction == 0.0
    assert rule in {trigger.rule for trigger in decision.triggers}


def test_risk_limits_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown_pct=0).validate()
