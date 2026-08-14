"""Causal market-regime attribution for single-asset strategy returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from evaluation.strategy_robustness import _path_metrics, _returns_frame


@dataclass(frozen=True)
class MarketRegimeConfig:
    """Trailing-only regime settings expressed in bars."""

    trend_window: int = 72
    volatility_window: int = 72
    threshold_history: int = 168
    trend_band: float = 0.5

    def validate(self) -> None:
        if min(self.trend_window, self.volatility_window, self.threshold_history) < 5:
            raise ValueError("regime windows must each be at least 5 bars")
        if not np.isfinite(self.trend_band) or self.trend_band < 0:
            raise ValueError("trend_band must be finite and non-negative")


def classify_market_regimes(
    frame: pd.DataFrame,
    *,
    config: MarketRegimeConfig | None = None,
) -> pd.DataFrame:
    """Classify each return bar using information available before that bar starts."""
    settings = config or MarketRegimeConfig()
    settings.validate()
    if "close" not in frame.columns:
        raise ValueError("market-regime input requires close")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("market-regime input requires a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("market-regime input index must be unique and increasing")

    close = pd.to_numeric(frame["close"], errors="coerce")
    bar_return = close.pct_change(fill_method=None)
    trend_return = close.pct_change(settings.trend_window, fill_method=None)
    trailing_volatility = bar_return.rolling(
        settings.volatility_window,
        min_periods=settings.volatility_window,
    ).std(ddof=0)
    trend_scale = trailing_volatility * np.sqrt(settings.trend_window)
    normalized_trend = trend_return.div(trend_scale.where(trend_scale > 0))
    trailing_volatility_median = trailing_volatility.rolling(
        settings.threshold_history,
        min_periods=settings.threshold_history,
    ).median()

    trend_regime = pd.Series(
        np.select(
            [
                normalized_trend > settings.trend_band,
                normalized_trend < -settings.trend_band,
            ],
            ["bull", "bear"],
            default="sideways",
        ),
        index=frame.index,
        dtype="object",
    ).where(normalized_trend.notna(), "warmup")
    volatility_regime = pd.Series(
        np.where(
            trailing_volatility > trailing_volatility_median,
            "high_volatility",
            "low_volatility",
        ),
        index=frame.index,
        dtype="object",
    ).where(trailing_volatility_median.notna(), "warmup")

    # The return recorded at t spans t-1 to t. Shift the state so its label was
    # already known at t-1 and cannot inspect the return it is used to attribute.
    return pd.DataFrame(
        {
            "trend_regime": trend_regime.shift(1).fillna("warmup"),
            "volatility_regime": volatility_regime.shift(1).fillna("warmup"),
            "normalized_trend": normalized_trend.shift(1),
            "trailing_volatility": trailing_volatility.shift(1),
        },
        index=frame.index,
    )


def _group_performance(
    joined: pd.DataFrame,
    group_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, group in joined.groupby(group_column, sort=True):
        if regime == "warmup":
            continue
        rows.append(
            {
                "regime": str(regime),
                **_path_metrics(group["net_return"]),
                "gross_total_return": _path_metrics(group["gross_return"])[
                    "total_return"
                ],
                "mean_turnover": float(group["turnover"].mean()),
                "mean_position": float(group["position"].mean()),
            }
        )
    return rows


def market_regime_analysis(
    frame: pd.DataFrame,
    result: dict[str, Any],
    *,
    config: MarketRegimeConfig | None = None,
) -> dict[str, Any]:
    """Attribute an existing strategy path to causal trend and volatility states."""
    settings = config or MarketRegimeConfig()
    regimes = classify_market_regimes(frame, config=settings)
    returns = _returns_frame(result)
    if "position" not in returns.columns:
        raise ValueError("strategy result is missing position")
    joined = returns.join(regimes, how="left")
    usable = joined.loc[
        joined["trend_regime"].ne("warmup")
        & joined["volatility_regime"].ne("warmup")
    ]
    combined = usable.assign(
        combined_regime=(
            usable["trend_regime"].astype(str)
            + " / "
            + usable["volatility_regime"].astype(str)
        )
    )
    return {
        "status": "computed" if not usable.empty else "insufficient_data",
        "config": asdict(settings),
        "coverage": float(len(usable) / len(joined)) if len(joined) else 0.0,
        "trend_performance": _group_performance(usable, "trend_regime"),
        "volatility_performance": _group_performance(
            usable, "volatility_regime"
        ),
        "combined_performance": _group_performance(
            combined, "combined_regime"
        ),
        "lookahead_status": "pass_by_one_bar_regime_lag",
        "note": (
            "Regimes are descriptive diagnostics, not trading signals. Every return bar is "
            "labelled with a trailing state shifted by one full bar; no full-sample quantile "
            "or future price is used."
        ),
    }
