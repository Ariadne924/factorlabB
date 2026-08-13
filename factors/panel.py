"""Point-in-time multi-symbol panel construction and cross-sectional operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

TIME_LEVEL = "timestamp"
SYMBOL_LEVEL = "symbol"
PANEL_INDEX_NAMES = (TIME_LEVEL, SYMBOL_LEVEL)
EPSILON = 1e-12


class PanelValidationError(ValueError):
    """Raised when a dataframe cannot safely represent a research panel."""


def _utc_index(index: pd.Index, symbol: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise PanelValidationError(f"{symbol} must use a DatetimeIndex")
    if index.tz is None:
        raise PanelValidationError(f"{symbol} timestamps must be timezone-aware")
    normalized = pd.DatetimeIndex(pd.to_datetime(index, utc=True), name=TIME_LEVEL)
    if normalized.has_duplicates:
        raise PanelValidationError(f"{symbol} contains duplicate timestamps")
    return normalized


def build_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    join: Literal["outer", "inner"] = "outer",
) -> pd.DataFrame:
    """Combine symbol frames without filling observations across time.

    ``outer`` retains every observed timestamp and exposes missing symbols via
    coverage metrics. ``inner`` retains timestamps observed for every symbol.
    Values are never forward- or backward-filled.
    """
    if not frames:
        raise PanelValidationError("frames cannot be empty")
    if join not in {"outer", "inner"}:
        raise PanelValidationError("join must be 'outer' or 'inner'")

    normalized_frames: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in frames.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            raise PanelValidationError("symbol cannot be empty")
        if symbol in normalized_frames:
            raise PanelValidationError(f"duplicate normalized symbol: {symbol}")
        local = frame.copy()
        local.index = _utc_index(local.index, symbol)
        normalized_frames[symbol] = local.sort_index()

    if join == "inner":
        common: pd.DatetimeIndex | None = None
        for frame in normalized_frames.values():
            frame_index = pd.DatetimeIndex(frame.index)
            common = (
                frame_index
                if common is None
                else pd.DatetimeIndex(common.intersection(frame_index))
            )
        assert common is not None
        normalized_frames = {
            symbol: frame.loc[common] for symbol, frame in normalized_frames.items()
        }

    panel = pd.concat(normalized_frames, names=[SYMBOL_LEVEL, TIME_LEVEL])
    panel = panel.reorder_levels(PANEL_INDEX_NAMES).sort_index()
    validate_panel(panel)
    return panel


def validate_panel(panel: pd.DataFrame | pd.Series) -> None:
    """Validate the canonical ``(timestamp, symbol)`` point-in-time index."""
    if not isinstance(panel.index, pd.MultiIndex):
        raise PanelValidationError("panel must use a MultiIndex")
    if tuple(panel.index.names) != PANEL_INDEX_NAMES:
        raise PanelValidationError(
            f"panel index names must be {PANEL_INDEX_NAMES}, got {panel.index.names}"
        )
    if panel.index.has_duplicates:
        raise PanelValidationError("panel contains duplicate timestamp-symbol rows")
    timestamps = panel.index.get_level_values(TIME_LEVEL)
    if not isinstance(timestamps, pd.DatetimeIndex) or timestamps.tz is None:
        raise PanelValidationError("panel timestamps must be timezone-aware")
    if not panel.index.is_monotonic_increasing:
        raise PanelValidationError("panel index must be sorted")


def panel_coverage(
    panel: pd.DataFrame | pd.Series,
    *,
    expected_symbols: int | None = None,
) -> pd.DataFrame:
    """Report per-timestamp symbol coverage without manufacturing observations."""
    validate_panel(panel)
    symbols = panel.index.get_level_values(SYMBOL_LEVEL)
    inferred = int(symbols.nunique())
    expected = inferred if expected_symbols is None else int(expected_symbols)
    if expected < 1 or expected < inferred:
        raise PanelValidationError("expected_symbols must cover all symbols in the panel")
    counts = symbols.to_series(index=panel.index).groupby(level=TIME_LEVEL).nunique()
    result = pd.DataFrame({"available_symbols": counts.astype(int)})
    result["expected_symbols"] = expected
    result["coverage_ratio"] = result["available_symbols"] / expected
    result["is_complete"] = result["available_symbols"] == expected
    return result


def _validate_series(values: pd.Series) -> pd.Series:
    validate_panel(values)
    return values.astype(float)


def cs_rank(
    values: pd.Series,
    *,
    method: Literal["average", "min", "max", "first", "dense"] = "average",
    center: bool = False,
) -> pd.Series:
    """Percentile rank assets independently at each timestamp."""
    series = _validate_series(values)
    ranked = series.groupby(level=TIME_LEVEL, sort=False).rank(method=method, pct=True)
    return ranked - 0.5 if center else ranked


def cs_scale(values: pd.Series, *, gross: float = 1.0) -> pd.Series:
    """Scale each timestamp so absolute exposures sum to ``gross``."""
    if not np.isfinite(gross) or gross <= 0:
        raise PanelValidationError("gross must be a positive finite number")
    series = _validate_series(values)
    denominator = series.abs().groupby(level=TIME_LEVEL, sort=False).transform("sum")
    return series.mul(gross).div(denominator.where(denominator > EPSILON))


def cs_zscore(values: pd.Series) -> pd.Series:
    """Population z-score calculated independently at each timestamp."""
    series = _validate_series(values)
    grouped = series.groupby(level=TIME_LEVEL, sort=False)
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform("std", ddof=0)
    return (series - mean).div(standard_deviation.where(standard_deviation > EPSILON))


def cs_winsorize(
    values: pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Clip each timestamp using cross-sectional empirical quantiles."""
    if not 0 <= lower < upper <= 1:
        raise PanelValidationError("winsorize bounds must satisfy 0 <= lower < upper <= 1")
    series = _validate_series(values)
    grouped = series.groupby(level=TIME_LEVEL, sort=False)
    lower_bound = grouped.transform(lambda group: group.quantile(lower))
    upper_bound = grouped.transform(lambda group: group.quantile(upper))
    return series.clip(lower=lower_bound, upper=upper_bound)


def cs_neutralize(
    values: pd.Series,
    exposures: pd.DataFrame,
    *,
    add_intercept: bool = True,
) -> pd.Series:
    """Return same-timestamp OLS residuals against supplied exposures.

    Rows with missing values remain missing. A timestamp needs more valid assets
    than regression coefficients; otherwise its residuals are left as NaN.
    """
    series = _validate_series(values)
    validate_panel(exposures)
    if exposures.empty or exposures.shape[1] == 0:
        raise PanelValidationError("at least one neutralization exposure is required")
    if not exposures.index.equals(series.index):
        exposures = exposures.reindex(series.index)

    result = pd.Series(np.nan, index=series.index, name=series.name, dtype=float)
    for _, group in series.groupby(level=TIME_LEVEL, sort=False):
        group_exposures = exposures.loc[group.index].astype(float)
        valid = group.notna() & group_exposures.notna().all(axis=1)
        y = group.loc[valid].to_numpy(dtype=float)
        x = group_exposures.loc[valid].to_numpy(dtype=float)
        if add_intercept:
            x = np.column_stack([np.ones(len(x), dtype=float), x])
        if len(y) <= x.shape[1]:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        result.loc[group.loc[valid].index] = y - x @ coefficients
    return result
