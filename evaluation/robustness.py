"""Small-sample robustness diagnostics for a single time-series factor."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from evaluation.ic_analysis import compute_ic, compute_rank_ic


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Return from the current close to ``horizon`` bars ahead."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return close.shift(-horizon).div(close).sub(1).rename(f"forward_return_{horizon}")


def trailing_window(data: pd.Series, days: int) -> pd.Series:
    """Take a calendar-day trailing sample when a DatetimeIndex is available."""
    if days < 1:
        raise ValueError("days must be at least 1")
    if data.empty:
        return data.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        cutoff = data.index.max() - pd.Timedelta(days=days)
        return data.loc[data.index >= cutoff]
    return data.tail(days)


def block_bootstrap_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    n_bootstrap: int = 500,
    block_size: int = 24,
    confidence: float = 0.95,
    random_state: int = 0,
) -> dict[str, float | int | None]:
    """Moving-block bootstrap CI for RankIC, preserving short serial dependence."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    aligned = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("forward_return")], axis=1
    ).dropna()
    n_obs = len(aligned)
    if n_obs < 5:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_obs": n_obs,
            "n_bootstrap": n_bootstrap,
            "block_size": min(block_size, max(n_obs, 1)),
        }

    values = aligned.to_numpy(dtype=float)
    effective_block = min(block_size, n_obs)
    starts = np.arange(n_obs - effective_block + 1)
    blocks_needed = int(np.ceil(n_obs / effective_block))
    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        positions = np.concatenate(
            [np.arange(start, start + effective_block) for start in chosen]
        )[:n_obs]
        sample = values[positions]
        value = pd.Series(sample[:, 0]).rank().corr(pd.Series(sample[:, 1]).rank())
        if pd.notna(value):
            samples.append(float(value))

    if not samples:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_obs": n_obs,
            "n_bootstrap": n_bootstrap,
            "block_size": effective_block,
        }
    distribution = np.asarray(samples)
    alpha = 1 - confidence
    # Add-one correction prevents an impossible exact p=0 in a finite bootstrap sample.
    non_positive = (int(np.count_nonzero(distribution <= 0)) + 1) / (len(distribution) + 1)
    non_negative = (int(np.count_nonzero(distribution >= 0)) + 1) / (len(distribution) + 1)
    p_value = min(1.0, 2 * min(non_positive, non_negative))
    return {
        "estimate": float(compute_rank_ic(aligned["factor"], aligned["forward_return"])),
        "ci_low": float(np.quantile(distribution, alpha / 2)),
        "ci_high": float(np.quantile(distribution, 1 - alpha / 2)),
        "p_value": p_value,
        "n_obs": n_obs,
        "n_bootstrap": len(samples),
        "block_size": effective_block,
    }


def benjamini_hochberg(p_values: pd.Series, *, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    clean = p_values.dropna().astype(float).clip(0, 1)
    result = pd.DataFrame(index=p_values.index, columns=["p_value", "q_value", "reject"])
    result["p_value"] = p_values
    result["reject"] = False
    if clean.empty:
        return result
    ordered = clean.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = ordered.to_numpy() * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result.loc[ordered.index, "q_value"] = adjusted
    result.loc[ordered.index, "reject"] = adjusted <= alpha
    return result


def window_horizon_robustness(
    factor_values: pd.Series,
    close: pd.Series,
    *,
    lookback_days: Iterable[int] = (30, 60, 90, 180),
    horizons: Iterable[int] = (1, 3, 6, 12, 24),
    min_obs: int = 20,
) -> pd.DataFrame:
    """IC/RankIC grid across calendar lookbacks and forward-return horizons."""
    records: list[dict[str, float | int | str | None]] = []
    for days in lookback_days:
        sliced_factor = trailing_window(factor_values, int(days))
        for horizon in horizons:
            target = forward_return(close, int(horizon)).reindex(sliced_factor.index)
            aligned = pd.concat([sliced_factor.rename("factor"), target], axis=1).dropna()
            enough = len(aligned) >= min_obs
            records.append(
                {
                    "lookback_days": int(days),
                    "horizon": int(horizon),
                    "n_obs": len(aligned),
                    "ic": compute_ic(aligned["factor"], aligned.iloc[:, 1]) if enough else None,
                    "rank_ic": (
                        compute_rank_ic(aligned["factor"], aligned.iloc[:, 1]) if enough else None
                    ),
                    "status": "computed" if enough else "insufficient_data",
                }
            )
    return pd.DataFrame.from_records(records)


def sign_consistency(grid: pd.DataFrame) -> float | None:
    """Share of valid RankIC cells agreeing with the median non-zero sign."""
    values = pd.to_numeric(grid.get("rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    values = values[values != 0]
    if values.empty:
        return None
    reference = float(np.sign(values.median()))
    if reference == 0:
        return None
    return float((np.sign(values) == reference).mean())


def group_monotonicity(group_returns: pd.DataFrame) -> float | None:
    """Spearman correlation between group number and mean group return."""
    if group_returns.empty or len(group_returns) < 2:
        return None
    value = group_returns["group"].rank().corr(group_returns["mean_return"].rank())
    return None if pd.isna(value) else float(value)
