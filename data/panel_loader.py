"""Load point-in-time multi-symbol panels from existing Silver parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from data.silver import silver_to_factor_input
from data.validator import DataValidator
from factors.panel import PanelValidationError, build_panel


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def load_silver_panel(
    data_dir: Path,
    *,
    interval: str,
    symbols: tuple[str, ...] | None = None,
    start: str | None = None,
    end: str | None = None,
    join: Literal["outer", "inner"] = "outer",
) -> pd.DataFrame:
    """Load one frequency across symbols without filling missing bars."""
    normalized_interval = "1d" if interval == "24h" else interval
    requested_symbols = {symbol.upper() for symbol in symbols or ()}
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("silver/**/klines.parquet")):
        silver = pd.read_parquet(path)
        DataValidator.validate_klines(silver)
        file_interval = str(silver["interval"].iloc[0])
        symbol = str(silver["symbol"].iloc[0]).upper()
        if file_interval != normalized_interval:
            continue
        if requested_symbols and symbol not in requested_symbols:
            continue
        if symbol in frames:
            raise PanelValidationError(
                f"multiple Silver files found for {symbol} {normalized_interval}"
            )
        frame = silver_to_factor_input(silver)
        if start is not None:
            frame = frame.loc[frame.index >= _utc(start)]
        if end is not None:
            frame = frame.loc[frame.index <= _utc(end)]
        if not frame.empty:
            frames[symbol] = frame.drop(columns=["symbol", "interval"], errors="ignore")
    if not frames:
        raise PanelValidationError(
            f"no Silver data found for interval={normalized_interval} and symbols={symbols}"
        )
    missing = requested_symbols - set(frames)
    if missing:
        raise PanelValidationError(f"missing requested Silver symbols: {sorted(missing)}")
    return build_panel(frames, join=join)
