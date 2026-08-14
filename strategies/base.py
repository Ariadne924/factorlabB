"""Minimal strategy interface; execution and costs stay in the backtest layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    category: str
    description: str
    required_columns: tuple[str, ...]


class BaseStrategy(ABC):
    metadata: StrategyMetadata

    @abstractmethod
    def generate_target(self, frame: pd.DataFrame) -> pd.Series:
        """Return desired exposure in [-1, 1] using information available at each row."""

    def parameters(self) -> dict[str, Any]:
        return dict(vars(self))

    def validate_frame(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.metadata.required_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{self.metadata.name} missing columns: {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("strategy input requires a timezone-aware DatetimeIndex")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("strategy input index must be unique and increasing")

