"""Registry dedicated to expressions that require a multi-symbol panel."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from factors.expression_engine import CompiledPanelExpression, compile_panel_expression

PANEL_FACTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_panel_expression(
    name: str,
    expression: str,
    *,
    category: str,
    description: str,
    source: str,
    source_url: str,
    source_formula: str = "",
    adaptation_note: str = "",
) -> CompiledPanelExpression:
    """Compile and register a cross-sectional expression with provenance."""
    if name in PANEL_FACTOR_REGISTRY:
        raise ValueError(f"panel factor '{name}' is already registered")
    compiled = compile_panel_expression(expression)
    PANEL_FACTOR_REGISTRY[name] = {
        "compiled": compiled,
        "name": name,
        "category": category,
        "description": description,
        "source": source,
        "source_url": source_url,
        "source_formula": source_formula,
        "adaptation_note": adaptation_note,
        **compiled.metadata(),
    }
    return compiled


def compute_panel_factor(name: str, panel: pd.DataFrame) -> pd.Series:
    """Evaluate a registered factor on a canonical panel."""
    metadata = PANEL_FACTOR_REGISTRY.get(name)
    if metadata is None:
        raise KeyError(f"panel factor '{name}' is not registered")
    compiled: CompiledPanelExpression = metadata["compiled"]
    return compiled.evaluate(panel).rename(name)


def get_panel_factor_metadata(name: str) -> dict[str, Any]:
    """Return serializable metadata without exposing the compiled object."""
    metadata = PANEL_FACTOR_REGISTRY.get(name)
    if metadata is None:
        raise KeyError(f"panel factor '{name}' is not registered")
    output = {key: value for key, value in metadata.items() if key != "compiled"}
    return deepcopy(output)


def list_panel_factors(category: str | None = None) -> list[str]:
    """List registered panel factors, optionally filtered by category."""
    if category is None:
        return list(PANEL_FACTOR_REGISTRY)
    return [
        name
        for name, metadata in PANEL_FACTOR_REGISTRY.items()
        if metadata.get("category") == category
    ]
