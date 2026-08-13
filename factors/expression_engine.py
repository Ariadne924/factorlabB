"""Safe, past-only expression engine for formulaic factor research.

The engine intentionally supports a small auditable language instead of Python
``eval``. Expressions can reference dataframe columns, numeric constants,
arithmetic operators, and the functions listed in :data:`SUPPORTED_FUNCTIONS`.
Attribute access, indexing, comprehensions, lambdas, and negative delays are
rejected during compilation.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factors.panel import (
    SYMBOL_LEVEL,
    cs_neutralize,
    cs_rank,
    cs_scale,
    cs_winsorize,
    cs_zscore,
    validate_panel,
)
from factors.registry import FACTOR_REGISTRY, register_factor

EPSILON = 1e-12
MAX_EXPRESSION_LENGTH = 2_000
MAX_COMPLEXITY = 500
MAX_DEPTH = 32
MAX_WINDOW = 100_000
MAX_ABS_EXPONENT = 10.0


class ExpressionError(ValueError):
    """Base exception for invalid or unsupported factor expressions."""


class ExpressionSyntaxError(ExpressionError):
    """Raised when an expression is not valid syntax."""


class ExpressionValidationError(ExpressionError):
    """Raised when syntax is valid but violates the safe language rules."""


def _protected_div(left: Any, right: Any) -> Any:
    if isinstance(right, pd.Series):
        return left / right.where(right.abs() > EPSILON)
    return np.nan if abs(float(right)) <= EPSILON else left / right


def _series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, (int, float, np.number)):
        return pd.Series(float(value), index=index)
    raise ExpressionValidationError("expression result must be numeric or a pandas Series")


def _last_rank(values: np.ndarray) -> float:
    series = pd.Series(values)
    return float(series.rank(pct=True).iloc[-1])


def _arg_position(values: np.ndarray, *, maximum: bool) -> float:
    position = int(np.nanargmax(values) if maximum else np.nanargmin(values))
    return float(position + 1)


def _decay(values: np.ndarray) -> float:
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


SUPPORTED_FUNCTIONS = frozenset(
    {
        "abs",
        "argmax",
        "argmin",
        "corr",
        "cov",
        "decay_linear",
        "delay",
        "delta",
        "eq",
        "ewm_mean",
        "ge",
        "gt",
        "le",
        "log",
        "logical_and",
        "logical_or",
        "lt",
        "maximum",
        "minimum",
        "protected_div",
        "returns",
        "rolling_max",
        "rolling_mad",
        "rolling_mean",
        "rolling_min",
        "rolling_rank",
        "rolling_std",
        "rolling_sum",
        "sign",
        "signed_power",
        "where",
        "zscore",
    }
)
PANEL_ONLY_FUNCTIONS = frozenset(
    {"cs_neutralize", "cs_rank", "cs_scale", "cs_winsorize", "cs_zscore"}
)
PANEL_SUPPORTED_FUNCTIONS = SUPPORTED_FUNCTIONS | PANEL_ONLY_FUNCTIONS

_WINDOW_FUNCTIONS = {
    "argmax",
    "argmin",
    "decay_linear",
    "ewm_mean",
    "rolling_max",
    "rolling_mad",
    "rolling_mean",
    "rolling_min",
    "rolling_rank",
    "rolling_std",
    "rolling_sum",
    "zscore",
}
_PAIR_WINDOW_FUNCTIONS = {"corr", "cov"}
_PERIOD_FUNCTIONS = {"delay", "delta", "returns"}


def _numeric_constant(node: ast.AST, label: str) -> float:
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        raise ExpressionValidationError(f"{label} must be a numeric constant")
    if not isinstance(node.value, (int, float)):
        raise ExpressionValidationError(f"{label} must be a numeric constant")
    value = float(node.value)
    if not np.isfinite(value):
        raise ExpressionValidationError(f"{label} must be finite")
    return value


def _positive_integer(node: ast.AST, label: str, *, allow_zero: bool = False) -> int:
    value = _numeric_constant(node, label)
    integer = int(value)
    minimum = 0 if allow_zero else 1
    if value != integer or integer < minimum:
        condition = "a non-negative" if allow_zero else "a positive"
        raise ExpressionValidationError(f"{label} must be {condition} integer")
    if integer > MAX_WINDOW:
        raise ExpressionValidationError(f"{label} exceeds limit {MAX_WINDOW}")
    return integer


class _Validator(ast.NodeVisitor):
    def __init__(self, supported_functions: frozenset[str]) -> None:
        self.dependencies: set[str] = set()
        self.supported_functions = supported_functions
        self.depth = 0
        self.max_depth = 0

    def visit(self, node: ast.AST) -> Any:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            return super().visit(node)
        finally:
            self.depth -= 1

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Constant,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )
        if not isinstance(node, allowed):
            raise ExpressionValidationError(
                f"unsupported syntax: {node.__class__.__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise ExpressionValidationError("private or dunder names are not allowed")
        self.dependencies.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        _numeric_constant(node, "constant")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Pow):
            exponent = _numeric_constant(node.right, "power exponent")
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise ExpressionValidationError(
                    f"power exponent exceeds absolute limit {MAX_ABS_EXPONENT:g}"
                )
        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError("only direct calls to supported functions are allowed")
        function = node.func.id
        if function not in self.supported_functions:
            raise ExpressionValidationError(f"unsupported function: {function}")
        if node.keywords:
            raise ExpressionValidationError("keyword arguments are not supported")

        if function in {"cs_rank", "cs_scale", "cs_zscore"}:
            if len(node.args) != 1:
                raise ExpressionValidationError(f"{function} expects one argument")
        elif function == "cs_winsorize":
            if len(node.args) != 3:
                raise ExpressionValidationError(
                    "cs_winsorize expects value, lower, and upper"
                )
            lower = _numeric_constant(node.args[1], "cs_winsorize lower")
            upper = _numeric_constant(node.args[2], "cs_winsorize upper")
            if not 0 <= lower < upper <= 1:
                raise ExpressionValidationError(
                    "cs_winsorize bounds must satisfy 0 <= lower < upper <= 1"
                )
        elif function == "cs_neutralize":
            if len(node.args) < 2:
                raise ExpressionValidationError(
                    "cs_neutralize expects value and at least one exposure"
                )
        elif function in _WINDOW_FUNCTIONS:
            if len(node.args) != 2:
                raise ExpressionValidationError(f"{function} expects value and window")
            _positive_integer(node.args[1], f"{function} window")
        elif function in _PAIR_WINDOW_FUNCTIONS:
            if len(node.args) != 3:
                raise ExpressionValidationError(
                    f"{function} expects left, right, and window"
                )
            _positive_integer(node.args[2], f"{function} window")
        elif function in _PERIOD_FUNCTIONS:
            if len(node.args) != 2:
                raise ExpressionValidationError(f"{function} expects value and period")
            _positive_integer(
                node.args[1],
                f"{function} period",
                allow_zero=function == "delay",
            )
        elif function == "signed_power":
            if len(node.args) != 2:
                raise ExpressionValidationError("signed_power expects value and exponent")
            exponent = _numeric_constant(node.args[1], "signed_power exponent")
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise ExpressionValidationError(
                    f"signed_power exponent exceeds absolute limit {MAX_ABS_EXPONENT:g}"
                )
        elif function in {
            "eq",
            "ge",
            "gt",
            "le",
            "logical_and",
            "logical_or",
            "lt",
            "maximum",
            "minimum",
            "protected_div",
        }:
            if len(node.args) != 2:
                raise ExpressionValidationError(f"{function} expects two arguments")
        elif function == "where":
            if len(node.args) != 3:
                raise ExpressionValidationError("where expects condition, true, and false")
        elif len(node.args) != 1:
            raise ExpressionValidationError(f"{function} expects one argument")

        for argument in node.args:
            self.visit(argument)


def _canonical(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _canonical(node.body)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp):
        operator = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"({operator}{_canonical(node.operand)})"
    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
        }
        operator = operators[type(node.op)]
        return f"({_canonical(node.left)}{operator}{_canonical(node.right)})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = ",".join(_canonical(argument) for argument in node.args)
        return f"{node.func.id}({arguments})"
    raise ExpressionValidationError(f"cannot serialize {node.__class__.__name__}")


def _complexity(tree: ast.AST) -> int:
    ignored = (ast.Expression, ast.Load)
    return sum(1 for node in ast.walk(tree) if not isinstance(node, ignored))


@dataclass(frozen=True)
class CompiledExpression:
    """Validated expression plus reproducibility metadata."""

    expression: str
    canonical: str
    dependencies: tuple[str, ...]
    complexity: int
    digest: str
    _tree: ast.Expression

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        """Evaluate against one time-ordered symbol dataframe."""
        missing = [column for column in self.dependencies if column not in frame.columns]
        if missing:
            raise ExpressionValidationError(f"missing expression dependencies: {missing}")
        result = _Evaluator(frame).evaluate(self._tree.body)
        return _series(result, frame.index).replace([np.inf, -np.inf], np.nan).rename(
            self.digest
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "canonical_expression": self.canonical,
            "expression_hash": self.digest,
            "expression_complexity": self.complexity,
            "data_dependencies": list(self.dependencies),
            "scope": "time_series",
            "lookahead_policy": "past_only_static_validation",
        }


@dataclass(frozen=True)
class CompiledPanelExpression:
    """Validated expression evaluated on a canonical multi-symbol panel."""

    expression: str
    canonical: str
    dependencies: tuple[str, ...]
    complexity: int
    digest: str
    _tree: ast.Expression

    def evaluate(self, panel: pd.DataFrame) -> pd.Series:
        validate_panel(panel)
        missing = [column for column in self.dependencies if column not in panel.columns]
        if missing:
            raise ExpressionValidationError(f"missing expression dependencies: {missing}")
        result = _PanelEvaluator(panel).evaluate(self._tree.body)
        return _series(result, panel.index).replace([np.inf, -np.inf], np.nan).rename(
            self.digest
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "canonical_expression": self.canonical,
            "expression_hash": self.digest,
            "expression_complexity": self.complexity,
            "data_dependencies": list(self.dependencies),
            "scope": "cross_sectional",
            "lookahead_policy": "same_timestamp_cross_section_and_past_only_time_series",
        }


class _Evaluator:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return self.frame[node.id].astype(float)
        if isinstance(node, ast.Constant):
            return _numeric_constant(node, "constant")
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self.evaluate(node.left), self.evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return _protected_div(left, right)
            if isinstance(node.op, ast.Pow):
                return left**right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self._call(node.func.id, node.args)
        raise ExpressionValidationError(f"cannot evaluate {node.__class__.__name__}")

    def _call(self, name: str, nodes: list[ast.expr]) -> Any:
        values = [self.evaluate(node) for node in nodes]
        value = values[0]
        if name == "abs":
            return abs(value)
        if name == "sign":
            return np.sign(value)
        if name == "log":
            magnitude = abs(value)
            if isinstance(magnitude, pd.Series):
                return np.log(magnitude.where(magnitude > EPSILON))
            return np.nan if magnitude <= EPSILON else np.log(magnitude)
        if name == "signed_power":
            return np.sign(value) * (abs(value) ** float(values[1]))
        if name == "protected_div":
            return _protected_div(value, values[1])
        if name in {"maximum", "minimum"}:
            left = _series(value, self.frame.index)
            right = _series(values[1], self.frame.index)
            combined = pd.concat([left, right], axis=1)
            return combined.max(axis=1) if name == "maximum" else combined.min(axis=1)
        if name in {"eq", "ge", "gt", "le", "lt"}:
            right = values[1]
            operations = {
                "eq": lambda: value == right,
                "ge": lambda: value >= right,
                "gt": lambda: value > right,
                "le": lambda: value <= right,
                "lt": lambda: value < right,
            }
            return operations[name]()
        if name in {"logical_and", "logical_or"}:
            left_condition = _series(value, self.frame.index).fillna(False).astype(bool)
            right_condition = _series(values[1], self.frame.index).fillna(False).astype(bool)
            return (
                left_condition & right_condition
                if name == "logical_and"
                else left_condition | right_condition
            )
        if name == "where":
            condition = _series(value, self.frame.index).fillna(False).astype(bool)
            true_value = _series(values[1], self.frame.index)
            false_value = _series(values[2], self.frame.index)
            selected = true_value.where(condition, false_value)
            # A warm-up NaN is not a usable selected value; fall back to the
            # explicit alternative branch instead of leaking an accidental NaN.
            return selected.where(selected.notna(), false_value)

        period = int(values[-1])
        series = _series(value, self.frame.index)
        if name == "delay":
            return series.shift(period)
        if name == "delta":
            return series.diff(period)
        if name == "returns":
            return _protected_div(series, series.shift(period)) - 1.0

        if name in _PAIR_WINDOW_FUNCTIONS:
            right = _series(values[1], self.frame.index)
            return (
                series.rolling(period, min_periods=period).corr(right)
                if name == "corr"
                else series.rolling(period, min_periods=period).cov(right)
            )

        rolling = series.rolling(period, min_periods=period)
        if name == "ewm_mean":
            return series.ewm(span=period, adjust=False, min_periods=period).mean()
        if name == "rolling_sum":
            return rolling.sum()
        if name == "rolling_mean":
            return rolling.mean()
        if name == "rolling_std":
            return rolling.std(ddof=0)
        if name == "rolling_min":
            return rolling.min()
        if name == "rolling_max":
            return rolling.max()
        if name == "rolling_mad":
            return rolling.apply(
                lambda items: float(np.mean(np.abs(items - np.mean(items)))), raw=True
            )
        if name == "rolling_rank":
            return rolling.apply(_last_rank, raw=True)
        if name == "argmin":
            return rolling.apply(lambda values: _arg_position(values, maximum=False), raw=True)
        if name == "argmax":
            return rolling.apply(lambda values: _arg_position(values, maximum=True), raw=True)
        if name == "decay_linear":
            return rolling.apply(_decay, raw=True)
        if name == "zscore":
            return _protected_div(series - rolling.mean(), rolling.std(ddof=0))
        raise ExpressionValidationError(f"unsupported function: {name}")


class _PanelEvaluator(_Evaluator):
    def _per_symbol(self, series: pd.Series, operation: Any) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in series.groupby(level=SYMBOL_LEVEL, sort=False):
            result.loc[group.index] = operation(group).to_numpy(dtype=float)
        return result

    def _per_symbol_pair(
        self,
        left: pd.Series,
        right: pd.Series,
        operation: Any,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=left.index, dtype=float)
        for symbol, group in left.groupby(level=SYMBOL_LEVEL, sort=False):
            other = right.xs(symbol, level=SYMBOL_LEVEL, drop_level=False)
            result.loc[group.index] = operation(group, other).to_numpy(dtype=float)
        return result

    def _call(self, name: str, nodes: list[ast.expr]) -> Any:
        values = [self.evaluate(node) for node in nodes]
        value = values[0]
        series = _series(value, self.frame.index)
        if name == "cs_rank":
            return cs_rank(series)
        if name == "cs_scale":
            return cs_scale(series)
        if name == "cs_zscore":
            return cs_zscore(series)
        if name == "cs_winsorize":
            return cs_winsorize(series, lower=float(values[1]), upper=float(values[2]))
        if name == "cs_neutralize":
            exposure_columns = {
                f"exposure_{position}": _series(exposure, self.frame.index)
                for position, exposure in enumerate(values[1:])
            }
            return cs_neutralize(series, pd.DataFrame(exposure_columns))
        if name in {
            "abs",
            "eq",
            "maximum",
            "minimum",
            "ge",
            "gt",
            "le",
            "log",
            "logical_and",
            "logical_or",
            "lt",
            "protected_div",
            "sign",
            "signed_power",
            "where",
        }:
            return super()._call(name, nodes)

        period = int(values[-1])
        if name == "delay":
            return self._per_symbol(series, lambda group: group.shift(period))
        if name == "delta":
            return self._per_symbol(series, lambda group: group.diff(period))
        if name == "returns":
            return self._per_symbol(
                series,
                lambda group: _protected_div(group, group.shift(period)) - 1.0,
            )

        if name in _PAIR_WINDOW_FUNCTIONS:
            right = _series(values[1], self.frame.index)
            if name == "corr":
                return self._per_symbol_pair(
                    series,
                    right,
                    lambda left, other: left.rolling(period, min_periods=period).corr(other),
                )
            return self._per_symbol_pair(
                series,
                right,
                lambda left, other: left.rolling(period, min_periods=period).cov(other),
            )

        def rolling_operation(group: pd.Series) -> pd.Series:
            rolling = group.rolling(period, min_periods=period)
            if name == "ewm_mean":
                return group.ewm(span=period, adjust=False, min_periods=period).mean()
            if name == "rolling_sum":
                return rolling.sum()
            if name == "rolling_mean":
                return rolling.mean()
            if name == "rolling_std":
                return rolling.std(ddof=0)
            if name == "rolling_min":
                return rolling.min()
            if name == "rolling_max":
                return rolling.max()
            if name == "rolling_mad":
                return rolling.apply(
                    lambda items: float(np.mean(np.abs(items - np.mean(items)))),
                    raw=True,
                )
            if name == "rolling_rank":
                return rolling.apply(_last_rank, raw=True)
            if name == "argmin":
                return rolling.apply(
                    lambda items: _arg_position(items, maximum=False), raw=True
                )
            if name == "argmax":
                return rolling.apply(
                    lambda items: _arg_position(items, maximum=True), raw=True
                )
            if name == "decay_linear":
                return rolling.apply(_decay, raw=True)
            if name == "zscore":
                return _protected_div(group - rolling.mean(), rolling.std(ddof=0))
            raise ExpressionValidationError(f"unsupported function: {name}")

        return self._per_symbol(series, rolling_operation)


def _validated_expression(
    expression: str,
    supported_functions: frozenset[str],
) -> tuple[ast.Expression, _Validator, int, str, str]:
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionSyntaxError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionValidationError("expression is too long")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionSyntaxError(f"invalid expression syntax: {exc.msg}") from exc

    validator = _Validator(supported_functions)
    try:
        validator.visit(parsed)
    except RecursionError as exc:
        raise ExpressionValidationError("expression nesting is too deep") from exc
    complexity = _complexity(parsed)
    if complexity > MAX_COMPLEXITY:
        raise ExpressionValidationError(
            f"expression complexity {complexity} exceeds limit {MAX_COMPLEXITY}"
        )
    if validator.max_depth > MAX_DEPTH:
        raise ExpressionValidationError(
            f"expression depth {validator.max_depth} exceeds limit {MAX_DEPTH}"
        )
    canonical = _canonical(parsed)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return parsed, validator, complexity, canonical, digest


def compile_expression(expression: str) -> CompiledExpression:
    """Parse and statically validate a past-only formula."""
    parsed, validator, complexity, canonical, digest = _validated_expression(
        expression, SUPPORTED_FUNCTIONS
    )
    dependencies = tuple(sorted(validator.dependencies - validator.supported_functions))
    return CompiledExpression(
        expression=expression,
        canonical=canonical,
        dependencies=dependencies,
        complexity=complexity,
        digest=digest,
        _tree=parsed,
    )


def compile_panel_expression(expression: str) -> CompiledPanelExpression:
    """Compile a formula with per-symbol history and same-time cross sections."""
    parsed, validator, complexity, canonical, digest = _validated_expression(
        expression, PANEL_SUPPORTED_FUNCTIONS
    )
    dependencies = tuple(sorted(validator.dependencies - validator.supported_functions))
    return CompiledPanelExpression(
        expression=expression,
        canonical=canonical,
        dependencies=dependencies,
        complexity=complexity,
        digest=digest,
        _tree=parsed,
    )


def register_expression_factor(
    name: str,
    expression: str,
    *,
    category: str = "formulaic",
    description: str = "",
    source: str = "expression_engine",
    source_url: str = "",
) -> CompiledExpression:
    """Compile an expression and add it to the existing factor registry."""
    compiled = compile_expression(expression)

    @register_factor(
        name=name,
        category=category,
        description=description or expression,
        default_params={},
        source=source,
        source_url=source_url,
        scope="time_series",
        data_dependencies=compiled.dependencies,
    )
    def expression_factory(params: dict[str, Any] | None = None):
        if params:
            raise ExpressionValidationError("registered expressions do not accept parameters")
        return compiled.evaluate

    FACTOR_REGISTRY[name].update(compiled.metadata())
    return compiled
