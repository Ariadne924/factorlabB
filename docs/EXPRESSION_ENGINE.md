# Factor Expression Engine

The expression engine is the controlled bridge between manually implemented
factors and automated factor mining. It evaluates a restricted formula language;
it does not execute arbitrary Python code.

## Example

```python
from factors.expression_engine import compile_expression

formula = compile_expression(
    "zscore(returns(close, 1), 24) + protected_div(delta(volume, 6), volume)"
)
values = formula.evaluate(frame)
```

An expression can also be connected to the existing registry:

```python
from factors.expression_engine import register_expression_factor

register_expression_factor(
    "formula_momentum_volume_v1",
    "zscore(returns(close, 1), 24) + protected_div(delta(volume, 6), volume)",
)
```

## Supported language

- Columns: any non-private dataframe column such as `open`, `close`, `volume`,
  `funding_rate`, or `open_interest`.
- Arithmetic: `+`, `-`, `*`, protected `/`, `**`, and unary signs.
- Element operations: `abs`, `sign`, `log`, `signed_power`, `protected_div`.
- Controlled conditions: `gt`, `ge`, `lt`, `le`, `eq`, `logical_and`,
  `logical_or`, and `where`.
- Past observations: `delay`, `delta`, `returns`.
- Rolling operations: `rolling_sum`, `rolling_mean`, `rolling_std`,
  `rolling_min`, `rolling_max`, `rolling_rank`, `corr`, `cov`, `argmin`,
  `argmax`, `decay_linear`, and `zscore`.

Periods and windows must be non-negative or positive integer constants as
appropriate. Negative delays, attribute access, indexing, keyword arguments,
comprehensions, lambdas, unlisted functions, oversized windows, and extreme
power exponents are rejected before evaluation.

## Reproducibility metadata

Compilation produces:

- a canonical expression used for whitespace-independent deduplication;
- a stable 16-character SHA-256 prefix;
- sorted data dependencies;
- expression complexity and depth limits;
- an explicit `past_only_static_validation` lookahead policy.

Static validation prevents unsupported future-looking syntax. Generated
candidates must still pass the project's truncation-invariance check before
they enter research reports.

## Multi-symbol expressions

`compile_panel_expression` accepts canonical `(timestamp, symbol)` panels.
Historical functions such as `delay`, `returns`, and `rolling_mean` run
independently within each symbol. `cs_rank`, `cs_scale`, `cs_zscore`,
`cs_winsorize`, and `cs_neutralize` run only across assets at the same
timestamp. The single-symbol compiler deliberately rejects these `cs_*`
operators so formula scope cannot change silently.
