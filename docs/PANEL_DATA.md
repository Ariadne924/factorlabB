# Multi-Symbol Panel Data

The canonical research panel uses a sorted, unique MultiIndex:

```text
(timestamp, symbol)
```

Timestamps must be timezone-aware and are normalized to UTC. `build_panel`
accepts one time-indexed dataframe per symbol and never forward- or
backward-fills missing bars.

## Alignment modes

- `outer` retains every observed bar. `panel_coverage` records how many symbols
  are available at each timestamp. This is the default for data diagnostics.
- `inner` retains only timestamps shared by every supplied symbol. This is
  useful for strict cross-sectional experiments, but the resulting universe is
  still not automatically survivorship-bias-free.

## Cross-sectional operators

- `cs_rank`: same-timestamp percentile ranks;
- `cs_scale`: same-timestamp gross exposure scaling;
- `cs_zscore`: same-timestamp population standardization;
- `cs_winsorize`: same-timestamp empirical-quantile clipping;
- `cs_neutralize`: same-timestamp OLS residualization against one or more
  supplied exposures.

No operator reads a later timestamp. Missing assets remain missing, and
neutralization returns `NaN` when a timestamp has too few valid assets for the
requested regression.

## Research limitation

The configured 12-symbol core universe is an engineering starter universe. It
does not reconstruct historical listings, delistings, or point-in-time
eligibility and therefore cannot support a claim of survivorship-bias-free OOS
validation by itself.
