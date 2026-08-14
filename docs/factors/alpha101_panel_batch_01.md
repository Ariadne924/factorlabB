# Alpha101 Panel Batch 01

## Source and scope

The first two implementation passes cover formulas #1 through #18 where the
required operators are available, plus #20, #21, #23, #24, #25, #26, and #28
from [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991). This currently
totals 25 registered panel candidates.

The formulas are registered under their original numbers, for example
`alpha101_002`. Original formula text, the executable canonical expression,
source URL, dependencies, hash, complexity, and adaptation note are retained in
the panel registry.

## Crypto adaptation

- Original daily windows are treated as bar counts. A window of 10 therefore
  means ten selected-frequency bars, not automatically ten days.
- `VWAP` is constructed from `quote_volume / volume`.
- `adv20` is implemented as the trailing 20-bar mean volume.
- Time-series operators run independently per symbol.
- Rank operators run across available symbols at the same timestamp.
- No industry neutralization or point-in-time universe reconstruction is
  inferred.

## Evaluation status

These are formula candidates, not validated crypto alpha. Their original paper
results concern a different asset class and sampling convention. IC, RankIC,
ICIR, decay, turnover, group returns, costs, and walk-forward behavior must be
computed from the project's crypto panel before any empirical conclusion.

The implementation tests alignment, finite output, dependency failures, and
historical invariance after changing the final timestamp. A passing lookahead
test establishes implementation discipline; it does not establish profitability.
