# Panel Research and Strategy Builder

## Offline panel research

```bash
python scripts/run_panel_research.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT \
  --intervals 1h 6h 24h
```

The script consumes existing Silver files only. At least three symbols at one
frequency are required. It produces same-timestamp IC and RankIC, ICIR, rolling
RankIC, explicit-horizon decay, group returns, turnover, cost sensitivity,
lookahead status, and BH-FDR fields.

LTW common-risk-factor returns are produced only when point-in-time
`market_cap` is present. Dollar volume is never used as a substitute.

## Reusable data coverage

`reports/data_catalog.json` records symbol, frequency, time range, rows,
coverage ratio, and internal gaps for every Silver dataset. The download script
uses it to request only uncovered edges and gaps. Fully covered requests are
marked `up_to_date`.

## Interactive strategy builder

The Streamlit Strategy Builder supports multiple registered panel factors,
weights, directions, an optional safe custom expression, symbol/frequency/date
selection (including 7/30/90-day presets), forward-return horizon, long-short
or long-only positions, rebalance frequency, fees, and slippage.

Inputs are standardized within each timestamp before weighted combination.
Positions formed at time t are evaluated against explicit forward returns. The
output includes gross/net returns, equity, drawdown, turnover, and input
coverage.

This is a research backtest, not an exchange execution simulator. It does not
model queue position, partial fills, borrow availability, liquidation, or market
impact, and it does not establish six-month OOS validation.
