# Cryptocurrency Common Risk Factors

The project implements the construction logic described in Liu, Tsyvinski, and
Wu, *Common Risk Factors in Cryptocurrency*:

- CMKT: market-capitalization-weighted cryptocurrency market return;
- CSMB: value-weighted bottom-30% size portfolio minus top-30%;
- CMOM: value-weighted top-30% momentum portfolio minus bottom-30%.

The paper forms portfolios weekly and uses three-week momentum for CMOM. The
implementation requires callers to map these periods into bars explicitly.

`market_cap` is a strict point-in-time dependency. Binance OHLCV or dollar volume
is not treated as a substitute. Until an external historical market-cap source is
merged backward by publication timestamp, the research pipeline reports this
benchmark as unavailable rather than manufacturing factor returns.
