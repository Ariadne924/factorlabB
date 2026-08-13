# CTREND Technical Inputs

This module implements the 28 technical inputs described in *A Trend Factor for
the Cross Section of Cryptocurrency Returns*:

- five momentum oscillators;
- seven scaled price SMAs plus MACD and its signal difference;
- seven scaled dollar-volume SMAs, volume MACD, its signal difference, and
  Chaikin money flow;
- four Bollinger-band measures.

Daily window integers from the paper are interpreted as bar counts at the
selected project frequency. The formulas therefore support engineering and
frequency-robustness research, but an hourly result is not the paper's daily or
weekly result.

The paper combines these inputs using a rolling cross-sectional combined elastic
net. This module deliberately registers the inputs only. It does not label an
equal-weight average or the existing Ridge baseline as the paper's fitted CTREND
signal. All empirical fields remain empty until real multi-symbol panel research
is run.
