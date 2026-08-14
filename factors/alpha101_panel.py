"""First cross-sectional Alpha101 batch adapted to crypto bar panels."""

from __future__ import annotations

from factors.panel_registry import register_panel_expression

SOURCE = "101 Formulaic Alphas (Kakushadze, 2016)"
SOURCE_URL = "https://arxiv.org/abs/1601.00991"
CATEGORY = "alpha101_cross_sectional"
ADAPTATION = (
    "Original daily-equity window integers are interpreted as bar counts at the "
    "selected crypto frequency. Industry neutralization is not inferred. VWAP is "
    "constructed as quote_volume / volume. Candidate status only; not validated alpha."
)
VWAP = "protected_div(quote_volume,volume)"
RETURNS = "returns(close,1)"


ALPHA101_PANEL_EXPRESSIONS: dict[str, tuple[str, str]] = {
    "alpha101_001": (
        f"cs_rank(argmax(signed_power(where(lt({RETURNS},0),"
        f"rolling_std({RETURNS},20),close),2),5))-0.5",
        "rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), "
        "2.), 5)) - 0.5",
    ),
    "alpha101_002": (
        "(-corr(cs_rank(delta(log(volume),2)),"
        "cs_rank(protected_div(close-open,open)),6))",
        "(-1 * correlation(rank(delta(log(volume), 2)), "
        "rank(((close - open) / open)), 6))",
    ),
    "alpha101_003": (
        "(-corr(cs_rank(open),cs_rank(volume),10))",
        "(-1 * correlation(rank(open), rank(volume), 10))",
    ),
    "alpha101_004": (
        "(-rolling_rank(cs_rank(low),9))",
        "(-1 * Ts_Rank(rank(low), 9))",
    ),
    "alpha101_005": (
        f"cs_rank(open-rolling_mean({VWAP},10))*(-abs(cs_rank(close-{VWAP})))",
        "rank((open - (sum(vwap, 10) / 10))) * "
        "(-1 * abs(rank((close - vwap))))",
    ),
    "alpha101_006": (
        "(-corr(open,volume,10))",
        "(-1 * correlation(open, volume, 10))",
    ),
    "alpha101_007": (
        "where(lt(rolling_mean(volume,20),volume),"
        "(-rolling_rank(abs(delta(close,7)),60))*sign(delta(close,7)),-1)",
        "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * "
        "sign(delta(close, 7))) : (-1 * 1))",
    ),
    "alpha101_008": (
        f"(-cs_rank((rolling_sum(open,5)*rolling_sum({RETURNS},5))-"
        f"delay(rolling_sum(open,5)*rolling_sum({RETURNS},5),10)))",
        "(-1 * rank(((sum(open, 5) * sum(returns, 5)) - "
        "delay((sum(open, 5) * sum(returns, 5)), 10))))",
    ),
    "alpha101_009": (
        "where(gt(rolling_min(delta(close,1),5),0),delta(close,1),"
        "where(lt(rolling_max(delta(close,1),5),0),delta(close,1),-delta(close,1)))",
        "((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : "
        "((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))",
    ),
    "alpha101_010": (
        "cs_rank(where(gt(rolling_min(delta(close,1),4),0),delta(close,1),"
        "where(lt(rolling_max(delta(close,1),4),0),delta(close,1),-delta(close,1))))",
        "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : "
        "((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : "
        "(-1 * delta(close, 1)))))",
    ),
    "alpha101_011": (
        f"(cs_rank(rolling_max({VWAP}-close,3))+"
        f"cs_rank(rolling_min({VWAP}-close,3)))*cs_rank(delta(volume,3))",
        "(rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) "
        "* rank(delta(volume, 3))",
    ),
    "alpha101_012": (
        "sign(delta(volume,1))*(-delta(close,1))",
        "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    ),
    "alpha101_013": (
        "(-cs_rank(cov(cs_rank(close),cs_rank(volume),5)))",
        "(-1 * rank(covariance(rank(close), rank(volume), 5)))",
    ),
    "alpha101_014": (
        f"(-cs_rank(delta({RETURNS},3)))*corr(open,volume,10)",
        "(-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)",
    ),
    "alpha101_015": (
        "(-rolling_sum(cs_rank(corr(cs_rank(high),cs_rank(volume),3)),3))",
        "(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))",
    ),
    "alpha101_016": (
        "(-cs_rank(cov(cs_rank(high),cs_rank(volume),5)))",
        "(-1 * rank(covariance(rank(high), rank(volume), 5)))",
    ),
    "alpha101_017": (
        "(-cs_rank(rolling_rank(close,10)))*"
        "cs_rank(delta(delta(close,1),1))*"
        "cs_rank(rolling_rank(protected_div(volume,rolling_mean(volume,20)),5))",
        "(-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * "
        "rank(ts_rank((volume / adv20), 5))",
    ),
    "alpha101_018": (
        "(-cs_rank(rolling_std(abs(close-open),5)+(close-open)+corr(close,open,10)))",
        "(-1 * rank(stddev(abs((close - open)), 5) + (close - open) + "
        "correlation(close, open, 10)))",
    ),
    "alpha101_020": (
        "(-cs_rank(open-delay(high,1)))*cs_rank(open-delay(close,1))*"
        "cs_rank(open-delay(low,1))",
        "(-1 * rank((open - delay(high, 1)))) * "
        "rank((open - delay(close, 1))) * rank((open - delay(low, 1)))",
    ),
    "alpha101_021": (
        "where(lt(rolling_mean(close,8)+rolling_std(close,8),"
        "rolling_mean(close,2)),-1,"
        "where(lt(rolling_mean(close,2),rolling_mean(close,8)-"
        "rolling_std(close,8)),1,"
        "where(ge(protected_div(volume,rolling_mean(volume,20)),1),1,-1)))",
        "Conditional comparison of 8-day mean/std, 2-day mean, and volume/adv20",
    ),
    "alpha101_023": (
        "where(lt(rolling_mean(high,20),high),-delta(high,2),0)",
        "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    ),
    "alpha101_024": (
        "where(le(protected_div(delta(rolling_mean(close,100),100),"
        "delay(close,100)),0.05),-(close-rolling_min(close,100)),-delta(close,3))",
        "Conditional 100-day moving-average change versus close history",
    ),
    "alpha101_025": (
        f"cs_rank((-{RETURNS})*rolling_mean(volume,20)*{VWAP}*(high-close))",
        "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))",
    ),
    "alpha101_026": (
        "(-rolling_max(corr(rolling_rank(volume,5),rolling_rank(high,5),5),3))",
        "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",
    ),
    "alpha101_028": (
        "cs_scale(corr(rolling_mean(volume,20),low,5)+(high+low)/2-close)",
        "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    ),
}


for _name, (_expression, _source_formula) in ALPHA101_PANEL_EXPRESSIONS.items():
    register_panel_expression(
        _name,
        _expression,
        category=CATEGORY,
        description=f"Crypto panel adaptation of {_name.replace('_', ' ').title()}",
        source=SOURCE,
        source_url=SOURCE_URL,
        source_formula=_source_formula,
        adaptation_note=ADAPTATION,
    )
