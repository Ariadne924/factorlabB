"""Technical inputs described by the cryptocurrency CTREND study."""

from __future__ import annotations

from factors.panel_registry import register_panel_expression

SOURCE = "A Trend Factor for the Cross Section of Cryptocurrency Returns"
SOURCE_URL = "https://doi.org/10.1017/S0022109024000747"
CATEGORY = "crypto_trend_technical"
ADAPTATION = (
    "Technical input to the CTREND research design, not the paper's fitted CS-C-ENet "
    "forecast. Daily window integers are bar counts at the selected frequency. "
    "Candidate status only; no paper performance is transferred to this dataset."
)

DELTA = "delta(close,1)"
GAIN = f"where(gt({DELTA},0),{DELTA},0)"
LOSS = f"where(lt({DELTA},0),-{DELTA},0)"
RSI = f"100-(100/(1+protected_div(rolling_mean({GAIN},14),rolling_mean({LOSS},14))))"
STOCH_K = (
    "100*protected_div(close-rolling_min(low,14),"
    "rolling_max(high,14)-rolling_min(low,14))"
)
TYPICAL = "(high+low+close)/3"
EMA_FAST = "ewm_mean(close,12)"
EMA_SLOW = "ewm_mean(close,26)"
MACD = f"protected_div({EMA_FAST}-{EMA_SLOW},{EMA_FAST})"
VOL_FAST = "ewm_mean(quote_volume,12)"
VOL_SLOW = "ewm_mean(quote_volume,26)"
VOL_MACD = f"protected_div({VOL_FAST}-{VOL_SLOW},{VOL_FAST})"
BOLL_MID = "rolling_mean(close,20)"
BOLL_STD = "rolling_std(close,20)"

CTREND_TECHNICAL_EXPRESSIONS: dict[str, str] = {
    "ctrend_rsi_14": RSI,
    "ctrend_stoch_k_14": STOCH_K,
    "ctrend_stoch_d_14_3": f"rolling_mean({STOCH_K},3)",
    "ctrend_stoch_rsi_14": (
        f"protected_div({RSI}-rolling_min({RSI},14),"
        f"rolling_max({RSI},14)-rolling_min({RSI},14))"
    ),
    "ctrend_cci_20": (
        f"protected_div({TYPICAL}-rolling_mean({TYPICAL},20),"
        f"0.015*rolling_mad({TYPICAL},20))"
    ),
    **{
        f"ctrend_sma_{window}": f"protected_div(rolling_mean(close,{window}),close)"
        for window in (3, 5, 10, 20, 50, 100, 200)
    },
    "ctrend_macd": MACD,
    "ctrend_macd_diff_signal": f"{MACD}-ewm_mean({MACD},9)",
    **{
        f"ctrend_volume_sma_{window}": (
            f"protected_div(rolling_mean(quote_volume,{window}),quote_volume)"
        )
        for window in (3, 5, 10, 20, 50, 100, 200)
    },
    "ctrend_volume_macd": VOL_MACD,
    "ctrend_volume_macd_diff_signal": f"{VOL_MACD}-ewm_mean({VOL_MACD},9)",
    "ctrend_chaikin_20": (
        "protected_div(rolling_sum(protected_div((close-low)-(high-close),"
        "high-low)*volume,20),rolling_sum(volume,20))"
    ),
    "ctrend_boll_low_20": f"protected_div({BOLL_MID}-2*{BOLL_STD},close)",
    "ctrend_boll_mid_20": f"protected_div({BOLL_MID},close)",
    "ctrend_boll_high_20": f"protected_div({BOLL_MID}+2*{BOLL_STD},close)",
    "ctrend_boll_width_20": f"protected_div(4*{BOLL_STD},{BOLL_MID})",
}


for _name, _expression in CTREND_TECHNICAL_EXPRESSIONS.items():
    register_panel_expression(
        _name,
        _expression,
        category=CATEGORY,
        description=f"CTREND technical input: {_name.removeprefix('ctrend_')}",
        source=SOURCE,
        source_url=SOURCE_URL,
        source_formula=_expression,
        adaptation_note=ADAPTATION,
    )
