# Qlib-style OHLCV 因子目录

## 来源与使用边界

本批 20 个因子参考 [Microsoft Qlib Alpha158 loader](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py) 的公开表达式，在本项目内独立实现。Qlib 使用 MIT License。

这里只采用可由单资产历史 OHLCV 计算的时间序列特征。依赖横截面排名、行业中性化或全市场可交易集合的表达式没有移植，也没有把短样本结果包装成已验证 alpha。所有 rolling 窗口都只包含当前及更早的 K 线。

## 正式因子

| 因子 | 定义摘要 | 默认参数 | 数据依赖 |
|---|---|---:|---|
| `k_mid` | `(close-open)/open` | 无 | OHLC |
| `k_length` | `(high-low)/open` | 无 | OHLC |
| `k_mid2` | `(close-open)/(high-low)` | 无 | OHLC |
| `upper_shadow` | 上影线/open | 无 | OHLC |
| `upper_shadow2` | 上影线/(high-low) | 无 | OHLC |
| `lower_shadow` | 下影线/open | 无 | OHLC |
| `lower_shadow2` | 下影线/(high-low) | 无 | OHLC |
| `k_shift` | `(2*close-high-low)/open` | 无 | OHLC |
| `k_shift2` | `(2*close-high-low)/(high-low)` | 无 | OHLC |
| `linear_trend_slope` | rolling 线性趋势斜率/close | 24 | close |
| `trend_r_squared` | rolling 线性趋势拟合优度 | 24 | close |
| `trend_residual` | 最新价相对 rolling 趋势的残差/close | 24 | close |
| `price_position_rsv` | close 在 rolling high-low 区间的位置 | 24 | HLC |
| `high_recency` | rolling 最高价出现位置，越接近 1 越新 | 24 | high |
| `low_recency` | rolling 最低价出现位置，越接近 1 越新 | 24 | low |
| `high_low_recency_diff` | high recency - low recency | 24 | HL |
| `price_volume_correlation` | close 与 log(volume) 的 rolling 相关性 | 24 | close, volume |
| `return_volume_correlation` | 收益率与 log-volume 变化的 rolling 相关性 | 24 | close, volume |
| `volume_weighted_volatility` | 成交量加权价格冲击的离散度 | 24 | close, volume |
| `volume_change_strength` | 有符号成交量变化/绝对成交量变化 | 24 | volume |

## 经济含义与局限

- K 线形态因子描述单根 K 线内买卖力量与收盘位置，不保证跨交易所或跨频率稳定。
- 趋势因子描述过去窗口的方向、拟合质量、区间位置和极值新近程度；高趋势拟合度不等于未来收益可预测。
- 量价因子描述价格与成交活动的同步关系，容易受成交量口径、异常成交和交易所结构变化影响。
- 默认窗口以 K 线根数计，不等于自然日。使用不同频率时必须重新做参数与 horizon 稳健性检验。
- 分母为零时输出 `NaN`，不使用未来数据填补。

## 评估字段

每个因子的 JSON 报告包含 IC、RankIC、ICIR、IC decay、分层收益、换手率、rolling IC、前视检查、成本假设，以及 30/60/90/180 天 × 1/3/6/12/24 根 K 线的稳健性网格、区块 bootstrap RankIC 区间、符号一致性、分层单调性和 BH-FDR 结果。没有真实 Silver 数据时这些实证字段保持空值。

