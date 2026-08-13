# 数字资产结构与派生因子目录

本轮新增20个候选因子。所有滚动、差分和交互仅使用当前及历史观测；它们是待检验特征，
不代表已经验证的交易 Alpha。

| 类别 | 因子 |
|---|---|
| Funding | `funding_rate_zscore`、`funding_rate_change`、`funding_rate_acceleration`、`funding_price_divergence` |
| Open Interest | `open_interest_zscore`、`open_interest_momentum`、`open_interest_price_divergence`、`open_interest_volume_confirmation` |
| Basis | `basis_zscore`、`basis_change`、`basis_funding_spread` |
| 主动成交 | `taker_imbalance`、`taker_imbalance_momentum` |
| 流动性 | `signed_price_impact`、`volume_shock_zscore`、`trade_intensity_zscore`、`average_trade_size_zscore` |
| 状态交互 | `volatility_term_ratio`、`momentum_volatility_interaction`、`momentum_liquidity_interaction` |

## 使用边界

- Funding、OI 和 Basis 必须经过 point-in-time 向后合并；
- Basis 只在来源统计周期结束后可用；
- 1m 研究中的 OI/Basis 使用已经结束的 5m 观测；
- 缺少可选数据列时因子显式返回依赖错误，不以未来值回填；
- 参数变体由机器学习入口受控生成，筛选与去相关只发生在训练折内。
