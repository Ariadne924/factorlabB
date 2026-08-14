# Amihud Illiquidity

## 定义与经济含义

- 定义：`abs(return) / quote_volume` 的 window 滚动均值。
- 经济含义：单位成交额对应的价格冲击代理。

## 数据依赖与参数

- 数据依赖：Silver `close`、`quote_volume`（缺失时以 `close*volume` 代理）；默认 `window=20`。
- 时间对齐：只使用当前及历史成交额。

## 评估结果

- IC / RankIC / ICIR、IC decay、分层测试、换手率：暂无真实报告结果。
- 前视检查：由截断重算验证；成本假设单边 0.10% + 0.05%。

## 局限

不同计价币与合约口径不可直接比较，零成交额产生缺失；未完成六个月 OOS。
