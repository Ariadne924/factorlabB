# Open Interest Change

## 定义与经济含义

- 定义：`OI_t / OI_{t-window} - 1`。
- 经济含义：新增杠杆仓位和参与度变化代理，不直接区分多空。

## 数据依赖与参数

- 数据依赖：永续 `open_interest`；默认 `window=12`。
- 时间对齐：API 时间戳按 backward as-of 合并，未来 OI 不会进入历史行。

## 评估结果

- IC / RankIC / ICIR、IC decay、分层测试、换手率：暂无真实样本结果。
- 前视检查：研究脚本执行并留痕；成本假设单边 0.15%。

## 局限

名义币量与美元价值口径不同，交易所迁移会扰动；六个月 OOS 未完成。
