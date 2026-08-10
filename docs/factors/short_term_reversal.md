# Short-Term Reversal

## 定义与经济含义

- 定义：`-(close_t / close_{t-window} - 1)`。
- 经济含义：检验短期过度反应后的均值回归。

## 数据依赖与参数

- 数据依赖：Silver `close`；默认 `window=3`。
- 时间对齐：无负 shift，只使用当前及历史 K 线。

## 评估结果

- IC / RankIC / ICIR、IC decay、分层测试、换手率：暂无真实样本结果。
- 前视检查：运行状态写入 `reports/single_factor/*.json`。
- 成本假设：手续费 0.10% + 滑点 0.05%（单边）。

## 局限

通常换手较高且易被成本吞噬；未完成六个月 OOS，不能称为已验证 alpha。
