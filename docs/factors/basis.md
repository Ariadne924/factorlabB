# Basis

## 定义与经济含义

- 定义：`(mark_price-index_price)/index_price` 的 window 历史均值。
- 经济含义：永续相对指数的溢折价和杠杆需求代理。

## 数据依赖与参数

- 数据依赖：永续标记价、指数价形成的 `basis`；默认 `window=3`。
- 时间对齐：仅 backward as-of 合并到 K 线；不得把当前快照回填历史。

## 评估结果

- IC / RankIC / ICIR、IC decay、分层测试、换手率：暂无真实数据结果。
- 前视检查：点时合并和截断不变性；成本假设单边 0.15%。

## 局限

当前 REST 方法仅提供实时快照，历史研究需可靠归档；六个月 OOS 未完成，不能声称 alpha 已验证。
