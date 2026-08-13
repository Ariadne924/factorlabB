# 前端数据契约

前端不应直接扫描 Bronze、Silver、Gold 或依赖 Python 内部对象。研究任务完成后只读取以下
UTF-8 JSON 文件。

## 常规因子入口

`reports/frontend_payload.json`

- `contract_version`：当前为 `1.2`；
- `status` 与 `disclaimer`：研究状态和强制风险说明；
- `overview`：因子数、已计算报告数和 FDR 通过数；
- `filters`：资产、频率、类别和来源枚举；
- `factor_results`：各资产 × 频率 × 因子的指标和报告路径；
- `panel_factor_results`：同一时点多币种截面因子的指标、稳健性和来源；
- `data_catalog`：Silver 覆盖目录的相对路径；
- `data_health`：数据健康报告路径及覆盖、新鲜度、ML 可用性摘要；
- `cross_frequency`：跨资产、跨频率方向一致性汇总。

单因子图表的完整时序数据继续从 `reports/single_factor/{SYMBOL}_{INTERVAL}_{FACTOR}.json`
按 `report` 字段加载。

## 机器学习入口

`reports/ml_factor_summary.json`

- `contract_version`：当前为 `1.0`；
- `model`：当前基线为 Ridge；
- `filters`：已生成结果的资产和交易所频率；
- `results`：状态、候选特征数、walk-forward 折数、指标及详细报告路径；
- `oos_6_months_completed`：当前固定为 `false`，除非未来真正完成锁定 OOS。

详细模型报告位于 `reports/ml_factor/{SYMBOL}_{INTERVAL}_ml_ridge_composite.json`，包含每折
训练/测试边界、训练折筛选分数、权重、`strategy_preset` 和 leakage controls。预设仅由训练折
推荐生成，包含正式因子、方向、参数变体及归一化候选权重。

## 兼容原则

- `1d` 在展示层转换为 `24h`，存储和交易所请求仍使用 `1d`；
- 前端必须展示 `disclaimer/research_note`，不得把候选结果标记为“已验证 Alpha”；
- 契约 `1.x` 版本只增加字段，不删除或改名；破坏性变化必须升级主版本；
- 缺失指标使用 JSON `null`，前端显示 `N/A`，不得显示为零。
