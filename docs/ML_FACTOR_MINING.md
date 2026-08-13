# 机器学习因子挖掘

## 当前范围

第一版不直接搜索不可解释的任意公式，而是把 registry 中已有候选因子及其受控窗口变体
作为特征，训练 Ridge 复合因子。每个输出值都来自严格的 expanding walk-forward 样本外预测。

```text
历史候选因子 → 参数变体 → 训练期筛选/去相关 → 训练期标准化 → Ridge → 下一测试窗复合因子
```

## 防泄漏约束

- 时间序列禁止随机打乱；
- 标准化均值和方差只在训练集拟合；
- 与训练目标的相关性筛选、覆盖率检查和高相关去冗余只使用当前训练折；
- 预测 horizon 对应的训练标签必须在测试窗开始前完全实现；
- 训练期和测试期之间保留 embargo；
- 报告只评价 walk-forward OOS 预测，不评价训练内拟合值；
- 结果仍需跨资产、跨频率以及锁定参数后的独立 OOS 验证。

## 运行

```bash
python scripts/run_ml_factor_mining.py \
  --symbols BTCUSDT ETHUSDT \
  --intervals 1m 5m 15m 1h 6h 24h \
  --min-train-days 90 \
  --test-days 14 \
  --embargo-bars 1 \
  --parameter-windows 6 12 24 48 \
  --max-features 80 \
  --max-pairwise-correlation 0.95
```

输出位于 `reports/ml_factor/`，汇总位于 `reports/ml_factor_summary.json`，并可在
Streamlit 的 `ML Factor Mining` 页面查看。

## 推荐与快速回测

页面只汇总训练折中的权重与筛选分数。推荐排序结合特征在各折的入选频率、权重方向
一致性和平均绝对权重，不使用测试折收益构造推荐分数。

随后把严格 OOS 的 Ridge 预测转换为透明的多空方向，并在对应未来收益上扣除单边手续费
和滑点，输出净值、回撤、换手率与方向命中率。这属于快速研究推断，不是交易执行模拟，
也不能单独证明因子有效。

## 后续方向

在 Ridge 基线稳定后，再考虑非线性树模型、符号表达式搜索和遗传规划。所有模型必须复用
相同的 walk-forward、purge/embargo、成本和多重检验口径，不能用随机 K-fold 结果替代。
