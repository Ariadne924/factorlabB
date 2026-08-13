# 开源因子来源与接入边界

## 已接入

### Microsoft Qlib

- 来源：https://github.com/microsoft/qlib
- 当前实现：20个 Alpha158-style OHLCV 特征与360个 Alpha360 滞后特征；
- 实现方式：根据公开表达式在本项目中独立实现，不把 Qlib 作为运行时依赖；
- Alpha360：`close/open/high/low/vwap/volume × lag 0..59`；
- Binance bar VWAP：`quote_volume / volume`；缺少字段时保持缺失或显式失败；
- Scope：当前是单资产时间序列特征，不冒充横截面因子。

### 101 Formulaic Alphas

- 论文：Zura Kakushadze, *101 Formulaic Alphas*；
- 当前实现：25 个可在加密货币 point-in-time 面板上忠实适配的公式；
- `rank`、`scale` 等在同一 UTC 时间截面计算，历史算子按币种分别计算；
- 依赖行业分类的公式尚未接入，不把行业中性化静默删除；
- 日频窗口被明确解释为当前频率的 K 线根数，不转移论文收益结论。

### CTREND technical inputs

- 论文：*A Trend Factor for the Cross Section of Cryptocurrency Returns*；
- 当前实现：论文描述的 28 个价格、成交额与技术指标输入；
- 当前不声称复现论文拟合后的 CS-C-ENet CTREND 因子；
- 公式、依赖和适配说明保存在独立 panel registry。

## 后续可选来源

### AlphaGen

- 来源：https://github.com/ICT-FinD-Lab/alphagen
- 用途：参考公式语法、协同因子池奖励和强化学习搜索设计；
- 本项目先实现可复现的表达式引擎与 walk-forward 评价，再决定是否引入 PyTorch/RL。

### AlphaForge

- 来源：https://github.com/DulyHao/AlphaForge
- 用途：参考“先挖掘因子池、再动态选择组合”的两阶段流程；
- 不直接引入其股票数据与实验配置。

### gplearn

- 来源：https://github.com/trevorstephens/gplearn
- License：BSD-3-Clause；
- 用途：后续作为遗传编程/符号变换基线；
- 会增加 scikit-learn、SciPy 等依赖，因此不在当前基础环境中强制安装。

## 数量口径

- 正式公式：唯一经济或数学定义；
- 参数变体：同一公式不同窗口，单独计入候选特征，不冒充独立因子；
- 自动生成公式：必须保存表达式、训练区间、随机种子和父代来源；
- 稳健候选：通过前视、覆盖率、相关性、成本和 walk-forward 门槛后才能升级状态。
