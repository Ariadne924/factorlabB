# 因子签名 v2：DataFrame 输入 + 参数化工厂

**上下文**：原骨架因子签名为 `(open, high, low, close, volume) -> pd.Series`，限制了因子只能使用 OHLCV 数据。加密货币特有因子（资金费率、链上数据）和多周期复合因子无法在这种签名下实现。

**决策**：

1. 因子签名改为 `(df: pd.DataFrame) -> pd.Series`，输入 DataFrame 必须包含 `open/high/low/close/volume` 五列，可额外包含任意列
2. 因子注册从存储裸函数改为存储**工厂函数**：`(params: dict | None) -> FactorFunction`
3. 使用时通过 `build_factor("RSI", params={"window": 7})` 获取参数化后的计算函数

**理由**：
- DataFrame 输入让因子自行选择需要的列（资金费率、taker_buy_volume、链上数据等），不修改签名即可扩展
- 工厂模式解决了 `rsi_7`/`rsi_14`/`rsi_21` 必须注册三次的问题——一套逻辑，可变参数
- `REQUIRED_COLUMNS` 校验保证最小输入合约不被破坏

**后果**：
- 旧版 `FactorFunction` Protocol 和 `factor_template` 需要更新
- 旧版 `FACTOR_REGISTRY` 的 `"func"` 键改为 `"factory"` 键
- `compute_factor` 现在接收 `df` 而非五个独立 Series
