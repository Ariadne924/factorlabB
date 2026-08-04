# 量化因子库 (Quant Factor Library)

数字资产（加密货币）量化因子研究与评估框架。为小组协作设计：统一接口、因子注册机制、多维评估体系。

## Language

**因子 (Factor)**：
从 OHLCV 行情数据中计算出的时序信号，用于预测未来收益。所有因子遵循统一签名 `(open, high, low, close, volume) -> pd.Series`。
_Avoid_：指标、信号、特征

**因子注册表 (Factor Registry)**：
全局字典 `FACTOR_REGISTRY`，存储所有已注册因子的元信息（函数、分类、描述、参数）。新因子通过 `@register_factor` 装饰器注册。
_Avoid_：因子列表、因子集合

**K 线 (Kline)**：
单根 OHLCV 柱线的标准化表示。时间戳统一 UTC，周期由 `KlineInterval` 枚举定义。
_Avoid_：蜡烛图（仅在可视化语境下使用）、Bar

**交易对 (Symbol)**：
以字符串表示的资产对，如 `'BTCUSDT'`。全项目统一使用字符串类型，不做类型转换。
_Avoid_：币对、Pair

**前视偏差 (Look-Ahead Bias)**：
因子计算中错误使用了未来信息的现象。`ForwardCheck` 是专门检查此问题的模块。
_Avoid_：未来函数、数据泄露（Data Leakage，在因子语境下指同一概念）

**IC (Information Coefficient)**：
因子值与未来收益的截面相关系数（Spearman 或 Pearson），衡量因子的预测能力。
_Avoid_：信息系数、Rank IC（特指 Spearman 版本时使用）

**分组测试 (Grouping Test)**：
将因子值排序后均分为 N 组，计算各组等权收益，观察多空收益差（Top-Bottom Spread）。
_Avoid_：分层测试、Quantile test

**资金费率 (Funding Rate)**：
永续合约定期在多空双方之间划转的费用比率，是加密货币市场特有的交易机制。
_Avoid_：Funding fee

**前向收益 (Forward Return)**：
因子值已知时刻 t 到未来时刻 t+Δ 之间的资产收益率。用于评估因子的预测能力。时间对齐必须严格：因子值的时间 t 必须早于收益区间。
_Avoid_：目标收益、未来收益

**数据 Schema**：
用 Pydantic BaseModel 定义的统一数据格式（KlineSchema / TradeSchema / FundingRateSchema），所有交易所数据进入系统后先标准化为 Schema 格式。
_Avoid_：数据模型、DTO

## Relationships

- **K 线 → 因子**：所有因子以 OHLCV 序列为输入，产出一个 pd.Series
- **因子 → 因子注册表**：每个因子必须注册后才能被评估框架和可视化模块发现
- **Schema → 交易所客户端**：所有交易所客户端的数据输出必须先转为 Schema 格式
- **因子 → 前向收益**：评估时因子值与未来收益对齐，时间顺序由 ForwardCheck 校验

## Flagged ambiguities

- "tick" 在市场微观结构语境中可能指逐笔成交 — 本项目统一用 **Trade** / 成交
- "volatility" 既指因子分类，也指波动率数值 — 分类用 **波动率因子**，数值用 **Vol**
