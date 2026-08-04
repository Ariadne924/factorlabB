# 统一 Schema + 抽象交易所接口

**上下文**：项目需要对接多个交易所（Binance 起步，后续扩展 OKX/Bybit）。各交易所 API 返回格式不同，如果因子层直接消费原始 API 数据，每加一个交易所就要改所有因子代码。

**决策**：
1. 用 Pydantic BaseModel 定义三套统一 Schema（KlineSchema / TradeSchema / FundingRateSchema），所有交易所数据先标准化再进入下游
2. 用 ABC 定义 `ExchangeBase` 抽象基类，统一 `fetch_klines` / `fetch_trades` / `fetch_funding_rate` / `ping` 四个接口
3. Schema 启用 `model_config = {"extra": "forbid"}` —— 交易所 API 新增字段时报错而非静默丢弃

**理由**：
- 因子层只依赖 Schema，不依赖具体交易所，实现了解耦
- `extra: forbid` 是防御性设计——交易所 API 变更是线上最高频的故障源，静默吞掉新字段比报错更危险
- ABC 模式让新增交易所的成本降到"实现四个方法"，不用碰其他代码

**后果**：
- Schema 字段变更影响全局，需全组通知 + ADR
- Pydantic 校验有运行时开销，但相对于网络 IO 可忽略
