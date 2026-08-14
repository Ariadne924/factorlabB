# Task 2 回测框架选型

## 结论

采用“保留自建研究内核 + 借鉴成熟框架执行语义”的混合方案。本轮不直接迁移到
Freqtrade、LEAN 或 Gekko，也不复制任何第三方回测结论。

原因是当前项目已经拥有统一 Silver/Gold 数据、因子 registry、无前视检查、截面研究、
ML Walk-Forward 和 Streamlit 工作流。整体迁移会破坏 Task 1 已验证接口，并增加数据格式
转换与结果口径不一致风险。新增 `strategies/` 和 `trading/` 作为解耦层即可满足 Task 2/3。

## 横向比较

| 方案 | 适合点 | 当前不直接采用的原因 | 借鉴内容 |
|---|---|---|---|
| Freqtrade | Python、数字资产优先，内置回测、dry-run、SQLite 和 WebUI | 交易机器人工作流较完整，但直接接入需要把现有因子、面板和报告迁入其策略/数据约定 | dry-run 优先、SQLite 状态、回测与前向对照、前视检查 |
| QuantConnect LEAN | 事件驱动、订单/费用/滑点/保证金模型成熟，回测与实盘接口完整 | 引擎主体为 C#，本地 CLI/Docker 工作流较重；对本项目当前 Python 研究链改造范围过大 | 订单事件、Brokerage Model、可替换费用/滑点/保证金模型 |
| Gekko | 结构简单，曾面向数字资产回测和交易 | 官方仓库已于 2020-02-16 归档并明确不再维护，不适合作为新项目基座 | 仅作为历史架构参考 |
| 自建轻量层 | 与 Task 1 完全兼容，口径透明，便于阈值与一致性研究 | 需要自行承担订单状态、风控、恢复和测试 | 本项目采用；严格限制范围，避免重造生产级交易所基础设施 |

## 一手来源

- Freqtrade 官方说明：<https://github.com/freqtrade/freqtrade>
- Freqtrade 回测文档：<https://www.freqtrade.io/en/stable/backtesting/>
- Freqtrade 策略与 dry-run 说明：<https://docs.freqtrade.io/en/stable/strategy-101/>
- LEAN 引擎说明：<https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/algorithm-engine>
- LEAN 订单与现实建模：<https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/key-concepts>
- Gekko 归档仓库：<https://github.com/askmike/gekko>

以上信息于 2026-08-14 核对。最终实现仍以本仓库测试和报告为准，第三方项目的宣传或回测
结果不作为本项目策略有效性的证据。

