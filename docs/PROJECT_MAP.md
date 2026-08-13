# 项目框架图

## 一条完整研究链路

```text
Binance / OKX
    ↓
Bronze 原始数据
    ↓ Schema + 质量检查
Silver UTC 标准数据 ── data_catalog.json（覆盖范围与缺口）
    ↓
单资产因子 registry（413）       多资产 panel registry（53）
    ↓                              ↓
Gold 因子值                    截面因子值
    └──────────────┬───────────────┘
                   ↓
IC / RankIC / ICIR / decay / 分组 / 换手 / 成本 / 前视 / FDR
                   ↓
JSON 报告 + frontend_payload.json
                   ↓
Streamlit Factor Explorer / Strategy Builder / Data Center
```

## 各目录负责什么

| 目录 | 职责 | 主要扩展 |
|---|---|---|
| `data/` | 交易所接口、Schema、Bronze/Silver、质量检查 | 多频率下载、增量覆盖目录、面板加载、逐笔成交、盘口、衍生品特征 |
| `factors/` | 因子定义与注册 | 413 个单资产候选、53 个截面候选、安全表达式引擎、截面算子 |
| `evaluation/` | 因子与策略检验 | 跨频率汇总、面板 IC、稳健性、机器学习 walk-forward、多因子回测 |
| `scripts/` | 可复现运行入口 | 数据计划/补缺、全部研究、面板研究、ML 因子挖掘 |
| `visualization/` | 报告与交互 | Factor Explorer、Panel Factors、Strategy Builder、Data Center、前端 JSON 契约 |
| `docs/` | 因子卡、方法与边界 | 来源、定义、数据依赖、前视规则、未验证声明 |
| `tests/` | 单元与回归测试 | mock API/WS、表达式安全、面板对齐、未来不变性、报告契约 |

Streamlit 使用深海军蓝背景、青绿色主色和浅色文字；配置位于 `.streamlit/config.toml`，
后续独立前端可以直接复用这组基础色。

## 因子口径

- 单资产候选 413 个：原有正式因子、数字资产结构/状态/交互、Alpha158-style、
  Qlib Alpha360 滞后特征。
- 截面候选 53 个：25 个 Alpha101 适配公式，以及 CTREND 论文的 28 个技术输入。
- 两类 registry 分开，避免把时间序列信号误当作同一时点截面信号。
- “候选”只表示公式已实现并可检验，不表示已经发现有效 Alpha。

## 数据口径

- 默认币池：BTC/ETH；扩展币池：12 个主流 USDT 永续。
- 默认频率：1m、5m、15m、1h、6h、24h（交易所存储为 `1d`）。
- 下载器读取数据目录，只补未覆盖边缘和内部缺口，已覆盖数据不会重复下载。
- OI/Basis 的 REST 历史长度受交易所限制；缺失保持为空，不使用未来值补齐。

## 交互回测目前是什么

用户选择币种、频率、7/30/90 天或自定义区间、多个因子、权重、方向、预测周期、
调仓周期、手续费与滑点。系统在每个时点做截面标准化和组合，再用明确的未来收益
计算组合表现。

这属于研究策略原型：它把因子分数转换为持仓规则。它还不是交易所执行系统，未模拟
排队、部分成交、冲击成本、爆仓或借币限制，也没有完成锁定六个月 OOS。
