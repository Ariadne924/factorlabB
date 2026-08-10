# crypto-factor-lab：数字资产量化因子库

> 小组协作项目 · Task 1：量化因子库构建

## 项目简介

本项目构建一套完整的数字资产（加密货币）量化因子研究与评估框架。涵盖：

- **数据层**：多交易所统一数据接口（Binance 优先），K 线 / 成交 / 资金费率标准化
- **因子层**：动量、波动率、成交量流动性、加密货币特有因子四大类，统一注册机制
- **评估层**：IC 分析、分组测试、稳定性检验、相关性分析、成本敏感性、前视偏差检查
- **可视化层**：单因子报告、交互式仪表盘

## 环境要求

- **Python 版本**：3.12（硬性要求，所有依赖均基于 Python 3.12 验证）
- **操作系统**：Windows / macOS / Linux 均可
- **编码**：所有文件统一 UTF-8
- **时区**：全局 UTC

## 快速开始

### 1. 进入项目

```bash
cd crypto-factor-lab
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 开发、测试和静态检查
pip install -r requirements-dev.txt
```

### 4. 运行

```bash
# 运行全部测试
pytest -q

# 离线运行全部已注册因子的研究与报告（只读取已有 Silver 数据）
python scripts/run_all_research.py

# 启动 Factor Explorer / Single Factor Report
streamlit run visualization/app.py

# 旧编排入口仍保留；目前只支持显式 dry-run
python scripts/run_all.py --dry-run
python scripts/incremental_update.py --dry-run

# CI 使用的质量检查
ruff check .
mypy config data factors utils scripts
```

## 项目结构

```
crypto-factor-lab/
├── config/              # 全局配置与常量
├── data/                # 数据层：Schema、交易所接口、下载器、校验器
├── factors/             # 因子层：基类、注册机制、四类因子子目录
├── evaluation/          # 评估层：IC、分组、稳定性、相关性、成本、前视检查
├── visualization/       # 可视化层：报告、仪表盘、绘图工具
├── utils/               # 工具层：时间转换、日志、IO
├── scripts/             # 运行脚本：全流程、增量更新
├── logs/                # 日志输出目录
├── reports/             # 报告输出目录
└── notebooks/           # Jupyter Notebook 实验目录
```

## 开发约定

- 所有时间戳统一使用 `pandas.Timestamp`，时区为 UTC
- 交易对 ID 使用字符串类型（如 `'BTCUSDT'`）
- 所有因子遵循统一函数签名（v2）：`(df: pd.DataFrame) -> pd.Series`，输入 DataFrame 必须含 OHLCV 列
- 新因子通过 `@register_factor` 装饰器注册
- 代码注释使用中文

## 协作规范

- 分支命名：`feature/<功能名>` 或 `fix/<问题名>`
- 提交信息：简明扼要描述改动
- 代码审查：所有 PR 需至少一人 Review

## 数据与报告约定

研究入口扫描 `data/silver/**/klines.parquet`。Silver K 线必须满足统一 UTC Schema；
数字资产特有因子还分别需要 `funding_rate`、`open_interest`、`basis` 列，这些列应通过
`data.silver.merge_point_in_time_features` 按发布时间向后合并，禁止未来回填。

如果没有真实 Silver 数据，研究入口仍会成功生成 `reports/research_manifest.json` 和空的
单因子报告，但所有 IC 等指标均为 `null`、状态为 `insufficient_data`。这不是实证结果。

## 当前实现范围与限制

- 13 个正式候选因子：4 个动量/反转、3 个波动率、3 个量价/流动性、3 个数字资产特有因子。
- Binance REST：K 线、逐笔聚合成交、盘口快照、资金费率、持仓量、当前基差；429/5xx 与网络错误采用有限退避重试。
- Binance WebSocket：基础同步消息流、自动重连和指数退避；不提供 exactly-once、持久化队列或完整生产级心跳状态机。
- 数据异常报告：极端价格跳变、stale/flat、零成交量、交易所时间缺口。
- 单因子 JSON/HTML 报告与 Streamlit Factor Explorer。
- `fetch_basis()` 只返回当前快照；历史基差研究需要先持续归档或接入可靠历史源。
- 当前仓库未附真实研究数据，未完成锁定六个月 OOS；任何短样本 IC 都不能描述为已验证 alpha。
- `scripts/run_all.py` 与 `scripts/incremental_update.py` 是保留的旧编排占位入口；实际研究请使用 `scripts/run_all_research.py`。
