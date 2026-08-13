# crypto-factor-lab：数字资产量化因子库

> 小组协作项目 · Task 1：量化因子库构建

## 项目简介

本项目构建一套完整的数字资产（加密货币）量化因子研究与评估框架。涵盖：

- **数据层**：多交易所统一数据接口（Binance 优先），K 线 / 成交 / 资金费率标准化
- **因子层**：413 个单资产正式候选因子与 53 个截面候选（25 个 Alpha101、28 个 CTREND 技术输入），覆盖动量、波动率、量价、数字资产特有、状态交互、Qlib 特征与公式因子；另含安全、无前视的表达式引擎
- **评估层**：IC、分组、相关性、成本、前视检查，以及多窗口/多 horizon、区块 bootstrap 与 BH-FDR
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

WSL 中若希望多个工作目录复用同一套依赖，可把环境放在仓库外：

```bash
python3.12 -m venv ~/.venvs/crypto-factor-lab
source ~/.venvs/crypto-factor-lab/bin/activate
python -m pip install -r requirements-dev.txt
```

以后进入项目只需重新执行 `source ~/.venvs/crypto-factor-lab/bin/activate`，无需重复安装。

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

# 默认复用数据、参数和研究源码均未变化的资产×频率报告；需要强制重算时关闭缓存
python scripts/run_all_research.py --no-cache

# 查看真实数据下载计划（不联网、不写行情文件）
python scripts/download_research_data.py

# 推荐：先比较 VPN 开/关时各数据源的真实连通性
python scripts/collect_real_data.py --probe

# 可恢复的一年真实数据采集：6币种 × 1m/5m/1h/6h/24h
# 默认仅显示摘要；1m 拉最近29天，其余频率从官方月度归档补一年，
# 最近29天再由 REST 补 Funding/OI/Basis。中断后重跑会续传；归档 ZIP 校验后本地缓存。
python scripts/collect_real_data.py
python scripts/collect_real_data.py --execute

# 若探测显示只有一个 Binance 入口可用，可分别执行；之后重跑 all 补齐
python scripts/collect_real_data.py --mode archive --execute
python scripts/collect_real_data.py --mode rest --execute

# 确认后下载 Binance 永续多频率数据（默认 BTC/ETH × 1m/5m/15m/1h/6h/24h）
python scripts/download_research_data.py --start 2025-01-01 --end 2026-01-01 --execute

# 扩展到12个主流永续；建议先 dry-run 并分频率执行
python scripts/download_research_data.py --universe core --start 2025-01-01 --end 2026-01-01

# 分批下载时可显式指定频率；24h 会映射为 Binance 1d
python scripts/download_research_data.py --intervals 5m 15m 1h 6h 24h --start 2025-01-01 --end 2026-01-01 --execute

# 可选：收窄主样本，并指定稳健性窗口、预测周期和 bootstrap 次数
python scripts/run_all_research.py --lookback-days 90 --robustness-windows 30 60 90 --horizons 1 3 6 12 --bootstrap-samples 500

# 严格 walk-forward 的机器学习复合因子（自动生成参数变体并在训练折筛选）
python scripts/run_ml_factor_mining.py --symbols BTCUSDT ETHUSDT --intervals 1m 5m 15m 1h 6h 24h

# 完整研究周期：数据资格 → 因子研究 → 多模型 Walk-Forward → 因子分级轮动
python scripts/run_research_cycle.py --symbols BTCUSDT ETHUSDT --intervals 1h 6h 24h

# 只更新全部保留因子的 A/B/C/D 研究优先级
python scripts/grade_factors.py --rotation-days 7

# 至少三个同频率币种的截面研究
python scripts/run_panel_research.py --symbols BTCUSDT ETHUSDT SOLUSDT --intervals 1h 6h 24h

# 启动 Factor Explorer / Single Factor Report
streamlit run visualization/app.py

# 另开一个终端启动公共实时行情（无需 API Key），前端进入“实时行情”
python scripts/run_live_market.py

# 长期运行入口：同时维护实时采集与前端，子进程退出后自动退避重启
python scripts/run_platform.py

# 独立检查当前研究汇总引用的报告是否完整
python scripts/check_reports.py

# 可选：刷新 BTC/ETH/SOL 的 1m/5m/1h 最近24小时（也可在 Data Center 点击刷新）
python scripts/refresh_recent_data.py

# 兼容的一键研究入口，以及近端数据增量刷新 + 报告更新
python scripts/run_all.py
python scripts/incremental_update.py --symbols BTCUSDT ETHUSDT --intervals 5m 1h

# 从 Binance 官方公开归档按月导入长历史 K 线（默认先展示计划）
python scripts/download_binance_archive.py --symbol BTCUSDT --interval 1h --year 2025 --months 1 2 3
python scripts/download_binance_archive.py --symbol BTCUSDT --interval 1h --year 2025 --months 1 2 3 --execute

# 在开发前创建至少 180 天的锁定 OOS 元数据；本命令不会读取或评估 OOS
python scripts/lock_oos.py --start 2025-01-01 --end 2025-07-01 --symbols BTCUSDT ETHUSDT --intervals 1h 24h

# 上述两个一键入口均支持显式 dry-run
python scripts/run_all.py --dry-run
python scripts/incremental_update.py --dry-run

# CI 使用的质量检查
ruff check .
mypy config data factors evaluation utils scripts visualization
```

## 项目结构

```
crypto-factor-lab/
├── config/              # 全局配置与常量
├── data/                # 数据层：Schema、交易所接口、下载器、校验器
├── factors/             # 因子层：基类、registry、原有因子与 Qlib-style 因子
├── evaluation/          # 评估层：IC、分组、稳健性、FDR、相关性、成本、前视检查
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

真实数据获取和 Binance REST 历史范围限制见 `docs/DATA_ACQUISITION.md`。其中 OI 与 Basis
只能直接取得最近约 30 天，长历史必须使用归档数据；系统不会用未来值回填缺口。

## 当前实现范围与限制

- 413 个正式候选因子：原有 13 个、20 个 Alpha158-style OHLCV 因子、20 个数字资产结构/状态/交互因子，以及360个本地实现的 Microsoft Qlib Alpha360 滞后特征；来源、依赖和参数写入 registry 与报告。
- 公式表达式引擎只执行白名单历史算子，生成规范化表达式、哈希、依赖和复杂度元数据，并拒绝负向位移、任意 Python 调用与过大计算窗口；详见 `docs/EXPRESSION_ENGINE.md`。
- 多币种面板使用严格的 `(timestamp, symbol)` UTC 索引，不填充缺失 K 线；支持同一时点 Rank、Scale、Z-score、Winsorize 与多暴露中性化，覆盖率和适用限制见 `docs/PANEL_DATA.md`。
- 独立 panel registry 已接入 25 个 Alpha101 公式与 CTREND 论文描述的 28 个技术输入；保留来源、依赖、哈希和适配说明，它们是待检验候选，不是已验证 alpha。
- `reports/data_catalog.json` 记录 Silver 覆盖与内部缺口；下载脚本只规划缺失范围，完整覆盖时返回 `up_to_date`。
- 截面研究输出同一时点 IC/RankIC、滚动 IC、IC decay、分组收益、换手率、成本敏感性、FDR 和前视检查；少于 3 个币种时明确返回 `insufficient_data`。
- Streamlit 左侧使用 6 个任务入口；新增“实时行情”，并由独立进程更新逐笔成交、盘口、当前 K 线、Funding 与 Basis 秒级快照。“策略研究”统一承载单币种择时、多币种选币和结果比较。两类策略均支持币种/频率/区间、多个因子、权重与方向、调仓和成本；ML 训练折推荐可以一键生成单币种策略候选预设。
- 首页使用数据、策略和验证产物动态生成流程完成度与下一步建议；“因子检验”统一展示单资产、截面和跨口径结果，并可后置，不再阻塞用户直接使用正式 registry 构建策略。数据中心将缺频率、K 线缺口、行情陈旧和衍生品缺失整理为 P0/P1 任务队列，具体交互口径见 `docs/FRONTEND_WORKFLOW.md`。
- 单币种择时包含成本重定价、分月/滚动表现、Buy & Hold、参数邻域检查、因果市场状态归因，以及训练期选参后冻结参数的策略 Walk-Forward；还支持显式保存和横向比较策略快照，保存结果默认留在本地报告目录。
- Data Center 每 30 秒读取本地覆盖目录，并提供用户主动触发的短窗口 REST 刷新；重复数据按时间键合并。该功能是近实时 K 线刷新，不冒充逐笔流式存储。
- Data Center 区分“历史可研究”“行情新鲜”“ML 候选”三类状态，并展示核心 6 币种 × 5 频率覆盖、内部缺口及 Funding/OI/Basis 新鲜度；机器学习入口只列出至少 120 天且覆盖率不低于 98% 的数据集。
- 默认研究样本收窄至末端 180 个自然日，并报告 30/60/90/180 天 × 1/3/6/12/24 根 K 线的 IC 稳健性网格、区块 bootstrap、符号一致性、分组单调性和 BH-FDR。
- Binance REST：K 线、逐笔聚合成交、盘口快照、资金费率、持仓量、当前基差；429/5xx 与网络错误采用有限退避重试。
- Binance 官方公开归档：支持月度现货/USD-M 永续 K 线 ZIP、SHA256 校验、微秒时间戳兼容及 Bronze/Silver 幂等导入。
- OKX 最小公共 REST 客户端：统一接入现货 K 线、成交、盘口与永续资金费率；不冒充完整生产级连接器。
- Binance WebSocket：公共 USD-M combined stream、独立采集进程、原子前端快照、已收盘 K 线幂等落盘、自动重连和指数退避；不提供 exactly-once、持久化队列或自动下单。详见 `docs/REALTIME_MARKET.md`。
- 实时页面包含分钟涨跌、点差、资金费率和数据陈旧四类市场异动提醒；提醒只用于研究观察，不触发下单。
- `scripts/run_platform.py` 可同时守护采集器和 Streamlit，记录 PID、重启次数与更新时间到 `reports/service_status.json`。它是单机研究环境守护器，不替代 Docker/systemd/Kubernetes。
- 研究入口会生成 `reports/report_health.json` 检查当前汇总引用是否完整；旧报告保留但不混入当前研究汇总。完整研究周期还会写入 `reports/training_status.json`，展示数据检查、因子研究、ML 和完成状态。
- 全量因子研究默认启用两级缓存：先复用未变化的资产×频率报告，选择范围、数据、参数和研究源码全部一致时直接复用最终汇总快照；任何相关条件变化都会自动降级为对应口径重算。进度写入 `reports/research_status.json`，并在数据中心显示完成比例、当前因子与缓存命中数。使用 `--no-cache` 可强制重算。
- 数据异常报告：极端价格跳变、stale/flat、零成交量、交易所时间缺口。
- 单因子 JSON/HTML 报告与 Streamlit Factor Explorer；面板支持资产/频率/类别筛选、跨口径汇总和机器学习报告。
- 研究入口额外生成版本化的 `reports/frontend_payload.json`，供下一阶段独立前端直接消费。
- 前端字段与兼容规则见 `docs/FRONTEND_DATA_CONTRACT.md`；ML 摘要使用独立的 `reports/ml_factor_summary.json`。
- 项目各层、研究链路与交互回测边界见 `docs/PROJECT_MAP.md`。
- 机器学习第一版使用无随机打乱的 expanding walk-forward Ridge，将候选因子及窗口变体组合成严格样本外复合因子；特征筛选、去相关、标准化、horizon purge 和 embargo 均显式记录。页面会汇总训练折中的入选频率、权重方向稳定性，并用严格 OOS 预测执行含成本的简单方向策略回测。
- 前端会从旧报告文件名恢复资产/频率，同时隐藏无法确定资产的遗留报告；`unknown` 不再作为可选资产。
- `fetch_basis()` 只返回当前快照；历史基差研究需要先持续归档或接入可靠历史源。
- 当前仓库未附真实研究数据，未完成锁定六个月 OOS；任何短样本 IC 都不能描述为已验证 alpha。
- `scripts/run_all.py` 已复用正式研究管道；`scripts/incremental_update.py` 复用短窗口刷新和研究入口。行情下载仍需显式执行，避免一键命令意外联网。
- 研究入口生成独立 `lookahead_report.json` 与分数据集因子相关性矩阵；单因子报告包含成本网格与盈亏平衡成本。
- `scripts/lock_oos.py` 只创建不可冒充完成的 OOS 锁定元数据。真实六个月 OOS 仍需完整历史数据，并在最终提交前只运行一次。
