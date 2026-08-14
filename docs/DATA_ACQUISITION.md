# 真实研究数据获取

下载计划会先扫描现有 Silver 文件并生成 `reports/data_catalog.json`。目录记录每个
币种/频率的起止时间、行数、覆盖率与内部缺口；`--execute` 只请求未覆盖区间。
如果请求已经完整覆盖，状态为 `up_to_date`，不会重复下载。

## 推荐起点

- 市场：Binance USDⓈ-M 永续
- 交易对：BTCUSDT、ETHUSDT
- K 线：1m、5m、15m、1h、6h、24h（Binance 参数为 `1d`）
- 主历史区间：至少 12 个月，理想为 18 个月以上
- 扩展字段：Funding Rate、Open Interest、Basis

先检查下载计划，不发网络请求：

```bash
python scripts/download_research_data.py
```

确认日期和磁盘位置后执行：

```bash
python scripts/download_research_data.py --start 2025-01-01 --end 2026-01-01 --execute
```

默认计划覆盖六个研究频率。也可以先分批执行，避免一次下载过多 1m 数据：

```bash
python scripts/download_research_data.py --intervals 5m 15m 1h 6h 24h --start 2025-01-01 --end 2026-01-01 --execute
python scripts/download_research_data.py --interval 1m --start 2025-08-01 --end 2026-01-01 --execute
```

默认 `starter` 币池是 BTCUSDT、ETHUSDT。扩展的 `core` 币池包含12个当前主流永续：

```bash
python scripts/download_research_data.py --universe core --start 2025-01-01 --end 2026-01-01
```

预设币池是当前固定清单，只适合工程验证和候选研究，不构成无幸存者偏差的历史币池。
正式横截面回测必须另建 point-in-time universe，记录每个时点可交易、上市和流动性状态。

完成后运行：

```bash
python scripts/run_all_research.py --lookback-days 180 --intervals 1m 5m 15m 1h 6h 24h
python scripts/run_ml_factor_mining.py --intervals 1m 5m 15m 1h 6h 24h
streamlit run visualization/app.py
```

研究期间若只需要把最近 K 线推进到最新位置，无需重复下载全部历史：

```bash
python scripts/refresh_recent_data.py --symbols BTCUSDT ETHUSDT SOLUSDT \
  --intervals 1m 5m 1h --lookback-hours 24
```

它会重叠拉取一个受限尾部窗口，并按时间键合并 Bronze/Silver。Streamlit Data Center
提供同一入口并每30秒重读本地目录。该模式用于近实时研究刷新，不是逐笔级流式存储。

## 数据路径

```text
data/bronze/binance/futures/{SYMBOL}/{INTERVAL}/
├── klines.parquet
├── funding_rate.parquet
├── open_interest.parquet
└── basis.parquet

data/silver/binance/futures/{SYMBOL}/{INTERVAL}/klines.parquet
```

Silver K 线会按发布时间向后合并 Funding/OI/Basis。历史 Basis 的官方时间戳是统计周期开始时间，本项目将其可用时间推迟至周期结束后再合并，防止周期内前视。

## 官方数据限制

Binance USDⓈ-M REST 的 Open Interest Statistics 和 Basis 只提供最近约 30 天，无法通过反复分页得到 12–18 个月历史。Funding 和 K 线可下载更长区间。长历史 OI/Basis 需要：

1. Binance 公共数据归档中存在的对应文件；或
2. 从现在开始定时归档；或
3. 用户提供的可靠第三方历史文件。

在长历史 OI/Basis 接入前，这两类因子的报告会保留缺失值，不进行未来回填。

## Git 规则

Parquet 数据已由 `.gitignore` 排除。不要把行情原始数据、API key、`.env` 或大型报告上传 GitHub。
## Binance 官方历史归档

REST 适合增量更新；长历史 K 线可从 Binance 官方 `data.binance.vision` 月度 ZIP 导入：

```bash
python scripts/download_binance_archive.py --symbol BTCUSDT --interval 1h --year 2025 --months 1 2 3
python scripts/download_binance_archive.py --symbol BTCUSDT --interval 1h --year 2025 --months 1 2 3 --execute
```

默认只打印计划。执行时会下载同名 `.CHECKSUM`、验证 SHA256、兼容 2025 年后现货归档的微秒时间戳，并幂等写入 Bronze/Silver。该归档不能补齐 Binance 未公开的长历史 OI/Basis。

## 推荐的一键真实数据采集

```bash
python scripts/collect_real_data.py --probe
python scripts/collect_real_data.py
python scripts/collect_real_data.py --execute
```

默认采集 BTC/ETH/SOL/BNB/XRP/DOGE 的 1m/5m/1h/6h/24h：1m 保存最近 29 天；其余频率通过官方月度归档补一年，并用 REST 覆盖最近 29 天及 Funding/OI/Basis。每个成功任务立即写入 `reports/real_data_collection_manifest.json`，断线后重跑相同命令只补未完成任务。

月度 ZIP 在 SHA256 校验通过后会缓存到 `data/cache/binance_archive/`。同一币种和频率的多个月份会批量合并，只重写一次 Bronze/Silver；后续重跑会优先读取校验后的缓存，不重复占用网络。该缓存和 Parquet 均由 `.gitignore` 排除，不上传 GitHub。

VPN 是否有帮助必须以 `--probe` 的结果判断：先在当前网络运行一次，再切换 VPN 状态运行一次，比较各源的 `ok` 和延迟。官方归档与 Binance Futures REST 是两个独立入口，可能一个可用、另一个不可用。

执行前默认有连通性门禁，不会在数据源整体不可用时逐任务等待超时。如果只有归档可用，使用 `--mode archive --execute`；如果只有 Futures REST 可用，使用 `--mode rest --execute`。网络恢复后再运行默认 `--execute` 补齐另一部分。

归档完成后，可在已激活的 Linux/WSL 虚拟环境中一次完成 REST 补数与项目验收：

```bash
bash scripts/finish_real_data.sh
```

脚本使用 `set -euo pipefail`，任一步失败都会停止并保留采集 manifest；再次运行会从未完成的 REST 任务续跑。
