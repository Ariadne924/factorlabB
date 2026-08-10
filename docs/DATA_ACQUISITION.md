# 真实研究数据获取

## 推荐起点

- 市场：Binance USDⓈ-M 永续
- 交易对：BTCUSDT、ETHUSDT
- K 线：1h
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

完成后运行：

```bash
python scripts/run_all_research.py --lookback-days 180
streamlit run visualization/app.py
```

## 数据路径

```text
data/bronze/binance/futures/{SYMBOL}/1h/
├── klines.parquet
├── funding_rate.parquet
├── open_interest.parquet
└── basis.parquet

data/silver/binance/futures/{SYMBOL}/1h/klines.parquet
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

