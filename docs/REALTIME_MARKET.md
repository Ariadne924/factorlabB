# 实时行情入口

实时行情采用“独立采集器 + 本地快照 + Streamlit 只读展示”的轻量结构。页面重载不会重启 WebSocket，研究模块也不会使用尚未收盘的当前 K 线。

## 启动

先在项目虚拟环境中开启采集器：

```bash
python scripts/run_live_market.py
```

再开一个终端启动前端：

```bash
streamlit run visualization/app.py
```

浏览器访问 `http://localhost:8501`，从左侧进入“实时行情”。页面每 2 秒读取 `reports/live_market.json`，展示逐笔聚合成交、best bid/ask、五档盘口、1 分钟 K 线、标记价格、资金费率与永续基差。采集器使用现货流取得成交、best bid/ask 和 K 线，并使用 USD-M 流取得五档盘口和永续指标；这样在部分 VPN 环境下 Futures 子流受限时，基础行情仍可用。

默认核心池为 `BTCUSDT、ETHUSDT、SOLUSDT、BNBUSDT、XRPUSDT、DOGEUSDT`。可通过 `--symbols` 改为任意明确列表；不建议把低流动性币种未经筛选直接加入机器学习训练。

## API 与数据边界

- 使用 Binance Spot 与 USD-M Futures 公共 WebSocket，不需要 API Key。
- 公开 WebSocket 地址为 `wss://fstream.binance.com`；VPN、代理或地区网络策略可能导致断连，采集器会有限指数退避并自动重连。
- 当前分钟 K 线只用于行情展示；仅 `closed=true` 的现货 K 线会幂等合并到 `Silver/binance/spot`，不会混入 Futures 历史口径。
- Funding 和实时基差优先来自 mark price stream，并由 60 秒公开 REST 轮询兜底。Open Interest 没有在本入口伪装成 WebSocket 实时流，仍由数据中心使用公开 REST 定期补充。
- 这是研究级秒级刷新，不是交易所托管、逐笔不丢失或 exactly-once 的生产交易系统，也不会自动下单。

若只想看实时快照、不更新 Silver，可使用：

```bash
python scripts/run_live_market.py --no-persist
```
