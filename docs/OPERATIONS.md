# 长期运行与排障

## 日常启动

在已经安装项目依赖的 Python 3.12 环境中运行：

```bash
python scripts/run_platform.py
```

该入口同时维护实时采集器和 Streamlit。任一子进程异常退出后会按 1、2、4 秒逐步退避重启，最长等待 30 秒；60 秒内连续崩溃 5 次将打开熔断器并以失败状态退出，避免无限重启。实时快照超过 45 秒未更新时会重启采集器。运行状态写入 `reports/service_status.json`，子进程输出写入 `logs/live_market.log` 和 `logs/frontend.log`，超过约 5 MB 会在下一次启动前轮转。按 `Ctrl+C` 或收到终止信号会依次关闭两个子进程。

如果前端由其他方式托管，可以只维护采集器：

```bash
python scripts/run_platform.py --no-frontend
```

## 报告排障

```bash
python scripts/check_reports.py
```

退出码为 0 表示当前 `research_summary.json` 引用的报告全部存在、JSON 可解析且因子名一致。`unreferenced_old_files` 只是历史运行留下的报告，不会混入当前汇总，也不代表生成失败。

## 增量研究缓存与进度

`python scripts/run_all_research.py` 默认启用两级缓存。Silver 文件、研究参数和因子/评估源码均未变化时，已有资产×频率报告会被复用；本次选择范围也完全一致时，最终汇总快照会直接复用。任何相关条件变化都会自动重新计算对应口径。任务进度写入 `reports/research_status.json`，数据中心每 30 秒显示完成比例、当前口径和缓存命中数。

需要强制重算时运行：

```bash
python scripts/run_all_research.py --no-cache
```

缓存只减少重复计算，不改变 IC、FDR、前视检查和报告生成规则；跨口径 FDR 仍会根据本次选择的完整报告集合重新汇总。

## 训练状态

完整研究周期使用：

```bash
python scripts/run_research_cycle.py
```

状态持续写入 `reports/training_status.json`。成功时为 `completed`，异常时为 `failed` 并保留阶段和错误原因。ML 只会进入满足固定切分口径的数据范围：120 天训练、30 天验证、30 天锁定 OOS，覆盖率至少 98%。锁定窗口存在不等于已完成 6 个月 OOS，也不代表已验证 alpha。

## 市场提醒边界

实时页当前提醒分钟涨跌、点差扩大、极端资金费率和数据停止更新。提醒只用于研究监控，不自动下单；页面关闭时不会向系统或手机推送通知。
