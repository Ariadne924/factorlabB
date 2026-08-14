# 本轮迭代交付记录

日期：2026-08-12

## 交付范围

- 因子：413 个单资产候选 + 53 个截面候选，共 466 个；加入 Qlib Alpha360、
  数字资产结构/状态因子、25 个 Alpha101 面板公式、28 个 CTREND 技术输入。
- 表达式：安全白名单语法、公式哈希、依赖、复杂度限制、单资产/面板执行和未来数据拒绝。
- 数据：12 币种预设，1m/5m/15m/1h/6h/24h，Silver 覆盖目录与只补缺口的下载计划。
- 研究：跨资产/频率汇总、面板 IC/RankIC/ICIR、decay、分组、换手、成本、FDR、
  前视检查、LTW 风险因子（仅在 point-in-time market_cap 可用时）。
- ML：训练折内筛选、去相关和标准化的 expanding walk-forward Ridge 复合因子。
- 可视化：Factor Explorer、Panel Factors、Strategy Builder、Data Center、ML 报告，
  以及版本化前端 JSON 契约。

## 修改与新增文件

### 配置与说明

- `.gitignore`
- `.streamlit/config.toml`
- `README.md`
- `docs/DATA_ACQUISITION.md`
- `docs/DERIVED_FACTOR_CATALOG.md`
- `docs/EXPRESSION_ENGINE.md`
- `docs/FRONTEND_DATA_CONTRACT.md`
- `docs/ML_FACTOR_MINING.md`
- `docs/OPEN_SOURCE_FACTOR_SOURCES.md`
- `docs/PANEL_DATA.md`
- `docs/PANEL_RESEARCH_AND_STRATEGY.md`
- `docs/PROJECT_MAP.md`
- `docs/factors/alpha101_panel_batch_01.md`
- `docs/factors/ctrend_technical_inputs.md`
- `docs/factors/ltw_common_risk_factors.md`

### 数据与因子

- `data/catalog.py`
- `data/downloader.py`
- `data/panel_loader.py`
- `factors/__init__.py`
- `factors/alpha101_panel.py`
- `factors/crypto_trend_panel.py`
- `factors/derived_signals.py`
- `factors/expression_engine.py`
- `factors/panel.py`
- `factors/panel_registry.py`
- `factors/qlib_alpha360.py`

### 评估、脚本与可视化

- `evaluation/cross_frequency.py`
- `evaluation/crypto_risk_factors.py`
- `evaluation/ml_factor_mining.py`
- `evaluation/panel_analysis.py`
- `evaluation/panel_strategy.py`
- `scripts/download_research_data.py`
- `scripts/run_all_research.py`
- `scripts/run_ml_factor_mining.py`
- `scripts/run_panel_research.py`
- `visualization/app.py`
- `visualization/frontend_payload.py`
- `visualization/report_discovery.py`
- `data/refresh.py`
- `scripts/refresh_recent_data.py`

### 测试

- `tests/test_alpha101_panel.py`
- `tests/test_cross_frequency.py`
- `tests/test_crypto_risk_factors.py`
- `tests/test_crypto_trend_panel.py`
- `tests/test_data_catalog.py`
- `tests/test_data_downloader.py`
- `tests/test_derived_signals.py`
- `tests/test_expression_engine.py`
- `tests/test_frontend_payload.py`
- `tests/test_ml_factor_mining.py`
- `tests/test_panel.py`
- `tests/test_panel_analysis.py`
- `tests/test_panel_research_script.py`
- `tests/test_panel_strategy.py`
- `tests/test_qlib_alpha360.py`
- `tests/test_qlib_style_factors.py`
- `tests/test_data_refresh.py`
- `tests/test_report_discovery.py`
- `tests/test_single_factor_report.py`

## 已执行验收

- Python 3.12 `compileall`：通过。
- 413 + 53 registry 数量与导入：通过。
- 53 个面板因子合成数据全量计算：通过。
- 面板策略、数据补缺规划合成 smoke：通过。
- 无 Silver 数据的 `python scripts/run_all_research.py`：通过，并正确输出
  `insufficient_data`、data catalog、panel summary 和 frontend payload。
- `git diff --check` 与 100 字符行长检查：通过；CRLF 提示不是内容错误。

用户的 WSL Python 3.12 环境在本轮扩展前曾通过 `74 passed`、Ruff 和 mypy。本轮新增代码后，
Codex 桌面沙箱无法访问该 WSL 服务，临时 Windows Python 又没有 pytest/ruff/mypy/pyarrow，
因此不能诚实声称新的完整测试集已经全部执行。回到 WSL 后必须运行：

```bash
source ~/.venvs/crypto-factor-lab/bin/activate
pytest -q
ruff check .
mypy config data factors evaluation utils scripts visualization
python scripts/run_all_research.py
```

## 当前数据事实

仓库当前只发现 `BTCUSDT / 1h` 的 Silver 文件，不足以计算真实截面结果，也没有覆盖要求的
多币种、多频率口径。因此本轮没有生成或伪造新的 IC 结论。

## 未完成项与风险

- 需要实际下载/导入 12 币种多频率数据；1m 长样本耗时和磁盘占用较大。
- Binance OI/Basis REST 历史有限，长历史需归档或第三方可靠数据。
- 当前固定币池有幸存者偏差；正式研究需要 point-in-time universe。
- CTREND 当前实现的是论文 28 个技术输入，不是论文拟合后的完整 CTREND 模型。
- Streamlit 是研究前端；独立 Web/Java 前端、账户系统、作业队列和生产部署尚未实现。
- 策略回测没有模拟盘口冲击、排队、部分成交、借币和爆仓。
- 尚未完成锁定六个月 OOS；短样本 IC 和当前样本收益不得表述为已验证 Alpha。
