# 量化因子库 — Agent 协作指南

## 项目本质

一个 Python 3.12 量化因子研究框架。核心职责：从交易所拉数据 → 标准化 → 算因子 → 评估 → 出报告。

## 关键约定

- **Python 3.12 硬性要求**。不要建议升级 Python 版本，不要在未确认兼容性的情况下新增依赖。所有新依赖必须锁定版本号。
- **时区统一 UTC**。所有 `datetime` 操作必须带时区，`naive` 时间戳视为 Bug。
- **交易对 ID 是字符串**。`'BTCUSDT'`，不是 `BTCUSDT` 也不是 `btc_usdt`。规范化：存储路径统一使用大写（`symbol.upper()`），用户输入不强制格式。文件系统路径中 `BTCUSDT` 与 `btcusdt` 在 Windows/macOS 上等效，但代码内部应保持一致的大写形式。
- **因子签名统一（v2）**。`(df: pd.DataFrame) -> pd.Series`。输入 DataFrame 必须含 OHLCV 列，可额外含 funding_rate 等扩展列。新增因子必须注册到 `FACTOR_REGISTRY`，否则评估框架发现不了。
- **所有注释用中文**。docstring 也走中文。这是小组协作规范，方便审阅。
- **不要过度设计**。G0 是门禁阶段——方法体写 `raise NotImplementedError` 比写一半的假逻辑强。`pass` 也可以。

## 项目语言

读 `CONTEXT.md` — 里面定义了项目的共享语言和术语映射。使用里面的术语，不要自己造词。尤其在写代码、命名变量、写注释时。

## 架构决策

`docs/adr/` 下记录了关键决策。新增不可逆的架构决定前，先写 ADR。

## 代码风格

- 用 `from __future__ import annotations` 在每个文件开头
- 类型注解用新式写法（`list[str]` 而非 `List[str]`，Python 3.12 支持）
- Pydantic v2 风格：`model_config` 而非 `class Config`
- 常量用 `Final` 标注

## 目录职责速查

| 目录 | 职责 | 可修改 |
|------|------|--------|
| `config/` | 全局配置、枚举、常量 | 加配置项可以，改已有常量需 ADR |
| `data/` | Schema 定义、交易所接口、下载/校验 | Schema 字段变更需全组通知 |
| `factors/` | 因子基类、注册机制、四类因子实现 | 新因子放对应子目录 |
| `evaluation/` | IC、分组、稳定性、相关性、成本、前视 | 各模块独立，互不依赖 |
| `visualization/` | 报告、仪表盘、绘图 | 依赖 evaluation 结果 |
| `utils/` | 时间、日志、IO 工具 | 可加工具，不要放业务逻辑 |
| `scripts/` | 入口脚本 | 只做编排，不放核心逻辑 |
