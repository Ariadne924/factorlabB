# 数字资产量化因子库 (Quant Factor Library)

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

### 1. 克隆项目

```bash
cd quant_factor_library
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
# 一键全流程当前尚未实现；dry-run 仅展示计划步骤
python scripts/run_all.py --dry-run

# 增量更新 dry-run
python scripts/incremental_update.py --dry-run

# 运行契约测试
python -m unittest discover -s tests -v

# CI 使用的质量检查
ruff check .
mypy config data factors utils scripts
```

## 项目结构

```
quant_factor_library/
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

## 当前状态

**G0 门禁（W1 第1天）**：项目骨架搭建完成，顶层接口规范已定义。后续迭代将逐步实现各模块的具体逻辑。

当前下载、评估和报告模块尚未实现。两个入口不带 `--dry-run` 时返回非零退出码，
避免自动化系统把占位流程误判为真实成功。
