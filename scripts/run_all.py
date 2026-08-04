"""
一键全流程脚本（v2 — Medallion 管道版）

完整流程：
  1. Bronze：从交易所下载原始数据，存入 data/bronze/
  2. Silver：清洗标准化，存入 data/silver/
  3. Gold：计算所有已注册因子，存入 data/gold/
  4. 评估：IC 分析、分组测试、稳定性等
  5. 报告：生成可视化报告

当前版本为占位框架，输出日志供 G0 门禁验收。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保模块导入正常
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from factors.registry import FACTOR_REGISTRY as _FACTOR_REGISTRY  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)


def step_1_download() -> None:
    """Bronze 层：下载原始数据"""
    logger.info("-" * 40)
    logger.info("[Bronze 层] 从交易所下载原始 K 线数据...")
    # TODO:
    #   1. 从 config 读取交易所列表、交易对列表、K 线周期列表
    #   2. 遍历调用 binance_client.BinanceClient.get_klines()
    #   3. 用 data.schema.KlineRaw.from_binance_row() 校验每一行
    #   4. 用 data.bronze.write_bronze_klines() 写入 parquet
    logger.info("[Bronze 层] 未执行（dry-run）")


def step_2_clean() -> None:
    """Silver 层：清洗标准化"""
    logger.info("-" * 40)
    logger.info("[Silver 层] Bronze → Silver 清洗转换...")
    # TODO:
    #   1. 遍历 data/bronze/ 下所有 parquet 文件
    #   2. 用 data.silver.build_silver_from_bronze() 转换
    #   3. 用 data.validator.DataValidator 校验完整性
    logger.info("[Silver 层] 未执行（dry-run）")


def step_3_compute_factors() -> None:
    """Gold 层：计算因子"""
    logger.info("-" * 40)
    logger.info("[Gold 层] 计算所有已注册因子...")
    # TODO:
    #   1. 读取 Silver 层数据 → data.silver.silver_to_factor_input()
    #   2. 遍历 FACTOR_REGISTRY，对每个因子族：
    #      - 用 factors.registry.build_factor(name, params) 构建计算函数
    #      - 用 data.gold.write_gold_factor() 写入结果
    #   3. 逐因子计算，支持增量追加
    logger.info(f"    已注册因子族: {len(_FACTOR_REGISTRY)} 个")
    logger.info("[Gold 层] 未执行（dry-run）")


def step_4_evaluate() -> None:
    """评估：IC / 分组 / 稳定性 / 相关性 / 成本敏感性 / 前视检查"""
    logger.info("-" * 40)
    logger.info("[评估] 因子表现评估...")
    # TODO:
    #   1. 从 Gold 层读取所有因子值
    #   2. 计算前向收益（与因子时间对齐）
    #   3. 依次调用 evaluation 模块各函数
    #   4. 汇总评估结果表
    logger.info("[评估] 未执行（dry-run）")


def step_5_report() -> None:
    """可视化报告"""
    logger.info("-" * 40)
    logger.info("[报告] 生成分析报告...")
    # TODO:
    #   1. 对每个因子调用 visualization.single_factor_report
    #   2. 汇总 IC 排名表、多空收益对比图
    logger.info("[报告] 未执行（dry-run）")


def main(*, dry_run: bool = False) -> int:
    """主流程入口（Medallion 五步管道）"""
    if not dry_run:
        logger.error(
            "全流程尚未实现。仅检查管道编排请使用: "
            "python scripts/run_all.py --dry-run"
        )
        return 2

    logger.info("=" * 50)
    logger.info("=== 量化因子库 · 全流程启动 ===")
    logger.info("=== Bronze → Silver → Gold → Evaluate → Report ===")
    logger.info("=" * 50)

    step_1_download()
    step_2_clean()
    step_3_compute_factors()
    step_4_evaluate()
    step_5_report()

    logger.info("=" * 50)
    logger.info("=== dry-run 完成：未下载、转换或写入任何数据 ===")
    logger.info("=" * 50)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化因子库全流程入口")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅展示计划中的管道步骤，不执行数据处理",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
