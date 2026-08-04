"""
增量更新脚本（占位框架）

用于每日增量更新数据、重新计算因子、更新评估结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)


def main(*, dry_run: bool = False) -> int:
    """增量更新入口"""
    if not dry_run:
        logger.error(
            "增量更新尚未实现。仅检查编排请使用: "
            "python scripts/incremental_update.py --dry-run"
        )
        return 2

    logger.info("=== 开始增量更新 ===")

    # ----------------------------------------------------------------
    # 1. 获取最新数据
    # ----------------------------------------------------------------
    logger.info("[1/4] 拉取增量数据...")
    # TODO: 实现增量数据拉取
    logger.info("[1/4] 未执行（dry-run）")

    # ----------------------------------------------------------------
    # 2. 更新因子值
    # ----------------------------------------------------------------
    logger.info("[2/4] 重新计算因子值...")
    # TODO: 只更新受新数据影响的因子
    logger.info("[2/4] 未执行（dry-run）")

    # ----------------------------------------------------------------
    # 3. 更新评估指标
    # ----------------------------------------------------------------
    logger.info("[3/4] 更新评估指标...")
    # TODO: 滚动计算最新 IC、分组收益等
    logger.info("[3/4] 未执行（dry-run）")

    # ----------------------------------------------------------------
    # 4. 刷新报告
    # ----------------------------------------------------------------
    logger.info("[4/4] 刷新分析报告...")
    # TODO: 更新图表和 HTML 报告
    logger.info("[4/4] 未执行（dry-run）")

    logger.info("=== dry-run 完成：未更新任何数据 ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化因子库增量更新入口")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅展示计划中的增量步骤，不执行数据处理",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
