"""
日志配置模块

统一项目日志输出格式和级别。
"""

from __future__ import annotations

import logging
import sys

from config.settings import LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str = "quant_factor_library") -> logging.Logger:
    """创建并配置一个命名日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        配置好的 Logger 实例

    示例：
        >>> logger = setup_logger(__name__)
        >>> logger.info("数据下载完成")
        >>> logger.warning("检测到缺失数据")
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "quant_factor_library") -> logging.Logger:
    """获取已有的日志记录器（不重新配置）"""
    return logging.getLogger(name)
