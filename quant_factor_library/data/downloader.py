"""
历史数据下载器

负责批量下载和管理历史行情数据。
支持断点续传、多交易对并行下载。
"""

from __future__ import annotations


class DataDownloader:
    """历史数据下载器

    功能（待实现）：
    - 按交易对和时间范围下载 K 线数据
    - 支持断点续传
    - 数据本地缓存管理
    """

    def __init__(self) -> None:
        # TODO: 初始化下载队列、缓存路径等
        pass

    def download_klines(self, symbol: str, interval: str, start_date: str, end_date: str) -> None:
        """下载指定交易对的 K 线数据"""
        raise NotImplementedError

    def download_all(self, symbols: list[str], interval: str) -> None:
        """批量下载多个交易对的 K 线数据"""
        raise NotImplementedError
