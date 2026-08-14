"""分页下载 Binance 研究数据并构建 point-in-time Silver 数据集。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config.constants import KlineInterval, get_interval_ms
from config.settings import DEFAULT_DATA_DIR
from data.binance_client import BinanceClient
from data.paths import normalize_path_segment, safe_data_path
from data.schema import bronze_to_silver
from data.silver import merge_point_in_time_features, silver_to_factor_input
from utils.io_utils import merge_parquet_safe

ClientFactory = Callable[[str], BinanceClient]


def _derivatives_feature_period(interval: str) -> str:
    """OI/Basis 不提供 1m 周期，分钟研究使用已结束的 5m 观测。"""
    return "5m" if interval == "1m" else interval


def _utc(value: str | datetime) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


class DataDownloader:
    """研究用历史数据下载器。

    K 线和 Funding 可按时间分页；Binance REST 对历史 OI/Basis 只开放最近约
    30 天，因此 bundle 会自动收窄这两类请求并在 manifest 中记录限制。
    """

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        client_factory: ClientFactory = BinanceClient,
    ) -> None:
        self.data_dir = (data_dir or DEFAULT_DATA_DIR).resolve()
        self.client_factory = client_factory

    def _path(
        self,
        *,
        layer: str,
        market: str,
        symbol: str,
        interval: str,
        filename: str,
    ) -> Path:
        return safe_data_path(
            self.data_dir,
            normalize_path_segment(layer, field_name="layer", case="lower"),
            "binance",
            normalize_path_segment(market, field_name="market", case="lower"),
            normalize_path_segment(symbol, field_name="symbol", case="upper"),
            normalize_path_segment(interval, field_name="interval", case="lower"),
            normalize_path_segment(filename, field_name="filename", case="lower"),
        )

    @staticmethod
    def _write_merged(
        path: Path,
        frame: pd.DataFrame,
        *,
        key: str,
    ) -> pd.DataFrame:
        return merge_parquet_safe(path, frame, key=key)

    @staticmethod
    def _paginate_datetime(
        fetch: Callable[[datetime, datetime], pd.DataFrame],
        *,
        start: datetime,
        end: datetime,
        timestamp_column: str = "timestamp",
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        cursor = start
        pages: list[pd.DataFrame] = []
        for _ in range(max_pages):
            if cursor > end:
                break
            page = fetch(cursor, end)
            if page.empty:
                break
            page = page.copy()
            page[timestamp_column] = pd.to_datetime(page[timestamp_column], utc=True)
            page = page.loc[page[timestamp_column].between(cursor, end)]
            if page.empty:
                break
            pages.append(page)
            next_cursor = page[timestamp_column].max().to_pydatetime() + timedelta(milliseconds=1)
            if next_cursor <= cursor:
                raise RuntimeError("分页时间戳没有前进")
            cursor = next_cursor
        else:
            raise RuntimeError("分页超过安全上限")
        if not pages:
            return pd.DataFrame()
        return (
            pd.concat(pages, ignore_index=True)
            .drop_duplicates(subset=[timestamp_column], keep="last")
            .sort_values(timestamp_column)
            .reset_index(drop=True)
        )

    def fetch_klines_range(
        self,
        *,
        symbol: str,
        interval: str,
        start: str | datetime,
        end: str | datetime,
        market: str = "futures",
    ) -> pd.DataFrame:
        start_dt, end_dt = _utc(start), _utc(end)
        if start_dt > end_dt:
            raise ValueError("start 不得晚于 end")
        interval_enum = KlineInterval(interval)
        step_ms = get_interval_ms(interval_enum.value)
        client = self.client_factory(symbol)
        cursor = start_dt
        pages: list[pd.DataFrame] = []
        while cursor <= end_dt:
            if market == "futures":
                page = client.fetch_futures_klines(
                    interval_enum, start_time=cursor, end_time=end_dt, limit=1000
                )
            elif market == "spot":
                page = client.fetch_klines(
                    interval_enum, start_time=cursor, end_time=end_dt, limit=1000
                )
            else:
                raise ValueError("market 仅支持 spot/futures")
            if page.empty:
                break
            page = page.loc[(page["open_time"] >= int(start_dt.timestamp() * 1000))]
            page = page.loc[(page["open_time"] <= int(end_dt.timestamp() * 1000))]
            if page.empty:
                break
            pages.append(page)
            next_ms = int(page["open_time"].max()) + step_ms
            next_cursor = datetime.fromtimestamp(next_ms / 1000, tz=UTC)
            if next_cursor <= cursor:
                raise RuntimeError("K 线分页时间没有前进")
            cursor = next_cursor
        if not pages:
            return pd.DataFrame()
        return (
            pd.concat(pages, ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    def download_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        *,
        market: str = "futures",
    ) -> Path:
        """下载指定交易对 K 线，幂等合并到 Bronze parquet。"""
        frame = self.fetch_klines_range(
            symbol=symbol,
            interval=interval,
            start=start_date,
            end=end_date,
            market=market,
        )
        if frame.empty:
            raise ValueError("请求范围内没有 K 线数据")
        path = self._path(
            layer="bronze",
            market=market,
            symbol=symbol,
            interval=interval,
            filename="klines.parquet",
        )
        self._write_merged(path, frame, key="open_time")
        return path

    def download_bundle(
        self,
        *,
        symbol: str,
        interval: str,
        start: str | datetime,
        end: str | datetime,
    ) -> dict[str, Any]:
        """下载一个永续研究 bundle，并生成带 point-in-time 特征的 Silver K 线。"""
        start_dt, end_dt = _utc(start), _utc(end)
        if start_dt > end_dt:
            raise ValueError("start 不得晚于 end")
        client = self.client_factory(symbol)
        server_time = client.fetch_server_time()
        effective_end = min(end_dt, server_time)
        if start_dt > effective_end:
            raise ValueError(
                f"start {start_dt.isoformat()} 晚于 Binance 服务器时间 "
                f"{server_time.isoformat()}"
            )
        kline_frame = self.fetch_klines_range(
            symbol=symbol,
            interval=interval,
            start=start_dt,
            end=effective_end,
            market="futures",
        )
        if kline_frame.empty:
            raise ValueError("请求范围内没有永续 K 线数据")

        feature_errors: dict[str, str] = {}
        feature_period = _derivatives_feature_period(interval)

        def optional_feature(name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
            try:
                return fetch()
            except (requests.RequestException, ValueError) as exc:
                feature_errors[name] = str(exc)
                return pd.DataFrame()

        # OI/Basis 的保留期相对 Binance 当前服务器时间计算，而不是请求 end。
        feature_start = max(start_dt, server_time - timedelta(days=29))
        funding = optional_feature(
            "funding_rate",
            lambda: self._paginate_datetime(
                lambda left, right: client.fetch_funding_rate(
                    start_time=left, end_time=right, limit=1000
                ),
                start=start_dt,
                end=effective_end,
            ),
        )
        open_interest = (
            optional_feature(
                "open_interest",
                lambda: self._paginate_datetime(
                    lambda left, right: client.fetch_open_interest(
                        period=feature_period, start_time=left, end_time=right, limit=500
                    ),
                    start=feature_start,
                    end=effective_end,
                ),
            )
            if feature_start <= effective_end
            else pd.DataFrame()
        )
        basis = (
            optional_feature(
                "basis",
                lambda: self._paginate_datetime(
                    lambda left, right: client.fetch_historical_basis(
                        period=feature_period, start_time=left, end_time=right, limit=500
                    ),
                    start=feature_start,
                    end=effective_end,
                ),
            )
            if feature_start <= effective_end
            else pd.DataFrame()
        )

        bronze_path = self._path(
            layer="bronze",
            market="futures",
            symbol=symbol,
            interval=interval,
            filename="klines.parquet",
        )
        bronze = self._write_merged(bronze_path, kline_frame, key="open_time")
        feature_frames = {
            "funding_rate": funding,
            "open_interest": open_interest,
            "basis": basis,
        }
        feature_paths: dict[str, str] = {}
        stored_features: dict[str, pd.DataFrame] = {}
        for name, feature_frame in feature_frames.items():
            path = self._path(
                layer="bronze",
                market="futures",
                symbol=symbol,
                interval=interval,
                filename=f"{name}.parquet",
            )
            if not feature_frame.empty:
                stored_features[name] = self._write_merged(
                    path, feature_frame, key="timestamp"
                )
                feature_paths[name] = str(path)
            elif path.exists():
                stored_features[name] = pd.read_parquet(path)
            else:
                stored_features[name] = pd.DataFrame()

        silver = bronze_to_silver(bronze)
        enriched = silver_to_factor_input(silver)
        stored_funding = stored_features["funding_rate"]
        stored_open_interest = stored_features["open_interest"]
        stored_basis = stored_features["basis"]
        if not stored_funding.empty:
            enriched = merge_point_in_time_features(
                enriched, stored_funding, feature_columns=["funding_rate"]
            )
        if not stored_open_interest.empty:
            enriched = merge_point_in_time_features(
                enriched, stored_open_interest, feature_columns=["open_interest"]
            )
        if not stored_basis.empty:
            available_basis = stored_basis.copy()
            available_basis["available_at"] = available_basis["timestamp"] + pd.to_timedelta(
                get_interval_ms(feature_period), unit="ms"
            )
            enriched = merge_point_in_time_features(
                enriched,
                available_basis,
                feature_columns=["basis"],
                feature_time_column="available_at",
            )

        silver_path = self._path(
            layer="silver",
            market="futures",
            symbol=symbol,
            interval=interval,
            filename="klines.parquet",
        )
        merge_parquet_safe(
            silver_path,
            enriched.reset_index(),
            key="open_time_utc",
        )
        return {
            "status": "downloaded",
            "symbol": symbol.upper(),
            "interval": interval,
            "market": "futures",
            "requested_start": start_dt.isoformat(),
            "requested_end": end_dt.isoformat(),
            "effective_end": effective_end.isoformat(),
            "binance_server_time": server_time.isoformat(),
            "rows": {
                "klines": len(kline_frame),
                "funding_rate": len(funding),
                "open_interest": len(open_interest),
                "basis": len(basis),
            },
            "paths": {
                "bronze_klines": str(bronze_path),
                "silver_klines": str(silver_path),
                **feature_paths,
            },
            "limitations": [
                "Binance REST historical open interest and basis are limited to about 30 days.",
                "Long-history OI/Basis must be imported from an archive; missing values stay null.",
                "Basis becomes available only after its source period ends to prevent lookahead.",
                f"OI/Basis source period is {feature_period}; 1m research uses closed 5m data.",
            ],
            "feature_errors": feature_errors,
        }

    def download_all(self, symbols: list[str], interval: str) -> None:
        """兼容旧入口：下载最近 30 天的研究 bundle。"""
        end = datetime.now(UTC)
        start = end - timedelta(days=30)
        for symbol in symbols:
            self.download_bundle(symbol=symbol, interval=interval, start=start, end=end)
