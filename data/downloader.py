"""分页下载 Binance 研究数据并构建 point-in-time Silver 数据集。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import KlineInterval, get_interval_ms
from config.settings import DEFAULT_DATA_DIR
from data.binance_client import BinanceClient
from data.paths import normalize_path_segment, safe_data_path
from data.schema import bronze_to_silver
from data.silver import merge_point_in_time_features, silver_to_factor_input

ClientFactory = Callable[[str], BinanceClient]


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
        combined = frame.copy()
        if path.exists():
            combined = pd.concat([pd.read_parquet(path), combined], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(subset=[key], keep="last").sort_values(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, compression="zstd", index=False)
        return combined.reset_index(drop=True)

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
        if start_dt >= end_dt:
            raise ValueError("start 必须早于 end")
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
        if start_dt >= end_dt:
            raise ValueError("start 必须早于 end")
        client = self.client_factory(symbol)
        kline_frame = self.fetch_klines_range(
            symbol=symbol,
            interval=interval,
            start=start_dt,
            end=end_dt,
            market="futures",
        )
        if kline_frame.empty:
            raise ValueError("请求范围内没有永续 K 线数据")

        feature_start = max(start_dt, end_dt - timedelta(days=30))
        funding = self._paginate_datetime(
            lambda left, right: client.fetch_funding_rate(
                start_time=left, end_time=right, limit=1000
            ),
            start=start_dt,
            end=end_dt,
        )
        open_interest = self._paginate_datetime(
            lambda left, right: client.fetch_open_interest(
                period=interval, start_time=left, end_time=right, limit=500
            ),
            start=feature_start,
            end=end_dt,
        )
        basis = self._paginate_datetime(
            lambda left, right: client.fetch_historical_basis(
                period=interval, start_time=left, end_time=right, limit=500
            ),
            start=feature_start,
            end=end_dt,
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
                get_interval_ms(interval), unit="ms"
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
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.reset_index().to_parquet(
            silver_path, compression="zstd", index=False
        )
        return {
            "status": "downloaded",
            "symbol": symbol.upper(),
            "interval": interval,
            "market": "futures",
            "requested_start": start_dt.isoformat(),
            "requested_end": end_dt.isoformat(),
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
            ],
        }

    def download_all(self, symbols: list[str], interval: str) -> None:
        """兼容旧入口：下载最近 30 天的研究 bundle。"""
        end = datetime.now(UTC)
        start = end - timedelta(days=30)
        for symbol in symbols:
            self.download_bundle(symbol=symbol, interval=interval, start=start, end=end)
