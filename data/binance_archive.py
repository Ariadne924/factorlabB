"""Binance 官方 data.binance.vision K 线归档下载与校验。"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import requests

from config.constants import KlineInterval
from config.settings import DEFAULT_DATA_DIR
from data.paths import normalize_path_segment, safe_data_path
from data.schema import KlineRaw, bronze_to_silver, raw_to_dataframe


class BinanceArchiveDownloader:
    """下载月度官方 ZIP，校验 SHA256 后幂等写入 Bronze。"""

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        session: requests.Session | None = None,
        base_url: str = "https://data.binance.vision",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.data_dir = (data_dir or DEFAULT_DATA_DIR).resolve()
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self.cache_dir = (
            cache_dir or self.data_dir / "cache" / "binance_archive"
        ).resolve()
        self.use_cache = use_cache

    @staticmethod
    def monthly_relative_path(
        *, symbol: str, interval: str, year: int, month: int, market: str
    ) -> str:
        if not 1 <= month <= 12:
            raise ValueError("month 必须在 1..12")
        if year < 2017:
            raise ValueError("year 早于 Binance 公开数据范围")
        norm_symbol = normalize_path_segment(symbol, field_name="symbol", case="upper")
        norm_interval = normalize_path_segment(interval, field_name="interval", case="lower")
        KlineInterval(norm_interval)
        if market == "spot":
            prefix = "data/spot/monthly/klines"
        elif market == "futures":
            prefix = "data/futures/um/monthly/klines"
        else:
            raise ValueError("market 仅支持 spot/futures")
        filename = f"{norm_symbol}-{norm_interval}-{year:04d}-{month:02d}.zip"
        return f"{prefix}/{norm_symbol}/{norm_interval}/{filename}"

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after
                        else self.backoff_base * 2**attempt
                    )
                    self._sleep(delay)
                    continue
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self._sleep(self.backoff_base * 2**attempt)
        raise RuntimeError("Binance 归档下载重试耗尽") from last_error

    @staticmethod
    def _verify_checksum(payload: bytes, checksum_text: str, filename: str) -> None:
        fields = checksum_text.strip().split()
        if not fields:
            raise ValueError("Binance 归档 checksum 为空")
        expected = fields[0].lower()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"Binance 归档 SHA256 不匹配: {filename}")

    def _cache_paths(self, relative: str) -> tuple[Path, Path]:
        archive_path = (self.cache_dir / Path(*relative.split("/"))).resolve()
        try:
            archive_path.relative_to(self.cache_dir)
        except ValueError as exc:
            raise ValueError("archive cache path escapes cache directory") from exc
        if archive_path == self.cache_dir:
            raise ValueError("archive cache path must name a file")
        return archive_path, Path(f"{archive_path}.CHECKSUM")

    def _load_archive(self, relative: str) -> tuple[bytes, bool]:
        """Return a checksum-verified payload and whether it came from cache."""
        filename = relative.rsplit("/", 1)[-1]
        cache_path, checksum_path = self._cache_paths(relative)
        if self.use_cache and cache_path.exists() and checksum_path.exists():
            payload = cache_path.read_bytes()
            checksum_text = checksum_path.read_text(encoding="utf-8")
            try:
                self._verify_checksum(payload, checksum_text, filename)
            except ValueError:
                pass
            else:
                return payload, True

        url = f"{self.base_url}/{relative}"
        payload = self._download(url)
        checksum_text = self._download(f"{url}.CHECKSUM").decode("utf-8")
        self._verify_checksum(payload, checksum_text, filename)
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            checksum_path.write_text(checksum_text, encoding="utf-8")
        return payload, False

    @staticmethod
    def _timestamp_ms(value: str) -> int:
        raw = int(value)
        # Spot 自 2025-01-01 起使用微秒；管道内部仍统一为毫秒。
        return raw // 1000 if raw >= 100_000_000_000_000 else raw

    @classmethod
    def parse_kline_zip(cls, payload: bytes, *, symbol: str, interval: str) -> pd.DataFrame:
        rows: list[KlineRaw] = []
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(names) != 1:
                    raise ValueError("Binance K 线 ZIP 必须恰好包含一个 CSV")
                content = archive.read(names[0]).decode("utf-8")
        except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
            raise ValueError("Binance K 线归档不是有效 ZIP/UTF-8 CSV") from exc
        for item in csv.reader(io.StringIO(content)):
            if not item or not item[0].lstrip("-").isdigit():
                continue
            if len(item) < 11:
                raise ValueError("Binance K 线归档行字段不足")
            rows.append(
                KlineRaw(
                    open_time=cls._timestamp_ms(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    close_time=cls._timestamp_ms(item[6]),
                    quote_asset_volume=float(item[7]),
                    number_of_trades=int(item[8]),
                    taker_buy_base_volume=float(item[9]),
                    taker_buy_quote_volume=float(item[10]),
                )
            )
        if not rows:
            raise ValueError("Binance K 线归档没有数据行")
        return raw_to_dataframe(rows, symbol=symbol, interval=interval).sort_values("open_time")

    def download_month(
        self, *, symbol: str, interval: str, year: int, month: int, market: str = "futures"
    ) -> dict[str, object]:
        relative = self.monthly_relative_path(
            symbol=symbol, interval=interval, year=year, month=month, market=market
        )
        payload, cache_hit = self._load_archive(relative)
        frame = self.parse_kline_zip(payload, symbol=symbol, interval=interval)
        path, silver_path = self._write_dataset(
            [frame], symbol=symbol, interval=interval, market=market
        )
        return self._result(
            frame=frame,
            symbol=symbol,
            interval=interval,
            market=market,
            year=year,
            month=month,
            bronze_path=path,
            silver_path=silver_path,
            cache_hit=cache_hit,
        )

    def _write_dataset(
        self,
        frames: Sequence[pd.DataFrame],
        *,
        symbol: str,
        interval: str,
        market: str,
    ) -> tuple[Path, Path]:
        """Merge a batch into Bronze and rebuild Silver exactly once."""
        path = safe_data_path(
            self.data_dir,
            "bronze",
            "binance",
            market,
            normalize_path_segment(symbol, field_name="symbol", case="upper"),
            normalize_path_segment(interval, field_name="interval", case="lower"),
            "klines.parquet",
        )
        combined = pd.concat(list(frames), ignore_index=True)
        if path.exists():
            combined = pd.concat([pd.read_parquet(path), combined], ignore_index=True)
        combined = combined.drop_duplicates("open_time", keep="last").sort_values("open_time")
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, compression="zstd", index=False)
        silver = bronze_to_silver(combined.reset_index(drop=True))
        silver_path = safe_data_path(
            self.data_dir,
            "silver",
            "binance",
            market,
            normalize_path_segment(symbol, field_name="symbol", case="upper"),
            normalize_path_segment(interval, field_name="interval", case="lower"),
            "klines.parquet",
        )
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(silver_path, compression="zstd", index=False)
        return path, silver_path

    @staticmethod
    def _result(
        *,
        frame: pd.DataFrame,
        symbol: str,
        interval: str,
        market: str,
        year: int,
        month: int,
        bronze_path: Path,
        silver_path: Path,
        cache_hit: bool,
    ) -> dict[str, object]:
        return {
            "status": "downloaded",
            "source": "data.binance.vision",
            "symbol": symbol.upper(),
            "interval": interval,
            "market": market,
            "year": year,
            "month": month,
            "rows": len(frame),
            "bronze_path": str(bronze_path),
            "silver_path": str(silver_path),
            "checksum_verified": True,
            "archive_cache_hit": cache_hit,
        }

    def download_months(
        self,
        *,
        tasks: Sequence[dict[str, object]],
        market: str = "futures",
    ) -> dict[str, object]:
        """Download one symbol/interval batch and write its Parquet files once."""
        if not tasks:
            return {"results": [], "errors": []}
        symbols = {str(task["symbol"]).upper() for task in tasks}
        intervals = {str(task["interval"]) for task in tasks}
        if len(symbols) != 1 or len(intervals) != 1:
            raise ValueError("download_months requires one symbol and interval per batch")
        symbol, interval = symbols.pop(), intervals.pop()
        downloaded: list[tuple[dict[str, object], pd.DataFrame, bool]] = []
        errors: list[dict[str, object]] = []
        for task in tasks:
            year, month = int(task["year"]), int(task["month"])
            relative = self.monthly_relative_path(
                symbol=symbol,
                interval=interval,
                year=year,
                month=month,
                market=market,
            )
            try:
                payload, cache_hit = self._load_archive(relative)
                frame = self.parse_kline_zip(payload, symbol=symbol, interval=interval)
            except (OSError, ValueError, requests.RequestException) as exc:
                errors.append({"task": dict(task), "error": str(exc)})
                continue
            downloaded.append((dict(task), frame, cache_hit))
        if not downloaded:
            return {"results": [], "errors": errors}
        bronze_path, silver_path = self._write_dataset(
            [item[1] for item in downloaded],
            symbol=symbol,
            interval=interval,
            market=market,
        )
        results = [
            self._result(
                frame=frame,
                symbol=symbol,
                interval=interval,
                market=market,
                year=int(task["year"]),
                month=int(task["month"]),
                bronze_path=bronze_path,
                silver_path=silver_path,
                cache_hit=cache_hit,
            )
            for task, frame, cache_hit in downloaded
        ]
        return {"results": results, "errors": errors}
