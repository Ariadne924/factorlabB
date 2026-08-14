from __future__ import annotations

import hashlib
import io
import zipfile
from unittest.mock import Mock

from data.binance_archive import BinanceArchiveDownloader


def archive_bytes(timestamp: int = 1_735_689_600_000_000) -> bytes:
    output = io.BytesIO()
    row = f"{timestamp},100,102,99,101,2,{timestamp + 3_599_999_999},202,7,1,101,0\n"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("BTCUSDT-1h-2025-01.csv", row)
    return output.getvalue()


def test_archive_url_and_microsecond_timestamp_normalization() -> None:
    relative = BinanceArchiveDownloader.monthly_relative_path(
        symbol="btcusdt", interval="1h", year=2025, month=1, market="futures"
    )
    assert relative == "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01.zip"
    frame = BinanceArchiveDownloader.parse_kline_zip(
        archive_bytes(), symbol="BTCUSDT", interval="1h"
    )
    assert frame.loc[0, "open_time"] == 1_735_689_600_000
    assert frame.loc[0, "number_of_trades"] == 7


def test_archive_download_verifies_checksum_and_writes_bronze(tmp_path) -> None:
    payload = archive_bytes(1_700_000_000_000)
    checksum = hashlib.sha256(payload).hexdigest().encode()
    session = Mock()
    first, second = Mock(), Mock()
    first.status_code = second.status_code = 200
    first.headers = second.headers = {}
    first.content, second.content = payload, checksum
    first.raise_for_status.return_value = second.raise_for_status.return_value = None
    session.get.side_effect = [first, second]
    result = BinanceArchiveDownloader(data_dir=tmp_path, session=session).download_month(
        symbol="BTCUSDT", interval="1h", year=2023, month=11
    )
    assert result["checksum_verified"] is True
    assert result["rows"] == 1
    assert (tmp_path / "bronze/binance/futures/BTCUSDT/1h/klines.parquet").exists()
    assert (tmp_path / "silver/binance/futures/BTCUSDT/1h/klines.parquet").exists()


def test_archive_download_retries_server_error() -> None:
    session = Mock()
    failed, recovered = Mock(), Mock()
    failed.status_code, recovered.status_code = 500, 200
    failed.headers = recovered.headers = {}
    recovered.content = b"ok"
    recovered.raise_for_status.return_value = None
    session.get.side_effect = [failed, recovered]
    sleeps: list[float] = []
    downloader = BinanceArchiveDownloader(
        session=session, max_retries=1, backoff_base=0.25, sleep=sleeps.append
    )
    assert downloader._download("https://example.test/file.zip") == b"ok"
    assert sleeps == [0.25]


def test_archive_cache_avoids_repeated_network_download(tmp_path) -> None:
    payload = archive_bytes(1_700_000_000_000)
    checksum = hashlib.sha256(payload).hexdigest().encode()
    session = Mock()
    archive_response, checksum_response = Mock(), Mock()
    archive_response.status_code = checksum_response.status_code = 200
    archive_response.headers = checksum_response.headers = {}
    archive_response.content, checksum_response.content = payload, checksum
    archive_response.raise_for_status.return_value = None
    checksum_response.raise_for_status.return_value = None
    session.get.side_effect = [archive_response, checksum_response]
    downloader = BinanceArchiveDownloader(data_dir=tmp_path, session=session)
    relative = downloader.monthly_relative_path(
        symbol="BTCUSDT", interval="1h", year=2023, month=11, market="futures"
    )

    first, first_hit = downloader._load_archive(relative)
    second, second_hit = downloader._load_archive(relative)

    assert first == second == payload
    assert first_hit is False
    assert second_hit is True
    assert session.get.call_count == 2


def test_archive_batch_writes_dataset_once(tmp_path) -> None:
    payloads = [
        archive_bytes(1_700_000_000_000),
        archive_bytes(1_702_678_400_000),
    ]
    downloader = BinanceArchiveDownloader(data_dir=tmp_path)
    downloader._load_archive = Mock(
        side_effect=[(payloads[0], False), (payloads[1], True)]
    )
    downloader._write_dataset = Mock(return_value=(tmp_path / "bronze", tmp_path / "silver"))

    result = downloader.download_months(
        tasks=[
            {"symbol": "BTCUSDT", "interval": "1h", "year": 2023, "month": 11},
            {"symbol": "BTCUSDT", "interval": "1h", "year": 2023, "month": 12},
        ]
    )

    assert len(result["results"]) == 2
    assert result["errors"] == []
    assert result["results"][1]["archive_cache_hit"] is True
    downloader._write_dataset.assert_called_once()
