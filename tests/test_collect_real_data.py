from __future__ import annotations

import json
from unittest.mock import Mock

import requests

from scripts.collect_real_data import (
    DATA_SOURCE_PROBES,
    build_collection_plan,
    collect_real_data,
    probe_data_sources,
)


def test_plan_uses_archives_for_closed_months_and_rest_for_recent_window() -> None:
    plan = build_collection_plan(
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1m", "1h", "24h"),
        start="2026-05-01",
        end="2026-08-12",
        recent_days=29,
    )
    assert plan["archive_task_count"] == 2 * 2 * 3
    assert plan["rest_task_count"] == 2 * 3
    assert {task["interval"] for task in plan["archive_tasks"]} == {"1h", "1d"}


def test_collection_checkpoints_success_and_keeps_failure_for_retry(tmp_path) -> None:
    plan = build_collection_plan(
        symbols=("BTCUSDT",),
        intervals=("1m", "1h"),
        start="2026-06-01",
        end="2026-08-12",
        recent_days=29,
    )

    class Archive:
        def __init__(self):
            self.calls = 0

        def download_month(self, **task):
            self.calls += 1
            if task["month"] == 7:
                raise OSError("temporary network error")
            return {"status": "downloaded", **task}

    class Rest:
        def download_bundle(self, **task):
            return {"status": "downloaded", "symbol": task["symbol"]}

    def catalog(data_dir, output):
        del data_dir
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"entry_count": 2}), encoding="utf-8")
        return {"entry_count": 2}

    archive = Archive()
    result = collect_real_data(
        plan,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        archive=archive,
        rest=Rest(),
        catalog_builder=catalog,
    )
    assert result["status"] == "partial"
    assert result["completed_task_count"] == result["expected_task_count"] - 1
    assert "temporary network error" in result["errors"][0]["error"]
    manifest = json.loads(
        (tmp_path / "reports/real_data_collection_manifest.json").read_text("utf-8")
    )
    assert manifest["catalog_entry_count"] == 2

    resumed = collect_real_data(
        plan,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        archive=archive,
        rest=Rest(),
        catalog_builder=catalog,
    )
    assert resumed["status"] == "partial"
    assert archive.calls == 3  # two first-run months, then only failed July


def test_source_probe_reports_each_endpoint_without_hiding_failures() -> None:
    session = Mock()
    good = Mock(status_code=200)
    session.get.side_effect = [good, requests.ConnectionError("blocked"), good, good]
    result = probe_data_sources(session=session)
    assert len(result["results"]) == len(DATA_SOURCE_PROBES)
    assert result["status"] == "partial_or_failed"
    assert sum(item["ok"] for item in result["results"]) == 3
    assert "blocked" in result["results"][1]["error"]


def test_collection_uses_archive_batch_when_supported(tmp_path) -> None:
    plan = build_collection_plan(
        symbols=("BTCUSDT",),
        intervals=("1h",),
        start="2026-06-01",
        end="2026-08-12",
        recent_days=29,
        mode="archive",
    )

    class Archive:
        def __init__(self) -> None:
            self.calls = 0

        def download_months(self, *, tasks, market):
            self.calls += 1
            assert market == "futures"
            return {
                "results": [
                    {"status": "downloaded", "market": market, **task}
                    for task in tasks
                ],
                "errors": [],
            }

    archive = Archive()
    result = collect_real_data(
        plan,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        archive=archive,
        catalog_builder=lambda *_, **__: {"entry_count": 0},
    )

    assert result["status"] == "completed"
    assert archive.calls == 1
