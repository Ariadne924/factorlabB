from __future__ import annotations

import json

from evaluation.research_runtime import (
    ResearchProgress,
    cached_report_paths,
    load_research_cache,
    research_code_fingerprint,
    research_run_signature,
    scope_signature,
    write_json_atomic,
)


def test_scope_cache_requires_matching_signature_and_existing_reports(tmp_path) -> None:
    silver = tmp_path / "data" / "silver" / "BTCUSDT" / "1h" / "klines.parquet"
    silver.parent.mkdir(parents=True)
    silver.write_bytes(b"first")
    report = tmp_path / "reports" / "single_factor" / "BTCUSDT_1h_probe.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    signature = scope_signature(
        silver, settings={"bootstrap_samples": 10}, code_fingerprint="code-v1"
    )
    cache = {
        "version": "1.0",
        "scopes": {
            "BTCUSDT_1h": {
                "signature": signature,
                "reports": ["single_factor/BTCUSDT_1h_probe.json"],
            }
        },
    }

    assert cached_report_paths(
        cache,
        scope="BTCUSDT_1h",
        signature=signature,
        reports_dir=tmp_path / "reports",
    ) == [report]
    silver.write_bytes(b"changed-size")
    changed = scope_signature(
        silver, settings={"bootstrap_samples": 10}, code_fingerprint="code-v1"
    )
    assert changed != signature
    assert (
        cached_report_paths(
            cache,
            scope="BTCUSDT_1h",
            signature=changed,
            reports_dir=tmp_path / "reports",
        )
        is None
    )


def test_research_code_fingerprint_changes_with_factor_source(tmp_path) -> None:
    for directory in ("factors", "evaluation", "visualization"):
        (tmp_path / directory).mkdir()
    factor = tmp_path / "factors" / "probe.py"
    factor.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "visualization" / "single_factor_report.py").write_text(
        "REPORT = 1\n", encoding="utf-8"
    )
    before = research_code_fingerprint(tmp_path)
    factor.write_text("VALUE = 2\n", encoding="utf-8")
    assert research_code_fingerprint(tmp_path) != before


def test_run_signature_is_order_stable_and_context_sensitive() -> None:
    first = research_run_signature(
        {"BTCUSDT_1h": "a", "ETHUSDT_1h": "b"}, context={"samples": 20}
    )
    reordered = research_run_signature(
        {"ETHUSDT_1h": "b", "BTCUSDT_1h": "a"}, context={"samples": 20}
    )
    changed = research_run_signature(
        {"BTCUSDT_1h": "a", "ETHUSDT_1h": "b"}, context={"samples": 21}
    )
    assert first == reordered
    assert first != changed


def test_progress_and_atomic_cache_are_recoverable(tmp_path) -> None:
    status_path = tmp_path / "reports" / "research_status.json"
    progress = ResearchProgress(status_path, total_scopes=2, factors_per_scope=10)
    progress.start_scope("BTCUSDT_1h")
    progress.advance(factor="probe", count=10)
    progress.complete_scope(cache_hit=False)
    progress.complete_scope(cache_hit=True, task_count=10)
    progress.complete()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["progress"] == 1.0
    assert status["cache_hit_scopes"] == 1
    assert status["cache_hit_tasks"] == 10

    cache_path = tmp_path / "reports" / "research_cache.json"
    write_json_atomic(cache_path, {"version": "1.0", "scopes": {}})
    assert load_research_cache(cache_path)["scopes"] == {}
