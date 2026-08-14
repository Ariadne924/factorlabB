from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from operations.runtime import (
    CrashCircuitBreaker,
    CrashPolicy,
    json_heartbeat_age,
    rotate_log,
    write_status,
)


def test_crash_circuit_breaker_opens_and_old_events_expire() -> None:
    breaker = CrashCircuitBreaker(CrashPolicy(max_crashes=3, window_seconds=10))
    assert breaker.record("live", 0) == 1
    assert breaker.record("live", 1) == 2
    assert breaker.is_open("live", 1) is False
    assert breaker.record("live", 2) == 3
    assert breaker.is_open("live", 2) is True
    assert breaker.is_open("live", 20) is False


def test_status_write_is_atomic_and_heartbeat_age_is_readable(tmp_path) -> None:
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    write_status(path, {"updated_at": (now - timedelta(seconds=4)).isoformat()})
    assert json.loads(path.read_text(encoding="utf-8"))["updated_at"]
    age = json_heartbeat_age(path, now=now)
    assert age is not None and 3.9 <= age <= 4.1


def test_log_rotation_keeps_bounded_backups(tmp_path) -> None:
    path = tmp_path / "service.log"
    path.write_text("12345", encoding="utf-8")
    rotate_log(path, max_bytes=4, backups=2)
    assert not path.exists()
    assert path.with_suffix(".log.1").exists()
