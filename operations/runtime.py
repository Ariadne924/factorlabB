"""Testable crash-loop, heartbeat and status primitives for the local supervisor."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CrashPolicy:
    max_crashes: int = 5
    window_seconds: float = 60.0

    def validate(self) -> None:
        if self.max_crashes < 1 or self.window_seconds <= 0:
            raise ValueError("crash policy values must be positive")


class CrashCircuitBreaker:
    def __init__(self, policy: CrashPolicy | None = None) -> None:
        self.policy = policy or CrashPolicy()
        self.policy.validate()
        self._crashes: dict[str, deque[float]] = defaultdict(deque)

    def record(self, service: str, now: float) -> int:
        events = self._crashes[service]
        events.append(now)
        cutoff = now - self.policy.window_seconds
        while events and events[0] < cutoff:
            events.popleft()
        return len(events)

    def is_open(self, service: str, now: float) -> bool:
        events = self._crashes[service]
        cutoff = now - self.policy.window_seconds
        while events and events[0] < cutoff:
            events.popleft()
        return len(events) >= self.policy.max_crashes


def json_heartbeat_age(path: Path, *, now: datetime | None = None) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(str(payload["updated_at"])).astimezone(UTC)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return max(0.0, ((now or datetime.now(UTC)) - updated).total_seconds())


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rotate_log(path: Path, *, max_bytes: int = 5_000_000, backups: int = 3) -> None:
    """Rotate only before process spawn, so an open child handle is never renamed."""
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    oldest = path.with_suffix(f"{path.suffix}.{backups}")
    if oldest.exists():
        oldest.unlink()
    for index in range(backups - 1, 0, -1):
        source = path.with_suffix(f"{path.suffix}.{index}")
        if source.exists():
            source.replace(path.with_suffix(f"{path.suffix}.{index + 1}"))
    path.replace(path.with_suffix(f"{path.suffix}.1"))
