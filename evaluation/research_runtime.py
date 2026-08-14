"""研究任务的口径级缓存和可观察状态。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESEARCH_CACHE_VERSION = "1.0"
RESEARCH_STATUS_VERSION = "1.0"


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def research_code_fingerprint(project_root: Path) -> str:
    """散列会影响单因子结果的源码；源码改变时自动失效缓存。"""
    roots = (project_root / "factors", project_root / "evaluation")
    files = [path for root in roots for path in root.rglob("*.py")]
    files.append(project_root / "visualization" / "single_factor_report.py")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scope_signature(
    silver_path: Path,
    *,
    settings: dict[str, Any],
    code_fingerprint: str,
) -> str:
    """使用文件身份、研究参数和源码生成口径缓存键。"""
    stat = silver_path.stat()
    return _json_hash(
        {
            "version": RESEARCH_CACHE_VERSION,
            "silver_path": str(silver_path.resolve()),
            "silver_size": stat.st_size,
            "silver_mtime_ns": stat.st_mtime_ns,
            "settings": settings,
            "code_fingerprint": code_fingerprint,
        }
    )


def research_run_signature(
    scope_signatures: dict[str, str], *, context: dict[str, Any]
) -> str:
    """为完全相同的研究选择生成最终汇总快照键。"""
    return _json_hash(
        {
            "version": RESEARCH_CACHE_VERSION,
            "scope_signatures": scope_signatures,
            "context": context,
        }
    )


def load_research_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": RESEARCH_CACHE_VERSION, "scopes": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": RESEARCH_CACHE_VERSION, "scopes": {}}
    if payload.get("version") != RESEARCH_CACHE_VERSION:
        return {"version": RESEARCH_CACHE_VERSION, "scopes": {}}
    payload.setdefault("scopes", {})
    return payload


def cached_report_paths(
    cache: dict[str, Any],
    *,
    scope: str,
    signature: str,
    reports_dir: Path,
) -> list[Path] | None:
    entry = cache.get("scopes", {}).get(scope, {})
    if entry.get("signature") != signature:
        return None
    paths = [reports_dir / relative for relative in entry.get("reports", [])]
    return paths if paths and all(path.is_file() for path in paths) else None


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class ResearchProgress:
    """把长任务状态持续写入 JSON，供前端和运维脚本读取。"""

    def __init__(self, path: Path, *, total_scopes: int, factors_per_scope: int) -> None:
        self.path = path
        self.state: dict[str, Any] = {
            "version": RESEARCH_STATUS_VERSION,
            "status": "running",
            "stage": "factor_research",
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "total_scopes": total_scopes,
            "completed_scopes": 0,
            "total_tasks": total_scopes * factors_per_scope,
            "completed_tasks": 0,
            "cache_hit_scopes": 0,
            "cache_hit_tasks": 0,
            "current_scope": None,
            "current_factor": None,
            "error": None,
        }
        self._write()

    def start_scope(self, scope: str) -> None:
        self.state.update({"current_scope": scope, "current_factor": None})
        self._write()

    def advance(self, *, factor: str | None = None, count: int = 1) -> None:
        self.state["current_factor"] = factor
        self.state["completed_tasks"] += count
        if self.state["completed_tasks"] % 10 == 0:
            self._write()

    def complete_scope(self, *, cache_hit: bool, task_count: int = 0) -> None:
        self.state["completed_scopes"] += 1
        if cache_hit:
            self.state["cache_hit_scopes"] += 1
            self.state["cache_hit_tasks"] += task_count
            self.state["completed_tasks"] += task_count
        self.state["current_factor"] = None
        self._write()

    def complete(self) -> None:
        self.state.update(
            {
                "status": "completed",
                "stage": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "current_scope": None,
                "current_factor": None,
            }
        )
        self._write()

    def fail(self, exc: Exception) -> None:
        self.state.update(
            {
                "status": "failed",
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        self._write()

    def _write(self) -> None:
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        total = int(self.state["total_tasks"])
        completed = int(self.state["completed_tasks"])
        self.state["progress"] = completed / total if total else 1.0
        write_json_atomic(self.path, self.state)
