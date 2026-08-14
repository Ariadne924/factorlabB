"""Supervise realtime collection and Streamlit with health checks and bounded restarts."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operations.runtime import (  # noqa: E402
    CrashCircuitBreaker,
    CrashPolicy,
    json_heartbeat_age,
    rotate_log,
    write_status,
)


def restart_delay(restarts: int, *, cap: float = 30.0) -> float:
    return min(cap, 2 ** max(0, restarts - 1))


def _stop_processes(processes: dict[str, subprocess.Popen[bytes]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="持续运行实时采集和 Streamlit 前端")
    parser.add_argument(
        "--status-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "service_status.json",
    )
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--no-live", action="store_true", help="不启动实时采集器")
    parser.add_argument("--no-frontend", action="store_true", help="不启动前端")
    parser.add_argument("--max-crashes", type=int, default=5)
    parser.add_argument("--crash-window", type=float, default=60.0)
    parser.add_argument("--live-stale-seconds", type=float, default=45.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    commands: dict[str, list[str]] = {
        "live_market": [sys.executable, str(PROJECT_ROOT / "scripts" / "run_live_market.py")],
        "frontend": [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "visualization" / "app.py"),
            "--server.headless",
            "true",
        ],
    }
    if args.no_live:
        commands.pop("live_market")
    if args.no_frontend:
        commands.pop("frontend")
    if not commands:
        parser.error("--no-live 和 --no-frontend 不能同时使用")
    if min(args.live_stale_seconds, args.startup_grace_seconds) <= 0:
        parser.error("heartbeat thresholds must be positive")

    breaker = CrashCircuitBreaker(
        CrashPolicy(max_crashes=args.max_crashes, window_seconds=args.crash_window)
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs: dict[str, BinaryIO] = {}
    restarts = {name: 0 for name in commands}
    started_at: dict[str, float] = {}
    last_exit_code: dict[str, int | None] = {name: None for name in commands}
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stopping:
            now_monotonic = time.monotonic()
            for name, command in commands.items():
                process = processes.get(name)
                if process is not None and process.poll() is not None:
                    last_exit_code[name] = process.returncode
                    restarts[name] += 1
                    breaker.record(name, now_monotonic)
                    if handle := logs.pop(name, None):
                        handle.close()
                    processes.pop(name, None)
                    if breaker.is_open(name, now_monotonic):
                        write_status(
                            args.status_path,
                            {
                                "status": "failed",
                                "reason": "crash_loop_circuit_open",
                                "service": name,
                                "updated_at": datetime.now(UTC).isoformat(),
                            },
                        )
                        _stop_processes(processes)
                        return 1
                    time.sleep(restart_delay(restarts[name]))
                    process = None

                if process is None:
                    log_path = args.log_dir / f"{name}.log"
                    rotate_log(log_path)
                    log_handle = log_path.open("ab", buffering=0)
                    logs[name] = log_handle
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    processes[name] = process
                    started_at[name] = time.monotonic()

            heartbeat_age = json_heartbeat_age(PROJECT_ROOT / "reports" / "live_market.json")
            live_process = processes.get("live_market")
            if (
                live_process is not None
                and live_process.poll() is None
                and time.monotonic() - started_at["live_market"] > args.startup_grace_seconds
                and (heartbeat_age is None or heartbeat_age > args.live_stale_seconds)
            ):
                live_process.terminate()

            write_status(
                args.status_path,
                {
                    "status": "running",
                    "updated_at": datetime.now(UTC).isoformat(),
                    "services": {
                        name: {
                            "pid": process.pid,
                            "restarts": restarts[name],
                            "last_exit_code": last_exit_code[name],
                            "uptime_seconds": round(
                                time.monotonic() - started_at[name], 1
                            ),
                            "health": (
                                "stale"
                                if name == "live_market"
                                and heartbeat_age is not None
                                and heartbeat_age > args.live_stale_seconds
                                else "running"
                            ),
                            "heartbeat_age_seconds": (
                                round(heartbeat_age, 1)
                                if name == "live_market" and heartbeat_age is not None
                                else None
                            ),
                            "log": str((args.log_dir / f"{name}.log").resolve()),
                        }
                        for name, process in processes.items()
                    },
                },
            )
            time.sleep(2)
    finally:
        _stop_processes(processes)
        for handle in logs.values():
            handle.close()
        write_status(
            args.status_path,
            {"status": "stopped", "updated_at": datetime.now(UTC).isoformat()},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
