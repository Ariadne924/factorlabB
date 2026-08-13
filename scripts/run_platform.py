"""长期运行实时采集器与 Streamlit；子进程退出后有限退避重启。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def restart_delay(restarts: int, *, cap: float = 30.0) -> float:
    return min(cap, 2 ** max(0, restarts - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="持续运行实时采集和 Streamlit 前端")
    parser.add_argument(
        "--status-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "service_status.json",
    )
    parser.add_argument("--no-live", action="store_true", help="不启动实时采集器")
    parser.add_argument("--no-frontend", action="store_true", help="不启动前端")
    args = parser.parse_args()
    status_path = args.status_path
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
        parser.error("--no-live 与 --no-frontend 不能同时使用")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[bytes]] = {}
    restarts = {name: 0 for name in commands}
    try:
        while True:
            for name, command in commands.items():
                process = processes.get(name)
                if process is None or process.poll() is not None:
                    if process is not None:
                        restarts[name] += 1
                        time.sleep(restart_delay(restarts[name]))
                    processes[name] = subprocess.Popen(command, cwd=PROJECT_ROOT)
            status = {
                "status": "running",
                "updated_at": datetime.now(UTC).isoformat(),
                "services": {
                    name: {"pid": process.pid, "restarts": restarts[name]}
                    for name, process in processes.items()
                },
            }
            temporary = status_path.with_suffix(f"{status_path.suffix}.tmp")
            temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
            temporary.replace(status_path)
            time.sleep(2)
    except KeyboardInterrupt:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        status_path.write_text(
            json.dumps(
                {"status": "stopped", "updated_at": datetime.now(UTC).isoformat()},
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
