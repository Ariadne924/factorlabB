"""创建至少六个月的锁定 OOS 元数据，不运行研究。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.oos_lock import create_oos_lock  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="创建锁定六个月 OOS 的元数据门禁")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/oos_lock.json")
    args = parser.parse_args(argv)
    result = create_oos_lock(
        args.output, start=args.start, end=args.end,
        symbols=tuple(args.symbols), intervals=tuple(args.intervals),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
