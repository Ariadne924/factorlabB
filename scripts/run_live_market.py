"""启动 Binance 公共实时行情采集器。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_market import DEFAULT_LIVE_SYMBOLS, BinanceLiveCollector  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="采集 Binance USD-M 公共实时行情")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_LIVE_SYMBOLS))
    parser.add_argument("--interval", default="1m")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_ROOT / "reports" / "live_market.json",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="只更新实时快照，不把已收盘 K 线合并到 Silver",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        help="收到指定消息数后退出，仅用于连通性诊断与测试",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(str(symbol).upper() for symbol in args.symbols)
    print("启动 Binance 公共 WebSocket（无需 API Key）")
    print(f"交易对: {', '.join(symbols)} | K线: {args.interval}")
    print(f"前端快照: {args.snapshot.resolve()}")
    collector = BinanceLiveCollector(
        symbols=symbols,
        interval=args.interval,
        snapshot_path=args.snapshot,
        persist_closed_bars=not args.no_persist,
    )
    collector.run(max_messages=args.max_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
