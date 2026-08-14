"""Safe Binance testnet readiness check; this command never places an order."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.binance_testnet import (  # noqa: E402
    BinanceTestnetClient,
    BinanceTestnetConfig,
)
from trading.ledger import TradingLedger  # noqa: E402
from trading.models import MarketType  # noqa: E402


class TestnetClientLike(Protocol):
    def sync_time(self) -> int: ...
    def account(self) -> dict[str, Any]: ...


ClientFactory = Callable[[BinanceTestnetConfig, TradingLedger], TestnetClientLike]


def _default_factory(
    config: BinanceTestnetConfig, ledger: TradingLedger
) -> BinanceTestnetClient:
    return BinanceTestnetClient(config, ledger=ledger)


def main(
    argv: list[str] | None = None,
    *,
    client_factory: ClientFactory = _default_factory,
) -> int:
    parser = argparse.ArgumentParser(description="检查 Binance 测试网与本地交易账本")
    parser.add_argument("--market", choices=["spot", "futures"], default="futures")
    parser.add_argument("--connect", action="store_true", help="访问公开测试网时间接口")
    parser.add_argument(
        "--account",
        action="store_true",
        help="读取测试网账户摘要；需要测试网环境变量，不会下单",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "data" / "state" / "testnet.sqlite3",
    )
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    market = MarketType.SPOT if args.market == "spot" else MarketType.USD_M_FUTURES
    config = BinanceTestnetConfig.from_env(market=market, dry_run=not args.account)
    ledger = TradingLedger(args.ledger)
    client = client_factory(config, ledger)
    output: dict[str, Any] = {
        "market": market.value,
        "base_url": config.base_url,
        "dry_run": config.dry_run,
        "credentials_configured": bool(config.api_key and config.api_secret),
        "ledger": ledger.health(),
        "network_checked": False,
        "account_checked": False,
        "orders_submitted": 0,
    }
    if args.connect or args.account:
        output["server_time_offset_ms"] = client.sync_time()
        output["network_checked"] = True
    if args.account:
        account = client.account()
        output["account_checked"] = True
        output["account_summary"] = {
            "can_trade": account.get("canTrade"),
            "wallet_balance": account.get("totalWalletBalance"),
            "asset_count": len(account.get("assets", account.get("balances", []))),
            "position_count": len(account.get("positions", [])),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
