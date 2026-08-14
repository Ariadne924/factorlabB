from __future__ import annotations

from typing import Any

from scripts.check_binance_testnet import main


class FakeClient:
    def sync_time(self) -> int:
        return 12

    def account(self) -> dict[str, Any]:
        return {
            "canTrade": True,
            "totalWalletBalance": "1000",
            "assets": [{"asset": "USDT"}],
            "positions": [],
        }


def test_testnet_check_is_offline_by_default(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    called = {"sync": False}

    class OfflineClient(FakeClient):
        def sync_time(self) -> int:
            called["sync"] = True
            return 0

    assert (
        main(
            ["--ledger", str(tmp_path / "state.sqlite3")],
            client_factory=lambda config, ledger: OfflineClient(),
        )
        == 0
    )
    assert called["sync"] is False
    assert '"orders_submitted": 0' in capsys.readouterr().out


def test_testnet_check_can_read_account_without_ordering(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert (
        main(
            ["--account", "--ledger", str(tmp_path / "state.sqlite3")],
            client_factory=lambda config, ledger: FakeClient(),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"account_checked": true' in output
    assert '"orders_submitted": 0' in output
