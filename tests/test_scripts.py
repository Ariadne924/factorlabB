from __future__ import annotations

import unittest
from pathlib import Path

from scripts.incremental_update import main as incremental_main
from scripts.run_all import main as run_all_main
from scripts.run_platform import restart_delay


class ScriptExitTests(unittest.TestCase):
    def test_platform_restart_delay_is_capped(self) -> None:
        self.assertEqual(restart_delay(1), 1)
        self.assertEqual(restart_delay(10), 30)

    def test_run_all_calls_real_research_entry(self) -> None:
        called: list[tuple[Path, Path]] = []

        def research(data_dir: Path, reports_dir: Path) -> dict[str, str]:
            called.append((data_dir, reports_dir))
            return {"status": "insufficient_data"}

        self.assertEqual(
            run_all_main(data_dir=Path("data"), reports_dir=Path("reports"), research=research),
            0,
        )
        self.assertEqual(called, [(Path("data"), Path("reports"))])

    def test_dry_run_is_explicitly_successful(self) -> None:
        self.assertEqual(run_all_main(dry_run=True), 0)
        self.assertEqual(incremental_main(dry_run=True), 0)

    def test_incremental_update_refreshes_then_runs_research(self) -> None:
        calls: list[str] = []

        def refresh(*args: object, **kwargs: object) -> dict[str, int | str]:
            calls.append("refresh")
            return {"status": "ok", "successful": 1}

        def research(*args: object, **kwargs: object) -> dict[str, str]:
            calls.append("research")
            return {"status": "ok"}

        self.assertEqual(
            incremental_main(
                data_dir=Path(".test-tmp/data"),
                reports_dir=Path(".test-tmp/reports"),
                refresh=refresh,
                research=research,
            ),
            0,
        )
        self.assertEqual(calls, ["refresh", "research"])


if __name__ == "__main__":
    unittest.main()
