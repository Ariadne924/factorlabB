from __future__ import annotations

import unittest

from scripts.incremental_update import main as incremental_main
from scripts.run_all import main as run_all_main


class ScriptExitTests(unittest.TestCase):
    def test_unimplemented_entrypoints_fail_by_default(self) -> None:
        self.assertEqual(run_all_main(), 2)
        self.assertEqual(incremental_main(), 2)

    def test_dry_run_is_explicitly_successful(self) -> None:
        self.assertEqual(run_all_main(dry_run=True), 0)
        self.assertEqual(incremental_main(dry_run=True), 0)


if __name__ == "__main__":
    unittest.main()
