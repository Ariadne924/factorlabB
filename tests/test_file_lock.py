from __future__ import annotations

import threading
import time

import pytest

from utils.file_lock import FileLock, FileLockTimeout


def test_file_lock_excludes_concurrent_writer(tmp_path) -> None:
    path = tmp_path / "data.lock"
    acquired = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with FileLock(path):
            acquired.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=1)
    with pytest.raises(FileLockTimeout):
        with FileLock(path, timeout=0.05, poll_interval=0.01):
            pass
    release.set()
    thread.join(timeout=1)

    with FileLock(path, timeout=0.2):
        time.sleep(0.001)


def test_file_lock_validates_timing_values(tmp_path) -> None:
    with pytest.raises(ValueError):
        FileLock(tmp_path / "x", timeout=-1)
