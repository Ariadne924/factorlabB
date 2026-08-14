"""Small cross-platform advisory file lock used by local data writers."""

from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import BinaryIO


class FileLockTimeout(TimeoutError):
    """Raised when an advisory lock cannot be acquired before the deadline."""


class FileLock(AbstractContextManager["FileLock"]):
    """Lock one byte in a sidecar file without adding a third-party dependency.

    The lock is advisory: every project writer touching the protected resource must
    use the same sidecar path. It works on Windows and POSIX/WSL.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.05,
    ) -> None:
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("timeout must be non-negative and poll_interval positive")
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._handle: BinaryIO | None = None

    def _try_lock(self) -> bool:
        assert self._handle is not None
        if os.name == "nt":
            import msvcrt

            handle = self._handle
            handle.seek(0)
            try:
                msvcrt.locking(  # type: ignore[attr-defined]
                    handle.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined]
                )
            except OSError:
                return False
            return True

        import fcntl

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def acquire(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        self._handle = handle
        deadline = time.monotonic() + self.timeout
        while not self._try_lock():
            if time.monotonic() >= deadline:
                handle.close()
                self._handle = None
                raise FileLockTimeout(f"timed out acquiring lock: {self.path}")
            time.sleep(self.poll_interval)
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                self._handle.fileno(),
                msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                1,
            )
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()
