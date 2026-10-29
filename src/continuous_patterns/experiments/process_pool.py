"""Process pool that tears down worker children on interrupt or parent death."""

from __future__ import annotations

import signal
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext
from types import TracebackType
from typing import Any


def _linux_parent_deathsig_to_term() -> None:
    """SIGTERM this worker when the parent process dies (Linux ``PR_SET_PDEATHSIG``)."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
            return
    except (OSError, AttributeError):
        return


def worker_bootstrap(user_initializer: Callable[[], None] | None = None) -> None:
    """Process-pool initializer: inherit parent-death signal, then optional user hook."""
    _linux_parent_deathsig_to_term()
    if user_initializer is not None:
        user_initializer()


class PoolInitializer:
    """Pickle-friendly ``ProcessPoolExecutor`` initializer (spawn-safe)."""

    def __init__(self, user_initializer: Callable[[], None] | None = None) -> None:
        self._user_initializer = user_initializer

    def __call__(self) -> None:
        worker_bootstrap(self._user_initializer)


class ManagedProcessPool:
    """``ProcessPoolExecutor`` that kills worker children on SIGINT/SIGTERM or pool exit."""

    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: BaseContext | None = None,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        self._max_workers = int(max_workers)
        self._mp_context = mp_context
        self._user_initializer = initializer
        self._executor: ProcessPoolExecutor | None = None
        self._old_sigint: Any = signal.SIG_DFL
        self._old_sigterm: Any = signal.SIG_DFL
        self._shutting_down = False

    def __enter__(self) -> ProcessPoolExecutor:
        self._executor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=self._mp_context,
            initializer=PoolInitializer(self._user_initializer),
        )
        self._old_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._old_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        return self._executor

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        force = (
            self._shutting_down
            or exc_type is KeyboardInterrupt
            or isinstance(exc, KeyboardInterrupt)
        )
        self._shutdown_workers(force=force)
        self._restore_signals()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._shutdown_workers(force=True)
        self._restore_signals()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    def _restore_signals(self) -> None:
        if self._old_sigint is not signal.SIG_DFL:
            signal.signal(signal.SIGINT, self._old_sigint)
            self._old_sigint = signal.SIG_DFL
        if self._old_sigterm is not signal.SIG_DFL:
            signal.signal(signal.SIGTERM, self._old_sigterm)
            self._old_sigterm = signal.SIG_DFL

    def _shutdown_workers(self, *, force: bool) -> None:
        ex = self._executor
        if ex is None:
            return
        self._executor = None
        try:
            ex.shutdown(wait=not force, cancel_futures=True)
        except Exception:
            pass
        processes = getattr(ex, "_processes", None)
        if not isinstance(processes, dict):
            return
        for proc in processes.values():
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    if force:
                        proc.kill()
                    else:
                        proc.terminate()
            except (ProcessLookupError, OSError, ValueError):
                pass
        if force:
            for proc in processes.values():
                if proc is None:
                    continue
                try:
                    proc.join(timeout=2.0)
                except Exception:
                    pass
