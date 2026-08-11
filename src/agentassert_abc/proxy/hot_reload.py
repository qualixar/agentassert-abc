# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""ContractWatcher — background hot-reload of a proxy's contract file.

Ported from agentassert-typec's `hot_reload.py`. Only the tracked type
changes: it watches/swaps a :class:`~agentassert_abc.gateway.SessionEnforcer`
(the port-delta §C4 rename of typec's `SessionMonitor`), not the abc v2
measurement-plane `SessionMonitor`.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

from agentassert_abc.gateway.enforcer import SessionEnforcer


class ContractWatcher:
    def __init__(self, contract_path: str, interval: float = 0.5) -> None:
        self._path = Path(contract_path)
        self._interval = interval
        self._current: SessionEnforcer | None = None
        self._pending: SessionEnforcer | None = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_monitor(self, enforcer: SessionEnforcer | None) -> None:
        with self._lock:
            self._current = enforcer

    def swap_if_pending(self) -> SessionEnforcer | None:
        with self._lock:
            if self._pending is not None:
                self._current, self._pending = self._pending, None
                return self._current
            return None

    def _watch_loop(self) -> None:
        try:
            last_hash = self._file_hash()
        except OSError:
            last_hash = ""

        while self._running:
            time.sleep(self._interval)
            try:
                current_hash = self._file_hash()
                if current_hash != last_hash:
                    last_hash = current_hash
                    self._try_reload()
            except OSError:
                pass

    def _try_reload(self) -> None:
        try:
            new_enforcer = SessionEnforcer.from_yaml(str(self._path))
            with self._lock:
                self._pending = new_enforcer
        except Exception:  # noqa: BLE001 — malformed contract must not crash the watcher.
            pass

    def _file_hash(self) -> str:
        content = self._path.read_bytes()
        return hashlib.sha256(content).hexdigest()
