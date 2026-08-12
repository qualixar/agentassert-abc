# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""ViolationLog — bounded in-memory log of enforcement decisions.

Ported unchanged from agentassert-typec.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class ViolationLog:
    """Thread-safe, bounded (deque maxlen) log of hard/soft violations."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._log: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, name: str, event_type: str, tool: str, reason: str) -> None:
        """Record a hard (critical) violation."""
        with self._lock:
            self._log.append(
                {
                    "name": name,
                    "event_type": event_type,
                    "tool": tool,
                    "reason": reason,
                    "kind": "hard",
                }
            )

    def record_soft(self, name: str, event_type: str, tool: str, reason: str) -> None:
        """Record a soft (warn-level) violation."""
        with self._lock:
            self._log.append(
                {
                    "name": name,
                    "event_type": event_type,
                    "tool": tool,
                    "reason": reason,
                    "kind": "soft",
                }
            )

    def all_violations(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._log)

    def hard_count(self) -> int:
        """Count of recorded hard-kind violations (used for DriftReport)."""
        with self._lock:
            return sum(1 for entry in self._log if entry.get("kind") == "hard")
