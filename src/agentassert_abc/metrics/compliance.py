# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""ComplianceTracker — running C_hard and C_soft averages.

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 3).
"""

from __future__ import annotations


class ComplianceTracker:
    """Tracks running averages of C_hard(t) and C_soft(t) across turns."""

    def __init__(self) -> None:
        self._c_hard_sum: float = 0.0
        self._c_soft_sum: float = 0.0
        self._turn_count: int = 0

    def record(self, c_hard: float, c_soft: float) -> None:
        """Record a turn's compliance scores."""
        self._c_hard_sum += c_hard
        self._c_soft_sum += c_soft
        self._turn_count += 1

    @property
    def mean_c_hard(self) -> float:
        """Running average of C_hard across all turns."""
        if self._turn_count == 0:
            return 1.0
        return self._c_hard_sum / self._turn_count

    @property
    def mean_c_soft(self) -> float:
        """Running average of C_soft across all turns."""
        if self._turn_count == 0:
            return 1.0
        return self._c_soft_sum / self._turn_count

    @property
    def turn_count(self) -> int:
        return self._turn_count
