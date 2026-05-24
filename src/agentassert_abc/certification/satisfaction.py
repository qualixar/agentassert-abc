# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""F2 (p, δ, k)-Satisfaction session-level check — Patent §5.3.

An agent A satisfies contract C with parameters (p, δ, k) iff:
  1. Pr[C_hard(t) = 1 for all t in session] ≥ p
  2. max_t |C_soft(t) - 1| ≤ δ
  3. For every soft violation at t₀, ∃ t₁ ≤ t₀ + k such that C_soft(t₁) = 1

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §5.3, F2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentassert_abc.models import SatisfactionParams


@dataclass(frozen=True)
class TurnRecord:
    """Single turn's compliance data within a session.

    Attributes:
        c_hard: Fraction of hard constraints satisfied (1.0 = all met).
        c_soft: Fraction of soft constraints satisfied (1.0 = all met).
        recovery_event: Whether a recovery was applied this turn.
    """

    c_hard: float = 1.0
    c_soft: float = 1.0
    recovery_event: bool = False


@dataclass(frozen=True)
class SessionLog:
    """Immutable record of a session's per-turn compliance.

    Attributes:
        turns: Ordered list of TurnRecord, one per interaction turn.
    """

    turns: tuple[TurnRecord, ...] = ()

    def __len__(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class SatisfactionVerdict:
    """Result of a (p, δ, k)-satisfaction check on a session log.

    Attributes:
        passed: True iff all three F2 conditions hold.
        p_observed: Observed probability of all-hard-compliance (1.0 or 0.0
            for a single session; aggregated across sessions by SPRT).
        max_soft_deviation: max_t |C_soft(t) - 1| observed in this session.
        max_recovery_window: Maximum turns-to-recovery observed (0 if none).
        failure_reasons: Human-readable reasons for failure (empty on pass).
    """

    passed: bool
    p_observed: float
    max_soft_deviation: float
    max_recovery_window: int
    failure_reasons: tuple[str, ...] = ()


class SatisfactionChecker:
    """Checks whether a session log satisfies F2 (p, δ, k) parameters.

    Usage:
        checker = SatisfactionChecker(params)
        verdict = checker.check_session(session_log)
        if verdict.passed:
            print("Session satisfies contract")
    """

    def __init__(self, params: SatisfactionParams) -> None:
        self._params = params

    def check_session(self, session_log: SessionLog) -> SatisfactionVerdict:
        """Evaluate all three F2 conditions on the session log.

        Args:
            session_log: The session's per-turn compliance record.

        Returns:
            SatisfactionVerdict with pass/fail and detailed metrics.
        """
        turns = session_log.turns
        reasons: list[str] = []

        # --- Condition 1: Pr[C_hard(t) = 1 for all t] >= p ---
        # For a single session: binary — either all turns had c_hard=1 or not.
        all_hard = all(t.c_hard == 1.0 for t in turns)
        p_observed = 1.0 if all_hard else 0.0
        cond1 = p_observed >= self._params.p
        if not cond1:
            hard_violation_turns = sum(1 for t in turns if t.c_hard < 1.0)
            reasons.append(
                f"F2/C1: hard compliance p_observed={p_observed:.2f} "
                f"< required p={self._params.p} "
                f"({hard_violation_turns} turn(s) with hard violation)"
            )

        # --- Condition 2: max_t |C_soft(t) - 1| <= delta ---
        max_dev = max((abs(1.0 - t.c_soft) for t in turns), default=0.0)
        cond2 = max_dev <= self._params.delta + 1e-9
        if not cond2:
            reasons.append(
                f"F2/C2: max soft deviation {max_dev:.4f} "
                f"> allowed delta={self._params.delta}"
            )

        # --- Condition 3: every soft violation recovered within k turns ---
        max_window = 0
        cond3 = True
        k = self._params.k
        for i, t in enumerate(turns):
            if t.c_soft < 1.0:  # soft violation at turn i
                recovered = False
                for j in range(i + 1, min(i + k + 1, len(turns))):
                    if turns[j].c_soft == 1.0:
                        recovered = True
                        max_window = max(max_window, j - i)
                        break
                if not recovered:
                    cond3 = False
                    reasons.append(
                        f"F2/C3: soft violation at turn {i} not recovered "
                        f"within k={k} turns"
                    )

        passed = cond1 and cond2 and cond3
        return SatisfactionVerdict(
            passed=passed,
            p_observed=p_observed,
            max_soft_deviation=max_dev,
            max_recovery_window=max_window,
            failure_reasons=tuple(reasons),
        )
