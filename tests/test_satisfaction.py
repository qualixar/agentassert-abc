# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for F2 (p, δ, k)-Satisfaction session-level check."""

import pytest

from agentassert_abc.certification.satisfaction import (
    SatisfactionChecker,
    SessionLog,
    TurnRecord,
)
from agentassert_abc.models import SatisfactionParams


def _make_params(p: float = 0.95, delta: float = 0.1, k: int = 3) -> SatisfactionParams:
    return SatisfactionParams(p=p, delta=delta, k=k)


class TestSatisfactionPass:
    """Cases where the session satisfies (p, δ, k)."""

    def test_all_perfect(self) -> None:
        params = _make_params()
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=1.0),
            TurnRecord(c_hard=1.0, c_soft=1.0),
            TurnRecord(c_hard=1.0, c_soft=1.0),
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True
        assert verdict.p_observed == 1.0
        assert verdict.max_soft_deviation == 0.0
        assert len(verdict.failure_reasons) == 0

    def test_soft_within_delta_recovered_within_k(self) -> None:
        """Soft deviation within δ, recovered within k turns."""
        params = _make_params(p=0.0, delta=0.2, k=3)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=1.0),
            TurnRecord(c_hard=1.0, c_soft=0.85),  # deviation = 0.15 < 0.2
            TurnRecord(c_hard=1.0, c_soft=1.0),   # recovered at t=2, window=1 ≤ k=3
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True
        assert verdict.max_soft_deviation == pytest.approx(0.15)
        assert verdict.max_recovery_window == 1

    def test_exactly_delta_boundary(self) -> None:
        """Deviation exactly at delta boundary should pass (≤ is inclusive)."""
        params = _make_params(p=0.0, delta=0.15, k=3)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.85),  # deviation = 0.15 == delta
            TurnRecord(c_hard=1.0, c_soft=1.0),   # recovered at t=1, window=1 ≤ k=3
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True

    def test_recovery_exactly_at_k(self) -> None:
        """Recovery at exactly k turns later should pass."""
        params = _make_params(p=0.0, delta=0.5, k=2)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.5),   # violation at t=0
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=1, still violated
            TurnRecord(c_hard=1.0, c_soft=1.0),    # t=2, recovered (window=2 == k)
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True
        assert verdict.max_recovery_window == 2

    def test_multiple_soft_violations_all_recovered(self) -> None:
        params = _make_params(p=0.0, delta=0.3, k=2)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.8),
            TurnRecord(c_hard=1.0, c_soft=1.0),    # recovered
            TurnRecord(c_hard=1.0, c_soft=0.75),
            TurnRecord(c_hard=1.0, c_soft=1.0),    # recovered
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True

    def test_single_turn_all_good(self) -> None:
        params = _make_params()
        log = SessionLog(turns=(TurnRecord(c_hard=1.0, c_soft=1.0),))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True


class TestSatisfactionFail:
    """Cases where the session fails (p, δ, k)."""

    def test_p_violation(self) -> None:
        """One hard violation → p_observed=0 < p=0.95."""
        params = _make_params(p=0.95, delta=0.5, k=5)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=1.0),
            TurnRecord(c_hard=0.5, c_soft=1.0),   # hard violation
            TurnRecord(c_hard=1.0, c_soft=1.0),
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False
        assert verdict.p_observed == 0.0
        assert any("C1" in r for r in verdict.failure_reasons)

    def test_delta_violation(self) -> None:
        """Soft deviation exceeds delta."""
        params = _make_params(p=0.0, delta=0.1, k=5)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.7),   # deviation = 0.3 > 0.1
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False
        assert verdict.max_soft_deviation == pytest.approx(0.3)
        assert any("C2" in r for r in verdict.failure_reasons)

    def test_k_violation(self) -> None:
        """Soft violation not recovered within k turns."""
        params = _make_params(p=0.0, delta=0.5, k=2)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.5),   # violation at t=0
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=1
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=2
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=3 — recovery at t=4, window=4 > k=2
            TurnRecord(c_hard=1.0, c_soft=1.0),    # recovered
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False
        assert any("C3" in r for r in verdict.failure_reasons)

    def test_recovery_one_turn_past_k(self) -> None:
        """Recovery at k+1 turns later should fail."""
        params = _make_params(p=0.0, delta=0.5, k=2)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=0.5),   # violation at t=0
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=1
            TurnRecord(c_hard=1.0, c_soft=0.5),   # t=2
            TurnRecord(c_hard=1.0, c_soft=1.0),    # t=3, window=3 > k=2
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False

    def test_multiple_failures(self) -> None:
        """Session fails multiple conditions simultaneously."""
        params = _make_params(p=0.95, delta=0.05, k=1)
        log = SessionLog(turns=(
            TurnRecord(c_hard=0.5, c_soft=0.5),   # hard + soft violation
            TurnRecord(c_hard=1.0, c_soft=0.5),   # still soft violated
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False
        # Should have reasons for C1 and C2 at minimum
        assert len(verdict.failure_reasons) >= 2


class TestSatisfactionEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_session_passes_vacuously(self) -> None:
        """Empty session: all conditions hold vacuously."""
        params = _make_params()
        log = SessionLog(turns=())
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True
        assert verdict.p_observed == 1.0  # all() of empty = True
        assert verdict.max_soft_deviation == 0.0

    def test_p_at_boundary(self) -> None:
        """p_observed=1.0 meets p=1.0 exactly."""
        params = _make_params(p=1.0, delta=0.1, k=3)
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=1.0),
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is True

    def test_p_just_below(self) -> None:
        """p_observed=0.0 fails p=0.01."""
        params = _make_params(p=0.01, delta=0.5, k=5)
        log = SessionLog(turns=(
            TurnRecord(c_hard=0.9, c_soft=1.0),
        ))
        verdict = SatisfactionChecker(params).check_session(log)
        assert verdict.passed is False
