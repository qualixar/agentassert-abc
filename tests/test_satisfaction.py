# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for F2 (p, δ, k)-Satisfaction — LLD-A v2 §3.4/§16 threshold semantics.

These validate the v2 excursion-onset-and-recovery reading (δ is the
acceptable-region boundary 1−δ; recovery target is C_soft ≥ 1−δ within k of
onset). The superseded v1 max-deviation cap / recover-to-exactly-1 reading is
NOT tested because it is not the semantics (LLD-A §16, §17). Reference-vector
names follow LLD-A §18.
"""

import pytest

from agentassert_abc.certification.satisfaction import (
    SatisfactionChecker,
    SessionLog,
    TurnRecord,
)
from agentassert_abc.models import SatisfactionParams


def _params(p: float = 0.95, delta: float = 0.1, k: int = 3) -> SatisfactionParams:
    return SatisfactionParams(p=p, delta=delta, k=k)


def _log(*c_soft: float, c_hard: float = 1.0) -> SessionLog:
    """Build a session log from soft scores (hard defaults to perfect)."""
    return SessionLog(turns=tuple(TurnRecord(c_hard=c_hard, c_soft=s) for s in c_soft))


def _passed(log: SessionLog, *, p: float = 0.0, delta: float = 0.1, k: int = 3) -> bool:
    return SatisfactionChecker(_params(p=p, delta=delta, k=k)).check_session(log).passed


class TestHardCondition:
    def test_all_perfect_passes(self) -> None:
        v = SatisfactionChecker(_params()).check_session(_log(1.0, 1.0, 1.0))
        assert v.passed is True
        assert v.passed_optimistic is True
        assert v.p_observed == 1.0
        assert v.soft_breaches == 0 and v.soft_pending == 0

    def test_hard_single_violation_fails_component(self) -> None:  # §18-1
        log = SessionLog(turns=(
            TurnRecord(c_hard=1.0, c_soft=1.0),
            TurnRecord(c_hard=0.5, c_soft=1.0),
            TurnRecord(c_hard=1.0, c_soft=1.0),
        ))
        v = SatisfactionChecker(_params(p=0.95)).check_session(log)
        assert v.passed is False
        assert v.p_observed == 0.0
        assert any("C1" in r for r in v.failure_reasons)

    def test_empty_constraint_sets_score_one(self) -> None:  # §18-3
        # Empty soft/hard constraints score 1.0 → always acceptable.
        v = SatisfactionChecker(_params()).check_session(_log(1.0))
        assert v.passed is True


class TestSoftExcursionV2:
    def test_soft_recovery_at_exact_deadline_passes(self) -> None:  # §18-4
        # δ=0.2 → threshold 0.8; onset at t0=0, recovery at t0+k=2.
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=2)).check_session(
            _log(0.5, 0.5, 0.8)
        )
        assert v.passed is True
        assert v.soft_breaches == 0
        assert v.max_recovery_window == 2

    def test_soft_recovery_after_deadline_fails(self) -> None:  # §18-5
        # Recovery at t0+k+1 (=3) is outside the window [0, 2].
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=2)).check_session(
            _log(0.5, 0.5, 0.5, 0.8)
        )
        assert v.passed is False
        assert v.passed_optimistic is False
        assert v.soft_breaches == 1
        assert any("C2" in r for r in v.failure_reasons)

    def test_overlapping_soft_obligations_share_recovery(self) -> None:  # §18-6
        # One uninterrupted below-episode = ONE onset discharged by one recovery
        # (v1 would have counted three separate violations).
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=3)).check_session(
            _log(0.5, 0.6, 0.7, 0.85)
        )
        assert v.passed is True
        assert v.soft_breaches == 0
        assert v.max_recovery_window == 3

    def test_threshold_boundary_is_acceptable(self) -> None:
        # C_soft exactly at 1−δ is inside the acceptable region → no onset.
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=1)).check_session(
            _log(1.0, 0.8, 1.0)
        )
        assert v.passed is True
        assert v.soft_breaches == 0

    def test_large_dip_recovered_passes_v2_no_deviation_cap(self) -> None:
        # Deviation 0.8 ≫ δ=0.1, but it recovers within k → v2 PASSES.
        # (Under the superseded v1 max-deviation cap this would have failed.)
        v = SatisfactionChecker(_params(p=0.0, delta=0.1, k=2)).check_session(
            _log(1.0, 0.2, 1.0)
        )
        assert v.passed is True
        assert v.max_soft_deviation == pytest.approx(0.8)  # informational only


class TestPendingBounds:
    """LLD-A §15/§18 — excursions running past the horizon are undecided."""

    def _pending_log(self) -> SessionLog:
        # Onset at the last turn; window [2, 2+3] runs past the horizon.
        return _log(1.0, 1.0, 0.5)

    def test_late_soft_obligation_is_pending_not_failed(self) -> None:  # §18-7
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=3)).check_session(
            self._pending_log()
        )
        assert v.soft_pending == 1
        assert v.soft_breaches == 0

    def test_pending_as_failure_is_lower_bound(self) -> None:  # §18-17
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=3)).check_session(
            self._pending_log()
        )
        assert v.passed is False  # pessimistic: pending counts as breach

    def test_pending_as_success_is_upper_bound(self) -> None:  # §18-18
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=3)).check_session(
            self._pending_log()
        )
        assert v.passed_optimistic is True  # optimistic: pending assumed recovered


class TestMonotonicity:
    def test_delta_monotonicity(self) -> None:  # §18-12
        # Same session: breaches at small δ, satisfied at large δ.
        log = _log(1.0, 0.75, 0.75, 0.75, 0.75)
        assert _passed(log, delta=0.1, k=2) is False
        assert _passed(log, delta=0.3, k=2) is True

    def test_recovery_window_monotonicity(self) -> None:  # §18-13
        # Same session: breaches at small k, satisfied at large k.
        log = _log(1.0, 0.5, 0.5, 0.5, 1.0)
        assert _passed(log, delta=0.2, k=1) is False
        assert _passed(log, delta=0.2, k=3) is True


class TestDegenerate:
    def test_soft_degenerates_to_hard_at_delta_zero_k_zero(self) -> None:  # §18-14
        # δ=0 → threshold 1.0; k=0 → must be acceptable at onset turn itself.
        # Effect: soft satisfied iff C_soft == 1 everywhere (hard-like).
        ok = SatisfactionChecker(_params(p=0.0, delta=0.0, k=0)).check_session(
            _log(1.0, 1.0, 1.0)
        )
        assert ok.passed is True
        bad = SatisfactionChecker(_params(p=0.0, delta=0.0, k=0)).check_session(
            _log(1.0, 0.99, 1.0)
        )
        assert bad.passed is False
        assert bad.soft_breaches == 1


class TestEdgeCases:
    def test_empty_session_passes_vacuously(self) -> None:
        v = SatisfactionChecker(_params()).check_session(SessionLog(turns=()))
        assert v.passed is True
        assert v.p_observed == 1.0

    def test_p_boundary_inclusive(self) -> None:
        v = SatisfactionChecker(_params(p=1.0, delta=0.1, k=3)).check_session(_log(1.0))
        assert v.passed is True

    def test_multiple_episodes_each_checked(self) -> None:
        # Two separate below-episodes: first recovers, second breaches.
        log = _log(1.0, 0.5, 1.0, 0.5, 0.5)  # onset t1 (recovers t2); onset t3 (no recovery)
        v = SatisfactionChecker(_params(p=0.0, delta=0.2, k=1)).check_session(log)
        assert v.soft_breaches == 1
        assert v.passed is False
