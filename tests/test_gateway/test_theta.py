# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_theta.py` +
`tests/test_coverage_gaps.py::TestThetaScorerGaps`.

Targets `agentassert_abc.metrics.theta.ThetaScorer` (Phase B) — the class
the gateway actually uses — NOT typec's discarded `ThetaScorer`, which
hardcoded `0.7*c_hard + 0.3*c_soft` (port-delta §C6, silent-break #1).
These tests verify the gateway-facing contract of the shared Phase B class:
`record_compliance`, `record_drift`, `record_violation`, `record_recovery`,
`apply_penalty`, `compute`.
"""

from __future__ import annotations

from agentassert_abc.metrics.theta import ThetaScorer
from agentassert_abc.models import ReliabilityWeights


class TestThetaScorer:
    def test_default_theta_is_one(self) -> None:
        scorer = ThetaScorer()
        assert scorer.compute() == 1.0

    def test_apply_penalty_reduces_score(self) -> None:
        scorer = ThetaScorer()
        scorer.apply_penalty(0.3)
        assert scorer.compute() < 1.0
        assert scorer.compute() == 0.7

    def test_compute_clamps_to_zero(self) -> None:
        scorer = ThetaScorer()
        scorer.apply_penalty(5.0)
        assert scorer.compute() == 0.0

    def test_compute_clamps_to_one(self) -> None:
        scorer = ThetaScorer()
        scorer.record_compliance(1.0, 1.0)
        assert scorer.compute() == 1.0

    def test_violations_reduce_score(self) -> None:
        scorer = ThetaScorer()
        scorer.record_violation()
        scorer.record_violation()
        assert scorer.compute() < 1.0

    def test_compliance_affects_score(self) -> None:
        scorer = ThetaScorer()
        scorer.record_compliance(0.5, 0.5)
        assert scorer.compute() < 1.0

    def test_recovery_success_improves_score(self) -> None:
        scorer = ThetaScorer()
        scorer.record_recovery(True)
        assert scorer.compute() == 1.0

    def test_multiple_penalties(self) -> None:
        scorer = ThetaScorer()
        scorer.apply_penalty(0.1)
        scorer.apply_penalty(0.1)
        scorer.apply_penalty(0.1)
        assert scorer.compute() == 0.7

    def test_compliance_formula_is_abc_not_typec(self) -> None:
        """Pins silent-break #1's fix: (c_hard+c_soft)/2, NOT 0.7*c_hard+0.3*c_soft."""
        scorer = ThetaScorer()
        scorer.record_compliance(c_hard=1.0, c_soft=0.5)
        assert scorer._compliance_scores == [0.75]  # (1.0+0.5)/2, NOT 0.85

    def test_record_drift(self) -> None:
        scorer = ThetaScorer()
        scorer.record_drift(0.3)
        scorer.record_drift(0.5)
        assert scorer.compute() < 1.0

    def test_custom_weights(self) -> None:
        weights = ReliabilityWeights(
            compliance=0.40, drift=0.20, event_freq=0.10, recovery_success=0.30
        )
        scorer = ThetaScorer(weights=weights)
        assert scorer.compute() == 1.0

    def test_record_violation(self) -> None:
        scorer = ThetaScorer()
        scorer.record_violation()
        scorer.record_violation()
        scorer.record_violation()
        assert scorer.compute() < 1.0

    def test_record_recovery_success_and_failure(self) -> None:
        scorer = ThetaScorer()
        scorer.record_recovery(True)
        scorer.record_recovery(False)
        assert scorer.compute() < 1.0

    def test_record_recovery_all_success(self) -> None:
        scorer = ThetaScorer()
        scorer.record_recovery(True)
        scorer.record_recovery(True)
        assert scorer.compute() == 1.0
