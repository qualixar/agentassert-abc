# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for DriftTracker — patent §5.1.

D(t) = w_c × D_compliance(t) + w_d × D_distributional(t)
D_compliance(t) = 1 - C(t)
D_distributional(t) = JSD(P_t || P_ref)
"""

import math


class TestDriftComputation:
    """D(t) = w_c × (1-C(t)) + w_d × JSD(P_t || P_ref)."""

    def test_perfect_compliance_zero_drift(self) -> None:
        from agentassert_abc.metrics.drift import DriftTracker
        from agentassert_abc.models import DriftConfig

        tracker = DriftTracker(config=DriftConfig())
        # Record perfect turn with reference-matching distribution
        d = tracker.compute_drift(c_total=1.0, action_dist={"rec": 0.5, "check": 0.5})
        # D_compliance = 1 - 1.0 = 0, D_distributional = JSD of same dist = 0
        assert d == 0.0 or d < 0.01  # Near zero

    def test_compliance_component(self) -> None:
        """D_compliance = 1 - C(t). w_c=0.6 by default."""
        from agentassert_abc.metrics.drift import DriftTracker
        from agentassert_abc.models import DriftConfig

        tracker = DriftTracker(config=DriftConfig())
        # Only compliance component (no distributional — same dist as ref)
        d = tracker.compute_drift(c_total=0.5, action_dist=None)
        # D = 0.6 × (1 - 0.5) + 0.4 × 0 = 0.3
        assert abs(d - 0.3) < 0.01

    def test_full_drift_with_jsd(self) -> None:
        """Test with both components."""
        from agentassert_abc.metrics.drift import DriftTracker
        from agentassert_abc.models import DriftConfig

        config = DriftConfig()
        tracker = DriftTracker(config=config)

        # Set reference distribution
        tracker.set_reference({"rec": 0.4, "check": 0.3, "promo": 0.2, "other": 0.1})

        # Shifted distribution (more promos, less recommendations)
        d = tracker.compute_drift(
            c_total=0.8,
            action_dist={"rec": 0.1, "check": 0.1, "promo": 0.7, "other": 0.1},
        )
        # D_compliance = 0.6 × 0.2 = 0.12
        # D_distributional = 0.4 × JSD > 0 (distributions are different)
        assert d > 0.12  # Must be higher than compliance-only

    def test_warning_threshold(self) -> None:
        from agentassert_abc.metrics.drift import DriftTracker
        from agentassert_abc.models import DriftConfig

        tracker = DriftTracker(config=DriftConfig())
        d = tracker.compute_drift(c_total=0.0, action_dist=None)
        # D = 0.6 × 1.0 = 0.6 — exceeds warning (0.3) AND critical (0.6)
        assert d >= 0.6
        assert tracker.is_warning(d)
        assert tracker.is_critical(d)


class TestJSDComputation:
    """Jensen-Shannon Divergence — must use scipy."""

    def test_identical_distributions_zero_jsd(self) -> None:
        from agentassert_abc.metrics.drift import compute_jsd

        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.25, 0.25, 0.25, 0.25]
        assert compute_jsd(p, q) < 1e-10

    def test_different_distributions_positive_jsd(self) -> None:
        from agentassert_abc.metrics.drift import compute_jsd

        p = [0.9, 0.1]
        q = [0.1, 0.9]
        jsd = compute_jsd(p, q)
        assert jsd > 0
        assert jsd <= 1.0  # JSD bounded by log(2) ≈ 0.693 for base e

    def test_jsd_symmetric(self) -> None:
        from agentassert_abc.metrics.drift import compute_jsd

        p = [0.7, 0.3]
        q = [0.3, 0.7]
        assert abs(compute_jsd(p, q) - compute_jsd(q, p)) < 1e-10

    def test_jsd_bounded(self) -> None:
        """JSD is always between 0 and ln(2)."""
        from agentassert_abc.metrics.drift import compute_jsd

        p = [1.0, 0.0]
        q = [0.0, 1.0]
        jsd = compute_jsd(p, q)
        assert 0 <= jsd <= math.log(2) + 1e-10
