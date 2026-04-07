# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for Reliability Index Θ — patent §5.7.

Θ = 0.35 × C̄ + 0.25 × (1 - D̄) + 0.20 × (1/(1+E)) + 0.20 × S

Where:
- C̄ = mean compliance score
- D̄ = mean drift score
- E = total violation events
- S = recovery success rate
"""


class TestThetaComputation:
    """Exact Θ formula from patent §5.7."""

    def test_perfect_session(self) -> None:
        """Perfect: C̄=1, D̄=0, E=0, S=1 → Θ=1.0."""
        from agentassert_abc.metrics.theta import compute_theta

        theta = compute_theta(c_bar=1.0, d_bar=0.0, events=0, recovery_rate=1.0)
        assert abs(theta - 1.0) < 0.01

    def test_patent_example(self) -> None:
        """Patent §6 Step 9: Θ = 0.35×0.97 + 0.25×(1-0.05) + 0.20×(1/2) + 0.20×1.0 = 0.877."""
        from agentassert_abc.metrics.theta import compute_theta

        theta = compute_theta(c_bar=0.97, d_bar=0.05, events=1, recovery_rate=1.0)
        expected = 0.35 * 0.97 + 0.25 * (1 - 0.05) + 0.20 * (1 / 2) + 0.20 * 1.0
        assert abs(theta - expected) < 0.001
        assert abs(theta - 0.877) < 0.01

    def test_zero_events(self) -> None:
        """1/(1+0) = 1.0 — no violations."""
        from agentassert_abc.metrics.theta import compute_theta

        theta = compute_theta(c_bar=0.9, d_bar=0.1, events=0, recovery_rate=0.0)
        # 0.35×0.9 + 0.25×0.9 + 0.20×1.0 + 0.20×0.0 = 0.315+0.225+0.2+0 = 0.74
        expected = 0.35 * 0.9 + 0.25 * 0.9 + 0.20 * 1.0 + 0.20 * 0.0
        assert abs(theta - expected) < 0.001

    def test_many_events_penalized(self) -> None:
        """More events → lower Θ. 1/(1+10) = 0.091."""
        from agentassert_abc.metrics.theta import compute_theta

        theta = compute_theta(c_bar=1.0, d_bar=0.0, events=10, recovery_rate=1.0)
        # 0.35×1 + 0.25×1 + 0.20×0.091 + 0.20×1 = 0.35+0.25+0.018+0.2 = 0.818
        assert theta < 0.85

    def test_deployment_threshold(self) -> None:
        """Θ >= 0.90 = deployment ready (patent default)."""
        from agentassert_abc.metrics.theta import compute_theta

        theta = compute_theta(c_bar=1.0, d_bar=0.0, events=0, recovery_rate=1.0)
        assert theta >= 0.90

    def test_custom_weights(self) -> None:
        from agentassert_abc.metrics.theta import compute_theta
        from agentassert_abc.models import ReliabilityWeights

        weights = ReliabilityWeights(
            compliance=0.5, drift=0.2, event_freq=0.2, recovery_success=0.1
        )
        theta = compute_theta(
            c_bar=1.0, d_bar=0.0, events=0, recovery_rate=1.0, weights=weights
        )
        assert abs(theta - 1.0) < 0.01
