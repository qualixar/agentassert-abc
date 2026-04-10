# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""C-11: Property-based tests for math functions using Hypothesis.

Tests mathematical invariants that must hold for ALL valid inputs:
- JSD: symmetry, boundedness, identity, non-negativity
- Theta: output range, monotonicity, boundary values
- Composition: output range, degradation, boundary values
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from agentassert_abc.certification.composition import (
    pipeline_composition_bound,
    sequential_composition_bound,
)
from agentassert_abc.metrics.drift import DriftTracker, compute_jsd
from agentassert_abc.metrics.theta import compute_theta
from agentassert_abc.models import DriftConfig

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

prob = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def distributions(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Two non-negative float lists of the same length."""
    n = draw(st.integers(min_value=1, max_value=10))
    p = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    q = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return p, q


# ---------------------------------------------------------------------------
# compute_jsd property tests
# ---------------------------------------------------------------------------


class TestJSDProperties:
    """Property-based tests for Jensen-Shannon Divergence."""

    @given(data=distributions())
    @settings(max_examples=200)
    def test_symmetry(self, data: tuple[list[float], list[float]]) -> None:
        """JSD(p, q) == JSD(q, p) for all valid distributions."""
        p, q = data
        assert abs(compute_jsd(p, q) - compute_jsd(q, p)) < 1e-10

    @given(data=distributions())
    @settings(max_examples=200)
    def test_boundedness(self, data: tuple[list[float], list[float]]) -> None:
        """0 <= JSD(p, q) <= ln(2) for all valid distributions."""
        p, q = data
        jsd = compute_jsd(p, q)
        assert jsd >= -1e-10, f"JSD was negative: {jsd}"
        assert jsd <= math.log(2) + 1e-10, f"JSD exceeded ln(2): {jsd}"

    @given(
        p=st.lists(
            st.floats(
                min_value=0.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=200)
    def test_identity(self, p: list[float]) -> None:
        """JSD(p, p) == 0 for any distribution."""
        jsd = compute_jsd(p, p)
        assert abs(jsd) < 1e-10, f"JSD(p, p) should be 0, got {jsd}"

    @given(data=distributions())
    @settings(max_examples=200)
    def test_non_negativity(
        self, data: tuple[list[float], list[float]]
    ) -> None:
        """JSD(p, q) >= 0 always."""
        p, q = data
        jsd = compute_jsd(p, q)
        assert jsd >= -1e-10, f"JSD was negative: {jsd}"

    def test_zero_sum_distributions(self) -> None:
        """C-04 fix: JSD([0,0], [0,0]) == 0."""
        assert compute_jsd([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_zero_sum_longer(self) -> None:
        """C-04 fix: zero-sum with more elements."""
        assert compute_jsd([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# compute_theta property tests
# ---------------------------------------------------------------------------


class TestThetaProperties:
    """Property-based tests for Reliability Index Theta."""

    @given(
        c_bar=prob,
        d_bar=prob,
        events=st.integers(min_value=0, max_value=1000),
        recovery_rate=prob,
    )
    @settings(max_examples=200)
    def test_output_in_unit_interval(
        self,
        c_bar: float,
        d_bar: float,
        events: int,
        recovery_rate: float,
    ) -> None:
        """Theta output is always in [0, 1] for valid inputs."""
        theta = compute_theta(
            c_bar=c_bar,
            d_bar=d_bar,
            events=events,
            recovery_rate=recovery_rate,
        )
        assert 0.0 - 1e-10 <= theta <= 1.0 + 1e-10, (
            f"Theta out of [0,1]: {theta}"
        )

    @given(
        c_bar_low=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
        c_bar_high=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_monotonicity_c_bar(
        self, c_bar_low: float, c_bar_high: float
    ) -> None:
        """Higher c_bar -> higher theta (all else equal)."""
        fixed_d = 0.2
        fixed_events = 3
        fixed_recovery = 0.5

        theta_low = compute_theta(c_bar_low, fixed_d, fixed_events, fixed_recovery)
        theta_high = compute_theta(c_bar_high, fixed_d, fixed_events, fixed_recovery)
        assert theta_high >= theta_low - 1e-10

    def test_perfect_inputs(self) -> None:
        """Perfect inputs (1.0, 0.0, 0, 1.0) -> theta = 1.0."""
        theta = compute_theta(c_bar=1.0, d_bar=0.0, events=0, recovery_rate=1.0)
        assert abs(theta - 1.0) < 1e-10

    def test_worst_inputs(self) -> None:
        """Worst inputs (0.0, 1.0, 1000, 0.0) -> theta near 0."""
        theta = compute_theta(
            c_bar=0.0, d_bar=1.0, events=1000, recovery_rate=0.0
        )
        assert theta < 0.01

    def test_boundary_high_events(self) -> None:
        """Theta with c_bar=0, d_bar=1.0, events=99999, recovery_rate=0."""
        theta = compute_theta(
            c_bar=0.0, d_bar=1.0, events=99999, recovery_rate=0.0
        )
        # 0.35*0 + 0.25*(1-1) + 0.20*(1/100000) + 0.20*0 ~ 0.000002
        assert theta < 0.001
        assert theta >= 0.0


# ---------------------------------------------------------------------------
# compose_guarantees / sequential_composition_bound property tests
# ---------------------------------------------------------------------------


class TestCompositionProperties:
    """Property-based tests for compositional guarantee bounds."""

    @given(p_a=prob, p_b=prob, p_h=prob)
    @settings(max_examples=200)
    def test_output_in_unit_interval(
        self, p_a: float, p_b: float, p_h: float
    ) -> None:
        """Composition bound is always in [0, 1] for valid inputs."""
        bound = sequential_composition_bound(p_a, p_b, p_h)
        assert -1e-10 <= bound <= 1.0 + 1e-10, (
            f"Bound out of [0,1]: {bound}"
        )

    @given(
        p_a=prob,
        p_b=prob,
        p_h=prob,
        p_c=prob,
        p_h2=prob,
    )
    @settings(max_examples=200)
    def test_degradation_adding_agents(
        self,
        p_a: float,
        p_b: float,
        p_h: float,
        p_c: float,
        p_h2: float,
    ) -> None:
        """Adding more agents decreases (or equals) the bound."""
        two_agent = pipeline_composition_bound([p_a, p_b], [p_h])
        three_agent = pipeline_composition_bound(
            [p_a, p_b, p_c], [p_h, p_h2]
        )
        assert three_agent <= two_agent + 1e-10

    def test_perfect_boundary(self) -> None:
        """p_a=1.0, p_b=1.0, p_h=1.0 -> bound = 1.0."""
        bound = sequential_composition_bound(1.0, 1.0, 1.0)
        assert abs(bound - 1.0) < 1e-10

    def test_zero_probability_zero_bound(self) -> None:
        """Any zero probability -> bound = 0."""
        assert sequential_composition_bound(0.0, 1.0, 1.0) == 0.0
        assert sequential_composition_bound(1.0, 0.0, 1.0) == 0.0
        assert sequential_composition_bound(1.0, 1.0, 0.0) == 0.0

    @given(
        probs=st.lists(prob, min_size=2, max_size=6),
    )
    @settings(max_examples=100)
    def test_pipeline_bound_in_unit_interval(
        self, probs: list[float]
    ) -> None:
        """Pipeline bound is always in [0, 1] for N agents."""
        agent_probs = probs
        # Generate handoff probs of correct length
        handoff_probs = probs[:-1]  # Use first n-1 as handoffs
        bound = pipeline_composition_bound(agent_probs, handoff_probs)
        assert -1e-10 <= bound <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# DriftTracker bounded history test
# ---------------------------------------------------------------------------


class TestDriftTrackerBoundedHistory:
    """Verify DriftTracker respects window size for history."""

    def test_history_bounded_by_window(self) -> None:
        """Create DriftTracker with window=5, add 20 entries, verify len <= 5."""
        config = DriftConfig(window=5)
        tracker = DriftTracker(config=config)

        for i in range(20):
            tracker.compute_drift(c_total=0.5 + i * 0.01, action_dist=None)

        assert len(tracker.history) <= 5
        assert len(tracker.history) == 5
