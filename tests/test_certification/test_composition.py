# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for Compositional Guarantees — Patent §5.5.

p_{A+B} >= p_A * p_B * p_h

Tests written FIRST (TDD RED phase).
"""

from __future__ import annotations

import pytest

from agentassert_abc.certification.composition import (
    pipeline_composition_bound,
    sequential_composition_bound,
)

# ---------------------------------------------------------------------------
# sequential_composition_bound — two agents
# ---------------------------------------------------------------------------


class TestSequentialCompositionBound:
    """p_{A+B} >= p_A * p_B * p_h."""

    def test_basic_bound(self) -> None:
        result = sequential_composition_bound(p_a=0.95, p_b=0.98, p_h=0.99)
        assert result == pytest.approx(0.95 * 0.98 * 0.99)

    def test_composition_example_from_patent(self) -> None:
        """Patent example: p_A=0.95, p_B=0.98, p_h=0.99 -> >= 0.921."""
        bound = sequential_composition_bound(p_a=0.95, p_b=0.98, p_h=0.99)
        assert bound >= 0.921
        # Exact: 0.95 * 0.98 * 0.99 = 0.921
        assert bound == pytest.approx(0.92169, rel=1e-3)

    def test_perfect_agents(self) -> None:
        """Perfect agents with perfect handoff -> 1.0."""
        assert sequential_composition_bound(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero_handoff(self) -> None:
        """Zero handoff compliance -> zero bound."""
        assert sequential_composition_bound(0.95, 0.98, 0.0) == pytest.approx(0.0)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            sequential_composition_bound(1.5, 0.98, 0.99)
        with pytest.raises(ValueError, match="probability"):
            sequential_composition_bound(0.95, -0.1, 0.99)
        with pytest.raises(ValueError, match="probability"):
            sequential_composition_bound(0.95, 0.98, 1.01)


# ---------------------------------------------------------------------------
# pipeline_composition_bound — N agents chained
# ---------------------------------------------------------------------------


class TestPipelineCompositionBound:
    """Chain of N agents: product of all agent probs * all handoff probs."""

    def test_two_agents_matches_sequential(self) -> None:
        """Pipeline of 2 should match sequential_composition_bound."""
        seq = sequential_composition_bound(0.95, 0.98, 0.99)
        pipe = pipeline_composition_bound([0.95, 0.98], [0.99])
        assert pipe == pytest.approx(seq)

    def test_composition_with_three_agents(self) -> None:
        """Chain of 3: p_A * p_B * p_C * p_h1 * p_h2."""
        p_a, p_b, p_c = 0.95, 0.98, 0.97
        p_h1, p_h2 = 0.99, 0.98
        expected = p_a * p_b * p_c * p_h1 * p_h2
        result = pipeline_composition_bound([p_a, p_b, p_c], [p_h1, p_h2])
        assert result == pytest.approx(expected)

    def test_single_agent_no_handoff(self) -> None:
        """Single agent pipeline = just p_A."""
        result = pipeline_composition_bound([0.95], [])
        assert result == pytest.approx(0.95)

    def test_handoff_count_must_be_agents_minus_one(self) -> None:
        """len(handoff_probs) must equal len(agent_probs) - 1."""
        with pytest.raises(ValueError, match="handoff"):
            pipeline_composition_bound([0.95, 0.98], [0.99, 0.98])

    def test_empty_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="agent"):
            pipeline_composition_bound([], [])

    def test_invalid_probability_in_pipeline(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            pipeline_composition_bound([0.95, 1.5], [0.99])

    def test_degradation_with_many_agents(self) -> None:
        """More agents in chain -> lower bound (monotonic degradation)."""
        bound_2 = pipeline_composition_bound([0.95, 0.95], [0.99])
        bound_5 = pipeline_composition_bound(
            [0.95] * 5, [0.99] * 4
        )
        assert bound_5 < bound_2
