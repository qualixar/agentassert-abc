# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Compositional Guarantees — Patent §5.5.

For sequential agent composition:
    p_{A+B} >= p_A * p_B * p_h

Where:
    p_A = Agent A's hard constraint satisfaction probability
    p_B = Agent B's hard constraint satisfaction probability
    p_h = Handoff compliance rate

For N-agent pipelines, the bound generalizes to:
    p_{pipeline} >= prod(p_i) * prod(p_h_j)

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §5.5
"""

from __future__ import annotations

import math


def _validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability in [0, 1].

    Args:
        value: The value to validate.
        name: Name for error messages.

    Raises:
        ValueError: If value is not in [0, 1].
    """
    if not (0.0 <= value <= 1.0):
        msg = f"{name} must be a valid probability in [0, 1], got {value}"
        raise ValueError(msg)


def sequential_composition_bound(
    p_a: float, p_b: float, p_h: float
) -> float:
    """Compute the lower bound for two sequential agents.

    p_{A+B} >= p_A * p_B * p_h

    Args:
        p_a: Agent A's hard constraint satisfaction probability.
        p_b: Agent B's hard constraint satisfaction probability.
        p_h: Handoff compliance rate between A and B.

    Returns:
        Lower bound on the composed system's satisfaction probability.

    Raises:
        ValueError: If any probability is outside [0, 1].
    """
    _validate_probability(p_a, "p_a probability")
    _validate_probability(p_b, "p_b probability")
    _validate_probability(p_h, "p_h probability")

    return p_a * p_b * p_h


def pipeline_composition_bound(
    agent_probs: list[float],
    handoff_probs: list[float],
) -> float:
    """Compute the lower bound for an N-agent pipeline.

    p_{pipeline} >= prod(agent_probs) * prod(handoff_probs)

    Args:
        agent_probs: List of per-agent satisfaction probabilities.
            Must have at least one element.
        handoff_probs: List of handoff compliance rates.
            Must have exactly len(agent_probs) - 1 elements.

    Returns:
        Lower bound on the pipeline's satisfaction probability.

    Raises:
        ValueError: If lists are empty, mismatched, or contain
            invalid probabilities.
    """
    if not agent_probs:
        msg = "agent_probs must contain at least one agent"
        raise ValueError(msg)

    expected_handoffs = len(agent_probs) - 1
    if len(handoff_probs) != expected_handoffs:
        msg = (
            f"handoff_probs must have exactly {expected_handoffs} "
            f"elements (agents - 1), got {len(handoff_probs)}"
        )
        raise ValueError(msg)

    for i, p in enumerate(agent_probs):
        _validate_probability(p, f"agent_probs[{i}] probability")

    for i, p in enumerate(handoff_probs):
        _validate_probability(p, f"handoff_probs[{i}] probability")

    return math.prod(agent_probs) * math.prod(handoff_probs)


# Alias for backward compatibility and readability in examples
compose_guarantees = sequential_composition_bound
