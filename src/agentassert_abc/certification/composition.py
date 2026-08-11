# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Extended compositional guarantees with C1-C5 condition checking.

Extends the existing `compose_guarantees(p_a, p_b, p_h)` with explicit
condition checkers for the five composition conditions (C1-C5) from the
patent, plus an aggregator `compose_guarantees_with_conditions`.

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §5.5, F5.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentassert_abc.models import ContractSpec


class ConditionVerdict(StrEnum):
    """Result of checking a single composition condition."""

    HOLDS = "holds"        # Condition satisfied
    FAILS = "fails"        # Condition violated
    INCONCLUSIVE = "inconclusive"  # Cannot decide statically


@dataclass(frozen=True)
class ConditionResult:
    """Result of a single composition condition check."""

    verdict: ConditionVerdict
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Truthy iff verdict == HOLDS."""
        return self.verdict == ConditionVerdict.HOLDS


@dataclass(frozen=True)
class CompositionResult:
    """Aggregated result of checking all C1-C5 conditions + bound."""

    bound: float                              # p_a * p_b * p_h
    # Ledger 3d: declared as Mapping so callers accept MappingProxyType; never mutable
    conditions: Mapping[str, ConditionResult] = field(default_factory=dict)
    all_hold: bool = False
    safety_label: str = ""

    def __post_init__(self) -> None:
        # Ledger 3d: wrap conditions in MappingProxyType so mutations (which would
        # silently make all_hold/safety_label stale) are caught at the call site.
        object.__setattr__(
            self, 'conditions', types.MappingProxyType(dict(self.conditions))
        )
        # Auto-calculate derived fields
        object.__setattr__(self, 'all_hold', all(c for c in self.conditions.values()))
        object.__setattr__(
            self,
            'safety_label',
            "verified" if self.all_hold
            else "violated"
            if any(c.verdict == ConditionVerdict.FAILS for c in self.conditions.values())
            else "pending",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_probability(name: str, value: float) -> None:
    """Validate that a value is a valid probability in [0, 1].

    Args:
        name: Parameter name for error messages
        value: Value to validate

    Raises:
        ValueError: If value is not in [0, 1]
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid probability {name}: {value}. Must be in [0, 1]")


# ---------------------------------------------------------------------------
# Composition bounds (F5)
# ---------------------------------------------------------------------------


def sequential_composition_bound(p_a: float, p_b: float, p_h: float) -> float:
    """Compute the lower bound for two sequential agents.

    p_{A+B} >= p_A * p_B * p_h

    Args:
        p_a: Agent A's hard constraint satisfaction probability
        p_b: Agent B's hard constraint satisfaction probability
        p_h: Handoff compliance rate between A and B

    Returns:
        Lower bound on the composition's hard constraint satisfaction probability
    """
    _validate_probability("p_a", p_a)
    _validate_probability("p_b", p_b)
    _validate_probability("p_h", p_h)
    return p_a * p_b * p_h


def pipeline_composition_bound(
    agent_probs: list[float],
    handoff_probs: list[float],
) -> float:
    """Compute the lower bound for an N-agent pipeline.

    p_{pipeline} >= prod(agent_probs) * prod(handoff_probs)

    Args:
        agent_probs: List of agent hard constraint satisfaction probabilities
        handoff_probs: List of handoff compliance rates between consecutive agents

    Returns:
        Lower bound on the pipeline's hard constraint satisfaction probability

    Raises:
        ValueError: If agent_probs is empty or handoff_probs length mismatch
    """
    if not agent_probs:
        msg = "agent_probs must contain at least one agent"
        raise ValueError(msg)

    expected_handoffs = len(agent_probs) - 1
    if len(handoff_probs) != expected_handoffs:
        msg = (
            f"Expected {expected_handoffs} handoff probabilities for {len(agent_probs)} agents, "
            f"got {len(handoff_probs)}"
        )
        raise ValueError(msg)

    for i, p in enumerate(agent_probs):
        _validate_probability(f"agent_probs[{i}]", p)
    for i, p in enumerate(handoff_probs):
        _validate_probability(f"handoff_probs[{i}]", p)

    return math.prod(agent_probs) * math.prod(handoff_probs)


# Backward-compatible alias: compose_guarantees IS sequential_composition_bound
compose_guarantees = sequential_composition_bound


# ---------------------------------------------------------------------------
# C1 — Type Compatibility (STATIC)
# ---------------------------------------------------------------------------


def check_c1_type_compatibility(
    contract_a: ContractSpec,
    contract_b: ContractSpec,
) -> ConditionResult:
    """C1: Type Compatibility — output of A compatible with input of B.

    Compares B's precondition fields against A's output schema (hard+soft invariants).
    HOLDS iff every field B expects as input is produced by A.
    INCONCLUSIVE if B has no preconditions.
    """
    # Collect what B expects as input (precondition fields)
    b_input_fields: set[str] = set()
    if contract_b.preconditions:
        for pre in contract_b.preconditions:
            b_input_fields.add(pre.check.field)

    # If B has no preconditions, nothing to check
    if not b_input_fields:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason="B has no preconditions to check",
            evidence={"unmatched_preconditions": []},
        )

    # Collect what A produces as output (invariant fields)
    a_output_fields: set[str] = set()
    if contract_a.invariants:
        for hc in contract_a.invariants.hard:
            a_output_fields.add(hc.check.field)
        for sc in contract_a.invariants.soft:
            a_output_fields.add(sc.check.field)

    unmatched = sorted(b_input_fields - a_output_fields)
    if not unmatched:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason="All B-precondition fields are produced by A",
            evidence={"unmatched_preconditions": []},
        )
    return ConditionResult(
        verdict=ConditionVerdict.FAILS,
        reason=f"B expects fields not produced by A: {unmatched}",
        evidence={"unmatched_preconditions": unmatched},
    )


# ---------------------------------------------------------------------------
# C2 — Invariant Preservation (STATIC)
# ---------------------------------------------------------------------------


def check_c2_invariant_preservation(
    contract_a: ContractSpec,
    contract_b: ContractSpec,
) -> ConditionResult:
    """C2: Invariant Preservation — A's postconditions include B's preconditions.

    Checks whether B's precondition fields are covered by A's HARD constraints only.
    HOLDS if every B-precondition field matches an A hard constraint field.
    FAILS if a B-precondition field is only in A's soft constraints.
    INCONCLUSIVE if B has no preconditions.
    """
    # Collect what B expects as input (precondition fields)
    b_input_fields: set[str] = set()
    if contract_b.preconditions:
        for pre in contract_b.preconditions:
            b_input_fields.add(pre.check.field)

    if not b_input_fields:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason="B has no preconditions to check",
            evidence={"uncovered_preconditions": []},
        )

    # Collect A's hard constraint fields only
    a_hard_fields: set[str] = set()
    a_soft_fields: set[str] = set()
    if contract_a.invariants:
        for hc in contract_a.invariants.hard:
            a_hard_fields.add(hc.check.field)
        for sc in contract_a.invariants.soft:
            a_soft_fields.add(sc.check.field)

    # Check: every B precondition must be in A's hard constraints
    uncovered = sorted(b_input_fields - a_hard_fields)
    if not uncovered:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason="All B-precondition fields are covered by A hard constraints",
            evidence={"uncovered_preconditions": []},
        )
    return ConditionResult(
        verdict=ConditionVerdict.FAILS,
        reason=f"B expects fields not in A hard constraints: {uncovered}",
        evidence={"uncovered_preconditions": uncovered},
    )


# ---------------------------------------------------------------------------
# C3 — Monotone Drift (DYNAMIC)
# ---------------------------------------------------------------------------

_EPSILON = 0.01  # tolerance for floating-point comparison


def check_c3_monotone_drift(
    drift_a: list[float] | None,
    drift_b: list[float] | None,
    drift_combined: list[float] | None,
) -> ConditionResult:
    """C3: Monotone Drift — D_{A∘B}(t) ≤ max(D_A(t), D_B(t)) + ε.

    Requires three drift sequences. HOLDS if the combined drift never exceeds
    the component-wise max by more than tolerance ε (default 0.01).
    INCONCLUSIVE if any input is None or sequences are empty.
    """
    if drift_a is None or drift_b is None or drift_combined is None:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason="Missing one or more drift sequences",
            evidence={"failing_timesteps": 0},
        )

    if not drift_a and not drift_b and not drift_combined:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason="Empty drift sequences",
            evidence={"failing_timesteps": 0},
        )

    if not (len(drift_a) == len(drift_b) == len(drift_combined)):
        return ConditionResult(
            verdict=ConditionVerdict.FAILS,
            reason="Drift sequences must have equal length",
            evidence={"failing_timesteps": 0},
        )

    failing = 0
    total = len(drift_a)

    # Ledger 2g: sequences shorter than 20 are too short for meaningful drift analysis;
    # max(1, int(total*0.05))=1 makes even 1-element sequences vacuously HOLD.
    if total < 20:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason=f"Drift sequence too short for analysis ({total} < 20 minimum)",
            evidence={"failing_timesteps": 0},
        )

    for da, db, dc in zip(drift_a, drift_b, drift_combined, strict=True):
        if dc > max(da, db) + _EPSILON:
            failing += 1

    # Allow up to 5% of timesteps to fail (grace threshold); no min-1 clamp
    grace = int(total * 0.05)
    if failing <= grace:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason=f"Combined drift within tolerance at {total - failing}/{total} timesteps",
            evidence={"failing_timesteps": failing},
        )

    return ConditionResult(
        verdict=ConditionVerdict.FAILS,
        reason=f"Combined drift exceeded max(component drifts) at {failing} timestep(s)",
        evidence={"failing_timesteps": failing},
    )


# ---------------------------------------------------------------------------
# C4 — Recovery Propagation (DYNAMIC)
# ---------------------------------------------------------------------------


def check_c4_recovery_propagation(
    pipeline_event_log: list[dict[str, Any]] | None,
) -> ConditionResult:
    """C4: Recovery Propagation — upstream recovery visible to downstream monitor.

    Requires pipeline event log with recovery events. HOLDS if every upstream
    recovery attempt is observed by the downstream monitor within 1 turn.
    INCONCLUSIVE if no event log provided or log is empty.
    """
    if pipeline_event_log is None or len(pipeline_event_log) == 0:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason="No pipeline event log provided",
            evidence={"unpropagated_recoveries": []},
        )

    # Parse recovery events
    upstream_recoveries: dict[int, bool] = {}  # turn -> has downstream observation
    for event in pipeline_event_log:
        agent = event.get("agent", "")
        evt_type = event.get("event", "")
        turn = event.get("turn", -1)

        if agent == "agent_a" and evt_type == "recovery_attempt":
            upstream_recoveries[turn] = False  # Mark as needing downstream observation
        elif agent == "agent_b" and evt_type == "recovery_success":
            # Ledger 2a: downstream recovery must FOLLOW upstream attempt (causal).
            # abs() counted acausal recoveries (B success BEFORE A attempt);
            # correct: B success must be at recovery_turn or recovery_turn+1.
            for recovery_turn in list(upstream_recoveries.keys()):
                if not upstream_recoveries[recovery_turn] and 0 <= turn - recovery_turn <= 1:
                    upstream_recoveries[recovery_turn] = True

    # Find unpropagated recoveries
    unpropagated = [turn for turn, observed in upstream_recoveries.items() if not observed]

    if not upstream_recoveries:
        # No upstream recovery events — vacuously HOLDS
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason="No upstream recovery events found",
            evidence={"unpropagated_recoveries": []},
        )

    if not unpropagated:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason="All upstream recoveries propagated to downstream",
            evidence={"unpropagated_recoveries": []},
        )

    return ConditionResult(
        verdict=ConditionVerdict.FAILS,
        reason=f"Unpropagated recoveries at turns: {unpropagated}",
        evidence={"unpropagated_recoveries": unpropagated},
    )


# ---------------------------------------------------------------------------
# C5 — Independence (DYNAMIC)
# ---------------------------------------------------------------------------


def check_c5_independence(
    pipeline_event_log: list[dict[str, Any]] | None,
) -> ConditionResult:
    """C5: Independence — A's violations don't cause B's violations.

    Performs a statistical independence check on violation events.
    HOLDS if P(B violates | A violated at t-1) ≈ P(B violates) within ±0.1.
    INCONCLUSIVE if log is None or has fewer than 10 events.
    """
    if pipeline_event_log is None or len(pipeline_event_log) < 10:
        return ConditionResult(
            verdict=ConditionVerdict.INCONCLUSIVE,
            reason=(
                "Event log too short for statistical test: "
                f"{len(pipeline_event_log) if pipeline_event_log else 0} < 10"
            ),
            evidence={"p_b_violates": 0.0, "p_b_given_a_prev": 0.0},
        )

    # Parse violation events
    a_violations: set[int] = set()
    b_violations: set[int] = set()
    max_turn = 0

    for event in pipeline_event_log:
        agent = event.get("agent", "")
        evt_type = event.get("event", "")
        turn = event.get("turn", 0)
        max_turn = max(max_turn, turn)

        if agent == "agent_a" and evt_type == "hard_violation":
            a_violations.add(turn)
        elif agent == "agent_b" and evt_type == "hard_violation":
            b_violations.add(turn)

    if not b_violations:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason="No B violations observed",
            evidence={"p_b_violates": 0.0, "p_b_given_a_prev": 0.0},
        )

    # Ledger 2b: turns are 0-indexed so the count is max_turn+1, not max_turn.
    # Using max_turn as denominator caused p_b=0 when all events on turn 0.
    p_b = len(b_violations) / (max_turn + 1)

    # P(B | A_prev) = P(B violates at t | A violated at the previous turn t-1).
    # Candidate turns are those whose PRECEDING turn had an A violation: {a+1}.
    # (Previously {a-1}, which tested whether B *leads* A — the reverse of A->B causation.)
    a_prev = {t + 1 for t in a_violations if t + 1 <= max_turn}
    if not a_prev:
        p_b_given_a = 0.0
    else:
        b_in_a_prev = len(b_violations & a_prev)
        p_b_given_a = b_in_a_prev / len(a_prev)

    diff = abs(p_b - p_b_given_a)

    if diff <= 0.1:
        return ConditionResult(
            verdict=ConditionVerdict.HOLDS,
            reason=(
                "B violations independent of A "
                f"(P(B)={p_b:.3f}, P(B|A_prev)={p_b_given_a:.3f}, diff={diff:.3f})"
            ),
            evidence={"p_b_violates": p_b, "p_b_given_a_prev": p_b_given_a},
        )

    return ConditionResult(
        verdict=ConditionVerdict.FAILS,
        reason=(
            "B violations correlated with A "
            f"(P(B)={p_b:.3f}, P(B|A_prev)={p_b_given_a:.3f}, diff={diff:.3f})"
        ),
        evidence={"p_b_violates": p_b, "p_b_given_a_prev": p_b_given_a},
    )


# ---------------------------------------------------------------------------
# Aggregator — condition-aware composition
# ---------------------------------------------------------------------------


def compose_guarantees_with_conditions(
    contract_a: ContractSpec,
    contract_b: ContractSpec,
    p_a: float,
    p_b: float,
    p_h: float,
    drift_a: list[float] | None = None,
    drift_b: list[float] | None = None,
    drift_combined: list[float] | None = None,
    pipeline_event_log: list[dict[str, Any]] | None = None,
) -> CompositionResult:
    """Check all five composition conditions (C1-C5) and return aggregated result.

    Combines the multiplicative bound `p_a * p_b * p_h` with explicit verdicts
    on each of C1 (Type Compatibility), C2 (Invariant Preservation),
    C3 (Monotone Drift), C4 (Recovery Propagation), and C5 (Independence).

    Args:
        contract_a: First agent's contract.
        contract_b: Second agent's contract.
        p_a: Agent A's hard constraint satisfaction probability.
        p_b: Agent B's hard constraint satisfaction probability.
        p_h: Handoff compliance rate between A and B.
        drift_a: Observed drift sequence for agent A (optional).
        drift_b: Observed drift sequence for agent B (optional).
        drift_combined: Observed drift sequence for the A→B pipeline (optional).
        pipeline_event_log: Timeline of pipeline-level events (optional).

    Returns:
        CompositionResult with bound, per-condition verdicts, and safety label.
    """
    bound = compose_guarantees(p_a, p_b, p_h)

    c1 = check_c1_type_compatibility(contract_a, contract_b)
    c2 = check_c2_invariant_preservation(contract_a, contract_b)
    c3 = check_c3_monotone_drift(drift_a, drift_b, drift_combined)
    c4 = check_c4_recovery_propagation(pipeline_event_log)
    c5 = check_c5_independence(pipeline_event_log)

    return CompositionResult(
        bound=bound,
        conditions={
            "C1_type_compatibility": c1,
            "C2_invariant_preservation": c2,
            "C3_monotone_drift": c3,
            "C4_recovery_propagation": c4,
            "C5_independence": c5,
        },
        all_hold=False,  # will be fixed in __post_init__
        safety_label="",  # will be fixed in __post_init__
    )