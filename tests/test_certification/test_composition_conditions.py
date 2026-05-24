# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for C1-C5 Composition Condition Checkers — G4.

Tests for the five composition condition checkers and the aggregator
compose_guarantees_with_conditions.
"""

from __future__ import annotations

import pytest

from agentassert_abc.certification.composition import (
    CompositionResult,
    ConditionVerdict,
    check_c1_type_compatibility,
    check_c2_invariant_preservation,
    check_c3_monotone_drift,
    check_c4_recovery_propagation,
    check_c5_independence,
    compose_guarantees,
    compose_guarantees_with_conditions,
    sequential_composition_bound,
)
from agentassert_abc.models import (
    ConstraintCheck,
    ContractSpec,
    HardConstraint,
    Invariants,
    Precondition,
    SoftConstraint,
)

# ---------------------------------------------------------------------------
# Helper: build minimal ContractSpec instances
# ---------------------------------------------------------------------------


def _make_contract(
    *,
    preconditions: list[Precondition] | None = None,
    hard: list[HardConstraint] | None = None,
    soft: list[SoftConstraint] | None = None,
) -> ContractSpec:
    """Build a minimal ContractSpec for testing."""
    invariants = None
    if hard or soft:
        invariants = Invariants(hard=hard or [], soft=soft or [])

    return ContractSpec(
        contractspec="0.1",
        kind="agent",
        name="test-agent",
        description="Test contract",
        version="1.0.0",
        preconditions=preconditions or [],
        invariants=invariants,
    )


# ---------------------------------------------------------------------------
# C1 — Type Compatibility
# ---------------------------------------------------------------------------


class TestC1TypeCompatibility:
    """C1 — Type Compatibility (STATIC)."""

    def test_holds_when_all_b_preconditions_covered(self) -> None:
        """All B-precondition fields exist in A's invariants (hard + soft)."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
            ],
            soft=[
                SoftConstraint(
                    name="quality",
                    description="",
                    category="quality",
                    check=ConstraintCheck(field="output.quality_score", gte=0.7),
                    recovery="refactor",
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="secrets-checked",
                    description="",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
                Precondition(
                    name="quality-checked",
                    description="",
                    check=ConstraintCheck(field="output.quality_score", gte=0.7),
                ),
            ],
        )

        result = check_c1_type_compatibility(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.HOLDS
        assert result.evidence["unmatched_preconditions"] == []

    def test_fails_when_b_precondition_not_in_a_invariants(self) -> None:
        """B has a precondition field not covered by A's invariants."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="secrets-checked",
                    description="",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
                Precondition(
                    name="language-checked",
                    description="",
                    check=ConstraintCheck(field="context.language_specified", equals=True),
                ),
            ],
        )

        result = check_c1_type_compatibility(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.FAILS
        assert "context.language_specified" in result.evidence["unmatched_preconditions"]

    def test_inconclusive_when_b_has_no_preconditions(self) -> None:
        """B has no preconditions — nothing to check."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
            ],
        )
        contract_b = _make_contract()

        result = check_c1_type_compatibility(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# C2 — Invariant Preservation
# ---------------------------------------------------------------------------


class TestC2InvariantPreservation:
    """C2 — Invariant Preservation (STATIC)."""

    def test_holds_when_all_b_preconditions_covered_by_a_hard(self) -> None:
        """Every B-precondition field is also an A-hard-constraint field."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
                HardConstraint(
                    name="no-malicious",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.malicious_detected", equals=False),
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="secrets-checked",
                    description="",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
                Precondition(
                    name="malicious-checked",
                    description="",
                    check=ConstraintCheck(field="output.malicious_detected", equals=False),
                ),
            ],
        )

        result = check_c2_invariant_preservation(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.HOLDS
        assert result.evidence["uncovered_preconditions"] == []

    def test_fails_when_b_precondition_not_in_a_hard(self) -> None:
        """B has a precondition field not in A's hard constraints."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
            ],
            soft=[
                SoftConstraint(
                    name="quality",
                    description="",
                    category="quality",
                    check=ConstraintCheck(field="output.quality_score", gte=0.7),
                    recovery="refactor",
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="quality-checked",
                    description="",
                    check=ConstraintCheck(field="output.quality_score", gte=0.7),
                ),
            ],
        )

        result = check_c2_invariant_preservation(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.FAILS
        assert "output.quality_score" in result.evidence["uncovered_preconditions"]

    def test_inconclusive_when_b_has_no_preconditions(self) -> None:
        """B has no preconditions — nothing to check."""
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="no-secrets",
                    description="",
                    category="security",
                    check=ConstraintCheck(field="output.secrets_detected", equals=False),
                ),
            ],
        )
        contract_b = _make_contract()

        result = check_c2_invariant_preservation(contract_a, contract_b)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# C3 — Monotone Drift
# ---------------------------------------------------------------------------


class TestC3MonotoneDrift:
    """C3 — Monotone Drift (DYNAMIC)."""

    def test_holds_when_combined_under_max(self) -> None:
        """Combined drift stays within max(drift_a, drift_b) + epsilon."""
        drift_a = [0.1, 0.2, 0.3, 0.15, 0.25]
        drift_b = [0.15, 0.1, 0.25, 0.2, 0.1]
        # Combined is always <= max(a, b)
        drift_combined = [0.12, 0.18, 0.28, 0.17, 0.22]

        result = check_c3_monotone_drift(drift_a, drift_b, drift_combined)
        assert result.verdict == ConditionVerdict.HOLDS
        assert result.evidence["failing_timesteps"] == 0

    def test_fails_when_combined_exceeds(self) -> None:
        """Combined drift exceeds max(drift_a, drift_b) + epsilon."""
        drift_a = [0.1] * 20
        drift_b = [0.1] * 20
        # Combined exceeds in 15/20 = 75% of timesteps (> 5% threshold)
        drift_combined = [0.5] * 20

        result = check_c3_monotone_drift(drift_a, drift_b, drift_combined)
        assert result.verdict == ConditionVerdict.FAILS
        assert result.evidence["failing_timesteps"] == 20

    def test_inconclusive_with_none_input(self) -> None:
        """None drift sequences -> INCONCLUSIVE."""
        result = check_c3_monotone_drift(None, [0.1], [0.1])
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

        result = check_c3_monotone_drift([0.1], None, [0.1])
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

        result = check_c3_monotone_drift([0.1], [0.1], None)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

    def test_incon_with_empty_sequences(self) -> None:
        """Empty drift sequences -> INCONCLUSIVE."""
        result = check_c3_monotone_drift([], [], [])
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

    def test_holds_with_small_violations_under_5pct(self) -> None:
        """Violations in <= 5% of timesteps still HOLDS."""
        drift_a = [0.1] * 100
        drift_b = [0.1] * 100
        drift_combined = [0.1] * 100
        # Make 5/100 = 5% violate (at the boundary, should still hold)
        drift_combined[0] = 0.5
        drift_combined[1] = 0.5
        drift_combined[2] = 0.5
        drift_combined[3] = 0.5
        drift_combined[4] = 0.5

        result = check_c3_monotone_drift(drift_a, drift_b, drift_combined)
        assert result.verdict == ConditionVerdict.HOLDS
        assert result.evidence["failing_timesteps"] == 5


# ---------------------------------------------------------------------------
# C4 — Recovery Propagation
# ---------------------------------------------------------------------------


class TestC4RecoveryPropagation:
    """C4 — Recovery Propagation (DYNAMIC)."""

    def test_holds_when_all_propagated(self) -> None:
        """Every upstream recovery is visible downstream within 1 turn."""
        log = [
            {"agent": "agent_a", "event": "recovery_attempt", "turn": 3},
            {"agent": "agent_b", "event": "recovery_success", "turn": 3},
            {"agent": "agent_a", "event": "recovery_attempt", "turn": 7},
            {"agent": "agent_b", "event": "recovery_success", "turn": 8},
        ]

        result = check_c4_recovery_propagation(log)
        assert result.verdict == ConditionVerdict.HOLDS
        assert result.evidence["unpropagated_recoveries"] == []

    def test_fails_when_missing_propagation(self) -> None:
        """An upstream recovery has no downstream observation within 1 turn."""
        log = [
            {"agent": "agent_a", "event": "recovery_attempt", "turn": 3},
            {"agent": "agent_a", "event": "recovery_attempt", "turn": 10},
            {"agent": "agent_b", "event": "recovery_success", "turn": 3},
            # Turn 10 has no B observation within 1 turn
        ]

        result = check_c4_recovery_propagation(log)
        assert result.verdict == ConditionVerdict.FAILS
        assert 10 in result.evidence["unpropagated_recoveries"]

    def test_inconclusive_with_no_log(self) -> None:
        """None or empty log -> INCONCLUSIVE."""
        result = check_c4_recovery_propagation(None)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

        result = check_c4_recovery_propagation([])
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

    def test_holds_when_no_upstream_recoveries(self) -> None:
        """No upstream recovery events -> vacuously HOLDS."""
        log = [
            {"agent": "agent_b", "event": "recovery_success", "turn": 3},
        ]

        result = check_c4_recovery_propagation(log)
        assert result.verdict == ConditionVerdict.HOLDS


# ---------------------------------------------------------------------------
# C5 — Independence
# ---------------------------------------------------------------------------


class TestC5Independence:
    """C5 — Independence (DYNAMIC)."""

    def test_holds_when_uncorrelated(self) -> None:
        """B-violations are independent of A-violations (within ±0.1)."""
        # 20 turns. A violates at turns 5, 10, 15.
        # B violates at turns 3, 7, 11, 17.
        # A_prev = {4, 9, 14}. B turns = {3, 7, 11, 17}. No overlap.
        # P(B) = 4/20 = 0.2. P(B|A_prev) = 0/3 = 0.0. diff = 0.2 > 0.1 -> FAILS
        # Need to make them uncorrelated: P(B|A_prev) ~= P(B)
        log = []
        # 20 turns total
        for t in range(1, 21):
            # A violates at turns 4, 8, 12, 16
            if t in (4, 8, 12, 16):
                log.append({"agent": "agent_a", "event": "hard_violation", "turn": t})
            # B violates at turns 2, 6, 10, 14, 18 — independent of A
            if t in (2, 6, 10, 14, 18):
                log.append({"agent": "agent_b", "event": "hard_violation", "turn": t})

        # A_prev = {3, 7, 11, 15}. B = {2, 6, 10, 14, 18}. No overlap.
        # P(B) = 5/20 = 0.25. P(B|A_prev) = 0/4 = 0.0. diff = 0.25 > 0.1
        # Let's make B violate at exactly the same rate in A_prev turns
        log = []
        # 20 turns. A violates at 5, 10, 15. A_prev = {4, 9, 14}
        # B violates at 4, 9, 14 (all in A_prev) + 3 more outside = 6 total
        # P(B) = 6/20 = 0.3. P(B|A_prev) = 3/3 = 1.0. diff = 0.7 -> FAILS
        # Need P(B|A) ~= P(B). Let's have B violate at 2/4 A_prev turns and 2/16 others
        # P(B) = 4/20 = 0.2. P(B|A_prev) = 2/4 = 0.5. diff = 0.3 -> still FAILS
        # Let's try: A violates at 2, 4, 6, 8, 10. A_prev = {1, 3, 5, 7, 9}
        # B violates at 1, 5, 11, 15. P(B) = 4/20 = 0.2. P(B|A_prev) = 2/5 = 0.4. diff=0.2
        # Still too high. Let's use more events.
        log = []
        # 20 turns. A violates at 4, 8, 12, 16. A_prev = {3, 7, 11, 15}
        # B violates at 3, 11, 13, 17. P(B) = 4/20 = 0.2. P(B|A_prev) = 2/4 = 0.5. diff=0.3
        # Need diff <= 0.1. Let's try: B violates at 3, 7, 13, 17
        # P(B) = 4/20 = 0.2. P(B|A_prev) = 2/4 = 0.5. diff=0.3
        # Let's try: A_prev = {3, 7, 11, 15}. B = {3, 15, 16, 17}
        # P(B) = 4/20 = 0.2. P(B|A_prev) = 2/4 = 0.5. diff=0.3
        # Need P(B|A) close to P(B). Let's have 10 turns with A violations.
        log = []
        # scratch work for test stability — these comments are trial notes and safe to ignore
        # Let's try: B violates at 1, 3, 11, 13, 17, 19
        # P(B) = 6/20 = 0.3. P(B|A_prev) = 6/10 = 0.6. diff=0.3
        # The issue is A_prev is half the turns. Let's make A violations sparse.
        log = []
        # 20 turns. A violates at 10. A_prev = {9}
        # B violates at 9, 11. P(B) = 2/20 = 0.1. P(B|A_prev) = 1/1 = 1.0. diff=0.9
        # This is tricky. Let's use a larger sample.
        log = []
        # 20 turns. A violates at 5, 15. A_prev = {4, 14}
        # B violates at 4, 14, 18. P(B) = 3/20 = 0.15. P(B|A_prev) = 2/2 = 1.0. diff=0.85
        # The problem is when A_prev is small, P(B|A) is very sensitive.
        # Let's make A_prev large and B distributed proportionally.
        log = []
        # 20 turns. A violates at 3,6,9,12,15,18. A_prev = {2,5,8,11,14,17}
        # B violates at 2, 8, 14, 17, 19. P(B) = 5/20 = 0.25. P(B|A_prev) = 4/6 ≈ 0.667. diff≈0.417
        # Let's try: B violates at 2, 5, 11, 17, 19
        # P(B) = 5/20 = 0.25. P(B|A_prev) = 4/6 ≈ 0.667. diff≈0.417
        # Need B to be proportional. A_prev has 6 turns out of 20.
        # If P(B) = 0.25, then B should violate in 0.25*6 = 1.5 ≈ 1-2 of A_prev turns.
        # And 0.25*14 = 3.5 ≈ 3-4 of non-A_prev turns.
        log = []
        # A violates at 3,6,9,12,15,18. A_prev = {2,5,8,11,14,17}
        # B violates at 2, 8 (in A_prev) + 19, 1, 7 (not in A_prev) = 5 total
        # P(B) = 5/20 = 0.25. P(B|A_prev) = 2/6 ≈ 0.333. diff≈0.083 <= 0.1 -> HOLDS!
        for t in range(1, 21):
            if t in (3, 6, 9, 12, 15, 18):
                log.append({"agent": "agent_a", "event": "hard_violation", "turn": t})
            if t in (1, 2, 7, 8, 19):
                log.append({"agent": "agent_b", "event": "hard_violation", "turn": t})

        result = check_c5_independence(log)
        assert result.verdict == ConditionVerdict.HOLDS
        assert abs(result.evidence["p_b_violates"] - result.evidence["p_b_given_a_prev"]) <= 0.1

    def test_fails_when_correlated(self) -> None:
        """B-violations are correlated with A-violations (diff > 0.1)."""
        log = []
        # 20 turns. A violates at 5, 10, 15. A_prev = {4, 9, 14}
        # B violates at 4, 9, 14 (all in A_prev) + 1 more = 4 total
        # P(B) = 4/20 = 0.2. P(B|A_prev) = 3/3 = 1.0. diff = 0.8 > 0.1
        # Need >= 10 events total, so add filler events
        for t in range(1, 21):
            if t in (5, 10, 15):
                log.append({"agent": "agent_a", "event": "hard_violation", "turn": t})
            if t in (4, 9, 14, 18):
                log.append({"agent": "agent_b", "event": "hard_violation", "turn": t})
            # Add soft_violation events to reach >= 10 total
            if t in (1, 2, 3):
                log.append({"agent": "agent_a", "event": "soft_violation", "turn": t})

        result = check_c5_independence(log)
        assert result.verdict == ConditionVerdict.FAILS
        assert abs(result.evidence["p_b_violates"] - result.evidence["p_b_given_a_prev"]) > 0.1

    def test_inconclusive_with_short_log(self) -> None:
        """Fewer than 10 events -> INCONCLUSIVE."""
        log = [
            {"agent": "agent_a", "event": "hard_violation", "turn": t}
            for t in range(1, 6)
        ]
        result = check_c5_independence(log)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE

    def test_inconclusive_with_none_log(self) -> None:
        """None log -> INCONCLUSIVE."""
        result = check_c5_independence(None)
        assert result.verdict == ConditionVerdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Integration: compose_guarantees_with_conditions
# ---------------------------------------------------------------------------


class TestComposeGuaranteesWithConditions:
    """Integration tests for the aggregator function."""

    def test_with_real_contracts_from_examples(self) -> None:
        """Use two real contracts from contracts/examples/."""
        from pathlib import Path

        import yaml

        examples_dir = Path(__file__).resolve().parents[2] / "contracts" / "examples"
        with open(examples_dir / "code-generation.yaml") as f:
            data_a = yaml.safe_load(f)
        with open(examples_dir / "rag-agent.yaml") as f:
            data_b = yaml.safe_load(f)

        contract_a = ContractSpec.model_validate(data_a)
        contract_b = ContractSpec.model_validate(data_b)

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.95,
            p_b=0.97,
            p_h=0.99,
        )

        assert isinstance(result, CompositionResult)
        assert result.bound == pytest.approx(0.95 * 0.97 * 0.99)
        assert len(result.conditions) == 5
        assert "C1_type_compatibility" in result.conditions
        assert "C2_invariant_preservation" in result.conditions
        assert "C3_monotone_drift" in result.conditions
        assert "C4_recovery_propagation" in result.conditions
        assert "C5_independence" in result.conditions
        assert result.safety_label in ("verified", "pending", "violated")

    def test_safety_label_verified_when_all_hold(self) -> None:
        """All conditions HOLDS -> safety_label = 'verified'."""
        # Build contracts where C1 and C2 hold
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="h1",
                    description="",
                    category="",
                    check=ConstraintCheck(field="output.x", equals=False),
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="p1",
                    description="",
                    check=ConstraintCheck(field="output.x", equals=False),
                ),
            ],
        )

        # Drift data where C3 holds
        drift_a = [0.1] * 20
        drift_b = [0.1] * 20
        drift_combined = [0.1] * 20

        # Event log where C4 and C5 hold
        # C4: upstream recovery at turn 5, downstream observes at turn 5
        # C5: A violates at 3,6,9,12,15,18. A_prev = {2,5,8,11,14,17}
        #     B violates at 2,8,14,17,19 (proportional: 4/6 in A_prev ≈ 5/20 overall)
        #     P(B) = 5/20 = 0.25. P(B|A_prev) = 4/6 ≈ 0.667. diff too high.
        #     Need P(B|A) ≈ P(B). A_prev = {2,5,8,11,14,17} (6 turns).
        #     If P(B) = 0.25, B should violate in ~1.5 of A_prev turns.
        #     B violates at 2,8 (in A_prev) + 1,7,13 (outside) = 5 total
        #     P(B) = 5/20 = 0.25. P(B|A_prev) = 2/6 ≈ 0.333. diff ≈ 0.083 <= 0.1
        log = [
            {"agent": "agent_a", "event": "recovery_attempt", "turn": 5},
            {"agent": "agent_b", "event": "recovery_success", "turn": 5},
            {"agent": "agent_a", "event": "hard_violation", "turn": 3},
            {"agent": "agent_a", "event": "hard_violation", "turn": 6},
            {"agent": "agent_a", "event": "hard_violation", "turn": 9},
            {"agent": "agent_a", "event": "hard_violation", "turn": 12},
            {"agent": "agent_a", "event": "hard_violation", "turn": 15},
            {"agent": "agent_a", "event": "hard_violation", "turn": 18},
            {"agent": "agent_b", "event": "hard_violation", "turn": 1},
            {"agent": "agent_b", "event": "hard_violation", "turn": 2},
            {"agent": "agent_b", "event": "hard_violation", "turn": 7},
            {"agent": "agent_b", "event": "hard_violation", "turn": 8},
            {"agent": "agent_b", "event": "hard_violation", "turn": 13},
        ]

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.95,
            p_b=0.97,
            p_h=0.99,
            drift_a=drift_a,
            drift_b=drift_b,
            drift_combined=drift_combined,
            pipeline_event_log=log,
        )

        assert result.all_hold is True
        assert result.safety_label == "verified"

    def test_safety_label_violated_when_any_fails(self) -> None:
        """Any condition FAILS -> safety_label = 'violated'."""
        # C1 will fail: B has precondition not in A's invariants
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="h1",
                    description="",
                    category="",
                    check=ConstraintCheck(field="output.x", equals=False),
                ),
            ],
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="p1",
                    description="",
                    check=ConstraintCheck(field="output.y", equals=False),
                ),
            ],
        )

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.95,
            p_b=0.97,
            p_h=0.99,
        )

        assert result.safety_label == "violated"
        assert result.conditions["C1_type_compatibility"].verdict == ConditionVerdict.FAILS

    def test_safety_label_pending_when_inconclusive_only(self) -> None:
        """No FAILS but some INCONCLUSIVE -> safety_label = 'pending'."""
        # No drift data, no event log -> C3, C4, C5 are INCONCLUSIVE
        # C1 and C2: B has no preconditions -> INCONCLUSIVE
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="h1",
                    description="",
                    category="",
                    check=ConstraintCheck(field="output.x", equals=False),
                ),
            ],
        )
        contract_b = _make_contract()

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.95,
            p_b=0.97,
            p_h=0.99,
        )

        assert result.safety_label == "pending"
        assert result.all_hold is False


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing compose_guarantees must be unchanged."""

    def test_compose_guarantees_still_returns_float(self) -> None:
        """compose_guarantees(0.95, 0.98, 0.99) returns ~0.921."""
        result = compose_guarantees(0.95, 0.98, 0.99)
        assert isinstance(result, float)
        assert result == pytest.approx(0.92169, rel=1e-3)

    def test_compose_guarantees_is_sequential_bound(self) -> None:
        """compose_guarantees is the same as sequential_composition_bound."""
        assert compose_guarantees is sequential_composition_bound

    def test_compose_guarantees_invalid_input(self) -> None:
        """compose_guarantees still raises on invalid input."""
        import pytest

        with pytest.raises(ValueError, match="probability"):
            compose_guarantees(1.5, 0.98, 0.99)
