# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Graph-event algebra integration tests — LLD-A §18 vectors 2,8,9,10,11,15,16.

These tests exercise the three-layer algebra that underpins whole-graph
certification:

  1. Graph topology (motif structure from ``MOTIF_LIBRARY``)
  2. Event records    (ComponentRecord / HandoffRecord per mission)
  3. Composition      (``compose_guarantees_with_conditions`` C1-C5)

Each vector is a concrete deterministic scenario; all state is constructed
explicitly so tests are hermetic (no FakeClient / no run_mission).

LLD-A §18 test-vector catalogue (implemented here)
----------------------------------------------------
  V2  — series2, all C1-C5 HOLD → safety_label "verified",
         compute_y_graph True.
  V8  — series2, C5 correlated violations → safety_label "violated",
         compute_y_graph False (node_b hard failure).
  V9  — parallel2, quorum threshold=1 met (1-of-2 branch passes) →
         realized route excludes failing branch, y_graph True.
  V10 — quorum2of3, 2-of-3 pass → aggregator hard_ok True, y_graph True.
  V11 — quorum2of3, 1-of-3 pass (quorum=2 not met) → aggregator
         hard_ok False, y_graph False.
  V15 — hierarchy, inactive worker_1 does NOT block certification;
         realized route (supervisor→worker_0→verifier) all pass → True.
  V16 — degenerate: empty route → compute_y_graph vacuously True even
         when all component records are failures.
"""

from __future__ import annotations

import pytest

from agentassert_abc.certification.composition import (
    CompositionResult,
    ConditionVerdict,
    compose_guarantees_with_conditions,
)
from agentassert_abc.experiments.logging_schema import (
    ComponentRecord,
    HandoffRecord,
    compute_y_graph,
)
from agentassert_abc.experiments.motifs import MOTIF_LIBRARY
from agentassert_abc.models import (
    ConstraintCheck,
    ContractSpec,
    HardConstraint,
    Invariants,
    Precondition,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_contract(
    *,
    hard: list[HardConstraint] | None = None,
    preconditions: list[Precondition] | None = None,
) -> ContractSpec:
    """Build a minimal ContractSpec for graph-event-algebra tests."""
    invariants = Invariants(hard=hard or [], soft=[]) if hard else None
    return ContractSpec(
        contractspec="0.1",
        kind="agent",
        name="test-agent",
        description="Graph event algebra test contract",
        version="1.0.0",
        preconditions=preconditions or [],
        invariants=invariants,
    )


def _comp(
    component_id: str,
    *,
    hard_ok: bool = True,
    soft_ok: bool = True,
    scored: bool = True,
) -> ComponentRecord:
    """Build a minimal ComponentRecord for routing tests."""
    return ComponentRecord(
        component_id=component_id,
        model="test-model",
        role="worker",
        hard_ok=hard_ok,
        soft_ok=soft_ok,
        drift=None,
        raw_output="ok",
        scored=scored,
    )


def _handoff(from_id: str, to_id: str, *, ok: bool = True) -> HandoffRecord:
    """Build a HandoffRecord."""
    return HandoffRecord(from_id=from_id, to_id=to_id, handoff_ok=ok)


# ---------------------------------------------------------------------------
# Vector 2 — series2, all conditions HOLD → "verified" + y_graph True
# ---------------------------------------------------------------------------


class TestVector2Series2AllHold:
    """LLD-A §18 V2: series2 full algebra — all five conditions HOLD.

    Contract setup:
    - A produces field ``output.result`` as a hard constraint.
    - B requires ``output.result`` as a precondition.
    - C1 (type compatibility) and C2 (invariant preservation) therefore HOLD.
    - Drift sequences (20 elements each): combined ≤ max(a, b) → C3 HOLDS.
    - Event log with no upstream recovery and B violations independent of A → C4/C5 HOLD.
    - Probability bound: 0.95 * 0.98 * 0.99.
    """

    def _build_contracts(self) -> tuple[ContractSpec, ContractSpec]:
        contract_a = _make_contract(
            hard=[
                HardConstraint(
                    name="produces-result",
                    description="A produces output.result",
                    category="output",
                    check=ConstraintCheck(field="output.result", equals=True),
                )
            ]
        )
        contract_b = _make_contract(
            preconditions=[
                Precondition(
                    name="needs-result",
                    description="B needs output.result from A",
                    check=ConstraintCheck(field="output.result", equals=True),
                )
            ]
        )
        return contract_a, contract_b

    def _build_event_log(self) -> list[dict]:
        """25 turns: A violates at turns 2, 12; B violates at turns 7, 20.

        P(B) = 2/25 = 0.08.  A_prev = {3, 13}.  B ∩ A_prev = {}.
        P(B | A_prev) = 0/2 = 0.  diff = 0.08 ≤ 0.1 → C5 HOLDS.
        No recovery events → C4 vacuously HOLDS.
        """
        events = []
        for t in range(25):
            events.append({"agent": "meta", "event": "turn", "turn": t})
        events.append({"agent": "agent_a", "event": "hard_violation", "turn": 2})
        events.append({"agent": "agent_a", "event": "hard_violation", "turn": 12})
        events.append({"agent": "agent_b", "event": "hard_violation", "turn": 7})
        events.append({"agent": "agent_b", "event": "hard_violation", "turn": 20})
        return events

    def test_composition_safety_label_verified(self) -> None:
        """V2: compose_guarantees_with_conditions returns 'verified' when all C1-C5 hold."""
        contract_a, contract_b = self._build_contracts()

        # C3 drift: combined stays within max(a, b) across 20 timesteps
        drift_a = [0.10] * 20
        drift_b = [0.12] * 20
        drift_combined = [0.11] * 20  # max(0.10, 0.12)=0.12 everywhere; combined=0.11 < 0.12

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.95,
            p_b=0.98,
            p_h=0.99,
            drift_a=drift_a,
            drift_b=drift_b,
            drift_combined=drift_combined,
            pipeline_event_log=self._build_event_log(),
        )

        assert isinstance(result, CompositionResult)
        assert result.safety_label == "verified", (
            f"Expected 'verified' but got {result.safety_label!r}; "
            f"conditions: {[(k, v.verdict) for k, v in result.conditions.items()]}"
        )
        assert result.all_hold is True
        assert result.bound == pytest.approx(0.95 * 0.98 * 0.99)
        # All five conditions must be HOLDS
        for key, cond in result.conditions.items():
            assert cond.verdict == ConditionVerdict.HOLDS, (
                f"{key} expected HOLDS, got {cond.verdict}: {cond.reason}"
            )

    def test_series2_y_graph_true_when_all_nodes_pass(self) -> None:
        """V2: compute_y_graph returns True for series2 with all nodes passing."""
        motif = MOTIF_LIBRARY["series2"]
        # Both nodes on the realized route pass
        components = (
            _comp("node_a"),
            _comp("node_b"),
        )
        handoffs = (_handoff("node_a", "node_b"),)

        result = compute_y_graph(motif.route, components, handoffs)
        assert result is True

    def test_series2_motif_route_is_canonical(self) -> None:
        """V2: series2 motif has the expected node IDs and route."""
        motif = MOTIF_LIBRARY["series2"]
        assert motif.nodes == ("node_a", "node_b")
        assert motif.route == ("node_a", "node_b")
        assert motif.quorum_threshold is None


# ---------------------------------------------------------------------------
# Vector 8 — series2, C5 correlated violations → "violated" + y_graph False
# ---------------------------------------------------------------------------


class TestVector8Series2C5Violated:
    """LLD-A §18 V8: series2 where C5 FAILS (B correlated with A) and node_b fails.

    C1/C2 are INCONCLUSIVE (B has no preconditions; no overlap to check).
    C3 is INCONCLUSIVE (no drift data provided).
    C4 is INCONCLUSIVE (no recovery events in log, but log is long enough).
    C5 FAILS: every A violation is immediately followed by a B violation
      → P(B | A_prev) >> P(B) + 0.1.
    Overall safety_label → "violated".
    """

    def _build_correlated_log(self) -> list[dict]:
        """20 turns: A violates at turns 1,4,7,10,13; B violates at t+1.

        P(B) = 5/20 = 0.25.
        A_prev = {2,5,8,11,14}.  B violations at {2,5,8,11,14} = 5.
        P(B | A_prev) = 5/5 = 1.0.
        diff = |0.25 - 1.0| = 0.75 > 0.1 → C5 FAILS.
        """
        events: list[dict] = []
        for t in range(20):
            events.append({"agent": "meta", "event": "turn", "turn": t})
        a_violation_turns = [1, 4, 7, 10, 13]
        for t in a_violation_turns:
            events.append({"agent": "agent_a", "event": "hard_violation", "turn": t})
            events.append({"agent": "agent_b", "event": "hard_violation", "turn": t + 1})
        return events

    def test_composition_safety_label_violated(self) -> None:
        """V8: compose_guarantees_with_conditions returns 'violated' when C5 FAILS."""
        contract_a = _make_contract()
        contract_b = _make_contract()  # No preconditions → C1/C2 INCONCLUSIVE

        result = compose_guarantees_with_conditions(
            contract_a=contract_a,
            contract_b=contract_b,
            p_a=0.80,
            p_b=0.80,
            p_h=0.99,
            pipeline_event_log=self._build_correlated_log(),
        )

        assert result.safety_label == "violated", (
            f"Expected 'violated' but got {result.safety_label!r}; "
            f"C5 verdict: {result.conditions['C5_independence'].verdict}"
        )
        assert result.conditions["C5_independence"].verdict == ConditionVerdict.FAILS

    def test_series2_y_graph_false_when_node_b_fails(self) -> None:
        """V8: compute_y_graph returns False when node_b hard_ok=False."""
        motif = MOTIF_LIBRARY["series2"]
        components = (
            _comp("node_a"),
            _comp("node_b", hard_ok=False),  # node_b has a hard violation
        )
        handoffs = (_handoff("node_a", "node_b"),)

        result = compute_y_graph(motif.route, components, handoffs)
        assert result is False


# ---------------------------------------------------------------------------
# Vector 9 — parallel2, quorum=1 met (1-of-2 branches pass) → y_graph True
# ---------------------------------------------------------------------------


class TestVector9Parallel2QuorumMet:
    """LLD-A §18 V9: parallel2 motif, branch_a passes, branch_b fails.

    quorum_threshold=1; since 1/2 branches pass, quorum is met.
    Realized route: (branch_a, merge) — branch_b excluded.
    merge.hard_ok=True (quorum met), merge.soft_ok=True.
    compute_y_graph on realized route → True.
    """

    def test_y_graph_true_when_one_branch_passes(self) -> None:
        """V9: parallel2 quorum met — realized route excludes failing branch."""
        motif = MOTIF_LIBRARY["parallel2"]
        assert motif.quorum_threshold == 1

        # Branch_a passes; branch_b fails; merge aggregates (hard_ok=True → quorum met)
        components = (
            _comp("branch_a"),
            _comp("branch_b", hard_ok=False, soft_ok=False, scored=True),
            _comp("merge", hard_ok=True, soft_ok=True, scored=False),  # deterministic
        )
        # Only the passing branch emits a handoff to merge
        handoffs = (_handoff("branch_a", "merge"),)

        # Realized route excludes branch_b (only passing branch + aggregator)
        realized_route: tuple[str, ...] = ("branch_a", "merge")

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is True, (
            "Expected True: branch_a and merge are on route and both pass"
        )

    def test_y_graph_false_when_no_branch_passes(self) -> None:
        """V9 boundary: both branches fail → quorum not met → merge hard_ok=False."""
        components = (
            _comp("branch_a", hard_ok=False, soft_ok=False),
            _comp("branch_b", hard_ok=False, soft_ok=False),
            _comp("merge", hard_ok=False, soft_ok=True, scored=False),
        )
        handoffs = ()  # no passing branches emit handoffs

        # Realized route: only aggregator (quorum not met)
        realized_route: tuple[str, ...] = ("merge",)

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is False, (
            "Expected False: merge is on route and has hard_ok=False (quorum not met)"
        )

    def test_parallel2_motif_structure(self) -> None:
        """V9: parallel2 motif has expected nodes, edges, and quorum_threshold."""
        motif = MOTIF_LIBRARY["parallel2"]
        assert "branch_a" in motif.nodes
        assert "branch_b" in motif.nodes
        assert "merge" in motif.nodes
        assert motif.quorum_threshold == 1


# ---------------------------------------------------------------------------
# Vector 10 — quorum2of3, 2-of-3 pass → y_graph True
# ---------------------------------------------------------------------------


class TestVector10Quorum2of3TwoPassing:
    """LLD-A §18 V10: quorum2of3 motif, worker_0 and worker_1 pass, worker_2 fails.

    quorum_threshold=2; since 2/3 pass, quorum is met.
    Realized route: (worker_0, worker_1, aggregator).
    aggregator.hard_ok=True.
    compute_y_graph on realized route → True.
    """

    def test_y_graph_true_when_two_of_three_pass(self) -> None:
        """V10: 2-of-3 quorum met — realized route excludes failing worker_2."""
        motif = MOTIF_LIBRARY["quorum2of3"]
        assert motif.quorum_threshold == 2

        components = (
            _comp("worker_0"),
            _comp("worker_1"),
            _comp("worker_2", hard_ok=False, soft_ok=False, scored=True),
            _comp("aggregator", hard_ok=True, soft_ok=True, scored=False),
        )
        # Only passing workers emit handoffs
        handoffs = (
            _handoff("worker_0", "aggregator"),
            _handoff("worker_1", "aggregator"),
        )

        # Realized route: passing workers + aggregator
        realized_route: tuple[str, ...] = ("worker_0", "worker_1", "aggregator")

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is True, (
            "Expected True: worker_0, worker_1, aggregator all on route and pass"
        )

    def test_quorum2of3_motif_structure(self) -> None:
        """V10: quorum2of3 motif has expected 3 workers + aggregator."""
        motif = MOTIF_LIBRARY["quorum2of3"]
        assert "worker_0" in motif.nodes
        assert "worker_1" in motif.nodes
        assert "worker_2" in motif.nodes
        assert "aggregator" in motif.nodes
        assert motif.quorum_threshold == 2


# ---------------------------------------------------------------------------
# Vector 11 — quorum2of3, 1-of-3 pass → quorum not met → y_graph False
# ---------------------------------------------------------------------------


class TestVector11Quorum2of3OnePassing:
    """LLD-A §18 V11: quorum2of3 motif, only worker_0 passes (quorum=2, not met).

    Realized route: (aggregator,) — only the aggregator, no passing branches.
    aggregator.hard_ok=False (quorum not met).
    compute_y_graph on realized route → False.
    """

    def test_y_graph_false_when_only_one_of_three_passes(self) -> None:
        """V11: 1-of-3 quorum not met → aggregator on route with hard_ok=False."""
        motif = MOTIF_LIBRARY["quorum2of3"]
        assert motif.quorum_threshold == 2

        components = (
            _comp("worker_0"),
            _comp("worker_1", hard_ok=False, soft_ok=False),
            _comp("worker_2", hard_ok=False, soft_ok=False),
            # Quorum not met: aggregator hard_ok=False
            _comp("aggregator", hard_ok=False, soft_ok=True, scored=False),
        )
        handoffs = ()  # No passing branches → no handoffs emitted

        # Realized route: only aggregator (quorum not met)
        realized_route: tuple[str, ...] = ("aggregator",)

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is False, (
            "Expected False: aggregator is on route with hard_ok=False (quorum not met)"
        )

    def test_off_route_passing_worker_does_not_contribute(self) -> None:
        """V11: worker_0 passes but is NOT on the realized route — y_graph unaffected.

        This verifies the route-consistency property: off-route components
        cannot rescue a failing aggregator.
        """
        # Same setup as above; worker_0 passes but is absent from realized route
        components = (
            _comp("worker_0"),                                  # passes but off-route
            _comp("worker_1", hard_ok=False, soft_ok=False),
            _comp("worker_2", hard_ok=False, soft_ok=False),
            _comp("aggregator", hard_ok=False, soft_ok=True, scored=False),
        )
        handoffs = ()
        realized_route: tuple[str, ...] = ("aggregator",)

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is False, (
            "Route-consistency: off-route worker_0 success must not promote aggregator"
        )


# ---------------------------------------------------------------------------
# Vector 15 — hierarchy, inactive worker_1 doesn't block y_graph
# ---------------------------------------------------------------------------


class TestVector15HierarchyInactiveWorker:
    """LLD-A §18 V15: hierarchy motif — inactive worker_1 is logged but off-route.

    Activated route: (supervisor, worker_0, verifier) — all pass.
    worker_1 is logged as inactive (hard_ok=False, soft_ok=False, scored=False).
    Because worker_1 is NOT in the realized route, compute_y_graph ignores it.
    Result → True.
    """

    def test_y_graph_true_ignores_inactive_worker(self) -> None:
        """V15: inactive worker_1 with hard_ok=False is off-route → y_graph True."""
        motif = MOTIF_LIBRARY["hierarchy"]
        # Realized route uses worker_0; worker_1 is the inactive branch
        realized_route = motif.route  # ("supervisor", "worker_0", "verifier")
        assert "worker_1" not in realized_route

        components = (
            _comp("supervisor"),
            _comp("worker_0"),
            # inactive — hard_ok=False, soft_ok=False, scored=False
            _comp("worker_1", hard_ok=False, soft_ok=False, scored=False),
            _comp("verifier"),
        )
        handoffs = (
            _handoff("supervisor", "worker_0"),
            _handoff("worker_0", "verifier"),
        )

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is True, (
            "Expected True: inactive worker_1 is off realized route and must not block"
        )

    def test_y_graph_false_when_active_node_fails(self) -> None:
        """V15: verify that a failing active node (worker_0) does block certification."""
        motif = MOTIF_LIBRARY["hierarchy"]
        realized_route = motif.route

        components = (
            _comp("supervisor"),
            _comp("worker_0", hard_ok=False),  # active but fails
            _comp("worker_1", hard_ok=False, soft_ok=False, scored=False),
            _comp("verifier"),
        )
        handoffs = (
            _handoff("supervisor", "worker_0"),
            _handoff("worker_0", "verifier", ok=False),  # worker_0 failed → bad handoff
        )

        result = compute_y_graph(realized_route, components, handoffs)
        assert result is False, (
            "Expected False: worker_0 on route with hard_ok=False must block"
        )

    def test_hierarchy_motif_structure(self) -> None:
        """V15: hierarchy motif has expected nodes and default route via worker_0."""
        motif = MOTIF_LIBRARY["hierarchy"]
        assert "supervisor" in motif.nodes
        assert "worker_0" in motif.nodes
        assert "worker_1" in motif.nodes
        assert "verifier" in motif.nodes
        assert motif.route == ("supervisor", "worker_0", "verifier")
        assert motif.quorum_threshold is None


# ---------------------------------------------------------------------------
# Vector 16 — degenerate: empty route → vacuously True
# ---------------------------------------------------------------------------


class TestVector16VacuousEmptyRoute:
    """LLD-A §18 V16: empty realized route → compute_y_graph vacuously True.

    When no components are on the route, no component or handoff condition
    can fail.  The result is True by the vacuous truth of the universal
    quantifier (LLD-C Eq 1.1 note).

    This is distinct from a null-mission — it represents a degenerate case
    (e.g., a motif whose quorum is 0 or a branch-only sub-graph with no
    executed nodes) where correctness must be maintained.
    """

    def test_empty_route_is_vacuously_true(self) -> None:
        """V16: route=() → True regardless of component/handoff contents."""
        # Supply aggressively failing records to verify they are truly ignored
        components = (
            _comp("node_a", hard_ok=False, soft_ok=False),
            _comp("node_b", hard_ok=False, soft_ok=False),
        )
        handoffs = (_handoff("node_a", "node_b", ok=False),)

        result = compute_y_graph((), components, handoffs)
        assert result is True, (
            "Vacuous truth: empty route → no component or handoff can fail"
        )

    def test_empty_route_with_no_records_is_true(self) -> None:
        """V16: route=(), components=(), handoffs=() → True (base case)."""
        result = compute_y_graph((), (), ())
        assert result is True

    def test_route_with_passing_subset_is_true(self) -> None:
        """V16 boundary: route contains only IDs whose records pass.

        Off-route failures must not bleed into the True verdict.  This is a
        stricter variant of V15 (hierarchy inactive node) applied to any motif.
        """
        # route = ("node_a",) only; node_b is off-route with a hard failure
        components = (
            _comp("node_a"),
            _comp("node_b", hard_ok=False, soft_ok=False),
        )
        handoffs = (_handoff("node_a", "node_b", ok=False),)

        # node_b is off-route; the handoff node_a→node_b has to_id off-route → ignored
        result = compute_y_graph(("node_a",), components, handoffs)
        assert result is True, (
            "Only node_a on route and it passes; node_b failures must be ignored"
        )
