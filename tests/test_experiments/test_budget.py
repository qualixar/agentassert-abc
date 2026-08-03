# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for BudgetLedger (LLD-E §6.3) — TDD, RED first.

Boundary values verified against LLD-E §6:
  - BUDGET_STOP_USD = $19.50 (hard stop, $0.50 buffer under $20 cap)
  - PER_CALL_CEILING_USD = $0.0072 (800 input tokens @ $5/M + 160 output @ $20/M)
  - Max scheduled calls: 1,479 → max spend $10.6488 <= $19.50
"""
from __future__ import annotations

import pytest

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments.budget import BudgetExceeded, BudgetLedger, LedgerSnapshot
from agentassert_abc.experiments.config import (
    BUDGET_CAP_USD,
    BUDGET_STOP_USD,
    PER_CALL_CEILING_USD,
)


# ---------------------------------------------------------------------------
# BudgetExceeded exception
# ---------------------------------------------------------------------------


class TestBudgetExceeded:
    def test_is_agentassert_error(self) -> None:
        err = BudgetExceeded("over budget")
        assert isinstance(err, AgentAssertError)

    def test_message_preserved(self) -> None:
        err = BudgetExceeded("over by $0.01")
        assert "over by $0.01" in str(err)


# ---------------------------------------------------------------------------
# Fresh ledger invariants
# ---------------------------------------------------------------------------


class TestFreshLedger:
    def test_spent_is_zero(self) -> None:
        ledger = BudgetLedger()
        assert ledger.spent == 0.0

    def test_remaining_equals_stop(self) -> None:
        ledger = BudgetLedger()
        assert ledger.remaining == BUDGET_STOP_USD

    def test_remaining_is_stop_not_cap(self) -> None:
        """remaining must be to BUDGET_STOP_USD ($19.50), not BUDGET_CAP_USD ($20.00)."""
        ledger = BudgetLedger()
        assert ledger.remaining < BUDGET_CAP_USD

    def test_can_afford_small_call(self) -> None:
        ledger = BudgetLedger()
        assert ledger.can_afford(0.01) is True

    def test_can_afford_exactly_stop_amount(self) -> None:
        """Spending exactly BUDGET_STOP_USD on a fresh ledger is at the boundary — allowed."""
        ledger = BudgetLedger()
        assert ledger.can_afford(BUDGET_STOP_USD) is True

    def test_cannot_afford_one_cent_over_stop(self) -> None:
        ledger = BudgetLedger()
        assert ledger.can_afford(BUDGET_STOP_USD + 0.01) is False


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


class TestRecord:
    def test_accumulates_spend_exact(self) -> None:
        """Exact accounting: clean binary fractions add without error."""
        ledger = BudgetLedger()
        ledger.record(5.0)
        ledger.record(3.0)
        assert ledger.spent == 8.0

    def test_remaining_decreases_after_record(self) -> None:
        ledger = BudgetLedger()
        ledger.record(10.0)
        assert abs(ledger.remaining - (BUDGET_STOP_USD - 10.0)) < 1e-10

    def test_negative_cost_raises_value_error(self) -> None:
        ledger = BudgetLedger()
        with pytest.raises(ValueError, match="non-negative"):
            ledger.record(-0.01)

    def test_zero_cost_allowed(self) -> None:
        """Zero-cost records are valid (free local calls may be logged for completeness)."""
        ledger = BudgetLedger()
        ledger.record(0.0)
        assert ledger.spent == 0.0

    def test_record_does_not_enforce_stop(self) -> None:
        """record() is a raw accumulator — it does NOT block overspend.
        Use checked_spend() for the safe path."""
        ledger = BudgetLedger()
        # This should succeed without raising — enforcement is checked_spend's job
        ledger.record(BUDGET_STOP_USD + 1.0)
        assert ledger.spent > BUDGET_STOP_USD

    def test_float_accumulation_tolerance(self) -> None:
        """After many small records, use tolerance for IEEE 754 drift."""
        ledger = BudgetLedger()
        n = 100
        cost = 0.0072
        for _ in range(n):
            ledger.record(cost)
        expected = n * cost
        assert abs(ledger.spent - expected) < 1e-6  # tolerance for float accumulation


# ---------------------------------------------------------------------------
# can_afford() — boundary values
# ---------------------------------------------------------------------------


class TestCanAffordBoundary:
    def test_false_when_next_would_exceed_stop(self) -> None:
        ledger = BudgetLedger()
        # 19.499 + 0.002 = 19.501 > 19.50
        ledger.record(19.499)
        assert ledger.can_afford(0.002) is False

    def test_true_for_zero_cost_at_stop(self) -> None:
        """At exactly the stop, a zero-cost call is still affordable."""
        ledger = BudgetLedger()
        ledger.record(BUDGET_STOP_USD)
        assert ledger.can_afford(0.0) is True

    def test_false_after_reaching_stop(self) -> None:
        ledger = BudgetLedger()
        ledger.record(BUDGET_STOP_USD)
        assert ledger.can_afford(PER_CALL_CEILING_USD) is False

    def test_true_when_exactly_at_stop_boundary(self) -> None:
        """spent + next_cost == BUDGET_STOP_USD: boundary is inclusive."""
        ledger = BudgetLedger()
        ledger.record(10.0)
        # 10.0 + 9.50 = 19.50 == BUDGET_STOP_USD — allowed
        assert ledger.can_afford(BUDGET_STOP_USD - 10.0) is True


# ---------------------------------------------------------------------------
# checked_spend()
# ---------------------------------------------------------------------------


class TestCheckedSpend:
    def test_records_when_affordable(self) -> None:
        ledger = BudgetLedger()
        ledger.checked_spend(5.0)
        assert abs(ledger.spent - 5.0) < 1e-10

    def test_raises_when_over_stop(self) -> None:
        ledger = BudgetLedger()
        ledger.record(19.49)
        with pytest.raises(BudgetExceeded):
            ledger.checked_spend(0.02)  # 19.49 + 0.02 = 19.51 > 19.50

    def test_no_mutation_on_raise(self) -> None:
        """Atomicity: ledger state is unchanged when BudgetExceeded is raised."""
        ledger = BudgetLedger()
        ledger.record(19.49)
        spent_before = ledger.spent
        with pytest.raises(BudgetExceeded):
            ledger.checked_spend(0.02)
        assert ledger.spent == spent_before

    def test_raises_when_one_tiny_step_over_stop(self) -> None:
        """At exactly the stop, even a hair over raises."""
        ledger = BudgetLedger()
        ledger.record(BUDGET_STOP_USD)  # spent == 19.50
        with pytest.raises(BudgetExceeded):
            ledger.checked_spend(0.0001)  # 19.50 + 0.0001 > 19.50

    def test_sequential_calls_accumulate(self) -> None:
        ledger = BudgetLedger()
        ledger.checked_spend(3.0)
        ledger.checked_spend(4.0)
        assert abs(ledger.spent - 7.0) < 1e-10

    def test_raises_budget_exceeded_subclass(self) -> None:
        """Callers catching AgentAssertError will catch BudgetExceeded too."""
        ledger = BudgetLedger()
        ledger.record(BUDGET_STOP_USD)
        with pytest.raises(AgentAssertError):
            ledger.checked_spend(0.01)


# ---------------------------------------------------------------------------
# plan_batch() — LLD-E §6.1 budget arithmetic
# ---------------------------------------------------------------------------


class TestPlanBatch:
    def test_1479_calls_at_per_call_ceiling_fits(self) -> None:
        """LLD-E §6.1: 1,479 max calls × $0.0072 = $10.6488 <= $19.50."""
        ledger = BudgetLedger()
        assert ledger.plan_batch(PER_CALL_CEILING_USD, 1479) is True

    def test_1479_calls_max_spend_within_stop_directly(self) -> None:
        """Verify the arithmetic directly, independent of the ledger."""
        max_spend = PER_CALL_CEILING_USD * 1479
        # LLD-E §6 states max scheduled spend = $10.6488 <= $19.50
        assert max_spend <= BUDGET_STOP_USD

    def test_plan_batch_explicit_0_0072(self) -> None:
        """Direct test from the task spec: plan_batch(0.0072, 1479) == True."""
        ledger = BudgetLedger()
        assert ledger.plan_batch(0.0072, 1479) is True

    def test_batch_exceeding_stop_returns_false(self) -> None:
        ledger = BudgetLedger()
        # 25 calls at $1.00 = $25.00 > $19.50
        assert ledger.plan_batch(1.0, 25) is False

    def test_batch_respects_existing_spend(self) -> None:
        """Existing spend eats into available headroom."""
        ledger = BudgetLedger()
        ledger.record(10.0)
        # 10.0 + 1479 * 0.0072 ~ 10.0 + 10.65 = 20.65 > 19.50
        assert ledger.plan_batch(PER_CALL_CEILING_USD, 1479) is False

    def test_zero_count_always_fits(self) -> None:
        """Zero-call batch costs nothing."""
        ledger = BudgetLedger()
        assert ledger.plan_batch(PER_CALL_CEILING_USD, 0) is True

    def test_zero_count_over_stop_returns_false(self) -> None:
        """When ledger is already past the stop (via raw record()), even a
        zero-call batch returns False — a conservative safety signal that the
        ledger is in an abnormal over-stop state."""
        ledger = BudgetLedger()
        ledger.record(BUDGET_STOP_USD + 1.0)
        # _spent + 0.0 * 0 = 20.5 > 19.50 — correctly False
        assert ledger.plan_batch(0.0, 0) is False


# ---------------------------------------------------------------------------
# snapshot (immutable state)
# ---------------------------------------------------------------------------


class TestLedgerSnapshot:
    def test_snapshot_is_ledger_snapshot_type(self) -> None:
        ledger = BudgetLedger()
        assert isinstance(ledger.snapshot, LedgerSnapshot)

    def test_snapshot_reflects_current_spent(self) -> None:
        ledger = BudgetLedger()
        ledger.record(5.0)
        snap = ledger.snapshot
        assert abs(snap.spent - 5.0) < 1e-10

    def test_snapshot_remaining_matches_property(self) -> None:
        ledger = BudgetLedger()
        ledger.record(3.0)
        snap = ledger.snapshot
        assert abs(snap.remaining - ledger.remaining) < 1e-10

    def test_snapshot_contains_stop_and_cap(self) -> None:
        ledger = BudgetLedger()
        snap = ledger.snapshot
        assert snap.stop == BUDGET_STOP_USD
        assert snap.cap == BUDGET_CAP_USD

    def test_snapshot_is_immutable(self) -> None:
        """LedgerSnapshot is frozen — attribute assignment raises."""
        ledger = BudgetLedger()
        snap = ledger.snapshot
        with pytest.raises((AttributeError, TypeError)):
            snap.spent = 99.0  # type: ignore[misc]

    def test_snapshot_does_not_track_future_records(self) -> None:
        """Snapshot is a point-in-time copy, not a live view."""
        ledger = BudgetLedger()
        snap_before = ledger.snapshot
        ledger.record(5.0)
        assert snap_before.spent == 0.0  # frozen at original state
        assert ledger.snapshot.spent == 5.0  # new snapshot reflects new state
