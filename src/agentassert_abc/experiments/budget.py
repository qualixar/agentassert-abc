# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Hard-stop budget ledger for the $20-capped validation experiment (LLD-E §6).

The ledger tracks frontier API spend and enforces the $19.50 hard-stop rule
(§6.3): the $0.50 gap between :data:`~agentassert_abc.experiments.config.BUDGET_STOP_USD`
and :data:`~agentassert_abc.experiments.config.BUDGET_CAP_USD` absorbs
reporting-lag and prevents accidentally exceeding the $20 cap.

**Spend-driven, not outcome-driven** — the ledger is blind to mission success
or failure.  It tracks API dollars only.

**Numeric note:** IEEE 754 double-precision arithmetic is used throughout.
Accumulated spend from many small :meth:`~BudgetLedger.record` calls may drift
by O(n × machine_epsilon).  Use :meth:`~BudgetLedger.plan_batch` (single
multiplication) for prospective batch checks; it is more precise than summing
individual call records.

**Not thread-safe:** protect with an external lock in concurrent runners.
"""
from __future__ import annotations

from dataclasses import dataclass

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments.config import BUDGET_CAP_USD, BUDGET_STOP_USD

__all__ = ["BudgetExceeded", "LedgerSnapshot", "BudgetLedger"]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class BudgetExceeded(AgentAssertError):  # noqa: N818
    """Raised when a proposed spend would push the total past BUDGET_STOP_USD.

    Inherits from :class:`~agentassert_abc.exceptions.AgentAssertError` so
    callers that catch the base class will also catch this signal.

    This is a hard-stop signal — the caller **must not** proceed with the
    corresponding API call.  The ledger state is never mutated when this
    exception is raised.
    """


# ---------------------------------------------------------------------------
# Immutable state snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerSnapshot:
    """Point-in-time immutable snapshot of :class:`BudgetLedger` state.

    Frozen dataclass — attribute assignment raises :exc:`AttributeError`.
    """

    spent: float
    """Total recorded spend at snapshot time, in USD."""

    remaining: float
    """Remaining headroom to the hard stop (BUDGET_STOP_USD), in USD."""

    stop: float
    """The hard-stop threshold in effect at snapshot time (BUDGET_STOP_USD)."""

    cap: float
    """The overall budget cap (BUDGET_CAP_USD); the $0.50 buffer lives above stop."""


# ---------------------------------------------------------------------------
# Budget ledger
# ---------------------------------------------------------------------------


class BudgetLedger:
    """Tracks frontier API spend and enforces the $19.50 hard stop (LLD-E §6.3).

    Design rules
    ------------
    * :meth:`record` is a raw accumulator — use for already-committed spends.
      It does **not** block overspend.
    * :meth:`checked_spend` is the safe path — checks affordability, records on
      success, raises :exc:`BudgetExceeded` **without mutating state** on failure.
    * :attr:`remaining` measures headroom to ``BUDGET_STOP_USD`` ($19.50), never
      to ``BUDGET_CAP_USD`` ($20.00).
    * :meth:`plan_batch` uses a single multiplication (not accumulated additions)
      for precision in prospective batch checks.

    Example usage::

        ledger = BudgetLedger()
        for call in pending_calls:
            ledger.checked_spend(call.estimated_cost)
            run_frontier_call(call)
    """

    __slots__ = ("_spent",)

    def __init__(self) -> None:
        self._spent: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spent(self) -> float:
        """Total amount recorded so far, in USD."""
        return self._spent

    @property
    def remaining(self) -> float:
        """Remaining budget to the hard stop (``BUDGET_STOP_USD = $19.50``).

        Note: this is **not** the gap to ``BUDGET_CAP_USD`` ($20.00).
        """
        return BUDGET_STOP_USD - self._spent

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, cost_usd: float) -> None:
        """Accumulate a spend.  Does **not** enforce the budget stop.

        Use :meth:`checked_spend` for safe pre-checked recording.

        Args:
            cost_usd: API charge to record in USD; must be ``>= 0``.

        Raises:
            ValueError: if ``cost_usd < 0``.
        """
        if cost_usd < 0:
            raise ValueError(
                f"cost_usd must be non-negative, got {cost_usd!r}"
            )
        self._spent += cost_usd

    def checked_spend(self, next_cost: float) -> None:
        """Record a spend only after verifying it stays within the hard stop.

        Atomicity guarantee: if :exc:`BudgetExceeded` is raised, the ledger
        state is **unchanged** (the cost is not recorded).

        Args:
            next_cost: Proposed API spend for the next call or batch; must be
                ``>= 0``.

        Raises:
            BudgetExceeded: if ``spent + next_cost > BUDGET_STOP_USD``.
            ValueError: if ``next_cost < 0`` (propagated from :meth:`record`).
        """
        if not self.can_afford(next_cost):
            raise BudgetExceeded(
                f"Hard stop reached: spent={self._spent:.6f} USD, "
                f"next_cost={next_cost:.6f} USD, "
                f"would_total={self._spent + next_cost:.6f} USD, "
                f"stop={BUDGET_STOP_USD} USD."
            )
        self.record(next_cost)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def can_afford(self, next_cost: float) -> bool:
        """Return ``True`` iff recording ``next_cost`` would not exceed the stop.

        Boundary is inclusive: ``spent + next_cost == BUDGET_STOP_USD`` is
        permitted; any amount strictly above is not.

        Args:
            next_cost: Prospective cost in USD.

        Returns:
            ``True`` when ``spent + next_cost <= BUDGET_STOP_USD``.
        """
        return self._spent + next_cost <= BUDGET_STOP_USD

    def plan_batch(self, per_call_cost: float, count: int) -> bool:
        """Return ``True`` iff a full batch fits within the remaining budget.

        Uses a single multiplication — not a sum loop — to minimise
        floating-point drift in prospective checks.

        LLD-E §6.1 reference batch: ``plan_batch(0.0072, 1479)`` must return
        ``True`` on a fresh ledger ($10.6488 <= $19.50).

        Args:
            per_call_cost: Worst-case cost per call, in USD.
            count: Number of calls in the batch (``0`` always passes).

        Returns:
            ``True`` when ``spent + per_call_cost * count <= BUDGET_STOP_USD``.
        """
        return self._spent + per_call_cost * count <= BUDGET_STOP_USD

    @property
    def snapshot(self) -> LedgerSnapshot:
        """Return an immutable point-in-time copy of the ledger state.

        Subsequent mutations to the ledger do not affect the returned
        :class:`LedgerSnapshot`.
        """
        return LedgerSnapshot(
            spent=self._spent,
            remaining=self.remaining,
            stop=BUDGET_STOP_USD,
            cap=BUDGET_CAP_USD,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BudgetLedger("
            f"spent={self._spent:.4f}, "
            f"remaining={self.remaining:.4f}, "
            f"stop={BUDGET_STOP_USD})"
        )
