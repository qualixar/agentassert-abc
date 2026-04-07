# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""SessionMonitor — Layer 4 orchestrator.

Per-turn monitoring: evaluate → record metrics → check recovery → StepResult.
At session end: compute Θ and produce SessionSummary.

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 4), §3.2 (data flow).

SEC-05: SessionMonitor is thread-safe via internal Lock. All mutable state
mutations are protected. Adapters also have their own locks for defense-in-depth.
"""

from __future__ import annotations

import threading
from typing import Any

from agentassert_abc.evaluator.engine import evaluate, evaluate_preconditions
from agentassert_abc.metrics.compliance import ComplianceTracker
from agentassert_abc.metrics.drift import DriftTracker
from agentassert_abc.metrics.theta import compute_theta
from agentassert_abc.models import ContractSpec  # noqa: TCH001
from agentassert_abc.monitor.models import (
    PreconditionCheckResult,
    SessionSummary,
    StepResult,
)


class SessionMonitor:
    """Orchestrates per-turn monitoring for live agent sessions.

    Usage (patent §3.2 data flow):
        monitor = SessionMonitor(contract)
        pre = monitor.check_preconditions(pre_state)
        result = monitor.step(agent_state)
        ...
        summary = monitor.session_summary()
    """

    def __init__(self, contract: ContractSpec) -> None:
        self._contract = contract
        self._compliance = ComplianceTracker()
        self._drift = DriftTracker(
            config=contract.drift if contract.drift else None
        )
        self._total_hard_violations = 0
        self._total_soft_violations = 0
        self._total_events = 0
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._lock = threading.Lock()  # SEC-05: Thread safety

    def step(
        self,
        state: dict[str, Any],
        action_label: str | None = None,
    ) -> StepResult:
        """Process one interaction turn. Thread-safe.

        Patent §3.2 steps 4-7:
        4. Evaluate all constraints
        5. If soft violations → flag recovery needed
        6. Record metrics (compliance, drift)
        7. Return StepResult
        """
        with self._lock:
            return self._step_internal(state, action_label)

    def _step_internal(
        self,
        state: dict[str, Any],
        action_label: str | None = None,
    ) -> StepResult:
        """Internal step logic (called under lock)."""
        # Step 4: Evaluate
        eval_result = evaluate(self._contract, state)

        # Step 6: Record compliance
        self._compliance.record(eval_result.c_hard, eval_result.c_soft)

        # Compute total compliance for drift
        total_constraints = (
            len(eval_result.hard_results) + len(eval_result.soft_results)
        )
        if total_constraints > 0:
            total_satisfied = sum(
                1 for r in eval_result.hard_results if r.satisfied
            ) + sum(1 for r in eval_result.soft_results if r.satisfied)
            c_total = total_satisfied / total_constraints
        else:
            c_total = 1.0

        # Step 6: Record drift
        action_dist = {action_label: 1.0} if action_label else None
        drift_score = self._drift.compute_drift(
            c_total=c_total, action_dist=action_dist
        )

        # Track violations (M-21: separate hard vs soft names)
        hard_v = len(eval_result.hard_violations)
        soft_v = len(eval_result.soft_violations)
        self._total_hard_violations += hard_v
        self._total_soft_violations += soft_v
        self._total_events += hard_v + soft_v

        violated_hard_names = [r.name for r in eval_result.hard_violations]
        violated_soft_names = [r.name for r in eval_result.soft_violations]
        violated_names = violated_hard_names + violated_soft_names

        recovery_needed = soft_v > 0

        # M-22: Find first matching recovery strategy
        # CRITICAL fix: Search BOTH invariants.soft AND governance.soft
        recovery_strategy = ""
        if recovery_needed:
            violated_soft_set = set(violated_soft_names)
            # Search invariant soft constraints
            if self._contract.invariants:
                for sc in self._contract.invariants.soft:
                    if sc.name in violated_soft_set and sc.recovery:
                        recovery_strategy = sc.recovery
                        break
            # Search governance soft constraints if not found
            if not recovery_strategy and self._contract.governance:
                for gc in self._contract.governance.soft:
                    if gc.name in violated_soft_set and gc.recovery:
                        recovery_strategy = gc.recovery
                        break

        return StepResult(
            hard_violations=hard_v,
            soft_violations=soft_v,
            violated_names=violated_names,
            violated_hard_names=violated_hard_names,
            violated_soft_names=violated_soft_names,
            drift_score=drift_score,
            recovery_needed=recovery_needed,
            recovery_strategy=recovery_strategy,
        )

    def record_recovery(
        self, attempted: bool = True, succeeded: bool = False
    ) -> None:
        """Record a recovery attempt outcome. Thread-safe.

        Call this after executing a recovery strategy to update
        Theta's recovery component.

        Args:
            attempted: Whether recovery was attempted.
            succeeded: Whether recovery resolved the violation.
        """
        with self._lock:
            if attempted:
                self._recovery_attempts += 1
                if succeeded:
                    self._recovery_successes += 1

    def reset(self) -> None:
        """Reset monitor for a new session. Thread-safe. (RF-26)"""
        with self._lock:
            self._compliance = ComplianceTracker()
            self._drift = DriftTracker(
                config=self._contract.drift if self._contract.drift else None
            )
            self._total_hard_violations = 0
            self._total_soft_violations = 0
            self._total_events = 0
            self._recovery_attempts = 0
            self._recovery_successes = 0

    def check_preconditions(
        self, state: dict[str, Any]
    ) -> PreconditionCheckResult:
        """Check all preconditions — patent §3.2 step 2. Thread-safe."""
        results = evaluate_preconditions(self._contract, state)
        failed = [r.name for r in results if not r.satisfied]
        return PreconditionCheckResult(
            all_met=len(failed) == 0,
            failed_names=failed,
        )

    def session_summary(self) -> SessionSummary:
        """Compute session-level aggregates + Θ — patent §3.2 step 8. Thread-safe."""
        with self._lock:
            return self._session_summary_internal()

    def _session_summary_internal(self) -> SessionSummary:
        """Internal summary logic (called under lock)."""
        # If no recovery was needed, treat as perfect (1.0).
        # If recoveries were attempted, compute actual rate.
        if self._recovery_attempts > 0:
            recovery_rate = self._recovery_successes / self._recovery_attempts
        elif self._total_soft_violations == 0:
            recovery_rate = 1.0  # No violations → nothing to recover → perfect
        else:
            recovery_rate = 0.0  # Violations occurred but no recovery attempted

        # Mean compliance = average of hard and soft
        mean_compliance = (
            self._compliance.mean_c_hard + self._compliance.mean_c_soft
        ) / 2.0

        theta = compute_theta(
            c_bar=mean_compliance,
            d_bar=self._drift.mean_drift,
            events=self._total_events,
            recovery_rate=recovery_rate,
        )

        return SessionSummary(
            turn_count=self._compliance.turn_count,
            total_hard_violations=self._total_hard_violations,
            total_soft_violations=self._total_soft_violations,
            total_events=self._total_events,
            mean_c_hard=self._compliance.mean_c_hard,
            mean_c_soft=self._compliance.mean_c_soft,
            mean_drift=self._drift.mean_drift,
            recovery_rate=recovery_rate,
            theta=theta,
        )
