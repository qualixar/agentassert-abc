# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Generic framework-agnostic adapter.

Works with ANY agent that produces dict output. No framework dependency required.
Internally delegates to SessionMonitor for stateful monitoring.

Patent §3.3: The ContractMiddleware wraps any agent graph and intercepts
state transitions. When a transition occurs, it extracts relevant fields,
evaluates them against the contract, and either allows (compliant),
triggers recovery (soft violation), or raises ContractBreachError (hard violation).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from agentassert_abc.exceptions import ContractBreachError, StateExtractionError
from agentassert_abc.monitor.session import SessionMonitor

if TYPE_CHECKING:
    from agentassert_abc.models import ContractSpec
    from agentassert_abc.monitor.models import SessionSummary, StepResult


class GenericAdapter:
    """Framework-agnostic adapter. Works with any agent that produces dict output.

    Usage:
        adapter = GenericAdapter(contract)
        result = adapter.check(agent_output)       # Evaluate, no raise
        result = adapter.check_and_raise(output)    # Evaluate, raise on hard breach
        summary = adapter.session_summary()          # Aggregated theta + metrics
    """

    def __init__(self, contract: ContractSpec) -> None:
        self._contract = contract
        self._monitor = SessionMonitor(contract)
        self._lock = threading.Lock()  # C-03: Thread safety

    def extract_state(self, output: Any) -> dict[str, Any]:
        """Extract state from output. For dicts, returns as-is.

        Args:
            output: Agent output — must be a dict.

        Returns:
            The output dict unchanged.

        Raises:
            TypeError: If output is not a dict.
        """
        if not isinstance(output, dict):
            msg = (
                f"GenericAdapter expects dict output, got {type(output).__name__}. "
                "Use a framework-specific adapter for non-dict outputs."
            )
            raise StateExtractionError(msg)
        return dict(output)  # Immutable copy

    def check(self, agent_output: dict[str, Any]) -> StepResult:
        """Evaluate agent output against the contract.

        Delegates to SessionMonitor.step() for stateful tracking.
        Does NOT raise on violations — use check_and_raise() for that.

        Args:
            agent_output: Flat dict of agent state fields.

        Returns:
            StepResult with violation counts and drift score.
        """
        state = self.extract_state(agent_output)
        with self._lock:
            return self._monitor.step(state)

    def check_and_raise(self, agent_output: dict[str, Any]) -> StepResult:
        """Evaluate and raise ContractBreachError on hard violations.

        Patent §3.3: "hard violation → raises ContractBreachError"

        Args:
            agent_output: Flat dict of agent state fields.

        Returns:
            StepResult if compliant or only soft violations.

        Raises:
            ContractBreachError: If any hard constraint is violated.
        """
        result = self.check(agent_output)

        if result.hard_violations > 0:
            # CRITICAL fix: Use only hard violation names, not all violated_names
            violated = ", ".join(result.violated_hard_names)
            msg = (
                f"Hard contract breach: {result.hard_violations} violation(s) "
                f"[{violated}]"
            )
            raise ContractBreachError(msg)

        return result

    def session_summary(self) -> SessionSummary:
        """Get aggregated session metrics including theta.

        Returns:
            SessionSummary with turn_count, violations, drift, and theta.
        """
        with self._lock:
            return self._monitor.session_summary()
