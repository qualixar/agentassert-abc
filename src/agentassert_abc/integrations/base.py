# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Base adapter protocol for framework integrations.

Patent §3.3: ContractMiddleware wraps any agent graph and intercepts
state transitions. All framework adapters share a common monitoring interface.

H-06: Redesigned Protocol to match what adapters actually implement.
The common interface is: check(), session_summary(), extract_state().
Framework-specific methods (wrap_node, guardrail, etc.) are NOT in the Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentassert_abc.monitor.models import StepResult  # noqa: TCH001


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol that all framework adapters satisfy.

    Common monitoring interface:
    1. check() — Evaluate agent output against a contract
    2. session_summary() — Get aggregated session metrics
    3. extract_state() — Normalize framework output to flat dict

    Framework-specific methods (wrap_node, guardrail, output_guardrail, etc.)
    are defined on individual adapter classes, not in this Protocol.
    """

    def check(self, agent_output: Any) -> StepResult:
        """Evaluate agent output against the contract.

        Args:
            agent_output: Framework-specific output (dict, TaskOutput, etc.)

        Returns:
            StepResult with violation counts and drift score.
        """
        ...

    def extract_state(self, output: Any) -> dict[str, Any]:
        """Extract a flat state dict from framework-specific output.

        Args:
            output: Raw output from the agent framework.

        Returns:
            Flat dict suitable for constraint evaluation.
        """
        ...

    def session_summary(self) -> Any:
        """Get aggregated session metrics including theta.

        Returns:
            SessionSummary with turn_count, violations, drift, and theta.
        """
        ...
