# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

# ruff: noqa: TC001
"""MCP Server Monitor — behavioral contract enforcement for MCP tool servers.

Monitors MCP tool calls against a ContractSpec,  # noqa: TC001 evaluating both the tool
invocation request (preconditions, governance) and the tool result (invariants).
Integrates with EventBus for violation/recovery/drift events.

Phase 4 — Layer 4: Runtime Monitor → MCP Server Monitor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from agentassert_abc.evaluator.engine import evaluate_check
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.models import (
    ContractSpec,  # noqa: TC001
)
from agentassert_abc.monitor.events import EventBus, MonitorEvent
from agentassert_abc.monitor.session import SessionMonitor


@dataclass(frozen=True)
class ToolCallVerdict:
    """Result of evaluating a single tool call against a contract.

    Attributes:
        allowed: True if all hard constraints and preconditions pass.
        hard_violations: Names of failed hard constraints.
        soft_violations: Names of failed soft constraints.
        recovery_needed: True if any soft constraint is violated.
    """

    allowed: bool
    hard_violations: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    recovery_needed: bool = False

    def __bool__(self) -> bool:
        return self.allowed


class MCPServerMonitor:
    """Monitors MCP tool calls against behavioral contracts.

    Evaluates both pre-invocation (preconditions, governance) and
    post-invocation (invariants) constraints. Supports both hard-fail
    mode (raise on hard violation) and non-blocking observation mode.

    Usage:
        contract = aa.load("mcp-server-contract.yaml")
        monitor = MCPServerMonitor(contract)

        # Before tool execution
        pre = monitor.check_pre_invoke("search", {"query": "..."})
        if not pre.allowed:
            raise RuntimeError(f"Blocked: {pre.hard_violations}")

        # ... execute tool ...

        # After tool execution
        result = monitor.check_post_invoke("search", tool_output_state)
        if result.recovery_needed:
            # Soft violation — caller decides whether to re-execute
            pass
    """

    def __init__(
        self,
        contract: ContractSpec,  # noqa: TC001
        event_bus: EventBus | None = None,
        raise_on_hard: bool = True,
    ) -> None:
        self._contract = contract
        self._session = SessionMonitor(contract)
        self._bus = event_bus or EventBus()
        self._raise_on_hard = raise_on_hard
        self._lock = threading.Lock()

    @property
    def contract(self) -> ContractSpec:
        return self._contract

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def session(self) -> SessionMonitor:
        return self._session

    def check_pre_invoke(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallVerdict:
        """Evaluate contract constraints BEFORE tool execution.

        Checks:
        - Governance constraints (authorization, budget, rate limits)
        - Preconditions (guard clauses for tool invocation)

        Args:
            tool_name: The MCP tool being invoked.
            arguments: Tool arguments as a flat dict.

        Returns:
            ToolCallVerdict — if not allowed, the tool should be blocked.

        Raises:
            ContractBreachError: If raise_on_hard=True and a hard constraint fails.
        """
        state = self._build_state(tool_name, arguments, phase="pre")
        hard_violations: list[str] = []
        soft_violations: list[str] = []

        with self._lock:
            # Governance checks
            if self._contract.governance:
                for gc in self._contract.governance.hard:
                    if not evaluate_check(gc.check, state):
                        hard_violations.append(gc.name)
                for gc in self._contract.governance.soft:
                    if not evaluate_check(gc.check, state):
                        hard_violations.append(gc.name)

            # Precondition checks
            if self._contract.preconditions:
                for pre in self._contract.preconditions:
                    if not evaluate_check(pre.check, state):
                        hard_violations.append(pre.name)

        allowed = len(hard_violations) == 0

        if not allowed and self._raise_on_hard:
            raise ContractBreachError(
                f"Tool '{tool_name}' blocked: {', '.join(hard_violations)}"
            )

        return ToolCallVerdict(
            allowed=allowed,
            hard_violations=hard_violations,
            soft_violations=soft_violations,
            recovery_needed=False,
        )

    def check_post_invoke(
        self, tool_name: str, result_state: dict[str, Any]
    ) -> ToolCallVerdict:
        """Evaluate contract constraints AFTER tool execution.

        Checks invariant constraints (hard and soft) against the tool's output.

        Args:
            tool_name: The MCP tool that was invoked.
            result_state: Flat dict extracted from the tool's result.

        Returns:
            ToolCallVerdict with violation details.
        """
        state = self._build_state(tool_name, result_state, phase="post")
        hard_violations: list[str] = []
        soft_violations: list[str] = []

        with self._lock:
            # Only check invariants (hard + soft); governance was pre-invoke
            if self._contract.invariants:
                for hc in self._contract.invariants.hard:
                    if not evaluate_check(hc.check, state):
                        hard_violations.append(hc.name)
                for sc in self._contract.invariants.soft:
                    if not evaluate_check(sc.check, state):
                        soft_violations.append(sc.name)

        return ToolCallVerdict(
            allowed=len(hard_violations) == 0,
            hard_violations=hard_violations,
            soft_violations=soft_violations,
            recovery_needed=len(soft_violations) > 0,
        )

    def emit(self, event: MonitorEvent) -> None:
        """Publish event to the instrumented EventBus."""
        self._bus.emit(event)

    def session_summary(self) -> Any:
        """Return aggregated session metrics from the underlying SessionMonitor."""
        return self._session.session_summary()

    @staticmethod
    def _build_state(
        tool_name: str, payload: dict[str, Any], *, phase: str
    ) -> dict[str, Any]:
        """Build a namespaced state dict for constraint evaluation.

        Prefixes fields with 'tool.' to match ContractSpec conventions.
        """
        state: dict[str, Any] = {"tool.name": tool_name, "tool.phase": phase}
        for k, v in payload.items():
            state[f"tool.{k}"] = v
        return state
