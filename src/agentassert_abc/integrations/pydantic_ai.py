# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Pydantic AI adapter — guardrail integration for Pydantic AI agents.

Wraps Pydantic AI agent runs with contract monitoring. Evaluates the
agent's final output against a ContractSpec after each run completes.

Patent §3.3: ContractMiddleware wraps any agent — PydanticAI.
Phase 7 — Integration & Marketplace → Pydantic AI Adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from agentassert_abc.integrations.base import AgentAdapter
from agentassert_abc.models import ContractSpec  # noqa: TC001
from agentassert_abc.monitor.session import SessionMonitor

_logger = logging.getLogger(__name__)


class PydanticAIAdapter(AgentAdapter):
    """Contract monitor for Pydantic AI agents.

    Injects constraint evaluation after each agent run. Works with both
    synchronous `agent.run_sync()` and async `agent.run()` patterns.

    Usage:
        import agentassert_abc as aa
        from pydantic_ai import Agent

        contract = aa.load("contract.yaml")
        adapter = PydanticAIAdapter(contract)
        agent = Agent("openai:gpt-5.2", system_prompt="...")

        # Run with contract monitoring
        result = adapter.guard(agent.run_sync("user query"))
        # result is the original agent result; violations are logged
        # and accessible via adapter.session_summary()
    """

    def __init__(self, contract: ContractSpec) -> None:
        self._contract = contract
        self._monitor = SessionMonitor(contract)

    # ------------------------------------------------------------------
    # AgentAdapter protocol
    # ------------------------------------------------------------------

    def check(self, agent_output: Any) -> Any:
        """Evaluate agent output against contract."""
        state = self.extract_state(agent_output)
        return self._monitor.step(state)

    def extract_state(self, output: Any) -> dict[str, Any]:
        """Extract flat state dict from Pydantic AI result.

        Supports: str (raw text), dict, Pydantic model, and
        PydanticAI RunResult objects.
        """
        if isinstance(output, str):
            return {"output.text": output}
        if isinstance(output, dict):
            flat: dict[str, Any] = {}
            for k, v in output.items():
                flat[f"output.{k}"] = v
            return flat
        # Pydantic model
        if hasattr(output, "model_dump"):
            d = output.model_dump()
            return {f"output.{k}": v for k, v in d.items()}
        # PydanticAI RunResult
        if hasattr(output, "data"):
            return self.extract_state(output.data)
        return {"output.raw": str(output)}

    def session_summary(self) -> Any:
        """Return session-level metrics."""
        return self._monitor.session_summary()

    # ------------------------------------------------------------------
    # Pydantic AI specific
    # ------------------------------------------------------------------

    def guard(self, agent_result: Any) -> Any:
        """Run contract checks on agent result and return it unchanged.

        Violations are recorded in the session monitor. Hard violations
        are logged but do NOT raise by default (Pydantic AI agents
        handle errors via their own retry mechanism).
        """
        step_result = self.check(agent_result)
        # Ledger 5c: log hard violations so they are observable even without raising.
        if step_result.hard_violations > 0:
            violated = ", ".join(step_result.violated_names)
            _logger.error(
                "PydanticAIAdapter.guard: %d hard violation(s) detected [%s]",
                step_result.hard_violations,
                violated,
            )
        return agent_result
