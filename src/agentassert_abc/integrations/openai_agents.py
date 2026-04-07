# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""OpenAI Agents SDK adapter — output/input guardrails + monitoring hooks.

Provides three integration modes:
1. Output guardrail: Triggers tripwire on hard violations (stops agent)
2. Input guardrail: Validates input before agent processes it
3. Run hooks: Monitors agent lifecycle without stopping execution

Patent §3.3: ContractMiddleware wraps any agent graph and intercepts
state transitions.

Usage:
    from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(contract)

    # Mode 1: Output guardrail (hard stop on violations)
    agent = Agent(
        name="my-agent",
        output_guardrails=[adapter.output_guardrail],
        input_guardrails=[adapter.input_guardrail],
        output_type=MyOutputModel,
    )

    # Mode 2: Run hooks (monitor without stopping)
    result = await Runner.run(agent, "input", hooks=adapter.run_hooks)

    # Get session metrics
    summary = adapter.session_summary()

Requires: openai-agents (pip install openai-agents)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from agentassert_abc.monitor.session import SessionMonitor

if TYPE_CHECKING:
    from agentassert_abc.models import ContractSpec
    from agentassert_abc.monitor.models import SessionSummary, StepResult

try:
    from agents import (
        GuardrailFunctionOutput,
        InputGuardrail,
        OutputGuardrail,
    )
    from agents.lifecycle import RunHooksBase

    _HAS_OPENAI_AGENTS = True
except ImportError:
    _HAS_OPENAI_AGENTS = False


def _require_openai_agents() -> None:
    if not _HAS_OPENAI_AGENTS:
        msg = (
            "OpenAI Agents SDK is required for this adapter. "
            "Install it with: pip install openai-agents"
        )
        raise ImportError(msg)


class OpenAIAgentsAdapter:
    """Adapter for OpenAI Agents SDK guardrails and run hooks.

    Evaluates agent output against a ContractSpec.
    Hard violations trigger the guardrail tripwire (raises exception).
    All violations are tracked for session-level metrics.

    Thread-safe (C-03).
    """

    def __init__(self, contract: ContractSpec) -> None:
        _require_openai_agents()
        self._contract = contract
        self._monitor = SessionMonitor(contract)
        self._lock = threading.Lock()  # C-03: Thread safety
        self._output_guardrail = self._create_output_guardrail()
        self._input_guardrail = self._create_input_guardrail()
        self._run_hooks = self._create_run_hooks()

    @property
    def output_guardrail(self) -> Any:
        """OutputGuardrail for Agent(output_guardrails=[...])."""
        return self._output_guardrail

    @property
    def input_guardrail(self) -> Any:
        """InputGuardrail for Agent(input_guardrails=[...]). (M-08)"""
        return self._input_guardrail

    @property
    def run_hooks(self) -> Any:
        """RunHooks for Runner.run(hooks=...)."""
        return self._run_hooks

    def check(self, agent_output: Any) -> StepResult:
        """Evaluate an agent output without raising.

        Args:
            agent_output: Agent output (Pydantic model, dict, or string).

        Returns:
            StepResult with violation counts and drift score.
        """
        state = self.extract_state(agent_output)
        with self._lock:
            return self._monitor.step(state)

    def session_summary(self) -> SessionSummary:
        """Get aggregated session metrics including theta."""
        with self._lock:
            return self._monitor.session_summary()

    def extract_state(self, output: Any) -> dict[str, Any]:
        """Extract flat state dict from OpenAI agent output.

        Handles:
        1. Pydantic BaseModel → model_dump() with output. prefix
        2. dict → as-is
        3. string → {"output.raw": text}

        Args:
            output: Agent output in any format.

        Returns:
            Flat dict suitable for constraint evaluation.
        """
        if isinstance(output, dict):
            return dict(output)

        if hasattr(output, "model_dump"):
            data = output.model_dump()
            return {f"output.{k}": v for k, v in data.items()}

        return {"output.raw": str(output)}

    def _create_output_guardrail(self) -> Any:
        """Create an OutputGuardrail that validates against the contract."""
        if not _HAS_OPENAI_AGENTS:
            return None

        monitor = self._monitor
        lock = self._lock
        extract = self.extract_state

        async def _guardrail_fn(
            ctx: Any,
            agent: Any,
            output: Any,
        ) -> GuardrailFunctionOutput:
            state = extract(output)
            with lock:
                step_result = monitor.step(state)

            tripwire = step_result.hard_violations > 0
            info: dict[str, Any] = {
                "hard_violations": step_result.hard_violations,
                "soft_violations": step_result.soft_violations,
                "violated_names": step_result.violated_names,
                "drift_score": step_result.drift_score,
            }

            return GuardrailFunctionOutput(
                output_info=info,
                tripwire_triggered=tripwire,
            )

        return OutputGuardrail(
            guardrail_function=_guardrail_fn,
            name="agentassert-contract-guardrail",
        )

    def _create_input_guardrail(self) -> Any:
        """Create an InputGuardrail for precondition checking. (M-08)"""
        if not _HAS_OPENAI_AGENTS:
            return None

        monitor = self._monitor
        lock = self._lock

        async def _input_guardrail_fn(
            ctx: Any,
            agent: Any,
            input_data: Any,
        ) -> GuardrailFunctionOutput:
            # Check preconditions against input state
            state: dict[str, Any] = {}
            if isinstance(input_data, dict):
                state = dict(input_data)
            elif isinstance(input_data, str):
                state = {"input.raw": input_data}

            with lock:
                pre_result = monitor.check_preconditions(state)

            return GuardrailFunctionOutput(
                output_info={
                    "all_met": pre_result.all_met,
                    "failed": pre_result.failed_names,
                },
                tripwire_triggered=not pre_result.all_met,
            )

        return InputGuardrail(
            guardrail_function=_input_guardrail_fn,
            name="agentassert-precondition-guardrail",
        )

    def _create_run_hooks(self) -> Any:
        """Create RunHooks that monitor agent outputs without stopping."""
        if not _HAS_OPENAI_AGENTS:
            return None

        monitor = self._monitor
        lock = self._lock
        extract = self.extract_state

        class _AgentAssertHooks(RunHooksBase):  # type: ignore[type-arg]
            """Lifecycle hooks for monitoring agent runs."""

            async def on_agent_end(
                self,
                context: Any,
                agent: Any,
                output: Any,
            ) -> None:
                state = extract(output)
                with lock:
                    monitor.step(state)

        return _AgentAssertHooks()
