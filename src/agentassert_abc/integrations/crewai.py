# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""CrewAI adapter — validate task output via guardrails.

Provides a guardrail function compatible with CrewAI's Task guardrail system.
When a task completes, the guardrail evaluates the output against a ContractSpec.
Hard violations reject the output (triggering retry), soft violations are logged.

Patent §3.3: ContractMiddleware wraps any agent graph and intercepts
state transitions.

Usage:
    from agentassert_abc.integrations.crewai import CrewAIAdapter

    adapter = CrewAIAdapter(contract)

    # Use as a task guardrail (CrewAI retries on failure)
    task = Task(
        description="...",
        agent=my_agent,
        guardrail=adapter.guardrail,
        guardrail_max_retries=3,
    )

    # Or use as a callback (monitor only, no retry)
    task = Task(
        description="...",
        agent=my_agent,
        callback=adapter.callback,
    )

    # Get session metrics after crew.kickoff()
    summary = adapter.session_summary()

Requires: crewai (pip install crewai)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from agentassert_abc.monitor.session import SessionMonitor

if TYPE_CHECKING:
    from agentassert_abc.models import ContractSpec
    from agentassert_abc.monitor.models import SessionSummary, StepResult

try:
    from crewai import TaskOutput  # noqa: F401

    _HAS_CREWAI = True
except ImportError:
    _HAS_CREWAI = False


def _require_crewai() -> None:
    if not _HAS_CREWAI:
        msg = (
            "CrewAI is required for this adapter. "
            "Install it with: pip install crewai"
        )
        raise ImportError(msg)


class CrewAIAdapter:
    """Adapter that monitors CrewAI task outputs against a ContractSpec.

    Provides two integration modes:
    1. guardrail: Rejects output on hard violations (CrewAI retries)
    2. callback: Logs all violations without rejecting

    Both modes track session-level metrics (drift, theta, recovery).
    Thread-safe (C-03).
    """

    def __init__(self, contract: ContractSpec) -> None:
        _require_crewai()
        self._contract = contract
        self._monitor = SessionMonitor(contract)
        self._lock = threading.Lock()  # C-03: Thread safety

    def guardrail(self, task_output: Any) -> tuple[bool, Any]:
        """CrewAI-compatible guardrail function.

        Evaluates task output against the contract.
        Returns (False, error_msg) on hard violations — CrewAI retries.
        Returns (True, task_output) when compliant or only soft violations.

        H-05: Returns the original task_output on success (not raw string)
        to preserve Pydantic/JSON structured output.

        Args:
            task_output: CrewAI TaskOutput object.

        Returns:
            Tuple of (is_valid, task_output_or_error_message).
        """
        state = self.extract_state(task_output)
        with self._lock:
            step_result = self._monitor.step(state)

        if step_result.hard_violations > 0:
            violated = ", ".join(step_result.violated_names)
            error_msg = (
                f"Contract breach: {step_result.hard_violations} hard "
                f"violation(s) [{violated}]. Please fix and retry."
            )
            return (False, error_msg)

        # H-05: Return original task_output to preserve structured output
        return (True, task_output)

    def callback(self, task_output: Any) -> None:
        """CrewAI-compatible callback function (monitor only).

        Evaluates task output and records metrics, but does NOT reject.
        Use this when you want monitoring without blocking.

        Args:
            task_output: CrewAI TaskOutput object.
        """
        state = self.extract_state(task_output)
        with self._lock:
            self._monitor.step(state)

    def check(self, agent_output: Any) -> StepResult:
        """Evaluate a TaskOutput without raising or rejecting.

        Args:
            agent_output: CrewAI TaskOutput object or plain dict.

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
        """Extract a flat state dict from CrewAI TaskOutput.

        Handles three output formats:
        1. Pydantic model → model_dump() with output. prefix
        2. JSON dict → with output. prefix (L-04: skip empty dicts)
        3. Raw string → {"output.raw": raw_text}

        Also extracts metadata (agent name, message count — M-06).

        Args:
            output: CrewAI TaskOutput or plain dict.

        Returns:
            Flat dict suitable for constraint evaluation.
        """
        if isinstance(output, dict):
            return dict(output)

        state: dict[str, Any] = {}

        # Extract structured output
        if hasattr(output, "pydantic") and output.pydantic is not None:
            pydantic_data = output.pydantic.model_dump()
            for key, value in pydantic_data.items():
                state[f"output.{key}"] = value
        elif (
            hasattr(output, "json_dict")
            and output.json_dict is not None
            and output.json_dict  # L-04: Skip empty dict
        ):
            for key, value in output.json_dict.items():
                state[f"output.{key}"] = value
        elif hasattr(output, "raw"):
            state["output.raw"] = output.raw

        # Extract metadata
        if hasattr(output, "agent"):
            state["session.agent_name"] = output.agent

        # M-06: Extract message count from LLM trace
        if hasattr(output, "messages") and output.messages:
            state["session.message_count"] = len(output.messages)

        return state
