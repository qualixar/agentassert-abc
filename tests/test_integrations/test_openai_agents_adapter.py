# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for OpenAI Agents SDK adapter — output guardrails and run hooks.

Uses mock objects to avoid requiring openai-agents as a test dependency.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from agentassert_abc.dsl.parser import loads_contract


def _make_contract():
    return loads_contract("""
contractspec: "0.1"
kind: agent
name: test-openai
description: OpenAI Agents SDK adapter test contract
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      description: No PII exposure
      check:
        field: output.pii_detected
        equals: false
    - name: no-harmful-content
      description: No harmful content
      check:
        field: output.harmful_content
        equals: false
  soft:
    - name: helpfulness
      description: Response helpfulness
      check:
        field: output.helpfulness_score
        gte: 0.6
      recovery: improve-response
      recovery_window: 2
recovery:
  strategies:
    - name: improve-response
      type: inject_correction
""")


@dataclass
class MockGuardrailFunctionOutput:
    output_info: object = None
    tripwire_triggered: bool = False


class MockOutputGuardrail:
    def __init__(self, guardrail_function=None, name=None):
        self.guardrail_function = guardrail_function
        self.name = name


class MockInputGuardrail:
    def __init__(self, guardrail_function=None, name=None):
        self.guardrail_function = guardrail_function
        self.name = name


class MockRunContextWrapper:
    def __init__(self):
        self.context = None


class MockAgent:
    def __init__(self, name="test-agent"):
        self.name = name


class MockRunHooksBase:
    """Mock base class for RunHooks."""

    async def on_agent_start(self, context, agent):
        pass

    async def on_agent_end(self, context, agent, output):
        pass

    async def on_llm_start(self, context, agent):
        pass

    async def on_llm_end(self, context, agent):
        pass

    async def on_tool_start(self, context, agent, tool):
        pass

    async def on_tool_end(self, context, agent, tool, result):
        pass

    async def on_handoff(self, context, agent, source):
        pass


@pytest.fixture(autouse=True)
def _mock_openai_agents():
    """Mock openai-agents SDK imports."""
    mock_agents = ModuleType("agents")
    mock_lifecycle = ModuleType("agents.lifecycle")

    mock_agents.Agent = MockAgent  # type: ignore[attr-defined]
    mock_agents.GuardrailFunctionOutput = MockGuardrailFunctionOutput  # type: ignore[attr-defined]
    mock_agents.OutputGuardrail = MockOutputGuardrail  # type: ignore[attr-defined]
    mock_agents.InputGuardrail = MockInputGuardrail  # type: ignore[attr-defined]
    mock_agents.RunContextWrapper = MockRunContextWrapper  # type: ignore[attr-defined]
    mock_lifecycle.RunHooksBase = MockRunHooksBase  # type: ignore[attr-defined]

    sys.modules["agents"] = mock_agents
    sys.modules["agents.lifecycle"] = mock_lifecycle

    yield

    sys.modules.pop("agents", None)
    sys.modules.pop("agents.lifecycle", None)
    mods_to_remove = [
        k for k in sys.modules
        if k.startswith("agentassert_abc.integrations.openai_agents")
    ]
    for mod in mods_to_remove:
        sys.modules.pop(mod, None)


class TestOpenAIAdapterOutputGuardrail:
    """Test output guardrail creation and behavior."""

    def test_guardrail_created(self) -> None:
        """Adapter creates an OutputGuardrail instance."""
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        guardrail = adapter.output_guardrail
        assert guardrail is not None
        assert guardrail.name == "agentassert-contract-guardrail"

    def test_guardrail_compliant_no_tripwire(self) -> None:
        """Compliant output -> tripwire_triggered=False."""
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        guardrail = adapter.output_guardrail
        ctx = MockRunContextWrapper()
        agent = MockAgent()

        output = {
            "output.pii_detected": False,
            "output.harmful_content": False,
            "output.helpfulness_score": 0.8,
        }

        result = asyncio.run(
            guardrail.guardrail_function(ctx, agent, output)
        )

        assert result.tripwire_triggered is False
        assert result.output_info["hard_violations"] == 0

    def test_guardrail_hard_violation_trips(self) -> None:
        """Hard violation -> tripwire_triggered=True."""
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        guardrail = adapter.output_guardrail
        ctx = MockRunContextWrapper()
        agent = MockAgent()

        output = {
            "output.pii_detected": True,  # VIOLATION
            "output.harmful_content": False,
            "output.helpfulness_score": 0.8,
        }

        result = asyncio.run(
            guardrail.guardrail_function(ctx, agent, output)
        )

        assert result.tripwire_triggered is True
        assert result.output_info["hard_violations"] == 1
        assert "no-pii" in result.output_info["violated_names"]

    def test_guardrail_soft_violation_no_tripwire(self) -> None:
        """Soft violation -> tripwire NOT triggered."""
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        guardrail = adapter.output_guardrail
        ctx = MockRunContextWrapper()
        agent = MockAgent()

        output = {
            "output.pii_detected": False,
            "output.harmful_content": False,
            "output.helpfulness_score": 0.3,  # Soft violation
        }

        result = asyncio.run(
            guardrail.guardrail_function(ctx, agent, output)
        )

        assert result.tripwire_triggered is False
        assert result.output_info["soft_violations"] == 1


class TestOpenAIAdapterExtractState:
    """Test state extraction from different output types."""

    def test_extract_from_dict(self) -> None:
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        state = adapter.extract_state({"key": "value"})
        assert state == {"key": "value"}

    def test_extract_from_pydantic(self) -> None:
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        mock_model = MagicMock()
        mock_model.model_dump.return_value = {
            "pii_detected": False,
            "harmful_content": False,
        }

        state = adapter.extract_state(mock_model)
        assert state["output.pii_detected"] is False
        assert state["output.harmful_content"] is False

    def test_extract_from_string(self) -> None:
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        state = adapter.extract_state("plain text output")
        assert state["output.raw"] == "plain text output"


class TestOpenAIAdapterRunHooks:
    """Test run hooks for monitoring."""

    def test_run_hooks_created(self) -> None:
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        hooks = adapter.run_hooks
        assert hooks is not None
        assert hasattr(hooks, "on_agent_end")

    def test_run_hooks_on_agent_end(self) -> None:
        """on_agent_end records metrics."""
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        hooks = adapter.run_hooks
        ctx = MagicMock()
        agent = MockAgent()

        output = {
            "output.pii_detected": False,
            "output.harmful_content": False,
            "output.helpfulness_score": 0.8,
        }

        asyncio.run(
            hooks.on_agent_end(ctx, agent, output)
        )

        summary = adapter.session_summary()
        assert summary.turn_count == 1
        assert summary.total_hard_violations == 0


class TestOpenAIAdapterSessionMetrics:
    """Test session-level metrics."""

    def test_session_summary_after_checks(self) -> None:
        from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

        contract = _make_contract()
        adapter = OpenAIAgentsAdapter(contract)

        compliant = {
            "output.pii_detected": False,
            "output.harmful_content": False,
            "output.helpfulness_score": 0.8,
        }
        adapter.check(compliant)
        adapter.check(compliant)
        adapter.check(compliant)

        summary = adapter.session_summary()
        assert summary.turn_count == 3
        assert summary.total_hard_violations == 0
        assert summary.theta >= 0.90
