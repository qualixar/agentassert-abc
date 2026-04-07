# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for LangGraph adapter — node interception and contract monitoring.

These tests use mock objects to avoid requiring langgraph as a test dependency.
The adapter's core logic (flatten state, evaluate contract, raise on hard) is
tested against the same contract engine as GenericAdapter.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from agentassert_abc.dsl.parser import loads_contract
from agentassert_abc.exceptions import ContractBreachError


def _make_contract():
    return loads_contract("""
contractspec: "0.1"
kind: agent
name: test-langgraph
description: LangGraph adapter test contract
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      description: No PII in output
      check:
        field: output.pii_detected
        equals: false
    - name: no-competitor
      description: No competitor references
      check:
        field: output.competitor_reference
        equals: false
  soft:
    - name: tone-score
      description: Brand tone quality
      check:
        field: output.tone_score
        gte: 0.7
      recovery: fix-tone
      recovery_window: 2
recovery:
  strategies:
    - name: fix-tone
      type: inject_correction
""")


@pytest.fixture(autouse=True)
def _mock_langgraph():
    """Mock langgraph imports so tests run without langgraph installed."""
    mock_graph = ModuleType("langgraph")
    mock_graph_graph = ModuleType("langgraph.graph")  # type: ignore[assignment]
    mock_types = ModuleType("langgraph.types")  # type: ignore[assignment]

    mock_graph_graph.StateGraph = MagicMock()  # type: ignore[attr-defined]
    mock_graph_graph.START = "__start__"  # type: ignore[attr-defined]
    mock_graph_graph.END = "__end__"  # type: ignore[attr-defined]

    class MockCommand:
        def __init__(self, update=None, goto=None):
            self.update = update or {}
            self.goto = goto

    mock_types.Command = MockCommand  # type: ignore[attr-defined]

    sys.modules["langgraph"] = mock_graph
    sys.modules["langgraph.graph"] = mock_graph_graph
    sys.modules["langgraph.types"] = mock_types

    yield

    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.graph", None)
    sys.modules.pop("langgraph.types", None)

    # Force reimport
    mods_to_remove = [
        k for k in sys.modules if k.startswith("agentassert_abc.integrations.langgraph")
    ]
    for mod in mods_to_remove:
        sys.modules.pop(mod, None)


class TestLangGraphAdapterWrapNode:
    """Test node wrapping and interception."""

    def test_wrap_node_compliant(self) -> None:
        """Wrapped node returns original result on compliant output."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        def my_node(state: dict) -> dict:
            return {
                "output.pii_detected": False,
                "output.competitor_reference": False,
                "output.tone_score": 0.85,
            }

        wrapped = adapter.wrap_node(my_node)
        state = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.9,
        }
        result = wrapped(state)

        assert result["output.pii_detected"] is False
        assert result["output.tone_score"] == 0.85

    def test_wrap_node_hard_violation_raises(self) -> None:
        """Hard violation in node output raises ContractBreachError."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        def bad_node(state: dict) -> dict:
            return {
                "output.pii_detected": True,  # VIOLATION
                "output.competitor_reference": False,
                "output.tone_score": 0.85,
            }

        wrapped = adapter.wrap_node(bad_node)
        state = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.9,
        }

        with pytest.raises(ContractBreachError, match="no-pii"):
            wrapped(state)

    def test_wrap_node_soft_violation_no_raise(self) -> None:
        """Soft violations do NOT raise — only tracked."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        def soft_node(state: dict) -> dict:
            return {
                "output.pii_detected": False,
                "output.competitor_reference": False,
                "output.tone_score": 0.3,  # Soft violation
            }

        wrapped = adapter.wrap_node(soft_node)
        state = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.9,
        }
        result = wrapped(state)

        assert result["output.tone_score"] == 0.3

    def test_wrap_node_raise_disabled(self) -> None:
        """raise_on_hard=False returns result even on hard violations."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        def bad_node(state: dict) -> dict:
            return {
                "output.pii_detected": True,
                "output.competitor_reference": False,
                "output.tone_score": 0.85,
            }

        wrapped = adapter.wrap_node(bad_node, raise_on_hard=False)
        state = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.9,
        }
        result = wrapped(state)

        assert result["output.pii_detected"] is True

    def test_wrap_node_command_return(self) -> None:
        """Node returning Command object is handled correctly."""
        from langgraph.types import Command

        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        def command_node(state: dict) -> object:
            return Command(
                update={
                    "output.pii_detected": False,
                    "output.competitor_reference": False,
                    "output.tone_score": 0.85,
                },
                goto="next_node",
            )

        wrapped = adapter.wrap_node(command_node)
        state = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.9,
        }
        result = wrapped(state)

        assert hasattr(result, "goto")
        assert result.goto == "next_node"


class TestLangGraphAdapterSessionMetrics:
    """Test session-level metrics after multiple checks."""

    def test_session_summary_after_checks(self) -> None:
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        compliant = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.85,
        }
        adapter.check(compliant)
        adapter.check(compliant)
        adapter.check(compliant)

        summary = adapter.session_summary()
        assert summary.turn_count == 3
        assert summary.total_hard_violations == 0
        assert summary.theta >= 0.90


class TestLangGraphAdapterFlattenOutput:
    """Test state flattening logic."""

    def test_flatten_dict(self) -> None:
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        pre = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        merged = LangGraphAdapter._flatten_node_output(pre, update)

        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_flatten_command(self) -> None:
        from langgraph.types import Command

        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        pre = {"a": 1}
        cmd = Command(update={"b": 2}, goto="next")
        merged = LangGraphAdapter._flatten_node_output(pre, cmd)

        assert merged == {"a": 1, "b": 2}
