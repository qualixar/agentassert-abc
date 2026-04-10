# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""C-12: _MonitoredGraph tests — wrap_graph invoke with compliance and violations.

Uses the same langgraph mock pattern as test_langgraph_adapter.py.
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
name: test-monitored-graph
description: MonitoredGraph test contract
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

    mods_to_remove = [
        k
        for k in sys.modules
        if k.startswith("agentassert_abc.integrations.langgraph")
    ]
    for mod in mods_to_remove:
        sys.modules.pop(mod, None)


class TestMonitoredGraphInvoke:
    """Test _MonitoredGraph.invoke() behavior."""

    def test_invoke_compliant_no_raise(self) -> None:
        """Compliant state from graph.invoke() passes without raising."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.85,
        }

        monitored = adapter.wrap_graph(mock_compiled)
        result = monitored.invoke({"input": "hello"})

        assert result["output.pii_detected"] is False
        assert result["output.tone_score"] == 0.85

    def test_invoke_hard_violation_raises(self) -> None:
        """Hard violation in graph output raises ContractBreachError."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "output.pii_detected": True,  # HARD VIOLATION
            "output.competitor_reference": False,
            "output.tone_score": 0.85,
        }

        monitored = adapter.wrap_graph(mock_compiled)

        with pytest.raises(ContractBreachError, match="no-pii"):
            monitored.invoke({"input": "hello"})

    def test_invoke_raise_on_hard_false(self) -> None:
        """raise_on_hard=False returns result even on violation."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "output.pii_detected": True,  # HARD VIOLATION
            "output.competitor_reference": False,
            "output.tone_score": 0.85,
        }

        monitored = adapter.wrap_graph(mock_compiled)
        result = monitored.invoke({"input": "hello"}, raise_on_hard=False)

        assert result["output.pii_detected"] is True

    def test_invoke_soft_violation_no_raise(self) -> None:
        """Soft violations do NOT raise — only tracked."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "output.pii_detected": False,
            "output.competitor_reference": False,
            "output.tone_score": 0.3,  # SOFT VIOLATION
        }

        monitored = adapter.wrap_graph(mock_compiled)
        result = monitored.invoke({"input": "hello"})

        assert result["output.tone_score"] == 0.3


class TestMonitoredGraphGetattr:
    """Test __getattr__ proxying to underlying graph."""

    def test_getattr_proxies_to_graph(self) -> None:
        """Accessing unknown attrs proxies to the underlying compiled graph."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.get_state.return_value = {"some": "state"}
        mock_compiled.nodes = ["node_a", "node_b"]

        monitored = adapter.wrap_graph(mock_compiled)

        # Access proxied attributes
        assert monitored.nodes == ["node_a", "node_b"]
        assert monitored.get_state() == {"some": "state"}

    def test_getattr_stream_proxied(self) -> None:
        """stream() is proxied without monitoring (M-11)."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(
            [{"step": 1}, {"step": 2}]
        )

        monitored = adapter.wrap_graph(mock_compiled)
        chunks = list(monitored.stream({"input": "hello"}))

        assert len(chunks) == 2
        mock_compiled.stream.assert_called_once()

    def test_invoke_non_dict_result_no_crash(self) -> None:
        """If graph.invoke returns non-dict, no crash (no monitoring)."""
        from agentassert_abc.integrations.langgraph import LangGraphAdapter

        contract = _make_contract()
        adapter = LangGraphAdapter(contract)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = "not a dict"

        monitored = adapter.wrap_graph(mock_compiled)
        result = monitored.invoke({"input": "hello"})

        assert result == "not a dict"
