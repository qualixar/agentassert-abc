# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""M-23: Adapter ImportError path tests + additional edge case tests.

Tests that _require_*() functions raise ImportError with helpful messages
when their respective libraries are not installed.

Also includes:
- _numeric() with non-coercible types (H-15)
- Empty json_dict in CrewAI extract_state (L-04)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


class TestRequireLangGraphImportError:
    """Test _require_langgraph() raises ImportError when langgraph absent."""

    def test_require_langgraph_raises(self) -> None:
        """_require_langgraph() raises ImportError with install hint."""
        # Remove any cached langgraph modules
        saved = {}
        keys_to_remove = [
            k for k in sys.modules if k.startswith("langgraph")
        ]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        # Also remove cached adapter module so it re-evaluates _HAS_LANGGRAPH
        adapter_keys = [
            k
            for k in sys.modules
            if k.startswith("agentassert_abc.integrations.langgraph")
        ]
        saved_adapters = {}
        for k in adapter_keys:
            saved_adapters[k] = sys.modules.pop(k)

        try:
            # Re-import the module fresh — _HAS_LANGGRAPH will be False
            import importlib

            mod = importlib.import_module(
                "agentassert_abc.integrations.langgraph"
            )
            # Call the private function directly
            with pytest.raises(ImportError, match="langgraph"):
                mod._require_langgraph()
        finally:
            # Restore modules
            sys.modules.update(saved)
            sys.modules.update(saved_adapters)
            # Clean up so other tests start fresh
            for k in list(sys.modules):
                if k.startswith("agentassert_abc.integrations.langgraph"):
                    sys.modules.pop(k, None)


class TestRequireCrewAIImportError:
    """Test _require_crewai() raises ImportError when crewai absent."""

    def test_require_crewai_raises(self) -> None:
        """_require_crewai() raises ImportError with install hint."""
        saved = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("crewai")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        adapter_keys = [
            k
            for k in sys.modules
            if k.startswith("agentassert_abc.integrations.crewai")
        ]
        saved_adapters = {}
        for k in adapter_keys:
            saved_adapters[k] = sys.modules.pop(k)

        try:
            import importlib

            mod = importlib.import_module(
                "agentassert_abc.integrations.crewai"
            )
            with pytest.raises(ImportError, match="crewai"):
                mod._require_crewai()
        finally:
            sys.modules.update(saved)
            sys.modules.update(saved_adapters)
            for k in list(sys.modules):
                if k.startswith("agentassert_abc.integrations.crewai"):
                    sys.modules.pop(k, None)


class TestRequireOpenAIAgentsImportError:
    """Test _require_openai_agents() raises ImportError when agents absent."""

    def test_require_openai_agents_raises(self) -> None:
        """_require_openai_agents() raises ImportError with install hint."""
        saved = {}
        keys_to_remove = [
            k for k in sys.modules if k.startswith("agents")
        ]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        adapter_keys = [
            k
            for k in sys.modules
            if k.startswith("agentassert_abc.integrations.openai_agents")
        ]
        saved_adapters = {}
        for k in adapter_keys:
            saved_adapters[k] = sys.modules.pop(k)

        try:
            import importlib

            mod = importlib.import_module(
                "agentassert_abc.integrations.openai_agents"
            )
            with pytest.raises(ImportError, match="openai-agents"):
                mod._require_openai_agents()
        finally:
            sys.modules.update(saved)
            sys.modules.update(saved_adapters)
            for k in list(sys.modules):
                if k.startswith(
                    "agentassert_abc.integrations.openai_agents"
                ):
                    sys.modules.pop(k, None)


class TestNumericNonCoercible:
    """Test _numeric() returns None for non-coercible types (H-15)."""

    def test_numeric_none(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric(None) is None

    def test_numeric_dict(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric({}) is None

    def test_numeric_list(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric([]) is None

    def test_numeric_object(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric(object()) is None

    def test_numeric_valid_string(self) -> None:
        """Valid numeric string should coerce."""
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric("3.14") == pytest.approx(3.14)

    def test_numeric_int(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric(42) == 42.0

    def test_numeric_float(self) -> None:
        from agentassert_abc.evaluator.operators import _numeric

        assert _numeric(0.5) == 0.5


class TestCrewAIEmptyJsonDict:
    """Test CrewAI extract_state with empty json_dict falls through to raw (L-04)."""

    @pytest.fixture(autouse=True)
    def _mock_crewai(self):
        """Mock crewai imports."""
        mock_crewai = ModuleType("crewai")

        class MockTaskOutput:
            pass

        mock_crewai.TaskOutput = MockTaskOutput  # type: ignore[attr-defined]
        sys.modules["crewai"] = mock_crewai

        yield

        sys.modules.pop("crewai", None)
        mods_to_remove = [
            k
            for k in sys.modules
            if k.startswith("agentassert_abc.integrations.crewai")
        ]
        for mod in mods_to_remove:
            sys.modules.pop(mod, None)

    def test_empty_json_dict_falls_to_raw(self) -> None:
        """TaskOutput with json_dict={} should fall through to raw."""
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-empty-json
description: test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      description: No PII
      check:
        field: output.raw
        not_contains: "SSN"
""")
        adapter = CrewAIAdapter(contract)

        task_output = MagicMock()
        task_output.pydantic = None
        task_output.json_dict = {}  # Empty dict — should be skipped (L-04)
        task_output.raw = "Here is the response"
        task_output.agent = "test-agent"
        task_output.messages = None

        state = adapter.extract_state(task_output)

        # Should have fallen through to raw, not used empty json_dict
        assert "output.raw" in state
        assert state["output.raw"] == "Here is the response"
