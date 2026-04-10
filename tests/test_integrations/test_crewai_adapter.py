# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for CrewAI adapter — task guardrail and callback integration.

Uses mock TaskOutput objects to avoid requiring crewai as a test dependency.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from agentassert_abc.dsl.parser import loads_contract


def _make_contract():
    return loads_contract("""
contractspec: "0.1"
kind: agent
name: test-crewai
description: CrewAI adapter test contract
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      description: No PII in output
      check:
        field: output.pii_detected
        equals: false
    - name: no-false-claims
      description: No false claims
      check:
        field: output.false_claim_detected
        equals: false
  soft:
    - name: quality-score
      description: Output quality
      check:
        field: output.quality_score
        gte: 0.6
      recovery: improve-quality
      recovery_window: 2
recovery:
  strategies:
    - name: improve-quality
      type: inject_correction
""")


class MockTaskOutput:
    """Mock CrewAI TaskOutput for testing."""

    def __init__(
        self,
        raw: str = "test output",
        pydantic: object | None = None,
        json_dict: dict | None = None,
        agent: str = "test-agent",
    ):
        self.raw = raw
        self.pydantic = pydantic
        self.json_dict = json_dict
        self.agent = agent
        self.output_format = "RAW"
        self.description = "test task"
        self.summary = "test"


@pytest.fixture(autouse=True)
def _mock_crewai():
    """Mock crewai imports."""
    mock_crewai = ModuleType("crewai")
    mock_crewai.TaskOutput = MockTaskOutput  # type: ignore[attr-defined]
    sys.modules["crewai"] = mock_crewai

    yield

    sys.modules.pop("crewai", None)
    mods_to_remove = [
        k for k in sys.modules if k.startswith("agentassert_abc.integrations.crewai")
    ]
    for mod in mods_to_remove:
        sys.modules.pop(mod, None)


class TestCrewAIAdapterGuardrail:
    """Test guardrail function (returns tuple for CrewAI retry mechanism)."""

    def test_guardrail_compliant_dict(self) -> None:
        """Dict state passes guardrail -> (True, output)."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        state = {
            "output.pii_detected": False,
            "output.false_claim_detected": False,
            "output.quality_score": 0.8,
        }
        is_valid, output = adapter.guardrail(state)

        assert is_valid is True

    def test_guardrail_hard_violation_rejects(self) -> None:
        """Hard violation -> (False, error_message) for retry."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        state = {
            "output.pii_detected": True,  # VIOLATION
            "output.false_claim_detected": False,
            "output.quality_score": 0.8,
        }
        is_valid, output = adapter.guardrail(state)

        assert is_valid is False
        assert "no-pii" in output

    def test_guardrail_soft_violation_passes(self) -> None:
        """Soft violations do NOT reject — only hard violations reject."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        state = {
            "output.pii_detected": False,
            "output.false_claim_detected": False,
            "output.quality_score": 0.3,  # Soft violation
        }
        is_valid, output = adapter.guardrail(state)

        assert is_valid is True

    def test_guardrail_multiple_hard_violations(self) -> None:
        """Multiple hard violations -> (False, message with count)."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        state = {
            "output.pii_detected": True,  # VIOLATION
            "output.false_claim_detected": True,  # VIOLATION
            "output.quality_score": 0.8,
        }
        is_valid, output = adapter.guardrail(state)

        assert is_valid is False
        assert "2" in output  # 2 violations


class TestCrewAIAdapterWithTaskOutput:
    """Test with mock TaskOutput objects (realistic CrewAI integration)."""

    def test_extract_state_from_json_dict(self) -> None:
        """TaskOutput with json_dict extracts fields with output. prefix."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        task_output = MockTaskOutput(
            json_dict={
                "pii_detected": False,
                "false_claim_detected": False,
                "quality_score": 0.9,
            }
        )

        state = adapter.extract_state(task_output)
        assert state["output.pii_detected"] is False
        assert state["output.quality_score"] == 0.9

    def test_extract_state_from_pydantic(self) -> None:
        """TaskOutput with pydantic model extracts via model_dump()."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        mock_model = MagicMock()
        mock_model.model_dump.return_value = {
            "pii_detected": False,
            "false_claim_detected": False,
            "quality_score": 0.85,
        }

        task_output = MockTaskOutput(pydantic=mock_model)
        state = adapter.extract_state(task_output)

        assert state["output.pii_detected"] is False
        assert state["output.quality_score"] == 0.85

    def test_extract_state_raw_string(self) -> None:
        """TaskOutput with only raw string extracts as output.raw."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        task_output = MockTaskOutput(raw="Here is my response")
        state = adapter.extract_state(task_output)

        assert state["output.raw"] == "Here is my response"

    def test_guardrail_with_task_output_json(self) -> None:
        """Full guardrail flow with TaskOutput json_dict."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        task_output = MockTaskOutput(
            json_dict={
                "pii_detected": False,
                "false_claim_detected": False,
                "quality_score": 0.9,
            }
        )
        is_valid, output = adapter.guardrail(task_output)

        assert is_valid is True


class TestCrewAIAdapterCallback:
    """Test callback function (monitor only, no reject)."""

    def test_callback_records_metrics(self) -> None:
        """Callback logs violations without rejecting."""
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        state = {
            "output.pii_detected": True,  # Hard violation
            "output.false_claim_detected": False,
            "output.quality_score": 0.8,
        }
        adapter.callback(state)  # Should NOT raise

        summary = adapter.session_summary()
        assert summary.total_hard_violations == 1


class TestCrewAIAdapterSessionMetrics:
    """Test session-level metrics."""

    def test_session_summary_tracks_multiple_checks(self) -> None:
        from agentassert_abc.integrations.crewai import CrewAIAdapter

        contract = _make_contract()
        adapter = CrewAIAdapter(contract)

        compliant = {
            "output.pii_detected": False,
            "output.false_claim_detected": False,
            "output.quality_score": 0.8,
        }
        adapter.check(compliant)
        adapter.check(compliant)

        summary = adapter.session_summary()
        assert summary.turn_count == 2
        assert summary.total_hard_violations == 0
        assert summary.theta >= 0.90
