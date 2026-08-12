# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for PydanticAIAdapter — contract monitoring for Pydantic AI agents.

Invariants pinned here:
  * extract_state() correctly flattens every supported output shape into a
    flat dict of "output.*" keys, without losing information.
  * guard() returns the original agent result unchanged regardless of
    violation outcome (Pydantic AI manages its own retry/error flow).
  * Hard violations are recorded in the session monitor and appear in the
    session summary — they are NOT silently dropped.
  * session_summary() reflects the cumulative metric across multiple guard()
    calls in the same session.
  * The adapter satisfies the AgentAdapter protocol.

These tests specifically target the 0% coverage of the module by exercising
every public method and every branch of extract_state().
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentassert_abc.dsl.parser import loads_contract
from agentassert_abc.integrations.base import AgentAdapter
from agentassert_abc.integrations.pydantic_ai import PydanticAIAdapter

# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


def _make_contract(field: str, value: Any = True) -> Any:
    """Build a minimal contract with one hard constraint on a given field."""
    return loads_contract(f"""
contractspec: "0.1"
kind: agent
name: pydantic-ai-test
description: pydantic ai adapter test
version: "1.0.0"
invariants:
  hard:
    - name: check-field
      check:
        field: {field}
        equals: {str(value).lower()}
""")


def _compliant_contract() -> Any:
    return _make_contract("output.text", True)


def _simple_contract() -> Any:
    """A contract with no hard constraints that always evaluates clean."""
    return loads_contract("""
contractspec: "0.1"
kind: agent
name: pydantic-ai-permissive
description: permissive
version: "1.0.0"
""")


# ---------------------------------------------------------------------------
# extract_state() — output shape flattening
# ---------------------------------------------------------------------------


class TestExtractState:
    def test_str_produces_output_text(self) -> None:
        """Plain string output → {"output.text": the_string}."""
        adapter = PydanticAIAdapter(_simple_contract())
        state = adapter.extract_state("agent said this")
        assert state == {"output.text": "agent said this"}

    def test_dict_prefixes_all_keys(self) -> None:
        """Dict output → every key prefixed with 'output.'."""
        adapter = PydanticAIAdapter(_simple_contract())
        state = adapter.extract_state({"decision": "buy", "confidence": 0.9})
        assert state["output.decision"] == "buy"
        assert state["output.confidence"] == 0.9
        assert len(state) == 2

    def test_empty_dict_produces_no_keys(self) -> None:
        """An empty dict output → empty state (no crash)."""
        adapter = PydanticAIAdapter(_simple_contract())
        assert adapter.extract_state({}) == {}

    def test_pydantic_model_via_model_dump(self) -> None:
        """Pydantic model output → model_dump() keys prefixed with 'output.'."""
        from pydantic import BaseModel

        class AgentResponse(BaseModel):
            safe: bool
            score: float

        adapter = PydanticAIAdapter(_simple_contract())
        state = adapter.extract_state(AgentResponse(safe=True, score=0.95))
        assert state["output.safe"] is True
        assert state["output.score"] == pytest.approx(0.95)

    def test_run_result_via_data_attribute(self) -> None:
        """PydanticAI RunResult-like objects (with .data) recurse into their .data."""
        adapter = PydanticAIAdapter(_simple_contract())
        run_result = MagicMock()
        del run_result.model_dump  # ensure no model_dump attribute
        run_result.data = "the actual answer"
        state = adapter.extract_state(run_result)
        assert state == {"output.text": "the actual answer"}

    def test_unrecognised_type_falls_back_to_output_raw(self) -> None:
        """Unrecognised objects → {"output.raw": str(obj)}.

        This must not raise. The fallback preserves the value so text-matching
        constraints can still run, just over the string representation.
        """
        adapter = PydanticAIAdapter(_simple_contract())

        class Unusual:
            def __repr__(self) -> str:
                return "unusual-repr"

        state = adapter.extract_state(Unusual())
        assert state == {"output.raw": "unusual-repr"}


# ---------------------------------------------------------------------------
# check() — step evaluation
# ---------------------------------------------------------------------------


class TestCheck:
    def test_check_compliant_state_no_violations(self) -> None:
        """check() runs extract_state() first, so input is raw agent output.

        Pass {"safe": True} (raw dict) → extract_state → {"output.safe": True}
        → constraint "output.safe equals true" → 0 violations.
        """
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: safe
      check:
        field: output.safe
        equals: true
""")
        adapter = PydanticAIAdapter(contract)
        # Raw dict: check() calls extract_state() internally, adding "output." prefix.
        result = adapter.check({"safe": True})
        assert result.hard_violations == 0
        assert result.soft_violations == 0

    def test_check_violating_state_records_hard_violation(self) -> None:
        """Violating raw output produces StepResult with hard_violations > 0.

        Pass {"safe": False} → extract_state → {"output.safe": False}
        → constraint "output.safe equals true" → 1 hard violation.
        """
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: safe
      check:
        field: output.safe
        equals: true
""")
        adapter = PydanticAIAdapter(contract)
        result = adapter.check({"safe": False})
        assert result.hard_violations == 1
        assert "safe" in result.violated_names


# ---------------------------------------------------------------------------
# guard() — wraps the full lifecycle
# ---------------------------------------------------------------------------


class TestGuard:
    def test_guard_returns_original_result_unchanged(self) -> None:
        """guard() must return the agent result as-is regardless of violations.

        Pydantic AI handles errors through its own retry/fallback mechanism.
        The adapter's job is to record violations, not to raise or block.
        """
        adapter = PydanticAIAdapter(_simple_contract())
        sentinel = object()
        returned = adapter.guard(sentinel)
        assert returned is sentinel

    def test_guard_with_string_result_passes_through(self) -> None:
        """String agent results are equally passed through unchanged."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: text-check
      check:
        field: output.text
        contains: "answer"
""")
        adapter = PydanticAIAdapter(contract)
        result = adapter.guard("the answer is 42")
        assert result == "the answer is 42"

    def test_guard_hard_violation_does_not_raise(self) -> None:
        """Hard violations in guard() must be LOGGED, never raised.

        Patent §3.3: PydanticAI agents handle errors via their own retry
        mechanism; the adapter must not interrupt that flow.
        """
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: must-be-safe
      check:
        field: output.safe
        equals: true
""")
        adapter = PydanticAIAdapter(contract)
        # This state violates the constraint — but guard() must not raise.
        result = adapter.guard({"safe": False})  # dict → output.safe is False
        assert result == {"safe": False}


# ---------------------------------------------------------------------------
# session_summary() — cumulative metrics
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_session_summary_accumulates_across_guard_calls(self) -> None:
        """session_summary() must reflect cumulative state across multiple guard()s."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: ok
      check:
        field: output.text
        equals: "yes"
""")
        adapter = PydanticAIAdapter(contract)
        adapter.guard("yes")    # compliant
        adapter.guard("yes")    # compliant
        adapter.guard("no")     # violation

        summary = adapter.session_summary()
        assert summary.turn_count == 3
        assert summary.total_hard_violations == 1
        assert 0.0 <= summary.theta <= 1.0

    def test_perfect_session_theta_near_one(self) -> None:
        """All-compliant session must produce theta near 1.0."""
        adapter = PydanticAIAdapter(_simple_contract())
        for _ in range(5):
            adapter.guard("anything")
        summary = adapter.session_summary()
        assert summary.theta >= 0.90


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_pydantic_ai_adapter_is_agent_adapter(self) -> None:
        """PydanticAIAdapter must satisfy the AgentAdapter structural protocol."""
        adapter = PydanticAIAdapter(_simple_contract())
        assert isinstance(adapter, AgentAdapter)
