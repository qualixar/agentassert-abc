# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for GenericAdapter — framework-agnostic contract middleware.

Patent §3.3: ContractMiddleware wraps any agent, intercepts state transitions,
evaluates against the contract, and either allows, triggers recovery, or raises.

TDD: RED phase — these tests are written BEFORE the implementation.
"""

from pathlib import Path

import pytest

from agentassert_abc.dsl.parser import load_contract, loads_contract
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.integrations.base import AgentAdapter
from agentassert_abc.integrations.generic import GenericAdapter

FIXTURES = Path(__file__).parent.parent / "test_dsl" / "fixtures"


class TestGenericAdapterCompliant:
    """Pass compliant state -> no violations."""

    def test_generic_adapter_compliant(self) -> None:
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        adapter = GenericAdapter(contract)
        result = adapter.check({"output.pii": False})

        assert result.hard_violations == 0
        assert result.soft_violations == 0
        assert result.recovery_needed is False


class TestGenericAdapterHardViolation:
    """Pass violating state -> hard violation count > 0."""

    def test_generic_adapter_hard_violation(self) -> None:
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        adapter = GenericAdapter(contract)
        result = adapter.check({"output.pii": True})

        assert result.hard_violations == 1
        assert "no-pii" in result.violated_names


class TestGenericAdapterRaisesOnBreach:
    """check_and_raise() raises ContractBreachError on hard violations."""

    def test_generic_adapter_raises_on_breach(self) -> None:
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        adapter = GenericAdapter(contract)

        with pytest.raises(ContractBreachError, match="no-pii"):
            adapter.check_and_raise({"output.pii": True})

    def test_check_and_raise_compliant_no_error(self) -> None:
        """check_and_raise on compliant state should return StepResult, not raise."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        adapter = GenericAdapter(contract)
        result = adapter.check_and_raise({"output.pii": False})

        assert result.hard_violations == 0

    def test_check_and_raise_soft_violation_no_error(self) -> None:
        """Soft violations do NOT raise — only hard violations do."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  soft:
    - name: tone
      check:
        field: output.tone
        gte: 0.7
      recovery: fix
recovery:
  strategies:
    - name: fix
      type: inject_correction
""")
        adapter = GenericAdapter(contract)
        result = adapter.check_and_raise({"output.tone": 0.3})

        assert result.soft_violations == 1
        assert result.recovery_needed is True
        # No error raised — soft violations are recoverable


class TestGenericAdapterSessionSummary:
    """After multiple checks, summary has correct theta."""

    def test_generic_adapter_session_summary(self) -> None:
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: check-x
      check:
        field: x
        equals: true
""")
        adapter = GenericAdapter(contract)

        # 2 compliant, 1 violation
        adapter.check({"x": True})
        adapter.check({"x": True})
        adapter.check({"x": False})

        summary = adapter.session_summary()
        assert summary.turn_count == 3
        assert summary.total_hard_violations == 1
        assert 0 < summary.theta < 1.0

    def test_perfect_session_high_theta(self) -> None:
        """All compliant turns -> theta near 1.0."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test-adapter
description: adapter test
version: "1.0.0"
invariants:
  hard:
    - name: check-x
      check:
        field: x
        equals: true
""")
        adapter = GenericAdapter(contract)
        adapter.check({"x": True})
        adapter.check({"x": True})
        adapter.check({"x": True})

        summary = adapter.session_summary()
        assert summary.theta >= 0.90


class TestGenericAdapterEcommerce:
    """Integration test with the patent e-commerce contract."""

    def test_generic_adapter_with_ecommerce_contract(self) -> None:
        contract = load_contract(FIXTURES / "ecommerce-product-recommendation.yaml")
        adapter = GenericAdapter(contract)

        # Compliant state matching the e-commerce contract
        state = {
            "output.competitor_reference_detected": False,
            "output.unverified_availability_claim": False,
            "output.unauthorized_discount_offered": False,
            "output.pii_detected": False,
            "output.sponsored_items_disclosed": True,
            "output.undisclosed_dynamic_pricing": False,
            "output.brand_tone_score": 0.85,
            "output.recommendation_relevance_score": 0.9,
            "output.completeness_score": 0.8,
            "output.upsell_count": 1,
            "response.latency_ms": 1200,
            "output.customer_satisfaction_score": 0.7,
            "tools.all_calls_authorized": True,
            "session.total_tokens": 5000,
            "session.total_cost_usd": 0.50,
        }
        result = adapter.check(state)

        assert result.hard_violations == 0
        assert result.soft_violations == 0

        summary = adapter.session_summary()
        assert summary.theta >= 0.90

    def test_ecommerce_pii_violation_raises(self) -> None:
        """PII detected in e-commerce contract -> hard violation -> raises."""
        contract = load_contract(FIXTURES / "ecommerce-product-recommendation.yaml")
        adapter = GenericAdapter(contract)

        state = {
            "output.competitor_reference_detected": False,
            "output.unverified_availability_claim": False,
            "output.unauthorized_discount_offered": False,
            "output.pii_detected": True,  # <-- violation
            "output.sponsored_items_disclosed": True,
            "output.undisclosed_dynamic_pricing": False,
            "output.brand_tone_score": 0.85,
            "output.recommendation_relevance_score": 0.9,
            "output.completeness_score": 0.8,
            "output.upsell_count": 1,
            "response.latency_ms": 1200,
            "output.customer_satisfaction_score": 0.7,
            "tools.all_calls_authorized": True,
            "session.total_tokens": 5000,
            "session.total_cost_usd": 0.50,
        }

        with pytest.raises(ContractBreachError):
            adapter.check_and_raise(state)


class TestBaseAdapterProtocol:
    """Verify GenericAdapter satisfies the AgentAdapter protocol."""

    def test_generic_adapter_is_agent_adapter(self) -> None:
        """GenericAdapter should be recognized as an AgentAdapter."""
        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        adapter = GenericAdapter(contract)
        # Protocol structural check — isinstance works with runtime_checkable
        assert isinstance(adapter, AgentAdapter)
