# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for SessionMonitor — patent §3.1 Layer 4, §3.2 data flow.

SessionMonitor.step(state) → evaluate → record metrics → check recovery → StepResult.
"""

from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "test_dsl" / "fixtures"


class TestSessionMonitorBasic:
    """Core step() loop."""

    def test_step_compliant_state(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        monitor = SessionMonitor(contract)
        result = monitor.step({"output.pii": False})
        assert result.hard_violations == 0
        assert result.soft_violations == 0
        assert result.drift_score >= 0.0

    def test_step_hard_violation(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      check:
        field: output.pii
        equals: false
""")
        monitor = SessionMonitor(contract)
        result = monitor.step({"output.pii": True})
        assert result.hard_violations == 1
        assert result.violated_names == ["no-pii"]

    def test_step_soft_violation(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
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
        monitor = SessionMonitor(contract)
        result = monitor.step({"output.tone": 0.3})
        assert result.soft_violations == 1
        assert result.recovery_needed is True

    def test_multiple_steps_track_metrics(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: check
      check:
        field: x
        equals: true
""")
        monitor = SessionMonitor(contract)
        monitor.step({"x": True})
        monitor.step({"x": True})
        monitor.step({"x": False})

        summary = monitor.session_summary()
        assert summary.turn_count == 3
        assert summary.total_hard_violations == 1
        assert 0 < summary.theta < 1.0


class TestSessionSummary:
    """session_summary() produces Theta and aggregates."""

    def test_perfect_session_theta(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: check
      check:
        field: x
        equals: true
""")
        monitor = SessionMonitor(contract)
        monitor.step({"x": True})
        monitor.step({"x": True})

        summary = monitor.session_summary()
        assert summary.theta >= 0.90  # Deployment ready


class TestCheckPreconditions:
    """check_preconditions() from patent §3.2 step 2."""

    def test_preconditions_pass(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: session-ok
    check:
      field: session.active
      equals: true
""")
        monitor = SessionMonitor(contract)
        result = monitor.check_preconditions({"session.active": True})
        assert result.all_met is True

    def test_preconditions_fail(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: session-ok
    check:
      field: session.active
      equals: true
""")
        monitor = SessionMonitor(contract)
        result = monitor.check_preconditions({"session.active": False})
        assert result.all_met is False
        assert "session-ok" in result.failed_names


class TestEcommerceMonitor:
    """Integration: monitor the patent e-commerce contract."""

    def test_ecommerce_compliant_session(self) -> None:
        from agentassert_abc.dsl.parser import load_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = load_contract(FIXTURES / "ecommerce-product-recommendation.yaml")
        monitor = SessionMonitor(contract)

        # Preconditions
        pre = monitor.check_preconditions({
            "session.customer_identified": True,
            "system.catalog_service_status": "available",
            "context.promotions_loaded": True,
        })
        assert pre.all_met

        # Compliant turn
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
        result = monitor.step(state)
        assert result.hard_violations == 0
        assert result.soft_violations == 0

        summary = monitor.session_summary()
        assert summary.theta >= 0.90
