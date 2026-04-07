# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Edge case tests for SessionMonitor — 100% coverage."""


class TestNoConstraintsContract:
    """Line 77: contract with zero constraints → c_total=1.0."""

    def test_step_no_constraints(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.monitor.session import SessionMonitor

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: empty
description: no constraints
version: "1.0.0"
""")
        monitor = SessionMonitor(contract)
        result = monitor.step({})
        assert result.hard_violations == 0
        assert result.drift_score == 0.0


class TestRecoveryRatePaths:
    """Lines 122, 126: recovery rate computation paths."""

    def test_soft_violations_no_recovery_attempted(self) -> None:
        """Line 126: violations occurred but no recovery was attempted."""
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
        field: x
        gte: 0.7
      recovery: fix
recovery:
  strategies:
    - name: fix
      type: inject_correction
""")
        monitor = SessionMonitor(contract)
        monitor.step({"x": 0.3})  # Soft violation, but no external recovery call

        summary = monitor.session_summary()
        # Violations occurred but recovery_attempts=0 → rate=0.0
        assert summary.recovery_rate == 0.0
        assert summary.total_soft_violations == 1

    def test_recovery_attempted_and_succeeded(self) -> None:
        """Line 122: recovery attempts > 0 → compute actual rate."""
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
        field: x
        gte: 0.7
      recovery: fix
recovery:
  strategies:
    - name: fix
      type: inject_correction
""")
        monitor = SessionMonitor(contract)
        monitor.step({"x": 0.3})  # Soft violation

        # L-13: Use public API instead of private attribute mutation
        monitor.record_recovery(attempted=True, succeeded=True)

        summary = monitor.session_summary()
        assert summary.recovery_rate == 1.0
