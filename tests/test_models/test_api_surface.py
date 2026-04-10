# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for top-level API surface — what users actually import."""


class TestLazyImports:
    """Convenience imports via __getattr__."""

    def test_load(self) -> None:
        import agentassert_abc as aa

        assert callable(aa.load)

    def test_loads(self) -> None:
        import agentassert_abc as aa

        contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: api-test
description: test
version: "1.0.0"
""")
        assert contract.name == "api-test"

    def test_session_monitor(self) -> None:
        import agentassert_abc as aa

        contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        monitor = aa.SessionMonitor(contract)
        result = monitor.step({})
        assert result.hard_violations == 0

    def test_compute_theta(self) -> None:
        import agentassert_abc as aa

        theta = aa.compute_theta(c_bar=1.0, d_bar=0.0, events=0, recovery_rate=1.0)
        assert abs(theta - 1.0) < 0.01

    def test_evaluate(self) -> None:
        import agentassert_abc as aa

        contract = aa.loads("""
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
        result = aa.evaluate(contract, {"x": True})
        assert result.c_hard == 1.0

    def test_invalid_attr_raises(self) -> None:
        import pytest

        import agentassert_abc as aa

        with pytest.raises(AttributeError):
            _ = aa.nonexistent_thing  # type: ignore[attr-defined]
