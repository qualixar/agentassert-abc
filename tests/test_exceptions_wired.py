# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for wired exception raising in v0.3.0."""

import pytest

from agentassert_abc import (
    DriftThresholdError,
    ExprEvaluationError,
    PreconditionFailedError,
    RecoveryFailedError,
)
from agentassert_abc.dsl.parser import loads_contract
from agentassert_abc.monitor.session import SessionMonitor


def test_drift_threshold_error_not_raised_when_below_threshold() -> None:
    """raise_on_drift flag enables check but does not raise for in-range drift."""
    contract = loads_contract("""
    contractspec: "0.1"
    kind: agent
    name: test
    description: test
    version: "1.0.0"
    invariants:
      hard:
        - name: always-true
          check:
            field: x
            equals: 1
    """)
    # With c_hard=1.0, drift score should be ~0, well below 0.3 threshold
    monitor = SessionMonitor(contract, raise_on_drift=True, drift_threshold=0.3)
    result = monitor.step({"x": 1})
    assert result is not None


def test_drift_threshold_error_not_raised_by_default() -> None:
    """By default (raise_on_drift=False) no exception is raised."""
    contract = loads_contract("""
    contractspec: "0.1"
    kind: agent
    name: test
    description: test
    version: "1.0.0"
    invariants:
      hard:
        - name: always-true
          check:
            field: x
            equals: 1
    """)
    monitor = SessionMonitor(contract)  # raise_on_drift=False by default
    # Should not raise regardless of drift score
    result = monitor.step({"x": 1})
    assert result is not None  # Returns StepResult, no exception


def test_recovery_failed_error_raised_after_max_attempts() -> None:
    """RecoveryFailedError raised when recovery attempts exceed max."""
    contract = loads_contract("""
    contractspec: "0.1"
    kind: agent
    name: test
    description: test
    version: "1.0.0"
    invariants:
      soft:
        - name: always-false
          check:
            field: x
            equals: 0   # we will pass x=1 -> violation
          recovery: fix-it
    recovery:
      strategies:
        - name: fix-it
          type: inject_correction
          actions: ["Try to fix"]
          max_attempts: 2
    """)
    monitor = SessionMonitor(contract, max_recovery_attempts=2)
    # First call: violation -> recovery attempted (count=1)
    result1 = monitor.step({"x": 1})  # violates soft constraint
    assert result1.recovery_needed is True
    # Record the recovery attempt as failed
    monitor.record_recovery(attempted=True, succeeded=False)
    # Second call: still violation -> recovery attempted (count=2)
    result2 = monitor.step({"x": 1})
    assert result2.recovery_needed is True
    monitor.record_recovery(attempted=True, succeeded=False)
    # Third call: still violation -> recovery attempted (count=3 > max=2) -> should raise
    with pytest.raises(RecoveryFailedError, match="Recovery failed after 3 attempts"):
        monitor.step({"x": 1})
        monitor.record_recovery(attempted=True, succeeded=False)


_PRECONDITION_YAML = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: must-be-true
    check:
      field: ready
      equals: true
invariants:
  hard:
    - name: always-true
      check:
        field: x
        equals: 1
"""


def test_precondition_failed_error_raised_by_default() -> None:
    """PreconditionFailedError raised when check_preconditions default raise."""
    contract = loads_contract(_PRECONDITION_YAML)
    monitor = SessionMonitor(contract)
    with pytest.raises(PreconditionFailedError, match="must-be-true"):
        monitor.check_preconditions({"ready": False})


def test_precondition_failed_error_not_raised_when_disabled() -> None:
    """When raise_on_failure=False, PreconditionFailedError is not raised."""
    contract = loads_contract(_PRECONDITION_YAML)
    monitor = SessionMonitor(contract)
    result = monitor.check_preconditions(
        {"ready": False}, raise_on_failure=False,
    )
    assert result.all_met is False
    assert "must-be-true" in result.failed_names


def test_all_exceptions_inherit_from_base() -> None:
    """All custom exceptions inherit from AgentAssertError for consistent catching."""
    assert issubclass(DriftThresholdError, Exception)
    assert issubclass(RecoveryFailedError, Exception)
    assert issubclass(PreconditionFailedError, Exception)
    assert issubclass(ExprEvaluationError, Exception)

    # Specifically from AgentAssertError
    from agentassert_abc.exceptions import AgentAssertError
    assert issubclass(DriftThresholdError, AgentAssertError)
    assert issubclass(RecoveryFailedError, AgentAssertError)
    assert issubclass(PreconditionFailedError, AgentAssertError)
    assert issubclass(ExprEvaluationError, AgentAssertError)


def test_expr_evaluation_error_can_be_raised() -> None:
    """ExprEvaluationError exists and can be instantiated."""
    err = ExprEvaluationError("test expression error")
    assert str(err) == "test expression error"
    assert isinstance(err, Exception)
