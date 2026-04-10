# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for typed exception hierarchy."""


def test_base_exception_exists() -> None:
    from agentassert_abc.exceptions import AgentAssertError

    assert issubclass(AgentAssertError, Exception)


def test_contract_parse_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, ContractParseError

    assert issubclass(ContractParseError, AgentAssertError)
    err = ContractParseError("bad yaml")
    assert str(err) == "bad yaml"


def test_contract_breach_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, ContractBreachError

    assert issubclass(ContractBreachError, AgentAssertError)


def test_contract_validation_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, ContractValidationError

    assert issubclass(ContractValidationError, AgentAssertError)


def test_drift_threshold_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, DriftThresholdError

    assert issubclass(DriftThresholdError, AgentAssertError)


def test_recovery_failed_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, RecoveryFailedError

    assert issubclass(RecoveryFailedError, AgentAssertError)


def test_precondition_failed_error() -> None:
    from agentassert_abc.exceptions import AgentAssertError, PreconditionFailedError

    assert issubclass(PreconditionFailedError, AgentAssertError)
