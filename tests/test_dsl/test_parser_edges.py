# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Edge case tests to close coverage gaps from audit Findings 2, 3."""


class TestParsesContractStructError:
    """Finding 2: parses_contract() structural error path."""

    def test_struct_error_returns_parse_result(self) -> None:
        from agentassert_abc.dsl.parser import parses_contract

        result = parses_contract("contractspec: '0.1'\nkind: agent\n")
        assert not result.is_valid
        assert result.contract is None
        assert any(e.code == "STRUCT_ERROR" for e in result.errors)

    def test_parses_semantic_error_returns_errors(self) -> None:
        """Contract with missing recovery strategy → semantic error in non-raising API."""
        from agentassert_abc.dsl.parser import parses_contract

        yaml = """
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
      recovery: ghost-strategy
"""
        result = parses_contract(yaml)
        assert not result.is_valid
        assert any(e.code == "MISSING_RECOVERY_STRATEGY" for e in result.errors)


class TestLoadsContractSemanticError:
    """Finding 3: loads_contract() semantic error raising path."""

    def test_semantic_error_raises_validation_error(self) -> None:
        import pytest

        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractValidationError

        yaml = """
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
      recovery: nonexistent
"""
        with pytest.raises(ContractValidationError, match="MISSING_RECOVERY_STRATEGY"):
            loads_contract(yaml)


class TestParseContractFromFile:
    """parse_contract() non-raising file API."""

    def test_file_not_found_returns_error(self) -> None:
        from agentassert_abc.dsl.parser import parse_contract

        result = parse_contract("/nonexistent/file.yaml")
        assert not result.is_valid
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)
