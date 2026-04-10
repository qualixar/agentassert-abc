# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests to close ALL remaining coverage gaps — targeting 100%.

Each test targets a specific uncovered line identified in the deep audit.
"""

from pathlib import Path
from unittest.mock import patch


class TestParserImportGuard:
    """parser.py lines 23-24: ImportError when ruamel.yaml missing."""

    def test_import_error_without_ruamel(self) -> None:
        import importlib
        import sys

        import pytest

        # Remove the cached parser module so it re-imports
        mod_name = "agentassert_abc.dsl.parser"
        saved = sys.modules.pop(mod_name, None)

        with patch.dict(sys.modules, {"ruamel.yaml": None, "ruamel": None}):
            # Force re-import to trigger the ImportError guard
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            with pytest.raises(ImportError, match="ruamel.yaml is required"):
                importlib.import_module(mod_name)

        # Restore
        if saved is not None:
            sys.modules[mod_name] = saved


class TestParseContractFile:
    """parser.py line 121: parse_contract() reading valid file."""

    def test_parse_contract_valid_file(self, tmp_path: Path) -> None:
        from agentassert_abc.dsl.parser import parse_contract

        f = tmp_path / "valid.yaml"
        f.write_text("""
contractspec: "0.1"
kind: agent
name: file-parse
description: test
version: "1.0.0"
""")
        result = parse_contract(f)
        assert result.is_valid
        assert result.contract is not None
        assert result.contract.name == "file-parse"


class TestValidatorOperatorTypeBypass:
    """validator.py lines 278, 287: operator type check bypassing Pydantic.

    Pydantic normally catches type mismatches, but model_construct() skips validation.
    """

    def test_numeric_operator_with_wrong_type(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract
        from agentassert_abc.models import (
            ConstraintCheck,
            ContractSpec,
            HardConstraint,
            Invariants,
        )

        # Use model_construct to bypass Pydantic validation
        bad_check = ConstraintCheck.model_construct(
            field="x", gte="not_a_number"
        )
        bad_constraint = HardConstraint.model_construct(
            name="bad", description="", category="", check=bad_check
        )
        invariants = Invariants.model_construct(hard=[bad_constraint], soft=[])
        contract = ContractSpec.model_construct(
            contractspec="0.1",
            kind="agent",
            name="test",
            description="test",
            version="1.0.0",
            preconditions=[],
            invariants=invariants,
            governance=None,
            recovery=None,
            satisfaction=None,
            drift=None,
            reliability=None,
            metadata=None,
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "INVALID_NUMERIC_OPERATOR" in codes

    def test_string_operator_with_wrong_type(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract
        from agentassert_abc.models import (
            ConstraintCheck,
            ContractSpec,
            HardConstraint,
            Invariants,
        )

        bad_check = ConstraintCheck.model_construct(
            field="x", contains=42
        )
        bad_constraint = HardConstraint.model_construct(
            name="bad", description="", category="", check=bad_check
        )
        invariants = Invariants.model_construct(hard=[bad_constraint], soft=[])
        contract = ContractSpec.model_construct(
            contractspec="0.1",
            kind="agent",
            name="test",
            description="test",
            version="1.0.0",
            preconditions=[],
            invariants=invariants,
            governance=None,
            recovery=None,
            satisfaction=None,
            drift=None,
            reliability=None,
            metadata=None,
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "INVALID_STRING_OPERATOR" in codes


class TestEvaluatorEdgeCases:
    """engine.py lines 109, 117: empty results + missing field evidence."""

    def test_evidence_missing_field(self) -> None:
        """Constraint on a field not in state → evidence says 'not found'."""
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.evaluator.engine import evaluate

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: missing-check
      check:
        field: nonexistent.field
        equals: true
""")
        result = evaluate(contract, {"other": "value"})
        assert len(result.hard_violations) == 1
        assert "not found" in result.hard_violations[0].evidence


class TestFractionSatisfiedEmpty:
    """engine.py line 109: empty list returns 1.0."""

    def test_empty_list(self) -> None:
        from agentassert_abc.evaluator.engine import _fraction_satisfied

        assert _fraction_satisfied([]) == 1.0


class TestOperatorEdgeCases:
    """operators.py lines 61, 68: no-operator fallback + numeric coerce."""

    def test_no_operator_returns_false(self) -> None:
        """ConstraintCheck with zero operators → False."""
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        # model_construct to bypass validation (normally validator catches this)
        check = ConstraintCheck.model_construct(field="x")
        assert evaluate_check(check, {"x": "anything"}) is False

    def test_numeric_string_coercion(self) -> None:
        """String '42' should coerce to float for numeric comparison."""
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="val", gte=40.0)
        # State has string "42" — _numeric should coerce it
        assert evaluate_check(check, {"val": "42"}) is True
