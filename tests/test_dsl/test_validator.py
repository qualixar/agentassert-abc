# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for semantic validation — LLD §4.2 rules.

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 1 semantic validation).
"""

import pytest


def _make_contract(**overrides):  # type: ignore[no-untyped-def]
    """Helper to build a ContractSpec with overrides."""
    from agentassert_abc.models import ContractSpec

    defaults = {
        "contractspec": "0.1",
        "kind": "agent",
        "name": "test",
        "description": "test",
        "version": "1.0.0",
    }
    defaults.update(overrides)
    return ContractSpec.model_validate(defaults)


class TestRecoveryCrossReferences:
    """Soft constraint recovery must reference existing strategies."""

    def test_valid_recovery_reference(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [],
                "soft": [
                    {"name": "tone", "check": {"field": "x", "gte": 0.7}, "recovery": "fix-tone"},
                ],
            },
            recovery={
                "strategies": [
                    {"name": "fix-tone", "type": "inject_correction"},
                ],
            },
        )
        errors = validate_contract(contract)
        assert len([e for e in errors if e.code == "MISSING_RECOVERY_STRATEGY"]) == 0

    def test_missing_recovery_strategy(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [],
                "soft": [
                    {
                        "name": "tone",
                        "check": {"field": "x", "gte": 0.7},
                        "recovery": "nonexistent",
                    },
                ],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "MISSING_RECOVERY_STRATEGY" in codes

    def test_missing_fallback_strategy(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            recovery={
                "strategies": [
                    {"name": "fix", "type": "inject_correction", "fallback": "ghost"},
                ],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "MISSING_FALLBACK" in codes


class TestOperatorValidation:
    """Each ConstraintCheck must have exactly one operator."""

    def test_no_operator_set(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "bad", "check": {"field": "x"}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "NO_OPERATOR" in codes

    def test_multiple_operators_set(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "bad", "check": {"field": "x", "equals": True, "gte": 0.5}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "MULTIPLE_OPERATORS" in codes

    def test_single_operator_valid(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "ok", "check": {"field": "x", "equals": True}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        op_errors = [e for e in errors if e.code in ("NO_OPERATOR", "MULTIPLE_OPERATORS")]
        assert len(op_errors) == 0


class TestInvalidRegex:
    """matches operator must have valid regex."""

    def test_invalid_regex_pattern(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "bad-regex", "check": {"field": "x", "matches": "[invalid("}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "INVALID_REGEX" in codes

    def test_valid_regex_pattern(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "ok", "check": {"field": "x", "matches": "^PROMO[0-9]{4}$"}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        assert len([e for e in errors if e.code == "INVALID_REGEX"]) == 0


class TestDuplicateNames:
    """Constraint and strategy names must be unique."""

    def test_duplicate_constraint_names(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            invariants={
                "hard": [
                    {"name": "dupe", "check": {"field": "a", "equals": True}},
                    {"name": "dupe", "check": {"field": "b", "equals": False}},
                ],
                "soft": [],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "DUPLICATE_CONSTRAINT_NAME" in codes

    def test_duplicate_strategy_names(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            recovery={
                "strategies": [
                    {"name": "dupe", "type": "inject_correction"},
                    {"name": "dupe", "type": "reduce_autonomy"},
                ],
            },
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "DUPLICATE_STRATEGY_NAME" in codes


class TestSatisfactionRanges:
    """(p, delta, k) parameter validation — patent §5.3."""

    def test_p_out_of_range(self) -> None:
        """SEC-10: p > 1.0 rejected at Pydantic model level."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="less than or equal to 1"):
            _make_contract(satisfaction={"p": 1.5, "delta": 0.1, "k": 3})

    def test_p_zero(self) -> None:
        """SEC-10: p = 0.0 rejected at Pydantic model level (must be > 0)."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="greater than 0"):
            _make_contract(satisfaction={"p": 0.0, "delta": 0.1, "k": 3})

    def test_delta_out_of_range(self) -> None:
        """SEC-10: delta > 1.0 rejected at Pydantic model level."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="less than or equal to 1"):
            _make_contract(satisfaction={"p": 0.95, "delta": 2.0, "k": 3})

    def test_k_too_small(self) -> None:
        """SEC-10: k = 0 rejected at Pydantic model level (must be >= 1)."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="greater than or equal to 1"):
            _make_contract(satisfaction={"p": 0.95, "delta": 0.1, "k": 0})

    def test_valid_satisfaction_no_errors(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(satisfaction={"p": 0.95, "delta": 0.1, "k": 3})
        errors = validate_contract(contract)
        sat_errors = [e for e in errors if e.code.startswith("SATISFACTION")]
        assert len(sat_errors) == 0


class TestWeightSums:
    """Drift and reliability weights should sum to ~1.0."""

    def test_drift_weights_bad_sum_warning(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            drift={"weights": {"compliance": 0.9, "distributional": 0.9}}
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "DRIFT_WEIGHTS_SUM" in codes

    def test_drift_weights_good_sum_no_warning(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            drift={"weights": {"compliance": 0.6, "distributional": 0.4}}
        )
        errors = validate_contract(contract)
        assert len([e for e in errors if e.code == "DRIFT_WEIGHTS_SUM"]) == 0

    def test_reliability_weights_bad_sum_warning(self) -> None:
        from agentassert_abc.dsl.validator import validate_contract

        contract = _make_contract(
            reliability={
                "weights": {
                    "compliance": 0.5,
                    "drift": 0.5,
                    "event_freq": 0.5,
                    "recovery_success": 0.5,
                }
            }
        )
        errors = validate_contract(contract)
        codes = [e.code for e in errors]
        assert "RELIABILITY_WEIGHTS_SUM" in codes


class TestOperatorTypeValidation:
    """Numeric/string type mismatches caught at structural level by Pydantic."""

    def test_gte_with_string_rejected_by_pydantic(self) -> None:
        """Pydantic catches type mismatch before semantic validator runs."""
        from pydantic import ValidationError as PydanticValidationError

        from agentassert_abc.models import ContractSpec

        with pytest.raises(PydanticValidationError):
            ContractSpec.model_validate({
                "contractspec": "0.1",
                "kind": "agent",
                "name": "t",
                "description": "t",
                "version": "1.0.0",
                "invariants": {
                    "hard": [{"name": "bad", "check": {"field": "x", "gte": "not_a_number"}}],
                    "soft": [],
                },
            })

    def test_contains_with_number_rejected_by_pydantic(self) -> None:
        """Pydantic catches type mismatch before semantic validator runs."""
        from pydantic import ValidationError as PydanticValidationError

        from agentassert_abc.models import ContractSpec

        with pytest.raises(PydanticValidationError):
            ContractSpec.model_validate({
                "contractspec": "0.1",
                "kind": "agent",
                "name": "t",
                "description": "t",
                "version": "1.0.0",
                "invariants": {
                    "hard": [{"name": "bad", "check": {"field": "x", "contains": 42}}],
                    "soft": [],
                },
            })


class TestEcommerceValidation:
    """The patent e-commerce contract must pass ALL validation."""

    def test_ecommerce_contract_valid(self) -> None:
        from agentassert_abc.dsl.parser import load_contract
        from agentassert_abc.dsl.validator import validate_contract

        contract = load_contract(
            "tests/test_dsl/fixtures/ecommerce-product-recommendation.yaml"
        )
        errors = validate_contract(contract)
        error_level = [e for e in errors if e.level == "error"]
        assert len(error_level) == 0, f"Errors: {error_level}"
