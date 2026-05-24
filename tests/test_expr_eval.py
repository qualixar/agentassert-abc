# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for sandboxed expr operator evaluator."""

import pytest

from agentassert_abc.evaluator.expr_eval import SafeExprEvaluator


@pytest.fixture
def ev() -> SafeExprEvaluator:
    return SafeExprEvaluator()


class TestExprFunctional:
    """Functional tests — expressions that should evaluate correctly."""

    def test_simple_comparison(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("value > 0.5", {"value": 0.7})
        assert result.value is True
        assert result.is_bool is True
        assert result.error is None

    def test_simple_comparison_false(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("value > 0.5", {"value": 0.3})
        assert result.value is False

    def test_arithmetic(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("value * 2 < 1.0", {"value": 0.4})
        assert result.value is True

    def test_boolean_composition(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("(a > 0) and (b < 100)", {"a": 5, "b": 50})
        assert result.value is True

    def test_function_len(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("len(value) > 0", {"value": "abc"})
        assert result.value is True

    def test_function_abs(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("abs(x) < 10", {"x": -5})
        assert result.value is True

    def test_equality(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate('status == "active"', {"status": "active"})
        assert result.value is True

    def test_string_in(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("value in allowed", {"value": "a", "allowed": ["a", "b", "c"]})
        assert result.value is True

    def test_numeric_result_not_bool(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("x + y", {"x": 3, "y": 4})
        assert result.value == 7
        assert result.is_bool is False


class TestExprAdversarial:
    """Security tests — malicious expressions must be rejected or return error."""

    def test_dunder_import(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("__import__('os').system('echo hacked')", {})
        assert result.error is not None or result.value is False

    def test_dunder_class(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("''.__class__.__base__", {})
        assert result.error is not None or result.value is False

    def test_import_keyword(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("import os", {})
        assert result.error is not None or result.value is False

    def test_exec_keyword(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("exec('print(1)')", {})
        assert result.error is not None or result.value is False

    def test_eval_keyword(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("eval('1+1')", {})
        assert result.error is not None or result.value is False

    def test_open_builtin(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("open('/etc/passwd').read()", {})
        assert result.error is not None or result.value is False

    def test_globals(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("globals()", {})
        assert result.error is not None or result.value is False

    def test_locals(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("locals()", {})
        assert result.error is not None or result.value is False

    def test_lambda(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("(lambda: 42)()", {})
        assert result.error is not None or result.value is False

    def test_power_blocked(self, ev: SafeExprEvaluator) -> None:
        """Power operator ** should be blocked."""
        result = ev.evaluate("2 ** 10000", {})
        assert result.error is not None or result.value is False

    def test_empty_expression(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("", {})
        assert result.error is not None

    def test_malformed_syntax(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("value >", {"value": 5})
        assert result.error is not None

    def test_expression_too_long(self, ev: SafeExprEvaluator) -> None:
        long_expr = "x + " * 200 + "1"
        result = ev.evaluate(long_expr, {"x": 1})
        assert result.error is not None

    def test_undefined_name(self, ev: SafeExprEvaluator) -> None:
        result = ev.evaluate("undefined_var > 0", {"x": 1})
        assert result.error is not None or result.value is False

    def test_state_immutability(self, ev: SafeExprEvaluator) -> None:
        """expr must not be able to mutate the caller's state dict."""
        state = {"x": 5}
        ev.evaluate("x + 1", state)
        assert state == {"x": 5}  # unchanged


class TestExprViaOperator:
    """Test expr through the evaluate_check integration point."""

    def test_expr_operator_in_check(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="x", expr="x > 5")
        assert evaluate_check(check, {"x": 10}) is True
        assert evaluate_check(check, {"x": 3}) is False

    def test_expr_operator_malicious_in_check(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="x", expr="__import__('os')")
        # Should return False (fail-closed), not crash
        assert evaluate_check(check, {"x": 1}) is False
