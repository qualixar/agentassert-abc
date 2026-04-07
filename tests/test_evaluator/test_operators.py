# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for constraint operators — patent §4.3 (12 operators + between).

Each operator tested independently with clear inputs/outputs.
"""



class TestEqualsOperator:
    def test_equals_true(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="output.pii", equals=False)
        assert evaluate_check(check, {"output.pii": False}) is True

    def test_equals_false(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="output.pii", equals=False)
        assert evaluate_check(check, {"output.pii": True}) is False

    def test_equals_string(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="status", equals="available")
        assert evaluate_check(check, {"status": "available"}) is True


class TestNotEqualsOperator:
    def test_not_equals(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="mode", not_equals="maintenance")
        assert evaluate_check(check, {"mode": "active"}) is True
        assert evaluate_check(check, {"mode": "maintenance"}) is False


class TestComparisonOperators:
    def test_gt(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="count", gt=0)
        assert evaluate_check(check, {"count": 5}) is True
        assert evaluate_check(check, {"count": 0}) is False

    def test_gte(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="score", gte=0.7)
        assert evaluate_check(check, {"score": 0.7}) is True
        assert evaluate_check(check, {"score": 0.69}) is False

    def test_lt(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="count", lt=3)
        assert evaluate_check(check, {"count": 2}) is True
        assert evaluate_check(check, {"count": 3}) is False

    def test_lte(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="latency", lte=3000)
        assert evaluate_check(check, {"latency": 3000}) is True
        assert evaluate_check(check, {"latency": 3001}) is False


class TestListOperators:
    def test_in(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="cat", in_=["electronics", "clothing"])
        assert evaluate_check(check, {"cat": "electronics"}) is True
        assert evaluate_check(check, {"cat": "food"}) is False

    def test_not_in(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="src", not_in=["admin"])
        assert evaluate_check(check, {"src": "api"}) is True
        assert evaluate_check(check, {"src": "admin"}) is False


class TestStringOperators:
    def test_contains(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="text", contains="sponsored")
        assert evaluate_check(check, {"text": "This is sponsored content"}) is True
        assert evaluate_check(check, {"text": "Normal content"}) is False

    def test_not_contains(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="text", not_contains="competitor")
        assert evaluate_check(check, {"text": "Our product"}) is True
        assert evaluate_check(check, {"text": "competitor product"}) is False

    def test_matches(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="code", matches="^PROMO[0-9]{4}$")
        assert evaluate_check(check, {"code": "PROMO1234"}) is True
        assert evaluate_check(check, {"code": "DISCOUNT50"}) is False


class TestExistsOperator:
    def test_exists_true(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="prefs", exists=True)
        assert evaluate_check(check, {"prefs": "anything"}) is True
        assert evaluate_check(check, {"other": "x"}) is False

    def test_exists_false(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="debug", exists=False)
        assert evaluate_check(check, {"other": "x"}) is True
        assert evaluate_check(check, {"debug": True}) is False


class TestBetweenOperator:
    def test_between(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="temp", between=(0.0, 1.0))
        assert evaluate_check(check, {"temp": 0.5}) is True
        assert evaluate_check(check, {"temp": 1.5}) is False
        assert evaluate_check(check, {"temp": 0.0}) is True
        assert evaluate_check(check, {"temp": 1.0}) is True


class TestMissingField:
    def test_missing_field_returns_false(self) -> None:
        from agentassert_abc.evaluator.operators import evaluate_check
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="nonexistent", equals=True)
        assert evaluate_check(check, {"other": "x"}) is False
