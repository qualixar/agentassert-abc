# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for the full evaluate() function — patent §3.1 Layer 2.

Evaluates contracts against agent state, produces C_hard(t) and C_soft(t).
"""



class TestEvaluateBasic:
    def test_all_hard_satisfied(self) -> None:
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
    - name: no-pii
      check:
        field: output.pii
        equals: false
    - name: no-competitor
      check:
        field: output.competitor
        equals: false
""")
        state = {"output.pii": False, "output.competitor": False}
        result = evaluate(contract, state)
        assert result.c_hard == 1.0
        assert len(result.hard_violations) == 0

    def test_one_hard_violated(self) -> None:
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
    - name: no-pii
      check:
        field: output.pii
        equals: false
    - name: no-competitor
      check:
        field: output.competitor
        equals: false
""")
        state = {"output.pii": True, "output.competitor": False}
        result = evaluate(contract, state)
        assert result.c_hard == 0.5
        assert len(result.hard_violations) == 1
        assert result.hard_violations[0].name == "no-pii"

    def test_soft_constraints(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.evaluator.engine import evaluate

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
    - name: complete
      check:
        field: output.complete
        gte: 0.6
      recovery: fix
recovery:
  strategies:
    - name: fix
      type: inject_correction
""")
        state = {"output.tone": 0.8, "output.complete": 0.3}
        result = evaluate(contract, state)
        assert result.c_soft == 0.5
        assert len(result.soft_violations) == 1
        assert result.soft_violations[0].name == "complete"

    def test_empty_contract_perfect_scores(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.evaluator.engine import evaluate

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        result = evaluate(contract, {})
        assert result.c_hard == 1.0
        assert result.c_soft == 1.0


class TestEvaluatePreconditions:
    def test_all_preconditions_met(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.evaluator.engine import evaluate_preconditions

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: session-valid
    check:
      field: session.active
      equals: true
""")
        results = evaluate_preconditions(contract, {"session.active": True})
        assert all(r.satisfied for r in results)

    def test_precondition_failed(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.evaluator.engine import evaluate_preconditions

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: session-valid
    check:
      field: session.active
      equals: true
""")
        results = evaluate_preconditions(contract, {"session.active": False})
        assert not results[0].satisfied


class TestEvaluateEcommerce:
    """Integration: evaluate the patent e-commerce contract."""

    def test_compliant_state(self) -> None:
        from agentassert_abc.dsl.parser import load_contract
        from agentassert_abc.evaluator.engine import evaluate

        contract = load_contract(
            "tests/test_dsl/fixtures/ecommerce-product-recommendation.yaml"
        )
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
        result = evaluate(contract, state)
        assert result.c_hard == 1.0, f"Hard violations: {result.hard_violations}"
        assert result.c_soft == 1.0, f"Soft violations: {result.soft_violations}"

    def test_pii_violation(self) -> None:
        from agentassert_abc.dsl.parser import load_contract
        from agentassert_abc.evaluator.engine import evaluate

        contract = load_contract(
            "tests/test_dsl/fixtures/ecommerce-product-recommendation.yaml"
        )
        state = {
            "output.competitor_reference_detected": False,
            "output.unverified_availability_claim": False,
            "output.unauthorized_discount_offered": False,
            "output.pii_detected": True,  # VIOLATION
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
        result = evaluate(contract, state)
        assert result.c_hard < 1.0
        violated_names = [v.name for v in result.hard_violations]
        assert "no-pii-leak" in violated_names
