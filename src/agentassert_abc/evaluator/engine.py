# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Constraint Evaluator Engine — Layer 2.

Stateless evaluation of ALL constraints against agent state.
Produces per-constraint results + aggregate C_hard(t), C_soft(t).

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 2).
"""

from __future__ import annotations

from typing import Any

from agentassert_abc.evaluator.models import ConstraintResult, EvaluationResult
from agentassert_abc.evaluator.operators import evaluate_check
from agentassert_abc.models import ContractSpec, HardConstraint, SoftConstraint  # noqa: TCH001


def evaluate(
    contract: ContractSpec, state: dict[str, Any]
) -> EvaluationResult:
    """Evaluate all constraints in a contract against agent state.

    Returns EvaluationResult with per-constraint results and aggregate scores.
    Stateless — can be called independently for any turn.
    """
    hard_results: list[ConstraintResult] = []
    soft_results: list[ConstraintResult] = []
    governance_results: list[ConstraintResult] = []

    # Hard invariants
    if contract.invariants:
        for c in contract.invariants.hard:
            hard_results.append(_eval_hard(c, state))

    # Soft invariants
    if contract.invariants:
        for c in contract.invariants.soft:
            soft_results.append(_eval_soft(c, state))

    # Governance hard — added to governance_results AND hard_results
    # (governance hard violations ARE hard violations for enforcement)
    if contract.governance:
        for c in contract.governance.hard:
            r = _eval_hard(c, state)
            governance_results.append(r)
            hard_results.append(r)

    # Governance soft — added to governance_results AND soft_results
    if contract.governance:
        for c in contract.governance.soft:
            r = ConstraintResult(
                name=c.name,
                satisfied=evaluate_check(c.check, state),
                evidence=_evidence(c.check, state),
                constraint_type="governance",
            )
            governance_results.append(r)
            soft_results.append(r)

    # RC-02 FIX: Compute C_hard/C_soft from INVARIANT constraints only
    # (not governance) to match patent's compliance metric definition.
    # Governance violations still appear in hard_violations/soft_violations
    # for enforcement purposes (ContractBreachError, recovery).
    inv_hard = [r for r in hard_results if r.constraint_type == "hard"]
    inv_soft = [r for r in soft_results if r.constraint_type == "soft"]
    c_hard = _fraction_satisfied(inv_hard) if inv_hard else 1.0
    c_soft = _fraction_satisfied(inv_soft) if inv_soft else 1.0

    return EvaluationResult(
        hard_results=hard_results,
        soft_results=soft_results,
        governance_results=governance_results,
        c_hard=c_hard,
        c_soft=c_soft,
        hard_violations=[r for r in hard_results if not r.satisfied],
        soft_violations=[r for r in soft_results if not r.satisfied],
    )


def evaluate_preconditions(
    contract: ContractSpec, state: dict[str, Any]
) -> list[ConstraintResult]:
    """Evaluate preconditions only. Returns list of results."""
    results: list[ConstraintResult] = []
    for pre in contract.preconditions:
        satisfied = evaluate_check(pre.check, state)
        results.append(ConstraintResult(
            name=pre.name,
            satisfied=satisfied,
            evidence=_evidence(pre.check, state),
            constraint_type="precondition",
        ))
    return results


def _eval_hard(c: HardConstraint, state: dict[str, Any]) -> ConstraintResult:
    return ConstraintResult(
        name=c.name,
        satisfied=evaluate_check(c.check, state),
        evidence=_evidence(c.check, state),
        constraint_type="hard",
    )


def _eval_soft(c: SoftConstraint, state: dict[str, Any]) -> ConstraintResult:
    return ConstraintResult(
        name=c.name,
        satisfied=evaluate_check(c.check, state),
        evidence=_evidence(c.check, state),
        constraint_type="soft",
    )


def _fraction_satisfied(results: list[ConstraintResult]) -> float:
    if not results:
        return 1.0
    return sum(1 for r in results if r.satisfied) / len(results)


def _evidence(check: Any, state: dict[str, Any]) -> str:
    """Build evidence string for why a constraint was satisfied/violated."""
    field = check.field
    if field not in state:
        return f"field '{field}' not found in state"
    return f"{field}={state[field]!r}"
