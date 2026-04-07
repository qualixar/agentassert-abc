# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Semantic validation for ContractSpec — LLD §4.2 rules.

Validates cross-references, operator consistency, parameter ranges,
and uniqueness constraints that Pydantic structural validation cannot catch.

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 1 semantic validation).
"""

from __future__ import annotations

import re

from agentassert_abc.dsl.models import ValidationError
from agentassert_abc.models import (  # noqa: TCH001
    ConstraintCheck,
    ContractSpec,
    GovernanceConstraint,
    SoftConstraint,
)

_OPERATOR_FIELDS = (
    "equals", "not_equals", "gt", "gte", "lt", "lte",
    "in_", "not_in", "contains", "not_contains", "matches",
    "exists", "expr", "between",
)


def _count_operators(check: ConstraintCheck) -> int:
    """Count how many operators are set on a ConstraintCheck."""
    return sum(1 for f in _OPERATOR_FIELDS if getattr(check, f) is not None)


def _collect_all_constraints(
    contract: ContractSpec,
) -> list[tuple[str, str, ConstraintCheck]]:
    """Collect (path, name, check) for all constraints in the contract."""
    results: list[tuple[str, str, ConstraintCheck]] = []

    for i, pre in enumerate(contract.preconditions):
        results.append((f"preconditions[{i}]", pre.name, pre.check))

    if contract.invariants:
        for i, c in enumerate(contract.invariants.hard):
            results.append((f"invariants.hard[{i}]", c.name, c.check))
        for i, c in enumerate(contract.invariants.soft):
            results.append((f"invariants.soft[{i}]", c.name, c.check))

    if contract.governance:
        for i, c in enumerate(contract.governance.hard):
            results.append((f"governance.hard[{i}]", c.name, c.check))
        for i, c in enumerate(contract.governance.soft):
            results.append((f"governance.soft[{i}]", c.name, c.check))

    return results


def validate_contract(contract: ContractSpec) -> list[ValidationError]:
    """Run all semantic validation rules. Returns list of findings."""
    errors: list[ValidationError] = []

    # Collect strategy names
    strategy_names: set[str] = set()
    if contract.recovery:
        strategy_names = {s.name for s in contract.recovery.strategies}

    # --- Recovery cross-references ---
    _validate_recovery_refs(contract, strategy_names, errors)

    # --- Fallback cross-references ---
    _validate_fallback_refs(contract, strategy_names, errors)

    # --- Operator validation ---
    _validate_operators(contract, errors)

    # --- Regex validation ---
    _validate_regex(contract, errors)

    # --- Duplicate names ---
    _validate_unique_names(contract, errors)

    # --- Satisfaction parameter ranges ---
    _validate_satisfaction_ranges(contract, errors)

    # --- Weight sums ---
    _validate_weight_sums(contract, errors)

    # --- Operator type compatibility ---
    _validate_operator_types(contract, errors)

    # --- RC-19: between tuple order ---
    _validate_between_order(contract, errors)

    return errors


def _validate_recovery_refs(
    contract: ContractSpec,
    strategy_names: set[str],
    errors: list[ValidationError],
) -> None:
    """Every soft constraint recovery must reference an existing strategy."""
    soft_constraints: list[tuple[str, SoftConstraint | GovernanceConstraint]] = []

    if contract.invariants:
        for i, c in enumerate(contract.invariants.soft):
            soft_constraints.append((f"invariants.soft[{i}]", c))
    if contract.governance:
        for i, c in enumerate(contract.governance.soft):
            soft_constraints.append((f"governance.soft[{i}]", c))

    for path, c in soft_constraints:
        if c.recovery and c.recovery not in strategy_names:
            errors.append(ValidationError(
                level="error",
                path=f"{path}.recovery",
                message=f"Recovery '{c.recovery}' not defined in recovery.strategies",
                code="MISSING_RECOVERY_STRATEGY",
            ))


def _validate_fallback_refs(
    contract: ContractSpec,
    strategy_names: set[str],
    errors: list[ValidationError],
) -> None:
    """Every strategy fallback must reference another existing strategy."""
    if not contract.recovery:
        return
    for i, s in enumerate(contract.recovery.strategies):
        if s.fallback and s.fallback not in strategy_names:
            errors.append(ValidationError(
                level="error",
                path=f"recovery.strategies[{i}].fallback",
                message=f"Fallback '{s.fallback}' referenced but not defined",
                code="MISSING_FALLBACK",
            ))


def _validate_operators(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """Each ConstraintCheck must have exactly one operator."""
    for path, name, check in _collect_all_constraints(contract):
        count = _count_operators(check)
        if count == 0:
            errors.append(ValidationError(
                level="error",
                path=f"{path}.check",
                message=f"Constraint '{name}': no operator set",
                code="NO_OPERATOR",
            ))
        elif count > 1:
            errors.append(ValidationError(
                level="error",
                path=f"{path}.check",
                message=f"Constraint '{name}': {count} operators set (expected 1)",
                code="MULTIPLE_OPERATORS",
            ))


def _validate_regex(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """matches operator must have valid regex."""
    for path, name, check in _collect_all_constraints(contract):
        if check.matches is not None:
            try:
                re.compile(check.matches)
            except re.error as e:
                errors.append(ValidationError(
                    level="error",
                    path=f"{path}.check.matches",
                    message=f"Constraint '{name}': invalid regex '{check.matches}': {e}",
                    code="INVALID_REGEX",
                ))


def _validate_unique_names(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """Constraint and strategy names must be unique within their section."""
    # Constraint names across all sections
    all_names: list[tuple[str, str]] = []
    for path, name, _ in _collect_all_constraints(contract):
        all_names.append((path, name))

    seen: set[str] = set()
    for path, name in all_names:
        if name in seen:
            errors.append(ValidationError(
                level="error",
                path=path,
                message=f"Duplicate constraint name '{name}'",
                code="DUPLICATE_CONSTRAINT_NAME",
            ))
        seen.add(name)

    # Strategy names
    if contract.recovery:
        strat_seen: set[str] = set()
        for i, s in enumerate(contract.recovery.strategies):
            if s.name in strat_seen:
                errors.append(ValidationError(
                    level="error",
                    path=f"recovery.strategies[{i}]",
                    message=f"Duplicate strategy name '{s.name}'",
                    code="DUPLICATE_STRATEGY_NAME",
                ))
            strat_seen.add(s.name)


def _validate_satisfaction_ranges(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """(p, delta, k) must be in valid ranges — patent §5.3."""
    if not contract.satisfaction:
        return
    s = contract.satisfaction
    if not (0 < s.p <= 1):
        errors.append(ValidationError(
            level="error",
            path="satisfaction.p",
            message=f"p must be in (0, 1], got {s.p}",
            code="SATISFACTION_P_RANGE",
        ))
    if not (0 < s.delta <= 1):
        errors.append(ValidationError(
            level="error",
            path="satisfaction.delta",
            message=f"delta must be in (0, 1], got {s.delta}",
            code="SATISFACTION_DELTA_RANGE",
        ))
    if s.k < 1:
        errors.append(ValidationError(
            level="error",
            path="satisfaction.k",
            message=f"k must be >= 1, got {s.k}",
            code="SATISFACTION_K_RANGE",
        ))


def _validate_weight_sums(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """Drift and reliability weights should sum to ~1.0."""
    if contract.drift:
        w = contract.drift.weights
        total = w.compliance + w.distributional
        if abs(total - 1.0) > 0.05:
            errors.append(ValidationError(
                level="warning",
                path="drift.weights",
                message=f"Drift weights sum to {total:.2f}, expected ~1.0",
                code="DRIFT_WEIGHTS_SUM",
            ))
    if contract.reliability:
        w = contract.reliability.weights
        total = w.compliance + w.drift + w.event_freq + w.recovery_success
        if abs(total - 1.0) > 0.05:
            errors.append(ValidationError(
                level="warning",
                path="reliability.weights",
                message=f"Reliability weights sum to {total:.2f}, expected ~1.0",
                code="RELIABILITY_WEIGHTS_SUM",
            ))


_NUMERIC_OPS = ("gt", "gte", "lt", "lte")
_STRING_OPS = ("contains", "not_contains", "matches")


def _validate_operator_types(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """Numeric operators need numeric values, string ops need strings."""
    for path, name, check in _collect_all_constraints(contract):
        for op in _NUMERIC_OPS:
            val = getattr(check, op)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(ValidationError(
                    level="error",
                    path=f"{path}.check.{op}",
                    message=f"'{name}': {op} requires numeric, got {type(val).__name__}",
                    code="INVALID_NUMERIC_OPERATOR",
                ))
        for op in _STRING_OPS:
            val = getattr(check, op)
            if val is not None and not isinstance(val, str):
                errors.append(ValidationError(
                    level="error",
                    path=f"{path}.check.{op}",
                    message=f"'{name}': {op} requires string, got {type(val).__name__}",
                    code="INVALID_STRING_OPERATOR",
                ))


def _validate_between_order(
    contract: ContractSpec,
    errors: list[ValidationError],
) -> None:
    """RC-19: between lower bound must be <= upper bound."""
    for path, name, check in _collect_all_constraints(contract):
        if check.between is not None and check.between[0] > check.between[1]:
                errors.append(ValidationError(
                    level="error",
                    path=f"{path}.check.between",
                    message=(
                        f"Constraint '{name}': between lower bound "
                        f"({check.between[0]}) > upper bound ({check.between[1]})"
                    ),
                    code="INVALID_BETWEEN_ORDER",
                ))
