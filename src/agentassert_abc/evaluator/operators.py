# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Constraint operators — patent §4.3 (14 operators).

Each operator: (field_value, check_value) → bool.
Stateless, pure functions. No side effects.

H-16: Field names use FLAT dotted notation (e.g., "output.pii_detected").
The evaluator looks up `state["output.pii_detected"]` as a flat key, NOT
as nested traversal `state["output"]["pii_detected"]`. Adapters' extract_state()
methods are responsible for flattening framework output into this format.

L-08 / SEC-02: The `matches` operator guards against ReDoS by capping
input length at 10_000 chars and regex pattern length at 1_000 chars.
Invalid regex patterns or type errors are caught and return False (SEC-06).
"""

from __future__ import annotations

import re
from typing import Any

from agentassert_abc.models import ConstraintCheck  # noqa: TCH001


def evaluate_check(check: ConstraintCheck, state: dict[str, Any]) -> bool:
    """Evaluate a single ConstraintCheck against agent state.

    Returns True if the constraint is satisfied, False otherwise.
    Missing fields return False (except exists=False which returns True).
    """
    field = check.field

    # exists operator — special: checks field presence, not value
    if check.exists is not None:
        field_present = field in state
        return field_present if check.exists else not field_present

    # All other operators need the field value
    if field not in state:
        return False

    val = state[field]

    if check.equals is not None:
        return val == check.equals
    if check.not_equals is not None:
        return val != check.not_equals
    if check.gt is not None:
        n = _numeric(val)
        return n is not None and n > check.gt
    if check.gte is not None:
        n = _numeric(val)
        return n is not None and n >= check.gte
    if check.lt is not None:
        n = _numeric(val)
        return n is not None and n < check.lt
    if check.lte is not None:
        n = _numeric(val)
        return n is not None and n <= check.lte
    if check.in_ is not None:
        return val in check.in_
    if check.not_in is not None:
        return val not in check.not_in
    if check.contains is not None:
        try:
            return check.contains in str(val)
        except (TypeError, AttributeError):
            return False
    if check.not_contains is not None:
        try:
            return check.not_contains not in str(val)
        except (TypeError, AttributeError):
            return False
    if check.matches is not None:
        try:
            text = str(val)
            # SEC-02: Guard against huge inputs that amplify ReDoS
            if len(text) > 10_000:
                return False
            # SEC-02: Guard against excessively long regex patterns
            if len(check.matches) > 1_000:
                return False
            return bool(re.search(check.matches, text))
        except (re.error, TypeError):
            return False
    if check.between is not None:
        n = _numeric(val)
        if n is None:
            return False
        return check.between[0] <= n <= check.between[1]

    # H-18: expr operator — not yet implemented, documented as future work.
    # Returns False (constraint unsatisfied) rather than silently passing.
    if check.expr is not None:
        return False

    # No operator set (should be caught by validator)
    return False


def _numeric(val: Any) -> float | None:
    """Coerce value to float for numeric comparisons.

    Fix H-15: Returns None for non-coercible values instead of raising.
    CRITICAL: Returns None for NaN and Infinity — these are not valid
    constraint values and would produce wrong comparison results.
    """
    if isinstance(val, (int, float)):
        import math

        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    try:
        import math

        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
