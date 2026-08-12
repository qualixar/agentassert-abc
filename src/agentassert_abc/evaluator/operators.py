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

import concurrent.futures
import re
import threading
from typing import Any

from agentassert_abc.models import ConstraintCheck  # noqa: TCH001

# Ledger 4a: wall-clock limit for a single re.search call to prevent ReDoS.
# Length caps alone (10K chars, 1K pattern) cannot stop exponential backtracking.
# concurrent.futures gives us a portable OS-independent timeout.
_RE_TIMEOUT_S: float = 1.0

# Shared worker for the ReDoS timeout guard. Built lazily and never shut down —
# a per-call ThreadPoolExecutor cost ~6x the regex evaluation itself under load.
# Several workers so one wedged catastrophic pattern cannot starve other checks.
_REGEX_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_REGEX_POOL_LOCK = threading.Lock()


def _regex_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared regex-timeout worker pool, creating it on first use."""
    global _REGEX_POOL  # noqa: PLW0603
    if _REGEX_POOL is None:
        with _REGEX_POOL_LOCK:
            if _REGEX_POOL is None:
                _REGEX_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="agentassert-regex"
                )
    return _REGEX_POOL


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
            # Ledger 4a: wall-clock timeout prevents exponential-backtracking ReDoS.
            # Length caps alone are insufficient for hand-crafted catastrophic patterns.
            # The pool is a module-level singleton: creating and tearing one down
            # per check cost ~6x the evaluation itself under load (audit H-03).
            _fut = _regex_pool().submit(re.search, check.matches, text)
            try:
                return bool(_fut.result(timeout=_RE_TIMEOUT_S))
            except concurrent.futures.TimeoutError:
                # The worker may still be spinning on a catastrophic pattern; the
                # pool absorbs it rather than blocking this caller.
                return False
        except (re.error, TypeError):
            return False
    if check.between is not None:
        n = _numeric(val)
        if n is None:
            return False
        return check.between[0] <= n <= check.between[1]

    # G5: expr operator — sandboxed Python expression evaluator.
    if check.expr is not None:
        from agentassert_abc.evaluator.expr_eval import SafeExprEvaluator

        evalr = SafeExprEvaluator()
        result = evalr.evaluate(check.expr, state)
        if result.error is not None:
            return False
        return bool(result.value)

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
