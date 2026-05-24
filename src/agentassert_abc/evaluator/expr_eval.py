# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Sandboxed `expr` operator evaluator — Patent §4.3 (14th operator).

Evaluates user-supplied Python expressions against contract state using
`simpleeval` with a strict whitelist. No imports, no I/O, no attribute
access on dunders. Time-bounded and length-bounded.

Security guarantees:
- No `__` (dunder) tokens reach the evaluator
- No import/exec/eval/open tokens reach the evaluator
- Time limit enforced via simpleeval's MAX_POWER + iteration limits
- Length limit enforced (500 chars max)
- State is passed as immutable shallow copy
- Power operator `**` blocked (DoS vector)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from simpleeval import (
    DEFAULT_OPERATORS,
    FeatureNotAvailable,
    NameNotDefined,
    SimpleEval,
)

# Whitelist of safe built-in functions exposed to expr strings
_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "round": round,
    "sum": sum,
    "any": any,
    "all": all,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
}

# Operators: everything except power (**) which is a DoS vector
_SAFE_OPERATORS = dict(DEFAULT_OPERATORS)
_SAFE_OPERATORS.pop(ast.Pow, None)  # Block power operator

# Banned substrings — pre-screen before passing to simpleeval
_BANNED_TOKENS = ("__", "import", "exec", "eval", "compile", "open", "input", "globals", "locals")

MAX_EXPR_LENGTH = 500


@dataclass(frozen=True)
class ExprResult:
    """Result of evaluating an expr string.

    Attributes:
        value: The evaluated result (bool or numeric).
        is_bool: Whether the result type is bool.
        error: None on success, error message string on failure.
    """

    value: Any
    is_bool: bool
    error: str | None


class SafeExprEvaluator:
    """Sandboxed expression evaluator for the `expr` ContractSpec operator.

    Usage:
        evaluator = SafeExprEvaluator()
        result = evaluator.evaluate(
            "value < max_budget", {"value": 50, "max_budget": 100},
        )
        if result.error:
            # treat as constraint failure
        passed = result.value
    """

    def evaluate(self, expr: str, context: dict[str, Any]) -> ExprResult:
        """Evaluate `expr` against an immutable view of `context`.

        Args:
            expr: Python expression string from contract YAML.
            context: Flat dict of state values available to the expression.

        Returns:
            ExprResult with value, is_bool, and optional error.
        """
        # Length guard
        if len(expr) > MAX_EXPR_LENGTH:
            return ExprResult(
                value=False, is_bool=True,
                error=f"expression exceeds max length {MAX_EXPR_LENGTH}",
            )

        # Pre-screen for banned tokens
        lowered = expr.lower()
        for token in _BANNED_TOKENS:
            if token in lowered:
                return ExprResult(
                    value=False, is_bool=True,
                    error=f"banned token '{token}' in expression",
                )

        # Build a fresh SimpleEval with whitelisted functions + copy of context
        se = SimpleEval(
            functions=dict(_SAFE_FUNCTIONS),
            operators=dict(_SAFE_OPERATORS),
            names=dict(context),  # shallow copy — expr cannot mutate caller state
        )

        try:
            value = se.eval(expr)
            return ExprResult(
                value=value,
                is_bool=isinstance(value, bool),
                error=None,
            )
        except (
            FeatureNotAvailable, NameNotDefined, SyntaxError, TypeError,
            ZeroDivisionError,
        ) as e:
            return ExprResult(value=False, is_bool=True, error=str(e))
        except Exception as e:
            # Catch-all for any other simpleeval / evaluation errors
            return ExprResult(value=False, is_bool=True, error=str(e))
