# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Typed exception hierarchy for AgentAssert.

All exceptions inherit from AgentAssertError for consistent catching.
"""


class AgentAssertError(Exception):
    """Base exception for all AgentAssert errors."""


class ContractParseError(AgentAssertError):
    """YAML parsing or schema validation failed."""


class ContractBreachError(AgentAssertError):
    """Hard constraint violated at runtime — critical."""


class ContractValidationError(AgentAssertError):
    """Semantic validation failed (bad cross-refs, invalid params)."""


class DriftThresholdError(AgentAssertError):
    """Drift exceeded critical threshold."""


class RecoveryFailedError(AgentAssertError):
    """Recovery re-prompting failed after max_attempts."""


class PreconditionFailedError(AgentAssertError):
    """Precondition not met — agent should not process request."""


class ExprEvaluationError(AgentAssertError):
    """Raised when expr operator evaluation fails for a reason other than a normal False result."""


class StateExtractionError(AgentAssertError, TypeError):
    """F-19: Output type not supported by adapter's extract_state().

    Inherits from both AgentAssertError (for uniform catching) and
    TypeError (for backward compatibility).
    """


class DependenceError(AgentAssertError):
    """Invalid input to a dependence estimator, or an underidentified model.

    Raised e.g. for mismatched paired lengths, out-of-range (eps, alpha),
    degenerate marginals, or a one-factor model with fewer than 3 indicators
    (loadings are not identified with only 2 indicators — LLD-E pre-lock fix).
    """
