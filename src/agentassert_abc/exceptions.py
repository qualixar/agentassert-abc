# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Typed exception hierarchy for AgentAssert.

All exceptions inherit from AgentAssertError for consistent catching.
"""

from __future__ import annotations

import json


class AgentAssertError(Exception):
    """Base exception for all AgentAssert errors."""


class ContractParseError(AgentAssertError):
    """YAML parsing or schema validation failed."""


class ContractLoadError(AgentAssertError):
    """Contract could not be loaded (file missing, unreadable, or from_yaml failure).

    Distinct from ContractParseError (which is a schema/YAML problem): this covers
    I/O and load-time failures. Used by the enforcement plane's from_yaml() paths.
    """


class ContractBreachError(AgentAssertError):
    """Hard constraint violated at runtime — critical.

    Backward compatible with the measurement plane, which raises it with a plain
    positional message: ``ContractBreachError("Tool 'x' blocked: no-pii")`` —
    ``str(err)`` returns that message. The enforcement plane (gateway) may instead
    raise it with structured fields as keyword args:

        ContractBreachError(violation_name="no-pii", reason="...", tool="search",
                            session_id="s1", contract_id="c1", decision="deny")

    All structured fields default to empty, so existing callers are unaffected.
    """

    def __init__(
        self,
        message: str = "",
        *,
        violation_name: str = "",
        reason: str = "",
        tool: str = "",
        session_id: str = "",
        contract_id: str = "",
        decision: str = "deny",
    ) -> None:
        self.violation_name = violation_name
        self.reason = reason
        self.tool = tool
        self.session_id = session_id
        self.contract_id = contract_id
        self.decision = decision
        # str(err) prefers an explicit message, else the reason/violation name.
        super().__init__(message or reason or violation_name or "contract breach")

    def to_dict(self) -> dict[str, str]:
        return {
            "violation_name": self.violation_name,
            "reason": self.reason,
            "tool": self.tool,
            "session_id": self.session_id,
            "contract_id": self.contract_id,
            "decision": self.decision,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_http_body(self) -> dict[str, str]:
        return {
            "error": "ContractBreachError",
            "violation": self.violation_name,
            "reason": self.reason,
            "tool": self.tool,
            "session_id": self.session_id,
            "contract_id": self.contract_id,
        }


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
