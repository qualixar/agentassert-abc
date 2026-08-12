# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Semantic validation for the enforcement-plane `process:` block.

Ported from agentassert-typec's `dsl/validator.py::validate_extended`
. Complements the base semantic validator
(`agentassert_abc.dsl.validator.validate_contract`): this one walks the
legacy list-of-operators form of `invariants.process` and produces
human-readable `ValidationError`s. The dict form is fully validated
structurally by Pydantic (`ProcessInvariants`), so this pass is a no-op
for it — it exists to give the same friendly diagnostics the base plane
gives, for contracts authored in the older list style.
"""

from __future__ import annotations

from typing import Any

from agentassert_abc.dsl.models import ValidationError

_VALID_BLOCKLIST_SCOPES = {"session", "turn"}
_VALID_CONTEXT_BUDGET_ACTIONS = {"warn", "deny", "compress"}
_VALID_DRIFT_ACTIONS = {"log", "warn", "theta_penalty"}
_VALID_JUDGE_ACTIONS = {"log", "warn", "theta_penalty", "deny"}
_VALID_MUST_PRECEDE_SCOPES = {"turn", "session"}


def validate_extended(data: dict[str, Any]) -> list[ValidationError]:
    """Semantic checks for `invariants.process` operators (list form).

    Returns a list of ValidationError (level 'error'/'warning'). An empty
    list means no semantic problems were found by this pass.
    """
    errors: list[ValidationError] = []

    inv = data.get("invariants", {}) or {}
    process = inv.get("process")
    # Only the legacy list-of-operators form is walked here; the dict form is
    # validated structurally by the ProcessInvariants Pydantic model.
    if not isinstance(process, list):
        return errors

    for i, op in enumerate(process):
        if not isinstance(op, dict):
            continue
        errors.extend(_validate_process_operator(op, f"invariants.process[{i}]"))

    return errors


def _validate_process_operator(op: dict[str, Any], path: str) -> list[ValidationError]:
    errors: list[ValidationError] = []

    if "tool_blocklist" in op:
        bl = op["tool_blocklist"] or {}
        if not isinstance(bl.get("tools"), list) or len(bl["tools"]) == 0:
            errors.append(ValidationError(
                level="error", path=f"{path}.tool_blocklist.tools",
                message="tool_blocklist.tools must be a non-empty list",
                code="BLOCKLIST_EMPTY_TOOLS",
            ))
        scope = bl.get("scope", "session")
        if scope not in _VALID_BLOCKLIST_SCOPES:
            errors.append(ValidationError(
                level="error", path=f"{path}.tool_blocklist.scope",
                message=f"scope must be one of {_VALID_BLOCKLIST_SCOPES}, got '{scope}'",
                code="INVALID_SCOPE",
            ))

    if "tool_allowlist" in op:
        al = op["tool_allowlist"] or {}
        if not isinstance(al.get("tools"), list) or len(al["tools"]) == 0:
            errors.append(ValidationError(
                level="error", path=f"{path}.tool_allowlist.tools",
                message="tool_allowlist.tools must be a non-empty list",
                code="ALLOWLIST_EMPTY_TOOLS",
            ))

    if "must_state" in op:
        ms = op["must_state"] or {}
        if not ms.get("field"):
            errors.append(ValidationError(
                level="error", path=f"{path}.must_state.field",
                message="must_state.field is required",
                code="MUST_STATE_NO_FIELD",
            ))
        if not ms.get("before_tool_pattern"):
            errors.append(ValidationError(
                level="error", path=f"{path}.must_state.before_tool_pattern",
                message="must_state.before_tool_pattern is required",
                code="MUST_STATE_NO_PATTERN",
            ))

    if "context_budget" in op:
        cb = op["context_budget"] or {}
        if not isinstance(cb.get("max_tokens_per_turn"), int) or cb["max_tokens_per_turn"] <= 0:
            errors.append(ValidationError(
                level="error", path=f"{path}.context_budget.max_tokens_per_turn",
                message="context_budget.max_tokens_per_turn must be a positive integer",
                code="INVALID_CONTEXT_BUDGET",
            ))
        action = cb.get("action_on_breach", "warn")
        if action not in _VALID_CONTEXT_BUDGET_ACTIONS:
            errors.append(ValidationError(
                level="error", path=f"{path}.context_budget.action_on_breach",
                message=f"action_on_breach must be one of {_VALID_CONTEXT_BUDGET_ACTIONS}",
                code="INVALID_ACTION",
            ))

    if "judge_predicate" in op:
        jp = op["judge_predicate"] or {}
        if not jp.get("rubric"):
            errors.append(ValidationError(
                level="error", path=f"{path}.judge_predicate.rubric",
                message="judge_predicate.rubric is required",
                code="JUDGE_NO_RUBRIC",
            ))
        sample_rate = jp.get("sample_rate", 0.2)
        if not (0.0 < sample_rate <= 1.0):
            errors.append(ValidationError(
                level="error", path=f"{path}.judge_predicate.sample_rate",
                message="sample_rate must be in (0.0, 1.0]",
                code="INVALID_SAMPLE_RATE",
            ))
        action = jp.get("action_on_fail", "theta_penalty")
        if action not in _VALID_JUDGE_ACTIONS:
            errors.append(ValidationError(
                level="error", path=f"{path}.judge_predicate.action_on_fail",
                message=f"action_on_fail must be one of {_VALID_JUDGE_ACTIONS}",
                code="INVALID_JUDGE_ACTION",
            ))

    if "must_precede" in op:
        mp = op["must_precede"] or {}
        if not mp.get("before"):
            errors.append(ValidationError(
                level="error", path=f"{path}.must_precede.before",
                message="must_precede.before is required",
                code="MUST_PRECEDE_NO_BEFORE",
            ))
        if not mp.get("after"):
            errors.append(ValidationError(
                level="error", path=f"{path}.must_precede.after",
                message="must_precede.after is required",
                code="MUST_PRECEDE_NO_AFTER",
            ))
        scope = mp.get("scope", "turn")
        if scope not in _VALID_MUST_PRECEDE_SCOPES:
            errors.append(ValidationError(
                level="error", path=f"{path}.must_precede.scope",
                message=f"must_precede.scope must be one of {_VALID_MUST_PRECEDE_SCOPES}",
                code="INVALID_MUST_PRECEDE_SCOPE",
            ))

    if "process_drift" in op:
        pd = op["process_drift"] or {}
        jsd = pd.get("jsd_threshold", 0.3)
        if not isinstance(jsd, (int, float)) or not (0.0 < jsd <= 1.0):
            errors.append(ValidationError(
                level="error", path=f"{path}.process_drift.jsd_threshold",
                message="jsd_threshold must be in (0.0, 1.0]",
                code="INVALID_JSD_THRESHOLD",
            ))
        action = pd.get("action", "log")
        if action not in _VALID_DRIFT_ACTIONS:
            errors.append(ValidationError(
                level="error", path=f"{path}.process_drift.action",
                message=f"process_drift.action must be one of {_VALID_DRIFT_ACTIONS}",
                code="INVALID_DRIFT_ACTION",
            ))

    return errors
