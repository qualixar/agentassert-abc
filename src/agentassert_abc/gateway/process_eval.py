# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Process-plane operator evaluators: tool_blocklist, must_precede,
tool_allowlist, must_state, context_budget, process_drift.

Ported from the Type C process evaluator. All five "PreAction gate"
evaluators are unchanged from
typec. `evaluate_turn_end_soft` (process_drift) is REWRITTEN —
§C7: abc v2's `DriftTracker` has no `current_jsd()`. The closest available
signal is the most recent composite D(t) recorded in `drift.history`
(mixing compliance + distributional shift, not pure JSD like typec's
tracker) — a documented behavior change, not a stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentassert_abc.process.models import DecisionResult, TypeCDecision

if TYPE_CHECKING:
    from agentassert_abc.gateway.compiler import CompiledContract
    from agentassert_abc.gateway.events import ContextWindow, PreAction, TurnEnd
    from agentassert_abc.gateway.violation_log import ViolationLog
    from agentassert_abc.metrics.drift import DriftTracker
    from agentassert_abc.metrics.theta import ThetaScorer

#: Θ penalty applied when process_drift's action is "theta_penalty".
#: Matches the value typec used at this call site (distinct from the
#: judge-predicate penalty of 0.03 in enforcer.py).
_PROCESS_DRIFT_PENALTY = 0.05


def evaluate_tool_blocklist(
    event: PreAction,
    compiled: CompiledContract,
    violations: ViolationLog,
) -> DecisionResult | None:
    tool = event.tool
    for pattern in compiled.tool_blocklist_patterns:
        if pattern.search(tool):
            violations.record(
                name="tool_blocklist",
                event_type="PreAction",
                tool=tool,
                reason=f"Tool '{tool}' matches blocklist pattern '{pattern.pattern}'",
            )
            return DecisionResult(
                decision=TypeCDecision.DENY,
                reason=f"ContractBreach: tool_blocklist — '{tool}' is blocked",
                violation_name="tool_blocklist",
            )
    return None


def evaluate_must_precede(
    event: PreAction,
    compiled: CompiledContract,
    seen_session: set[str],
    seen_turn: set[str],
    violations: ViolationLog,
) -> DecisionResult | None:
    tool = event.tool
    for rule in compiled.must_precede_rules:
        if rule["after"] != tool:
            continue
        seen = seen_session if rule["scope"] == "session" else seen_turn
        if rule["before"] not in seen:
            reason = (
                f"ContractBreach: must_precede — '{rule['before']}' must be called "
                f"before '{tool}' (scope={rule['scope']})"
            )
            violations.record(
                name="must_precede", event_type="PreAction", tool=tool, reason=reason
            )
            return DecisionResult(
                decision=TypeCDecision.DENY,
                reason=reason,
                violation_name="must_precede",
            )
    return None


def evaluate_tool_allowlist(
    event: PreAction,
    compiled: CompiledContract,
    violations: ViolationLog,
) -> DecisionResult | None:
    if not compiled.tool_allowlist_patterns:
        return None
    tool = event.tool
    for _scope, patterns in compiled.tool_allowlist_patterns:
        for pattern in patterns:
            if pattern.search(tool):
                return None
    violations.record(
        name="tool_allowlist",
        event_type="PreAction",
        tool=tool,
        reason=f"Tool '{tool}' not in any allowlist",
    )
    return DecisionResult(
        decision=TypeCDecision.DENY,
        reason=f"ContractBreach: tool_allowlist — '{tool}' is not permitted",
        violation_name="tool_allowlist",
    )


def evaluate_must_state(
    event: PreAction,
    compiled: CompiledContract,
    violations: ViolationLog,
) -> DecisionResult | None:
    tool = event.tool
    for rule in compiled.must_state_rules:
        for pattern in rule["patterns"]:
            if not pattern.search(tool):
                continue
            ctx = event.context
            if ctx is None or not ctx.has_stated_field(rule["field"]):
                violations.record(
                    name="must_state",
                    event_type="PreAction",
                    tool=tool,
                    reason=(
                        f"Field '{rule['field']}' not stated before '{tool}'. "
                        f"Rationale: {rule['rationale']}"
                    ),
                )
                return DecisionResult(
                    decision=TypeCDecision.DENY,
                    reason=(
                        f"ContractBreach: must_state — field '{rule['field']}' "
                        f"must be stated before calling '{tool}'. "
                        f"Rationale: {rule['rationale']}"
                    ),
                    violation_name="must_state",
                )
    return None


def evaluate_context_budget(
    event: ContextWindow,
    compiled: CompiledContract,
    violations: ViolationLog,
) -> DecisionResult:
    if compiled.context_budget_limit and event.token_count > compiled.context_budget_limit:
        action = compiled.context_budget_action
        if action == "deny":
            return DecisionResult(
                decision=TypeCDecision.DENY,
                reason=(
                    f"ContractBreach: context_budget — {event.token_count} tokens "
                    f"exceeds limit {compiled.context_budget_limit}"
                ),
                violation_name="context_budget",
            )
        if action == "warn":
            violations.record_soft(
                "context_budget",
                "ContextWindow",
                "context",
                f"{event.token_count} tokens > {compiled.context_budget_limit}",
            )
        elif action == "compress":
            return DecisionResult(decision=TypeCDecision.ALLOW, reason="compress_hint")
    return DecisionResult(decision=TypeCDecision.ALLOW)


def evaluate_turn_end_soft(
    event: TurnEnd,
    compiled: CompiledContract,
    drift: DriftTracker,
    theta: ThetaScorer,
    violations: ViolationLog,
) -> DecisionResult:
    """process_drift check at turn end.

    Uses `drift.history[-1]` (the most recently recorded composite D(t)) as
    the "current" drift signal, because `DriftTracker` exposes no isolated,
    side-effect-free "peek at current JSD" call.
    """
    if compiled.process_drift_config:
        current_drift = drift.history[-1] if drift.history else 0.0
        config = compiled.process_drift_config
        if current_drift > config.jsd_threshold:
            action = config.action
            if action == "log":
                violations.record_soft(
                    "process_drift",
                    "TurnEnd",
                    "n/a",
                    f"D(t) {current_drift:.3f} > threshold {config.jsd_threshold}",
                )
            elif action == "warn":
                violations.record_soft(
                    "process_drift",
                    "TurnEnd",
                    "n/a",
                    f"D(t) {current_drift:.3f} exceeds threshold",
                )
            elif action == "theta_penalty":
                theta.apply_penalty(_PROCESS_DRIFT_PENALTY)

    return DecisionResult(decision=TypeCDecision.ALLOW)
