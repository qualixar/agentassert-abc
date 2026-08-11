# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Event dispatch — routes a TypeCEvent to the right evaluator chain.

Ported from agentassert-typec's `evaluator/engine.py` (port-delta §A.2,
item #20). `_eval_pre_action` and the PreAction gate order are unchanged.

`_eval_post_action` is REWRITTEN (port-delta §C6/§C7, silent-breaks #1/#3):
typec's original called `drift.update(tool, state)` (AttributeError against
abc v2's DriftTracker) and `theta.record_action(tool)` (a documented
no-op — typec NEVER actually fed compliance/drift into ThetaScorer in
production; see CRIT note below). This version:

1. Evaluates the contract's abc-plane hard/soft invariants against
   `event.state` via `agentassert_abc.evaluator.engine.evaluate()` — the
   SAME real per-turn evaluation abc's own `SessionMonitor` uses. This is
   the source of `c_hard`/`c_soft`/`c_total` (never stubbed to 0/1 by
   fabrication — it is the live evaluation result).
2. Feeds `c_hard`/`c_soft` into `theta.record_compliance()`.
3. Feeds `c_total` + the tool label into `drift.compute_drift()` (abc v2's
   API — replaces typec's `update()`), which returns the real D(t) for
   this turn.
4. Feeds that same D(t) into `theta.record_drift()`. NOTE: this closes a
   gap that pre-existed in typec itself — `ThetaScorer.record_drift()` was
   defined but never called anywhere in typec v0.6.2's production path, so
   its `d_bar` term was silently always 0.0 there too. Wiring it here is a
   deliberate improvement made explicit, not a silent side effect.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from agentassert_abc.evaluator.engine import evaluate as evaluate_constraints
from agentassert_abc.gateway import process_eval
from agentassert_abc.gateway.events import (
    ContextWindow,
    PostAction,
    PreAction,
    SessionEnd,
    SessionStart,
    TurnEnd,
)
from agentassert_abc.process.models import DecisionResult, TypeCDecision

if TYPE_CHECKING:
    from agentassert_abc.gateway.compiler import CompiledContract
    from agentassert_abc.gateway.events import TypeCEvent
    from agentassert_abc.gateway.violation_log import ViolationLog
    from agentassert_abc.metrics.drift import DriftTracker
    from agentassert_abc.metrics.theta import ThetaScorer


def dispatch_event(
    event: TypeCEvent,
    compiled: CompiledContract,
    drift: DriftTracker,
    theta: ThetaScorer,
    violations: ViolationLog,
    seen_session: set[str] | None = None,
    seen_turn: set[str] | None = None,
    accumulated_cost: float = 0.0,
    tool_history: deque[str] | None = None,
    seq_hash_counts: dict[str, int] | None = None,
) -> DecisionResult:
    _seen_session = seen_session if seen_session is not None else set()
    _seen_turn = seen_turn if seen_turn is not None else set()
    _tool_history = tool_history if tool_history is not None else deque()
    _seq_hash_counts = seq_hash_counts if seq_hash_counts is not None else defaultdict(int)

    if isinstance(event, PreAction):
        return _eval_pre_action(
            event,
            compiled,
            _seen_session,
            _seen_turn,
            violations,
            accumulated_cost,
            _tool_history,
            _seq_hash_counts,
        )
    if isinstance(event, PostAction):
        return _eval_post_action(event, compiled, drift, theta, violations)
    if isinstance(event, TurnEnd):
        return process_eval.evaluate_turn_end_soft(event, compiled, drift, theta, violations)
    if isinstance(event, ContextWindow):
        return process_eval.evaluate_context_budget(event, compiled, violations)
    if isinstance(event, SessionStart):
        return DecisionResult(decision=TypeCDecision.ALLOW, reason="session started")
    if isinstance(event, SessionEnd):
        return DecisionResult(decision=TypeCDecision.ALLOW, reason="session ended")
    # TurnStart (and any unrecognized future event type) — no gate applies.
    return DecisionResult(decision=TypeCDecision.ALLOW)


def _eval_pre_action(
    event: PreAction,
    compiled: CompiledContract,
    seen_session: set[str],
    seen_turn: set[str],
    violations: ViolationLog,
    accumulated_cost: float,
    tool_history: deque[str],
    seq_hash_counts: dict[str, int],
) -> DecisionResult:
    from agentassert_abc.gateway.content import (
        evaluate_cost_ceiling,
        evaluate_repetition_guard,
    )

    result = process_eval.evaluate_tool_blocklist(event, compiled, violations)
    if result is not None:
        return result

    result = process_eval.evaluate_must_precede(
        event, compiled, seen_session, seen_turn, violations
    )
    if result is not None:
        return result

    result = process_eval.evaluate_tool_allowlist(event, compiled, violations)
    if result is not None:
        return result

    result = evaluate_cost_ceiling(event, compiled, accumulated_cost, violations)
    if result is not None:
        return result

    result = evaluate_repetition_guard(event, compiled, tool_history, seq_hash_counts, violations)
    if result is not None:
        return result

    result = process_eval.evaluate_must_state(event, compiled, violations)
    if result is not None:
        return result

    return DecisionResult(decision=TypeCDecision.ALLOW)


def _eval_post_action(
    event: PostAction,
    compiled: CompiledContract,
    drift: DriftTracker,
    theta: ThetaScorer,
    violations: ViolationLog,
) -> DecisionResult:
    eval_result = evaluate_constraints(compiled.spec, event.state)
    theta.record_compliance(eval_result.c_hard, eval_result.c_soft)

    # A PostAction is observed after the response exists, so it cannot be blocked
    # (decision stays ALLOW) — but an unsatisfied abc-plane constraint IS a
    # violation event and MUST be recorded, so the reliability event term (E) and
    # the ViolationLog reflect it. This mirrors the PreAction process-operator path
    # (which records denials via record_violation()); without it, continuous
    # PostAction hard-constraint failure would leave E=0 and inflate Θ.
    for r in eval_result.hard_results:
        if not r.satisfied:
            theta.record_violation()
            violations.record(
                name=r.name, event_type="post_action", tool=event.tool, reason=r.evidence
            )
    for r in eval_result.soft_results:
        if not r.satisfied:
            violations.record_soft(
                name=r.name, event_type="post_action", tool=event.tool, reason=r.evidence
            )

    total_constraints = len(eval_result.hard_results) + len(eval_result.soft_results)
    if total_constraints > 0:
        total_satisfied = sum(1 for r in eval_result.hard_results if r.satisfied) + sum(
            1 for r in eval_result.soft_results if r.satisfied
        )
        c_total = total_satisfied / total_constraints
    else:
        c_total = 1.0

    d_t = drift.compute_drift(c_total=c_total, action_dist={event.tool: 1.0})
    theta.record_drift(d_t)

    return DecisionResult(decision=TypeCDecision.ALLOW)
