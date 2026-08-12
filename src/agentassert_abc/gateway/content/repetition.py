# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Content operator: repetition_guard — detect repeated tool-call sequences.

Ported from the Type C content evaluator; behaviour unchanged, import paths
moved.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from agentassert_abc.process.models import DecisionResult, TypeCDecision

if TYPE_CHECKING:
    from collections import deque

    from agentassert_abc.gateway.compiler import CompiledContract
    from agentassert_abc.gateway.events import PreAction
    from agentassert_abc.gateway.violation_log import ViolationLog


def evaluate_repetition_guard(
    event: PreAction,
    compiled: CompiledContract,
    tool_history: deque[str],
    seq_hash_counts: dict[str, int],
    violations: ViolationLog,
) -> DecisionResult | None:
    """Deny/warn/log when the tool-call window repeats beyond `max_repeats`."""
    if compiled.repetition_guard_config is None:
        return None

    config = compiled.repetition_guard_config
    tool = event.tool

    for pattern in compiled.repetition_guard_ignore_patterns:
        if pattern.search(tool):
            return None

    # Build candidate history — don't mutate the real deque (commit happens
    # only when the caller allows the call through).
    candidate_history = [*list(tool_history), tool]

    window_size = config.window_size
    if len(candidate_history) < window_size:
        return None

    window = tuple(candidate_history[-window_size:])
    seq_key = hashlib.md5("|".join(window).encode()).hexdigest()  # noqa: S324

    current_count = seq_hash_counts.get(seq_key, 0) + 1  # +1 for this potential call

    if current_count > config.max_repeats:
        reason = (
            f"Repetition detected: [{' → '.join(window)}] "
            f"seen {current_count} times (max {config.max_repeats})"
        )
        if config.action == "deny":
            violations.record("repetition_guard", "PreAction", tool, reason)
            return DecisionResult(
                decision=TypeCDecision.DENY,
                reason=reason,
                violation_name="repetition_guard",
            )
        if config.action == "warn":
            violations.record_soft("repetition_guard", "PreAction", tool, reason)
            return None
        return None  # log: no violation entry, just pass through

    return None
