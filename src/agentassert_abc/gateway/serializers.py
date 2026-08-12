# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Serializers for SessionEnforcer sub-systems <-> SessionStore.

Ported from the Type C persistence serializers. `dump_theta`/`load_theta`
are unchanged (abc v2's
`ThetaScorer`, from Phase B, keeps the exact same private field names as
typec's). `dump_drift`/`load_drift` are REWRITTEN — abc v2's
`agentassert_abc.metrics.drift.DriftTracker` has a completely different
internal shape (`_history`, `_action_window`, `_reference`, `_config`) from
typec's discarded tracker (`_call_sequence`, `_baseline_counts`,
`_current_counts`, `_total_updates`); see

Rules (carried over from typec):
- `_seen_tools_turn` is NOT persisted (per-turn, resets every turn).
- ViolationLog's `_log` is a deque[dict] — serialized directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer
    from agentassert_abc.gateway.violation_log import ViolationLog
    from agentassert_abc.metrics.drift import DriftTracker
    from agentassert_abc.metrics.theta import ThetaScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ThetaScorer (abc v2 — same private field names as typec's)
# ---------------------------------------------------------------------------


def dump_theta(theta: ThetaScorer) -> dict[str, Any]:
    return {
        "compliance_scores": list(theta._compliance_scores),
        "drift_scores": list(theta._drift_scores),
        "violation_count": theta._violation_count,
        "recovery_attempts": theta._recovery_attempts,
        "recovery_successes": theta._recovery_successes,
        "penalty_sum": theta._penalty_sum,
    }


def load_theta(theta: ThetaScorer, data: dict[str, Any]) -> None:
    try:
        theta._compliance_scores = list(data.get("compliance_scores", []))
        theta._drift_scores = list(data.get("drift_scores", []))
        theta._violation_count = int(data.get("violation_count", 0))
        theta._recovery_attempts = int(data.get("recovery_attempts", 0))
        theta._recovery_successes = int(data.get("recovery_successes", 0))
        penalty_sum = float(data.get("penalty_sum", 0.0))
        if not (0.0 <= penalty_sum <= 10.0):
            logger.warning("Loaded penalty_sum %s is out of range, clamping", penalty_sum)
            penalty_sum = max(0.0, min(10.0, penalty_sum))
        theta._penalty_sum = penalty_sum
    except Exception as exc:
        logger.warning("load_theta failed: %s — using fresh state", exc)


# ---------------------------------------------------------------------------
# DriftTracker (abc v2 API rewrite, NOT typec's shape)
# ---------------------------------------------------------------------------


def dump_drift(drift: DriftTracker) -> dict[str, Any]:
    return {
        "history": list(drift.history),
        "action_window": list(drift._action_window),
        "reference": dict(drift._reference) if drift._reference is not None else None,
    }


def load_drift(drift: DriftTracker, data: dict[str, Any]) -> None:
    try:
        window = drift._config.window
        drift._history = deque(data.get("history", []), maxlen=window)
        drift._action_window = deque(data.get("action_window", []), maxlen=window)
        reference = data.get("reference")
        drift._reference = dict(reference) if reference is not None else None
    except Exception as exc:
        logger.warning("load_drift failed: %s — using fresh state", exc)


# ---------------------------------------------------------------------------
# ViolationLog
# ---------------------------------------------------------------------------


def dump_violations(log: ViolationLog) -> list[dict[str, Any]]:
    # _log is deque[dict] — dump the last 10,000 entries to bound growth.
    entries = list(log._log)
    if len(entries) > 10_000:
        entries = entries[-10_000:]
    return entries


def load_violations(log: ViolationLog, data: list[dict[str, Any]]) -> None:
    try:
        for entry in data:
            if isinstance(entry, dict):
                log._log.append(entry)
    except Exception as exc:
        logger.warning("load_violations failed: %s — using empty log", exc)


# ---------------------------------------------------------------------------
# SessionEnforcer meta
# ---------------------------------------------------------------------------


def dump_meta(enforcer: SessionEnforcer) -> dict[str, Any]:
    return {
        "turn_count": enforcer._turn_count,
        "deny_count": enforcer._deny_count,
        "seen_tools_session": sorted(enforcer._seen_tools_session),
    }


def load_meta(enforcer: SessionEnforcer, data: dict[str, Any]) -> None:
    try:
        enforcer._turn_count = int(data.get("turn_count", 0))
        enforcer._deny_count = int(data.get("deny_count", 0))
        enforcer._seen_tools_session = set(data.get("seen_tools_session", []))
        # _seen_tools_turn is NOT restored — per-turn state.
    except Exception as exc:
        logger.warning("load_meta failed: %s — using fresh state", exc)


# ---------------------------------------------------------------------------
# Phase 3: Cost state
# ---------------------------------------------------------------------------


def dump_cost(enforcer: SessionEnforcer) -> dict[str, Any]:
    return {"accumulated_usd": enforcer._accumulated_cost_usd}


def load_cost(enforcer: SessionEnforcer, data: dict[str, Any]) -> None:
    try:
        enforcer._accumulated_cost_usd = float(data.get("accumulated_usd", 0.0))
    except Exception as exc:
        logger.warning("load_cost failed: %s — using 0.0", exc)


# ---------------------------------------------------------------------------
# Phase 3: Repetition guard state
# ---------------------------------------------------------------------------


def dump_repetition(enforcer: SessionEnforcer) -> dict[str, Any]:
    return {
        "history": list(enforcer._tool_call_history),
        "hash_counts": dict(enforcer._sequence_hash_counts),
    }


def load_repetition(enforcer: SessionEnforcer, data: dict[str, Any]) -> None:
    try:
        history = data.get("history", [])
        enforcer._tool_call_history = deque(history, maxlen=1000)
        enforcer._sequence_hash_counts = defaultdict(int, data.get("hash_counts", {}))
    except Exception as exc:
        logger.warning("load_repetition failed: %s — using fresh state", exc)
