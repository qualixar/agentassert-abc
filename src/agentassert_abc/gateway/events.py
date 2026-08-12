# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Gateway event types (Type C consolidation).

Ported from agentassert-typec's `models/events.py` + `models/session.py`
. These are the hot-path event dataclasses the
enforcement plane (:mod:`agentassert_abc.gateway`) dispatches on — distinct
from the measurement plane's plain ``state: dict`` input to
:class:`agentassert_abc.monitor.session.SessionMonitor`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionContext:
    """Per-session context passed with a PreAction event.

    Tracks which fields the agent has "stated" this session/turn, used by
    the ``must_state`` process operator.
    """

    session_id: str
    stated_fields: frozenset[str] = field(default_factory=frozenset)
    turn_index: int = 0
    token_count: int = 0

    def has_stated_field(self, name: str) -> bool:
        return name in self.stated_fields


@dataclass
class HistoryDigest:
    """Compact summary of conversation history, attached to TurnStart."""

    turn_count: int = 0
    total_tokens: int = 0
    role_pattern: str = ""


@dataclass
class DriftReport:
    """Session-end drift summary.

    Port note: typec's original ``DriftTracker``
    exposed ``current_jsd()`` (pure Jensen-Shannon divergence over a tool
    window) and per-tool frequency via ``report()``. abc v2's
    ``agentassert_abc.metrics.drift.DriftTracker`` (the keeper — typec's own
    tracker is discarded) has no equivalent: it only exposes the *composite*
    D(t) = w_c*(1-C(t)) + w_d*JSD(t) via ``mean_drift``/``history``, mixing
    compliance and distributional shift.

    ``SessionEnforcer.close()`` therefore builds this report from:
    - ``current_jsd`` -> abc DriftTracker's ``mean_drift`` (composite score,
      NOT pure JSD — documented behavior change, not a silent stub).
    - ``tool_distribution`` -> computed locally from the enforcer's own
      bounded tool-call history (abc's tracker keeps no public per-tool
      distribution).
    """

    current_jsd: float = 0.0
    tool_distribution: dict[str, float] = field(default_factory=dict)
    window_size: int = 0
    violation_count: int = 0


@dataclass(frozen=True, kw_only=True)
class TypeCEvent:
    """Base for all enforcement-plane events."""

    session_id: str
    contract_id: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, kw_only=True)
class PreAction(TypeCEvent):
    """Fired before a tool call executes — the enforcer may DENY it."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    context: SessionContext | None = None


@dataclass(frozen=True, kw_only=True)
class PostAction(TypeCEvent):
    """Fired after a tool call executes — drives compliance/drift metrics."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TurnStart(TypeCEvent):
    user_input: str
    history_summary: HistoryDigest | None = None


@dataclass(frozen=True, kw_only=True)
class TurnEnd(TypeCEvent):
    assistant_output: str
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SessionStart(TypeCEvent):
    workdir: str
    model: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SessionEnd(TypeCEvent):
    theta: float
    theta_penalty: float = 0.0
    drift_report: DriftReport | None = None


@dataclass(frozen=True, kw_only=True)
class ContextWindow(TypeCEvent):
    token_count: int
    prefix_hash: str
