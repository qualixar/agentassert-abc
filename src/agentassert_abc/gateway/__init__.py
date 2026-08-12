# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Gateway — the real-time enforcement plane (Type C consolidation).

Ported from `agentassert-typec` (MIT) into `agentassert-abc` (AGPL-3.0-or-later)
.

Where `agentassert_abc.monitor` MEASURES agent behavior (state-diff inputs,
formal proofs), this package ENFORCES it at the hot path — intercepting tool
calls before/after execution and returning ALLOW/DENY/REDACT/MODIFY/WARN
decisions in real time.

Naming note: typec's `SessionMonitor` (event-driven
enforcer) collided with abc v2's existing `SessionMonitor` (step-driven
measurement monitor). The enforcer is exported here as `SessionEnforcer`.
`TypeCDecision`/`DecisionResult`/`ContractSpecExtended`/`ProcessInvariants`
etc. live in `agentassert_abc.process.models` (Phase A) and are re-exported
here, not duplicated.
"""

from __future__ import annotations

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.content import (
    evaluate_cost_ceiling,
    evaluate_pii_filter,
    evaluate_repetition_guard,
)
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import (
    ContextWindow,
    DriftReport,
    HistoryDigest,
    PostAction,
    PreAction,
    SessionContext,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
    TypeCEvent,
)
from agentassert_abc.gateway.judge import JudgeDispatcher
from agentassert_abc.gateway.persistence import SessionStore
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.process.models import (
    ContractSpecExtended,
    DecisionResult,
    ProcessInvariants,
    TypeCDecision,
)

__all__ = [
    "CompiledContract",
    "ContextWindow",
    "ContractSpecExtended",
    "DecisionResult",
    "DriftReport",
    "HistoryDigest",
    "JudgeDispatcher",
    "PostAction",
    "PreAction",
    "ProcessInvariants",
    "SessionContext",
    "SessionEnd",
    "SessionEnforcer",
    "SessionStart",
    "SessionStore",
    "TurnEnd",
    "TurnStart",
    "TypeCDecision",
    "TypeCEvent",
    "ViolationLog",
    "evaluate_cost_ceiling",
    "evaluate_pii_filter",
    "evaluate_repetition_guard",
]
