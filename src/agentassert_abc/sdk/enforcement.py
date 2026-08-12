# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Shared pre-call enforcement helpers for the SDK wrappers.

Ported from agentassert-typec-sdk's `enforcement.py`. `SessionMonitor` ->
`SessionEnforcer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.gateway.events import PreAction

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer


def build_pre_action(
    tool_name: str, args: dict[str, Any], session_id: str, contract_id: str
) -> PreAction:
    return PreAction(
        session_id=session_id,
        contract_id=contract_id,
        tool=tool_name,
        args=args,
    )


def check_and_raise(enforcer: SessionEnforcer, event: PreAction) -> dict[str, Any] | None:
    """Evaluate `event`; raise on DENY, return modified args on MODIFY, else None."""
    result = enforcer.evaluate(event)
    if result.is_deny():
        raise ContractBreachError(
            violation_name=result.violation_name,
            reason=result.reason,
            tool=event.tool,
            session_id=event.session_id,
            contract_id=event.contract_id,
        )
    if result.is_modify() and result.modified_args:
        return result.modified_args
    return None
