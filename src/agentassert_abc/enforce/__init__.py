# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Framework-neutral enforcement.

``agentassert_abc.integrations`` *measures* — it scores a session that already
happened. This package *enforces*: it decides, before a tool runs, whether it
may run at all.

The whole public surface is :class:`~agentassert_abc.enforce.EnforcementBridge`
plus the two verdict types it returns. Framework shims in
:mod:`agentassert_abc.enforce.shims` translate a native hook into those calls
and the verdict back into that framework's veto convention; each is short enough
to read in one sitting, and short enough to throw away when a framework changes
its hook API.

Typical use, framework-agnostic::

    from agentassert_abc.enforce import bridge_from_yaml

    guard = bridge_from_yaml("contract.yaml", surface="my-agent")

    decision = guard.before_tool("delete_table", {"name": "users"})
    if not decision:
        return f"blocked: {decision.reason}"
    result = run_tool(decision.arguments)

    outcome = guard.after_tool("delete_table", decision.arguments, result)
    if not outcome:
        return f"output withheld: {outcome.reason}"
    return outcome.redacted_text if outcome.redacted else result
"""

from __future__ import annotations

from typing import Any

from agentassert_abc.enforce.bridge import EnforcementBridge, ToolDecision, ToolOutcome

__all__ = [
    "EnforcementBridge",
    "ToolDecision",
    "ToolOutcome",
    "bridge_from_yaml",
]


def bridge_from_yaml(
    contract_path: str,
    *,
    surface: str = "bridge",
    session_id: str | None = None,
    fail_closed: bool = False,
    base_state: dict[str, Any] | None = None,
) -> EnforcementBridge:
    """Load a contract and return a bridge over it.

    Args:
        contract_path: path to the contract YAML.
        surface: short name for the integration, recorded on events.
        session_id: stable session id; generated when omitted.
        fail_closed: deny what the bridge cannot evaluate.
        base_state: fields merged into every post-tool state.

    Raises:
        ContractLoadError: if the contract cannot be loaded or is invalid.
    """
    from agentassert_abc.gateway.enforcer import SessionEnforcer

    return EnforcementBridge(
        SessionEnforcer.from_yaml(contract_path),
        surface=surface,
        session_id=session_id,
        fail_closed=fail_closed,
        base_state=base_state,
    )
