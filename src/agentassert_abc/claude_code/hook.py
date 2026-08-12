# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Stdin/stdout hook for Claude Code's PreToolUse/PostToolUse events.

Ported from agentassert-typec-claude-code's `hook.py`. `SessionMonitor` ->
`SessionEnforcer` (port-delta §C4). Fail-open by design: any error reading
stdin, loading the contract, or evaluating the event allows the tool call
through rather than blocking the user's session — matches typec's original
behavior, appropriate for a hook that must never hang or crash Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from agentassert_abc.exceptions import ContractLoadError
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import PostAction, PreAction, TypeCEvent
from agentassert_abc.gateway.state import (
    HOOK_PROVIDED_FIELDS,
    assert_evaluable_on_response_surface,
    flatten_output,
)

_enforcer_cache: dict[str, SessionEnforcer] = {}


def _get_enforcer(contract_path: str) -> SessionEnforcer | None:
    if not contract_path:
        return None
    if contract_path in _enforcer_cache:
        return _enforcer_cache[contract_path]
    try:
        enforcer = SessionEnforcer.from_yaml(contract_path)
        # Refuse a contract this surface can never evaluate. The hook only sees
        # the tool call and its output, so an invariant over anything else would
        # score as a violation on every turn no matter how the agent behaved.
        assert_evaluable_on_response_surface(
            enforcer._contract, "Claude Code hook", HOOK_PROVIDED_FIELDS
        )
        _enforcer_cache[contract_path] = enforcer
        return enforcer
    except ContractLoadError as exc:
        # Still fail-open (the hook must never block the user's session), but say
        # why on stderr rather than silently enforcing nothing.
        print(f"[agentassert] contract not loaded: {exc}", file=sys.stderr)
        return None
    except Exception:  # noqa: BLE001 — fail-open, hook must never block on a bad contract.
        return None


def _event_from_hook(
    hook_type: str, data: dict[str, Any], session_id: str, contract_id: str
) -> TypeCEvent | None:
    tool_name = data.get("tool_name", data.get("tool_name_input", {}).get("tool_name", ""))
    if not tool_name:
        tool_name = str(data.get("tool_name_input", ""))

    if hook_type == "PreToolUse":
        return PreAction(
            session_id=session_id,
            contract_id=contract_id,
            tool=tool_name,
            args=data.get("tool_input", {}),
        )
    if hook_type == "PostToolUse":
        tool_output = data.get("tool_output", data.get("tool_response"))
        # Flatten the tool output into `output.*`. Passing no state at all scored
        # every semantic invariant as a violation regardless of agent behaviour.
        state = {"tool.name": tool_name}
        state.update(flatten_output(tool_output))
        return PostAction(
            session_id=session_id,
            contract_id=contract_id,
            tool=tool_name,
            state=state,
            result=tool_output,
        )
    return None


def main() -> None:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        print(json.dumps({"action": "allow"}))
        return

    contract_path = os.environ.get("AGENTASSERT_CONTRACT", "")
    enforcer = _get_enforcer(contract_path)

    if enforcer is None:
        print(json.dumps({"action": "allow"}))
        return

    hook_type = data.get("hook_type", data.get("hook_event_name", ""))
    session_id = data.get("session_id", "default")
    contract_id = enforcer._contract.name

    event = _event_from_hook(hook_type, data, session_id, contract_id)

    if event is None:
        print(json.dumps({"action": "allow"}))
        return

    try:
        result = enforcer.evaluate(event)
        if result.is_deny():
            print(
                json.dumps(
                    {
                        "action": "block",
                        "reason": result.reason,
                        "violation": result.violation_name,
                    }
                )
            )
        elif result.is_modify() and result.modified_args:
            print(json.dumps({"action": "modify", "tool_input": result.modified_args}))
        else:
            print(json.dumps({"action": "allow"}))
    except Exception:  # noqa: BLE001 — fail-open, hook must never block the user's session.
        print(json.dumps({"action": "allow"}))


if __name__ == "__main__":
    main()
