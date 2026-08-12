# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""AgentScope shim.

Targets AgentScope's hooking functions. A ``ReActAgent`` supports hooks around
``reply``, ``observe``, ``print``, ``_reasoning`` and ``_acting``; pre-hooks
receive ``(agent, kwargs)`` and may return a replacement ``kwargs`` dict, and
post-hooks additionally receive the core function's ``output``.

Tool enforcement attaches to ``_acting``, which is where a ReAct agent executes
the tool it just chose.

Usage::

    from agentassert_abc.enforce import bridge_from_yaml
    from agentassert_abc.enforce.shims import register_agentscope_hooks

    guard = bridge_from_yaml("contract.yaml", surface="agentscope")
    register_agentscope_hooks(agent, guard)

AgentScope pre-hooks have no "return False to block" convention, so a denied
call raises :class:`~agentassert_abc.exceptions.ContractBreachError` — which is
the framework-idiomatic way to abort the act, and carries the reason with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentassert_abc.enforce.bridge import EnforcementBridge

__all__ = [
    "agentscope_post_acting_hook",
    "agentscope_pre_acting_hook",
    "register_agentscope_hooks",
]

_HOOK_NAME = "agentassert_contract"


def agentscope_pre_acting_hook(
    bridge: EnforcementBridge,
) -> Callable[[Any, dict[str, Any]], dict[str, Any] | None]:
    """Build a ``pre__acting`` hook enforcing ``bridge``'s contract.

    Returns:
        ``(agent, kwargs) -> kwargs | None``. Returns a replacement ``kwargs``
        when the contract rewrote the arguments, ``None`` to leave them alone,
        and raises ``ContractBreachError`` to block the act.
    """

    def pre_acting(agent: Any, kwargs: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG001
        tool, args, holder = _read_tool_call(kwargs)
        if not tool:
            return None

        decision = bridge.before_tool(tool, args)
        decision.raise_if_denied()

        if decision.modified:
            return _rewrite(kwargs, holder, decision.arguments)
        return None

    return pre_acting


def agentscope_post_acting_hook(
    bridge: EnforcementBridge,
) -> Callable[[Any, dict[str, Any], Any], Any]:
    """Build a ``post__acting`` hook that scores the tool's result."""

    def post_acting(agent: Any, kwargs: dict[str, Any], output: Any) -> Any:  # noqa: ARG001
        tool, args, _ = _read_tool_call(kwargs)
        if not tool:
            return output

        outcome = bridge.after_tool(tool, args, output, text=_as_text(output))
        outcome.raise_if_denied()
        if outcome.redacted:
            return outcome.redacted_text
        return output

    return post_acting


def register_agentscope_hooks(
    agent: Any, bridge: EnforcementBridge, *, name: str = _HOOK_NAME
) -> None:
    """Attach both acting hooks to one agent instance.

    Args:
        agent: an AgentScope agent exposing ``register_instance_hook``.
        bridge: the loaded enforcement bridge.
        name: hook name, so the pair can be replaced or removed later.

    Raises:
        TypeError: if the object does not expose ``register_instance_hook``.
    """
    register = getattr(agent, "register_instance_hook", None)
    if not callable(register):
        msg = (
            f"{type(agent).__name__} does not expose register_instance_hook; "
            "AgentScope hook registration needs an agent instance"
        )
        raise TypeError(msg)
    register("pre__acting", name, agentscope_pre_acting_hook(bridge))
    register("post__acting", name, agentscope_post_acting_hook(bridge))


# ---------------------------------------------------------------------------
# Structural accessors
# ---------------------------------------------------------------------------


def _read_tool_call(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """Find the tool call inside an ``_acting`` kwargs dict.

    AgentScope passes the chosen tool as a ``ToolUseBlock``-shaped object under
    one of a few keys depending on version, so the holder key is returned too:
    a rewrite has to put the arguments back where they came from.
    """
    for key in ("tool_call", "tool_use_block", "block", "tool"):
        candidate = kwargs.get(key)
        name, args = _unpack(candidate)
        if name:
            return name, args, key
    # Flat form: name and input sit directly on kwargs.
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        raw = kwargs.get("input", kwargs.get("arguments"))
        return name, dict(raw) if isinstance(raw, dict) else {}, None
    return "", {}, None


def _unpack(candidate: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(candidate, dict):
        name = candidate.get("name")
        raw = candidate.get("input", candidate.get("arguments"))
    else:
        name = getattr(candidate, "name", None)
        raw = getattr(candidate, "input", getattr(candidate, "arguments", None))
    if not isinstance(name, str) or not name:
        return "", {}
    return name, dict(raw) if isinstance(raw, dict) else {}


def _rewrite(kwargs: dict[str, Any], holder: str | None, args: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``kwargs`` with rewritten tool arguments."""
    if holder is None:
        return {**kwargs, "input": args}
    candidate = kwargs.get(holder)
    if isinstance(candidate, dict):
        return {**kwargs, holder: {**candidate, "input": args}}
    # A structured block: mutate the copy's field, since we cannot rebuild an
    # arbitrary framework type.
    try:
        candidate.input = args
    except AttributeError:
        return kwargs
    return {**kwargs, holder: candidate}


def _as_text(output: Any) -> str:
    return output if isinstance(output, str) else ""
