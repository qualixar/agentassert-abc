# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""CrewAI shim.

Targets ``crewai.hooks`` — a ``BeforeToolCallHook`` receives a
``ToolCallHookContext`` (``tool_name``, ``tool_input``, ``agent``, ``task``,
``crew``) and blocks the call by returning ``False``.

Usage::

    from agentassert_abc.enforce import bridge_from_yaml
    from agentassert_abc.enforce.shims import (
        crewai_before_tool_hook, crewai_after_tool_hook,
    )

    guard = bridge_from_yaml("contract.yaml", surface="crewai")
    crew = Crew(
        agents=[...],
        before_tool_call_hooks=[crewai_before_tool_hook(guard)],
        after_tool_call_hooks=[crewai_after_tool_hook(guard)],
    )

Because CrewAI signals refusal with a bare ``False``, the reason is written to
``context.tool_input`` metadata where CrewAI surfaces it, and always logged
through the violation log the contract already owns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentassert_abc.enforce.bridge import EnforcementBridge

__all__ = ["crewai_after_tool_hook", "crewai_before_tool_hook"]


def crewai_before_tool_hook(bridge: EnforcementBridge) -> Callable[[Any], bool]:
    """Build a ``BeforeToolCallHook`` that enforces ``bridge``'s contract.

    Returns:
        A callable taking CrewAI's ``ToolCallHookContext`` and returning
        ``False`` to block the tool call, ``True`` to permit it.
    """

    def before_tool_call(context: Any) -> bool:
        tool = _tool_name(context)
        args = _tool_input(context)

        decision = bridge.before_tool(tool, args)
        if not decision.allowed:
            _annotate(context, decision.reason, decision.violation)
            return False

        if decision.modified:
            # CrewAI reads the (possibly mutated) tool_input after the hook
            # chain, so rewriting it here is what makes MODIFY take effect.
            _set_tool_input(context, decision.arguments)
        return True

    return before_tool_call


def crewai_after_tool_hook(bridge: EnforcementBridge) -> Callable[[Any], bool]:
    """Build an ``AfterToolCallHook`` that scores the tool's result.

    A withheld result is replaced with an explanatory message rather than
    dropped, so the agent learns why and can choose another action.
    """

    def after_tool_call(context: Any) -> bool:
        tool = _tool_name(context)
        args = _tool_input(context)
        result = getattr(context, "tool_result", None)

        outcome = bridge.after_tool(tool, args, result, text=_as_text(result))
        if not outcome.allowed:
            _set_tool_result(
                context,
                f"[AgentAssert] output withheld: {outcome.reason}. "
                "The tool executed, but its result was not returned.",
            )
            return False
        if outcome.redacted:
            _set_tool_result(context, outcome.redacted_text or "")
        return True

    return after_tool_call


# ---------------------------------------------------------------------------
# Structural accessors — tolerate field renames without crashing an agent run
# ---------------------------------------------------------------------------


def _tool_name(context: Any) -> str:
    return str(getattr(context, "tool_name", "") or "")


def _tool_input(context: Any) -> dict[str, Any]:
    raw = getattr(context, "tool_input", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _set_tool_input(context: Any, args: dict[str, Any]) -> None:
    try:
        context.tool_input = args
    except AttributeError:
        return


def _set_tool_result(context: Any, value: str) -> None:
    try:
        context.tool_result = value
    except AttributeError:
        return


def _annotate(context: Any, reason: str, violation: str) -> None:
    """Record why the call was blocked where CrewAI will surface it."""
    message = f"[AgentAssert] blocked: {reason}" + (f" [{violation}]" if violation else "")
    for attr in ("blocked_reason", "error", "message"):
        if hasattr(context, attr):
            try:
                setattr(context, attr, message)
            except AttributeError:  # pragma: no cover - read-only property
                continue
            return


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else ""
