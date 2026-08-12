# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""LangChain / LangGraph shim — also covers DeerFlow, which is built on LangGraph.

Targets LangChain's middleware system (v1+), where ``wrap_tool_call(request,
handler)`` wraps each tool invocation. The handler may be called zero times
(short-circuit), once (normal), or many (retry); denying a call is simply
returning without ever calling it.

Usage::

    from langchain.agents.middleware import wrap_tool_call
    from agentassert_abc.enforce import bridge_from_yaml
    from agentassert_abc.enforce.shims import langchain_tool_middleware

    guard = bridge_from_yaml("contract.yaml", surface="langchain")
    agent = create_agent(model, tools, middleware=[
        wrap_tool_call(langchain_tool_middleware(guard)),
    ])

A denied call returns a ``ToolMessage`` carrying the reason, so the model reads
it as ordinary tool output and can pick another action. ``ToolMessage`` is
imported lazily — this module stays importable with no LangChain installed, and
``message_factory`` lets a test inject a stand-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentassert_abc.enforce.bridge import EnforcementBridge

__all__ = ["langchain_tool_middleware"]


def langchain_tool_middleware(
    bridge: EnforcementBridge,
    *,
    message_factory: Callable[[str, str], Any] | None = None,
) -> Callable[[Any, Callable[[Any], Any]], Any]:
    """Build a ``wrap_tool_call`` handler enforcing ``bridge``'s contract.

    Args:
        bridge: the loaded enforcement bridge.
        message_factory: builds the object returned for a blocked call, given
            ``(content, tool_call_id)``. Defaults to ``ToolMessage``, imported
            lazily.

    Returns:
        A callable ``(request, handler) -> result`` for LangChain's
        ``wrap_tool_call``.
    """
    build_message = message_factory or _default_message_factory

    def wrap(request: Any, handler: Callable[[Any], Any]) -> Any:
        tool, args, call_id = _read_call(request)

        decision = bridge.before_tool(tool, args)
        if not decision.allowed:
            # The handler is never invoked, so the tool never runs.
            return build_message(
                f"[AgentAssert] blocked: {decision.reason}. "
                "The tool did not run. Choose a different action.",
                call_id,
            )

        if decision.modified:
            request = _with_args(request, decision.arguments)

        result = handler(request)

        if not decision.evaluated:
            # The contract already failed to evaluate for this call; scoring the
            # result would report a violation that is our fault, not the agent's.
            return result

        outcome = bridge.after_tool(
            tool,
            decision.arguments,
            result,
            text=_result_text(result),
            force_redact=decision.redact_result,
        )
        if not outcome.allowed:
            return build_message(
                f"[AgentAssert] output withheld: {outcome.reason}. "
                "The tool executed, but its result was not returned.",
                call_id,
            )
        if outcome.redacted:
            return build_message(outcome.redacted_text or "", call_id)
        return result

    return wrap


def _default_message_factory(content: str, call_id: str) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, tool_call_id=call_id)


def _read_call(request: Any) -> tuple[str, dict[str, Any], str]:
    """Pull tool name, args and id out of a ToolCallRequest."""
    call = getattr(request, "tool_call", None)
    if isinstance(call, dict):
        args = call.get("args")
        return (
            str(call.get("name", "")),
            dict(args) if isinstance(args, dict) else {},
            str(call.get("id", "")),
        )
    # Some versions expose the fields directly on the request.
    args = getattr(request, "args", None)
    return (
        str(getattr(request, "name", "") or ""),
        dict(args) if isinstance(args, dict) else {},
        str(getattr(request, "id", "") or ""),
    )


def _with_args(request: Any, args: dict[str, Any]) -> Any:
    """Return a request carrying rewritten arguments.

    Prefers the framework's own copy helper so the rest of the request stays
    intact; falls back to mutation only when no such helper exists.
    """
    call = getattr(request, "tool_call", None)
    if isinstance(call, dict):
        new_call = {**call, "args": args}
        replace = getattr(request, "replace", None)
        if callable(replace):
            try:
                return replace(tool_call=new_call)
            except TypeError:
                pass
        try:
            request.tool_call = new_call
        except AttributeError:
            return request
        return request
    try:
        request.args = args
    except AttributeError:
        return request
    return request


def _result_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return result if isinstance(result, str) else ""
