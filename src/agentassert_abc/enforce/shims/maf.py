# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Microsoft Agent Framework shim.

Targets ``agent_framework.FunctionMiddleware``: each middleware receives a
``FunctionInvocationContext`` scoped to one tool invocation and a ``next``
callback. Declining to call ``next`` is how a call is blocked.

Usage::

    from agentassert_abc.enforce import bridge_from_yaml
    from agentassert_abc.enforce.shims import ContractFunctionMiddleware

    guard = bridge_from_yaml("contract.yaml", surface="maf")
    agent = ChatAgent(..., middleware=[ContractFunctionMiddleware(guard)])

Written as a class with an async ``process`` so it satisfies MAF's class-based
middleware protocol; ``__call__`` is aliased to it for the function-based form.
It does not subclass ``FunctionMiddleware``, which keeps this module importable
without ``agent-framework`` installed — MAF accepts any object with the right
shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agentassert_abc.enforce.bridge import EnforcementBridge

__all__ = ["ContractFunctionMiddleware"]


class ContractFunctionMiddleware:
    """Enforces a behavioral contract on every MAF function invocation."""

    def __init__(self, bridge: EnforcementBridge) -> None:
        self._bridge = bridge

    async def process(
        self,
        context: Any,
        next: Callable[[Any], Awaitable[None]],  # noqa: A002 - MAF's name
    ) -> None:
        tool = _function_name(context)
        args = _arguments(context)

        decision = self._bridge.before_tool(tool, args)
        if not decision.allowed:
            # `next` is never awaited, so the function never runs. MAF returns
            # whatever sits on `context.result` to the model.
            _set_result(
                context,
                f"[AgentAssert] blocked: {decision.reason}. "
                "The tool did not run. Choose a different action.",
            )
            _mark_terminated(context)
            return

        if decision.modified:
            _set_arguments(context, decision.arguments)

        await next(context)

        if not decision.evaluated:
            return

        result = getattr(context, "result", None)
        outcome = self._bridge.after_tool(
            tool,
            decision.arguments,
            result,
            text=result if isinstance(result, str) else "",
            force_redact=decision.redact_result,
        )
        if not outcome.allowed:
            _set_result(
                context,
                f"[AgentAssert] output withheld: {outcome.reason}. "
                "The tool executed, but its result was not returned.",
            )
        elif outcome.redacted:
            _set_result(context, outcome.redacted_text or "")

    # MAF also accepts plain callables as function middleware.
    __call__ = process


def _function_name(context: Any) -> str:
    function = getattr(context, "function", None)
    name = getattr(function, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(getattr(context, "function_name", "") or "")


def _arguments(context: Any) -> dict[str, Any]:
    raw = getattr(context, "arguments", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _set_arguments(context: Any, args: dict[str, Any]) -> None:
    try:
        context.arguments = args
    except AttributeError:
        return


def _set_result(context: Any, value: str) -> None:
    try:
        context.result = value
    except AttributeError:
        return


def _mark_terminated(context: Any) -> None:
    """Signal MAF to stop the pipeline where the version supports it."""
    if hasattr(context, "terminate"):
        try:
            context.terminate = True
        except AttributeError:  # pragma: no cover - read-only property
            return
