# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""WrappedOpenAI — enforced drop-in for `openai.OpenAI`/`AsyncOpenAI`.

Ported from agentassert-typec-sdk's `openai_wrapper.py`. `SessionMonitor` ->
`SessionEnforcer` (port-delta §C4). Sync/async dispatch added (see
`anthropic_wrapper.py` module docstring for the rationale) via
`inspect.iscoroutinefunction()` introspection of the wrapped
`chat.completions.create`.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from agentassert_abc.gateway.events import PostAction
from agentassert_abc.sdk.enforcement import build_pre_action, check_and_raise

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer


class WrappedOpenAI:
    def __init__(self, client: Any, enforcer: SessionEnforcer) -> None:
        self._client = client
        self._enforcer = enforcer
        self._chat = _WrappedChat(client, enforcer)

    @property
    def chat(self) -> _WrappedChat:
        return self._chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"WrappedOpenAI(contract={self._enforcer._contract.name})"


class _WrappedChat:
    def __init__(self, client: Any, enforcer: SessionEnforcer) -> None:
        self._inner = client.chat
        self._enforcer = enforcer
        self.completions = _WrappedCompletions(client.chat.completions, enforcer)


class _WrappedCompletions:
    def __init__(self, completions: Any, enforcer: SessionEnforcer) -> None:
        self._inner = completions
        self._enforcer = enforcer
        self._is_async = inspect.iscoroutinefunction(self._inner.create)

    def create(self, **kwargs: Any) -> Any:
        if self._is_async:
            return self._acreate(**kwargs)
        return self._screate(**kwargs)

    def _screate(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        response = self._inner.create(**kwargs)
        self._post(tool_name, response)
        return response

    async def _acreate(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        response = await self._inner.create(**kwargs)
        self._post(tool_name, response)
        return response

    def _pre(self, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        messages = kwargs.get("messages", [])
        tool_name = _extract_openai_tool(messages)
        event = build_pre_action(tool_name, kwargs, "sdk-session", self._enforcer._contract.name)
        modified = check_and_raise(self._enforcer, event)
        if modified:
            kwargs = {**kwargs, **modified}
        return tool_name, kwargs

    def _post(self, tool_name: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self._enforcer.evaluate(
            PostAction(
                session_id="sdk-session",
                contract_id=self._enforcer._contract.name,
                tool=tool_name,
                state={
                    "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                },
            )
        )


def _extract_openai_tool(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name:
                return name
    return "openai.chat.completion"
