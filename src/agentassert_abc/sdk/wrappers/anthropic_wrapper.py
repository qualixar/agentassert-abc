# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""WrappedAnthropic — enforced drop-in for `anthropic.Anthropic`/`AsyncAnthropic`.

Ported from agentassert-typec-sdk's `anthropic_wrapper.py`. `SessionMonitor`
-> `SessionEnforcer` (port-delta §C4).

Sync/async dispatch: typec's original only ever called the wrapped client's
methods synchronously, which silently returns an un-awaited coroutine (never
executed) when the caller passes an `AsyncAnthropic` client — despite the
docstring claiming async support. This port fixes that: `_WrappedMessages`
introspects `client.messages.create` with `inspect.iscoroutinefunction()`
once at construction and dispatches to a sync or async implementation
accordingly, so `wrap(AsyncAnthropic(), ...)` returns a client whose
`.messages.create(...)` is itself awaitable, mirroring the real SDK.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from agentassert_abc.gateway.events import PostAction
from agentassert_abc.sdk.enforcement import build_pre_action, check_and_raise

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer


class WrappedAnthropic:
    def __init__(self, client: Any, enforcer: SessionEnforcer) -> None:
        self._client = client
        self._enforcer = enforcer
        self._messages = _WrappedMessages(client, enforcer)

    @property
    def messages(self) -> _WrappedMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"WrappedAnthropic(contract={self._enforcer._contract.name})"


class _WrappedMessages:
    def __init__(self, client: Any, enforcer: SessionEnforcer) -> None:
        self._client = client.messages
        self._enforcer = enforcer
        self._is_async = inspect.iscoroutinefunction(self._client.create)

    def create(self, **kwargs: Any) -> Any:
        if self._is_async:
            return self._acreate(**kwargs)
        return self._screate(**kwargs)

    def _screate(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        response = self._client.create(**kwargs)
        self._post(tool_name, response)
        return response

    async def _acreate(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        response = await self._client.create(**kwargs)
        self._post(tool_name, response)
        return response

    def stream(self, **kwargs: Any) -> Any:
        if self._is_async:
            return self._astream(**kwargs)
        return self._sstream(**kwargs)

    def _sstream(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        stream = self._client.stream(**kwargs)

        def monitored_stream() -> Any:
            yield from stream
            self._post(tool_name, None)

        return monitored_stream()

    async def _astream(self, **kwargs: Any) -> Any:
        tool_name, kwargs = self._pre(kwargs)
        stream = self._client.stream(**kwargs)

        async def monitored_stream() -> Any:
            async for chunk in stream:
                yield chunk
            self._post(tool_name, None)

        return monitored_stream()

    def _pre(self, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        messages = kwargs.get("messages", [])
        tool_name = _extract_anthropic_tool(messages)
        event = build_pre_action(tool_name, kwargs, "sdk-session", self._enforcer._contract.name)
        modified = check_and_raise(self._enforcer, event)
        if modified:
            kwargs = {**kwargs, **modified}
        return tool_name, kwargs

    def _post(self, tool_name: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        self._enforcer.evaluate(
            PostAction(
                session_id="sdk-session",
                contract_id=self._enforcer._contract.name,
                tool=tool_name,
                state={"input_tokens": input_tokens, "output_tokens": output_tokens},
            )
        )


def _extract_anthropic_tool(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return block.get("name", "unknown")
    return "anthropic.messages"
