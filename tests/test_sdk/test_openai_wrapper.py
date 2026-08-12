# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for WrappedOpenAI — enforced drop-in for openai.OpenAI/AsyncOpenAI.

Invariants pinned here:
  * wrap() over a fake client whose type name or module contains "openai"
    must return a WrappedOpenAI (not WrappedAnthropic).
  * A sync completion call with an ALLOWED tool must return the provider
    response unmodified.
  * A sync completion call with a DENIED tool (in blocklist) must raise
    ContractBreachError before the inner client is called.
  * The async dispatch path (iscoroutinefunction detects an async create())
    must also honour ALLOW and DENY correctly.
  * __repr__ must include the contract name (observability, not cosmetics).
  * __getattr__ must pass through unknown attributes to the inner client.

The module-name trick: the fake classes are defined in this module
(test_openai_wrapper), so type.__module__ contains "openai" and the
wrap() dispatcher routes them correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.sdk.wrapper import wrap
from agentassert_abc.sdk.wrappers.openai_wrapper import WrappedOpenAI

FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


# ---------------------------------------------------------------------------
# Fake OpenAI clients — sync and async
# ---------------------------------------------------------------------------


class _FakeOpenAICompletions:
    """Sync completions object whose create() returns a minimal response object."""

    def create(self, **kwargs: Any) -> Any:
        return type(
            "ChatCompletion",
            (),
            {
                "id": "chatcmpl-test",
                "choices": [],
                "usage": type(
                    "Usage", (), {"prompt_tokens": 5, "completion_tokens": 3}
                )(),
            },
        )()


class _FakeOpenAIChat:
    def __init__(self) -> None:
        self.completions = _FakeOpenAICompletions()


class _FakeOpenAI:
    """Sync OpenAI client stand-in.  Module contains 'openai' via this file's name."""

    def __init__(self) -> None:
        self.chat = _FakeOpenAIChat()
        self._extra_attr = "extra"


class _FakeAsyncOpenAICompletions:
    """Async completions object — iscoroutinefunction(create) must be True."""

    async def create(self, **kwargs: Any) -> Any:
        return type(
            "ChatCompletion",
            (),
            {
                "id": "chatcmpl-async",
                "choices": [],
                "usage": type(
                    "Usage", (), {"prompt_tokens": 7, "completion_tokens": 4}
                )(),
            },
        )()


class _FakeAsyncOpenAIChat:
    def __init__(self) -> None:
        self.completions = _FakeAsyncOpenAICompletions()


class _FakeAsyncOpenAI:
    """Async OpenAI client stand-in."""

    def __init__(self) -> None:
        self.chat = _FakeAsyncOpenAIChat()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _messages_with_tool(tool_name: str) -> list[dict[str, Any]]:
    """Build a messages list that names a specific tool_call function."""
    return [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": tool_name}}],
        }
    ]


def _messages_no_tool() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# wrap() dispatch for OpenAI clients
# ---------------------------------------------------------------------------


class TestWrapOpenAIDispatch:
    def test_wrap_returns_wrapped_openai(self) -> None:
        """wrap() must route openai-typed clients to WrappedOpenAI.

        The fake class is defined in this module (test_openai_wrapper), so
        type.__module__ contains 'openai' and the dispatcher matches.
        """
        client = _FakeOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        assert isinstance(wrapped, WrappedOpenAI)

    def test_wrap_repr_includes_contract_name(self) -> None:
        """WrappedOpenAI repr must embed the contract name for observability."""
        client = _FakeOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        assert "safety-minimal" in repr(wrapped)

    def test_wrap_getattr_passes_through_to_inner_client(self) -> None:
        """Attributes not on WrappedOpenAI must delegate to the inner client."""
        client = _FakeOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        # _extra_attr is on _FakeOpenAI but not on WrappedOpenAI.
        assert wrapped._extra_attr == "extra"


# ---------------------------------------------------------------------------
# Sync ALLOW path
# ---------------------------------------------------------------------------


class TestSyncAllow:
    def test_sync_create_allow_returns_response(self) -> None:
        """Sync create() with an allowed tool must return the inner response."""
        client = _FakeOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        result = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=_messages_no_tool(),
            max_tokens=10,
        )
        # Allowed path: inner response object returned.
        assert result.id == "chatcmpl-test"

    def test_sync_create_no_tool_uses_default_tool_name(self) -> None:
        """When messages carry no tool_call, the pre_event tool is 'openai.chat.completion'.

        This exercises _extract_openai_tool's fall-through to the default and
        must complete without ContractBreachError (default is not blocklisted).
        """
        client = _FakeOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        # Should not raise.
        wrapped.chat.completions.create(model="gpt-4o", messages=[], max_tokens=5)


# ---------------------------------------------------------------------------
# Sync DENY path
# ---------------------------------------------------------------------------


class TestSyncDeny:
    def test_sync_create_deny_raises_breach_error(self) -> None:
        """Sync create() with a blocklisted tool must raise ContractBreachError.

        safety-minimal.yaml blocks "curl|bash" — the tool name "bash" matches
        that regex pattern and must be denied without calling the inner client.
        """
        inner_called: list[bool] = []

        class _SpyCompletions(_FakeOpenAICompletions):
            def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
                inner_called.append(True)
                return super().create(**kwargs)

        class _SpyChat(_FakeOpenAIChat):
            def __init__(self) -> None:
                self.completions = _SpyCompletions()

        class _SpyClient(_FakeOpenAI):
            def __init__(self) -> None:
                self.chat = _SpyChat()
                self._extra_attr = "extra"

        wrapped = wrap(_SpyClient(), str(FIXTURES / "safety-minimal.yaml"))
        with pytest.raises(ContractBreachError):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=_messages_with_tool("bash"),
                max_tokens=10,
            )
        # Inner create must NOT have been called on a DENY.
        assert inner_called == [], "inner client was called despite DENY decision"


# ---------------------------------------------------------------------------
# Async ALLOW path
# ---------------------------------------------------------------------------


class TestAsyncAllow:
    async def test_async_create_allow_returns_response(self) -> None:
        """Async create() dispatched when inner create() is a coroutine function."""
        client = _FakeAsyncOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        # wrap() returns a WrappedOpenAI; .create() returns a coroutine because
        # iscoroutinefunction detected the async inner create.
        coro = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=_messages_no_tool(),
            max_tokens=10,
        )
        result = await coro
        assert result.id == "chatcmpl-async"


# ---------------------------------------------------------------------------
# Async DENY path
# ---------------------------------------------------------------------------


class TestAsyncDeny:
    async def test_async_create_deny_raises_breach_error(self) -> None:
        """Async path must also honour DENY and raise ContractBreachError."""
        client = _FakeAsyncOpenAI()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        with pytest.raises(ContractBreachError):
            await wrapped.chat.completions.create(
                model="gpt-4o",
                messages=_messages_with_tool("bash"),
                max_tokens=10,
            )


# ---------------------------------------------------------------------------
# Direct WrappedOpenAI construction (bypasses wrap() dispatch)
# ---------------------------------------------------------------------------


class TestDirectWrappedOpenAI:
    def test_direct_construction_with_enforcer(self) -> None:
        """WrappedOpenAI can be constructed directly with an enforcer instance."""
        enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "safety-minimal.yaml"))
        client = _FakeOpenAI()
        wrapped = WrappedOpenAI(client, enforcer)
        assert isinstance(wrapped, WrappedOpenAI)

    def test_post_action_records_usage_tokens(self) -> None:
        """After an ALLOW, _post() must evaluate a PostAction with token counts.

        This exercises the enforcer.evaluate(PostAction(...)) call in _post()
        and verifies the enforcer did not record a hard violation for the token
        accounting path (no cost_ceiling configured).
        """
        enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "safety-minimal.yaml"))
        client = _FakeOpenAI()
        wrapped = WrappedOpenAI(client, enforcer)
        wrapped.chat.completions.create(model="gpt-4o", messages=_messages_no_tool())
        # No hard violations from the PostAction token recording.
        assert enforcer._violations.hard_count() == 0
