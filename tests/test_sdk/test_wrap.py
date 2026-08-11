# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `packages/sdk/tests/test_wrap.py`.

`SessionMonitor` -> `SessionEnforcer` (port-delta §C4). Adds async-client
coverage: typec's original wrapper only ever called the inner client
synchronously (silently broken for `AsyncAnthropic`/`AsyncOpenAI` despite
the docstring claiming async support) — this port fixes that via
`asyncio.iscoroutinefunction()` dispatch, and these new tests pin it down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.sdk.wrapper import wrap

FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


class _FakeAnthropic:
    class messages:  # noqa: N801 — mirrors the real client's `.messages` attribute name.
        @staticmethod
        def create(**kwargs):
            return {
                "status": "ok",
                "usage": type("Usage", (), {"input_tokens": 100, "output_tokens": 50})(),
            }


class _FakeAsyncAnthropic:
    class messages:  # noqa: N801 — mirrors the real client's `.messages` attribute name.
        @staticmethod
        async def create(**kwargs):
            return {
                "status": "ok",
                "usage": type("Usage", (), {"input_tokens": 100, "output_tokens": 50})(),
            }


class TestWrapAnthropic:
    def test_wrap_anthropic_type(self) -> None:
        client = _FakeAnthropic()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        assert "WrappedAnthropic" in repr(wrapped)

    def test_messages_create_passes_with_allowed_tool(self) -> None:
        client = _FakeAnthropic()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        resp = wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
                }
            ],
            max_tokens=1024,
        )
        assert resp["status"] == "ok"

    def test_messages_create_denies_blocked_tool(self) -> None:
        client = _FakeAnthropic()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        with pytest.raises(ContractBreachError):
            wrapped.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
                    }
                ],
                max_tokens=1024,
            )

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            wrap("not_a_client", str(FIXTURES / "safety-minimal.yaml"))


class TestWrapAnthropicAsync:
    async def test_async_messages_create_passes(self) -> None:
        client = _FakeAsyncAnthropic()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        result = wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        resp = await result
        assert resp["status"] == "ok"

    async def test_async_messages_create_denies_blocked_tool(self) -> None:
        client = _FakeAsyncAnthropic()
        wrapped = wrap(client, str(FIXTURES / "safety-minimal.yaml"))
        with pytest.raises(ContractBreachError):
            await wrapped.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
                    }
                ],
                max_tokens=1024,
            )
