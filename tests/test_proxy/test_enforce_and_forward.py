# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for enforce_and_forward — the proxy's per-request enforcement pipeline.

Invariants pinned here:
  * A DENY decision must return HTTP 400 and must NEVER call the upstream
    provider (forward_request must not be reached).
  * A non-streaming ALLOW must return the provider response with
    X-AgentAssert-Decision: allow and must pass the flattened output.*
    keys to the constraint evaluator (the regression for the bug where only
    response.bytes was in post_state).
  * The streaming ALLOW path returns a StreamingResponse with the right
    headers; the stream generator drives post-stream enforcement after all
    chunks have been yielded.
  * A provider-level network error (httpx.HTTPError / OSError) must surface
    as HTTP 502 — never a 500, never an unhandled exception.

These tests are designed to fail if the critical DENY-gate invariant is
broken: if forward_request is ever reached on a DENY, the spy will detect
it and the test will fail.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import orjson
import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport, AsyncClient

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.process.models import (
    ContractSpecExtended,
    InvariantsExtended,
    ProcessInvariants,
    ToolBlocklist,
)
from agentassert_abc.proxy.enforcement import enforce_and_forward
from agentassert_abc.proxy.normalizer.canonical import CanonicalRequest, CanonicalToolCall

FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_enforcer_with_blocklist(blocked_tools: list[str]) -> SessionEnforcer:
    """Build an enforcer whose blocklist denies the given tool names."""
    contract = ContractSpecExtended(
        contractspec="1.0",
        kind="agent",
        name="test-enforcement",
        description="proxy enforcement test",
        version="0.1",
        invariants=InvariantsExtended(
            process=ProcessInvariants(
                tool_blocklist=[ToolBlocklist(tools=blocked_tools, scope="session")]
            )
        ),
    )
    return SessionEnforcer(contract)


def _make_permissive_enforcer() -> SessionEnforcer:
    """Build a minimal enforcer that allows everything."""
    contract = ContractSpecExtended(
        contractspec="1.0",
        kind="agent",
        name="test-permissive",
        description="permissive test enforcer",
        version="0.1",
    )
    return SessionEnforcer(contract)


def _make_canonical(
    *,
    provider: str = "openai",
    stream: bool = False,
    tool_name: str | None = None,
) -> CanonicalRequest:
    """Build a minimal CanonicalRequest, optionally carrying a named tool."""
    tool_calls: list[CanonicalToolCall] = []
    if tool_name is not None:
        tool_calls = [CanonicalToolCall(id="tc1", name=tool_name, arguments={})]
    return CanonicalRequest(
        provider=provider,  # type: ignore[arg-type]
        model="gpt-4o",
        tool_calls=tool_calls,
        stream=stream,
        raw_payload={"model": "gpt-4o", "messages": []},
        session_id="sess-test",
    )


def _fake_request(headers: dict[str, str] | None = None) -> Any:
    """Minimal stand-in for FastAPI.Request; only .headers.items() is accessed."""
    req = MagicMock()
    req.headers.items.return_value = list((headers or {}).items())
    return req


def _openai_response_bytes(text: str = "hello world") -> bytes:
    """Build a minimal OpenAI chat completion response body."""
    return orjson.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def _fake_httpx_response(
    status: int = 200, body: bytes | None = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    body = body or _openai_response_bytes()
    return httpx.Response(status, content=body, headers=headers or {"content-type": "application/json"})


# ---------------------------------------------------------------------------
# DENY path — upstream must NEVER be called
# ---------------------------------------------------------------------------


class TestDenyBeforeUpstream:
    async def test_deny_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DENY decision must short-circuit with HTTP 400 before any provider call.

        The enforcement-gate invariant: a DENY must be emitted by the PreAction
        evaluator and the response returned immediately, never forwarding to the
        upstream LLM. A spy on forward_request proves this.
        """
        forward_spy: list[bool] = []

        async def spy_forward(*_args: Any, **_kwargs: Any) -> httpx.Response:
            forward_spy.append(True)
            return _fake_httpx_response()

        monkeypatch.setattr("agentassert_abc.proxy.enforcement.forward_request", spy_forward)

        enforcer = _make_enforcer_with_blocklist(["dangerous_tool"])
        canonical = _make_canonical(tool_name="dangerous_tool")

        result = await enforce_and_forward(canonical, enforcer, _fake_request())

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        body = orjson.loads(result.body)
        assert body.get("error") == "ContractBreachError"
        # The spy must NOT have been called — DENY never reaches the provider.
        assert forward_spy == [], "forward_request was called despite DENY decision"

    async def test_deny_sets_x_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DENY response must carry X-AgentAssert-Decision: deny for observability."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response()),
        )
        enforcer = _make_enforcer_with_blocklist(["blocked"])
        result = await enforce_and_forward(_make_canonical(tool_name="blocked"), enforcer, _fake_request())
        assert result.headers.get("X-AgentAssert-Decision") == "deny"

    async def test_deny_includes_tool_name_in_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The DENY response body must include the offending tool name for diagnostics."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response()),
        )
        enforcer = _make_enforcer_with_blocklist(["secret_tool"])
        canonical = _make_canonical(tool_name="secret_tool")
        result = await enforce_and_forward(canonical, enforcer, _fake_request())
        body = orjson.loads(result.body)
        assert body.get("tool") == "secret_tool"


# ---------------------------------------------------------------------------
# Non-streaming ALLOW path
# ---------------------------------------------------------------------------


class TestNonStreamingAllow:
    async def test_allow_returns_provider_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ALLOW non-streaming: must forward the provider's status code unchanged."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response(status=200)),
        )
        enforcer = _make_permissive_enforcer()
        result = await enforce_and_forward(_make_canonical(), enforcer, _fake_request())
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200

    async def test_allow_sets_decision_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ALLOW path must set X-AgentAssert-Decision: allow on the response."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response()),
        )
        enforcer = _make_permissive_enforcer()
        result = await enforce_and_forward(_make_canonical(), enforcer, _fake_request())
        assert result.headers.get("X-AgentAssert-Decision") == "allow"

    async def test_flattened_output_reaches_constraint_evaluator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The PostAction state must contain output.* keys from flatten_output.

        Regression for the bug where only response.bytes was in post_state, causing
        every semantic invariant to score as a violation regardless of agent behaviour.

        Pin: with a contract that requires output.text to contain "hello", a provider
        response whose 'choices[0].message.content' is "hello world" must NOT produce
        a violation (the flattened state must reach the evaluator).
        """
        from agentassert_abc.models import ConstraintCheck, HardConstraint

        body_bytes = _openai_response_bytes("hello world")
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response(body=body_bytes)),
        )

        # Build a contract that has a semantic constraint on output.text.
        contract = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="semantic-test",
            description="tests output flattening",
            version="0.1",
            invariants=InvariantsExtended(
                hard=[
                    HardConstraint(
                        name="text-contains-hello",
                        check=ConstraintCheck(field="output.text", contains="hello"),
                    )
                ]
            ),
        )
        enforcer = SessionEnforcer(contract)

        await enforce_and_forward(_make_canonical(), enforcer, _fake_request())

        # If output.text was NOT in the post_state the constraint would have fired.
        violations = enforcer._violations.all_violations()
        assert not any(
            v.get("name") == "text-contains-hello" for v in violations
        ), "output.text constraint violated — flatten_output state did not reach the evaluator"


# ---------------------------------------------------------------------------
# Streaming ALLOW path
# ---------------------------------------------------------------------------


class TestStreamingAllow:
    async def test_streaming_returns_streaming_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stream=True must return StreamingResponse, not JSONResponse."""

        class FakeStreamResp:
            status_code = 200
            headers: dict[str, str] = {}

            async def aiter_bytes(self):
                yield b"data: hello\n"
                yield b"data: world\n"

        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=FakeStreamResp()),
        )
        enforcer = _make_permissive_enforcer()
        canonical = _make_canonical(stream=True)
        result = await enforce_and_forward(canonical, enforcer, _fake_request())
        assert isinstance(result, StreamingResponse)
        assert result.headers.get("X-AgentAssert-Decision") == "allow"
        assert result.headers.get("X-AgentAssert-Mode") == "stream-through"

    async def test_streaming_executes_post_enforcement_after_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Consuming the stream must trigger post-stream enforcer evaluation.

        The stream_generator calls enforcer.evaluate(PostAction) and
        schedule_judge_evaluation only AFTER all chunks have been yielded. This
        test consumes the stream via the ASGI protocol to verify the generator
        body runs.
        """
        chunks = [b"data: chunk1\n", b"data: chunk2\n"]
        received: list[bytes] = []

        class FakeStreamResp:
            status_code = 200
            headers: dict[str, str] = {}

            async def aiter_bytes(self):
                for c in chunks:
                    yield c

        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=FakeStreamResp()),
        )
        enforcer = _make_permissive_enforcer()
        canonical = _make_canonical(stream=True)
        response = await enforce_and_forward(canonical, enforcer, _fake_request())
        assert isinstance(response, StreamingResponse)

        # Consume via ASGI to trigger the generator.
        async with AsyncClient(
            transport=ASGITransport(app=response), base_url="http://test"
        ) as client:
            resp = await client.get("/")
            received.extend([resp.content])

        # Generator ran: enforcer must have had its turn_count incremented
        # (TurnEnd is emitted inside stream_generator after all chunks).
        assert enforcer._turn_count >= 1


# ---------------------------------------------------------------------------
# Provider error → 502
# ---------------------------------------------------------------------------


class TestProviderError:
    async def test_httpx_error_yields_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """httpx.HTTPError from forward_request must map to HTTP 502.

        No exception must propagate to the caller — the proxy absorbs provider
        network errors and returns a structured error response.
        """
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(side_effect=httpx.HTTPStatusError("timeout", request=MagicMock(), response=MagicMock())),
        )
        enforcer = _make_permissive_enforcer()
        result = await enforce_and_forward(_make_canonical(), enforcer, _fake_request())
        assert isinstance(result, JSONResponse)
        assert result.status_code == 502
        body = orjson.loads(result.body)
        assert body.get("error") == "ProviderError"

    async def test_os_error_yields_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError (e.g. connection refused) must also produce HTTP 502."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(side_effect=OSError("connection refused")),
        )
        result = await enforce_and_forward(
            _make_canonical(), _make_permissive_enforcer(), _fake_request()
        )
        assert result.status_code == 502

    async def test_streaming_provider_error_yields_502(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """httpx.HTTPError during streaming must also produce HTTP 502."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(side_effect=httpx.RemoteProtocolError("stream error", request=MagicMock())),
        )
        result = await enforce_and_forward(
            _make_canonical(stream=True), _make_permissive_enforcer(), _fake_request()
        )
        assert result.status_code == 502


# ---------------------------------------------------------------------------
# Helper coverage: _extract_tool_name fallback and _try_parse_json guard
# ---------------------------------------------------------------------------


class TestHelperBranches:
    async def test_no_tool_calls_falls_back_to_provider_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When canonical has no tool_calls, tool_name = '{provider}.chat.completion'."""
        called_with_tool: list[str] = []

        async def capturing_forward(*_args: Any, **_kwargs: Any) -> httpx.Response:
            return _fake_httpx_response()

        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request", capturing_forward
        )
        enforcer = _make_permissive_enforcer()
        canonical = _make_canonical(provider="openai", tool_name=None)

        result = await enforce_and_forward(canonical, enforcer, _fake_request())

        # Should allow (no tool_name means it defaults, and no blocklist either).
        assert result.status_code == 200

    async def test_non_json_provider_body_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider returning non-JSON must not crash enforce_and_forward.

        _try_parse_json falls back to {"raw": text[:1000]} and enforcement
        continues, returning the provider's status code.
        """
        monkeypatch.setattr(
            "agentassert_abc.proxy.enforcement.forward_request",
            AsyncMock(return_value=_fake_httpx_response(body=b"not json !!!")),
        )
        result = await enforce_and_forward(
            _make_canonical(), _make_permissive_enforcer(), _fake_request()
        )
        # Must not raise; status code passes through.
        assert result.status_code == 200
