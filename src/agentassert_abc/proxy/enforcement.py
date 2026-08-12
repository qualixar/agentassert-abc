# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""enforce_and_forward — the proxy's per-request enforcement pipeline.

Ported from agentassert-typec's `enforcement.py`. Renamed `monitor` (typec's
`SessionMonitor`) to `enforcer` (:class:`~agentassert_abc.gateway.SessionEnforcer`)
throughout — port-delta §C4. Cost/PII helpers are the already-ported,
public-named versions in `agentassert_abc.gateway.content` (`update_cost`,
not typec's private `_update_cost`; same for `extract_usage`,
`parse_streaming_usage`, `apply_pii_redaction`).

Enforcement-gate invariant (CRIT self-review): every path that reaches
`forward_request`/`forward_raw` is preceded by a `PreAction` DENY check that
returns *before* any upstream call. A DENY never reaches the provider.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any

import httpx
import orjson
from fastapi.responses import JSONResponse, StreamingResponse

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.gateway.content.cost import parse_streaming_usage, update_cost
from agentassert_abc.gateway.content.pii import apply_pii_redaction, evaluate_pii_filter
from agentassert_abc.gateway.events import PostAction, PreAction, TurnEnd
from agentassert_abc.gateway.state import flatten_output
from agentassert_abc.proxy.forwarder import forward_request

if TYPE_CHECKING:
    from fastapi import Request

    from agentassert_abc.gateway.enforcer import SessionEnforcer
    from agentassert_abc.proxy.normalizer.canonical import CanonicalRequest

__all__ = ["enforce_and_forward"]


async def enforce_and_forward(
    canonical: CanonicalRequest,
    enforcer: SessionEnforcer,
    raw_request: Request,
    provider_path: str = "/v1/messages",
    upstream_overrides: dict[str, str] | None = None,
) -> JSONResponse | StreamingResponse:
    tool_name = _extract_tool_name(canonical)
    pre_event = PreAction(
        session_id=canonical.session_id,
        contract_id=enforcer._contract.name,
        tool=tool_name,
        args={"model": canonical.model, "stream": canonical.stream},
    )

    result = enforcer.evaluate(pre_event)

    if result.is_deny():
        # DENY never reaches the upstream LLM — return immediately.
        breach = ContractBreachError(
            violation_name=result.violation_name,
            reason=result.reason,
            tool=tool_name,
            session_id=canonical.session_id,
            contract_id=enforcer._contract.name,
        )
        return JSONResponse(
            status_code=400,
            content=breach.to_http_body(),
            headers={
                "X-AgentAssert-Decision": "deny",
                "Content-Type": "application/json",
            },
        )

    if canonical.stream:
        return await _forward_streaming(
            canonical, enforcer, pre_event, raw_request, provider_path, upstream_overrides
        )

    _t0 = time.perf_counter()
    try:
        provider_resp = await forward_request(
            provider=canonical.provider,
            payload=canonical.raw_payload,
            raw_request=raw_request,
            path=provider_path,
            upstream_overrides=upstream_overrides,
        )
    except (httpx.HTTPError, OSError) as e:
        return JSONResponse(
            status_code=502,
            content={"error": "ProviderError", "detail": str(e)},
            headers={"X-AgentAssert-Decision": "allow"},
        )
    latency_ms = (time.perf_counter() - _t0) * 1000.0

    resp_data = _try_parse_json(provider_resp)
    response_text = _extract_text_content(resp_data)

    # Flatten the response into `output.*` so semantic invariants can actually be
    # evaluated. Passing only a byte count scored every such constraint as a
    # violation, however well the agent behaved.
    post_state: dict[str, Any] = {
        "response.bytes": len(provider_resp.content),
        "response.status": provider_resp.status_code,
        "response.latency_ms": latency_ms,
        "latency_ms": latency_ms,
        "tool.name": tool_name,
    }
    post_state.update(flatten_output(resp_data))
    if response_text:
        post_state.setdefault("output.text", response_text)

    post_event = PostAction(
        session_id=canonical.session_id,
        contract_id=enforcer._contract.name,
        tool=tool_name,
        args={"status": provider_resp.status_code},
        state=post_state,
        result=resp_data,
    )
    enforcer.evaluate(post_event)

    update_cost(resp_data, canonical, enforcer)

    pii_result = evaluate_pii_filter(
        response_text, enforcer._compiled, enforcer._violations, is_streaming=False
    )
    if pii_result is not None and pii_result.is_deny():
        return JSONResponse(
            status_code=400,
            content={"error": "ContractBreach", "detail": pii_result.reason},
            headers={"X-AgentAssert-Decision": "deny"},
        )
    if pii_result is not None and pii_result.is_redact():
        redacted_text = apply_pii_redaction(
            response_text, enforcer._compiled.pii_compiled_patterns
        )
        resp_data = _inject_redacted_content(resp_data, redacted_text, canonical.provider)

    turn_end = TurnEnd(
        session_id=canonical.session_id,
        contract_id=enforcer._contract.name,
        assistant_output=response_text,
    )
    enforcer.evaluate(turn_end)
    enforcer.schedule_judge_evaluation(response_text, canonical.session_id)

    headers = dict(provider_resp.headers)
    headers["X-AgentAssert-Decision"] = "allow"

    return JSONResponse(
        status_code=provider_resp.status_code,
        content=resp_data,
        headers=headers,
    )


async def _forward_streaming(
    canonical: CanonicalRequest,
    enforcer: SessionEnforcer,
    pre_event: PreAction,
    raw_request: Request,
    provider_path: str,
    upstream_overrides: dict[str, str] | None = None,
) -> JSONResponse | StreamingResponse:
    try:
        provider_resp = await forward_request(
            provider=canonical.provider,
            payload=canonical.raw_payload,
            raw_request=raw_request,
            path=provider_path,
            upstream_overrides=upstream_overrides,
        )
    except (httpx.HTTPError, OSError) as e:
        return JSONResponse(
            status_code=502,
            content={"error": "ProviderError", "detail": str(e)},
            headers={"X-AgentAssert-Decision": "allow"},
        )

    async def stream_generator():
        accumulated = ""
        async for chunk in provider_resp.aiter_bytes():
            accumulated += _accumulate_chunk(chunk)
            yield chunk

        stream_state: dict[str, Any] = {
            "response.bytes": len(accumulated),
            "response.streamed": True,
            "tool.name": pre_event.tool,
            "output.text": accumulated[:4096],
            "output.raw": accumulated[:4096],
        }

        post_event = PostAction(
            session_id=canonical.session_id,
            contract_id=enforcer._contract.name,
            tool=pre_event.tool,
            args=pre_event.args,
            state=stream_state,
            result={"content": accumulated[:4096]},
        )
        enforcer.evaluate(post_event)

        usage_data = parse_streaming_usage(accumulated)
        if usage_data:
            update_cost(usage_data, canonical, enforcer)

        # PII filter post-stream (log/warn only — cannot block already-yielded data).
        evaluate_pii_filter(
            accumulated, enforcer._compiled, enforcer._violations, is_streaming=True
        )

        turn_end = TurnEnd(
            session_id=canonical.session_id,
            contract_id=enforcer._contract.name,
            assistant_output=accumulated[:4096],
        )
        enforcer.evaluate(turn_end)
        enforcer.schedule_judge_evaluation(accumulated, canonical.session_id)

    return StreamingResponse(
        stream_generator(),
        status_code=provider_resp.status_code,
        headers={
            "X-AgentAssert-Decision": "allow",
            "X-AgentAssert-Mode": "stream-through",
            "Content-Type": "text/event-stream",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_name(canonical: CanonicalRequest) -> str:
    if canonical.tool_calls:
        return canonical.tool_calls[-1].name
    return f"{canonical.provider}.chat.completion"


def _try_parse_json(resp: httpx.Response) -> Any:
    try:
        return orjson.loads(resp.content)
    except orjson.JSONDecodeError:
        return {"raw": resp.text[:1000]}


def _accumulate_chunk(chunk: bytes) -> str:
    return chunk.decode("utf-8", errors="replace")


def _extract_text_content(resp_data: Any) -> str:
    """Best-effort extraction of assistant text from a parsed LLM response."""
    if not isinstance(resp_data, dict):
        return ""
    # Anthropic format.
    content = resp_data.get("content", [])
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        if parts:
            return " ".join(parts)
    # OpenAI / OpenRouter format.
    choices = resp_data.get("choices", [])
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        return msg.get("content", "") or ""
    return ""


def _inject_redacted_content(resp_data: Any, redacted_text: str, provider: str) -> Any:
    """Reconstruct resp_data with redacted text content in-place (immutable copy)."""
    if not isinstance(resp_data, dict):
        return resp_data
    data = copy.deepcopy(resp_data)

    if "content" in data and isinstance(data["content"], list):
        for block in data["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = redacted_text
                break
        return data

    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        msg = data["choices"][0].get("message", {})
        if isinstance(msg, dict):
            msg["content"] = redacted_text
        return data

    return data
