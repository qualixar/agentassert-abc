# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Anthropic-compatible proxy routes.

Ported from typec (import paths only), plus one correctness fix: `count_tokens`
returned the raw `httpx.Response` directly where FastAPI expects a Starlette
`Response` — harmless in typec's own test (which never asserted on body/
headers), but broken against a real ASGI server. Wrapped into a proper
`fastapi.responses.Response` here.
"""

from __future__ import annotations

import uuid

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response

from agentassert_abc.proxy.enforcement import enforce_and_forward
from agentassert_abc.proxy.forwarder import forward_raw
from agentassert_abc.proxy.normalizer.anthropic_norm import normalize_anthropic

router = APIRouter()


@router.api_route("/v1/messages", methods=["POST"])
@router.api_route("/v1/messages/{path:path}", methods=["POST"])
async def messages(request: Request, path: str = "") -> Response:
    enforcer = request.app.state.monitor
    body = await request.body()
    payload = orjson.loads(body)
    session_id = request.headers.get("X-AgentAssert-Session", str(uuid.uuid4()))
    request_id = str(uuid.uuid4())
    canonical = normalize_anthropic(payload, session_id, request_id)
    return await enforce_and_forward(
        canonical, enforcer, request, "/v1/messages", request.app.state.upstream_overrides
    )


#: Headers httpx already applied (decompression) or that Starlette recomputes
#: itself — forwarding them verbatim would corrupt or double-encode the body.
_STRIP_ON_REWRAP = frozenset({"content-encoding", "content-length", "transfer-encoding"})


@router.api_route("/v1/messages/count_tokens", methods=["POST"])
async def count_tokens(request: Request) -> Response:
    body = await request.body()
    payload = orjson.loads(body)
    upstream_resp = await forward_raw(
        "anthropic", "/v1/messages/count_tokens", request, payload=payload
    )
    headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in _STRIP_ON_REWRAP
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=headers,
        media_type=upstream_resp.headers.get("content-type"),
    )
