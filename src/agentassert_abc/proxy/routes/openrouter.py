# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""OpenRouter-compatible proxy routes. Ported unchanged (import paths only)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import orjson
from fastapi import APIRouter, Request

from agentassert_abc.proxy.enforcement import enforce_and_forward
from agentassert_abc.proxy.normalizer.openrouter_norm import normalize_openrouter

if TYPE_CHECKING:
    from fastapi.responses import Response

router = APIRouter()


@router.api_route("/v1/chat/completions", methods=["POST"])
async def completions(request: Request) -> Response:
    enforcer = request.app.state.monitor
    body = await request.body()
    payload = orjson.loads(body)
    session_id = request.headers.get("X-AgentAssert-Session", str(uuid.uuid4()))
    request_id = str(uuid.uuid4())
    canonical = normalize_openrouter(payload, session_id, request_id)
    return await enforce_and_forward(
        canonical, enforcer, request, "/v1/chat/completions", request.app.state.upstream_overrides
    )
