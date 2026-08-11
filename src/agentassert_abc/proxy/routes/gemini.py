# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Gemini-compatible proxy route.

Port-delta bug fix: typec's `gemini.py` called
`enforce_and_forward(canonical, monitor, request, upstream_overrides)` —
`upstream_overrides` bound positionally to the `provider_path` parameter, so
the contract's `upstream.gemini` override was silently never applied. Fixed
here by passing `upstream_overrides` as the keyword it is.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import orjson
from fastapi import APIRouter, Request

from agentassert_abc.proxy.enforcement import enforce_and_forward
from agentassert_abc.proxy.normalizer.gemini_norm import normalize_gemini

if TYPE_CHECKING:
    from fastapi.responses import Response

router = APIRouter()


@router.api_route("/v1/models/{model}:generateContent", methods=["POST"])
async def gemini_generate(request: Request, model: str) -> Response:
    enforcer = request.app.state.monitor
    body = await request.body()
    payload = orjson.loads(body)
    payload["model"] = model
    session_id = request.headers.get("X-AgentAssert-Session", str(uuid.uuid4()))
    request_id = str(uuid.uuid4())
    canonical = normalize_gemini(payload, session_id, request_id)
    return await enforce_and_forward(
        canonical,
        enforcer,
        request,
        upstream_overrides=request.app.state.upstream_overrides,
    )
