# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""OpenAI Chat Completions API -> CanonicalRequest. Ported unchanged from typec."""

from __future__ import annotations

import json
from typing import Any

from agentassert_abc.proxy.normalizer.canonical import CanonicalRequest, CanonicalToolCall


def normalize_openai(
    payload: dict[str, Any], session_id: str, request_id: str
) -> CanonicalRequest:
    tool_calls: list[CanonicalToolCall] = []
    for msg in payload.get("messages", []):
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                CanonicalToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                    provider_raw=tc,
                )
            )

    return CanonicalRequest(
        provider="openai",
        model=payload.get("model", ""),
        messages=payload.get("messages", []),
        tool_calls=tool_calls,
        stream=payload.get("stream", False),
        max_tokens=payload.get("max_tokens"),
        temperature=payload.get("temperature"),
        raw_payload=payload,
        session_id=session_id,
        request_id=request_id,
    )
