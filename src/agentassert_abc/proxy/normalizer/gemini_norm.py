# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Gemini generateContent API -> CanonicalRequest. Ported unchanged from typec."""

from __future__ import annotations

from typing import Any

from agentassert_abc.proxy.normalizer.canonical import CanonicalRequest, CanonicalToolCall


def normalize_gemini(
    payload: dict[str, Any], session_id: str, request_id: str
) -> CanonicalRequest:
    tool_calls: list[CanonicalToolCall] = []
    for content in payload.get("contents", []):
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            fc = part.get("functionCall") or part.get("function_call")
            if fc:
                tool_calls.append(
                    CanonicalToolCall(
                        id=fc.get("id", ""),
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}),
                        provider_raw=fc,
                    )
                )

    return CanonicalRequest(
        provider="gemini",
        model=payload.get("model", ""),
        messages=payload.get("contents", []),
        tool_calls=tool_calls,
        stream=payload.get("stream", False),
        raw_payload=payload,
        session_id=session_id,
        request_id=request_id,
    )
