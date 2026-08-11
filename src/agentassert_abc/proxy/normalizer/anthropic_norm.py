# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Anthropic Messages API -> CanonicalRequest. Ported unchanged from typec."""

from __future__ import annotations

from typing import Any

from agentassert_abc.proxy.normalizer.canonical import CanonicalRequest, CanonicalToolCall


def normalize_anthropic(
    payload: dict[str, Any], session_id: str, request_id: str
) -> CanonicalRequest:
    tool_calls: list[CanonicalToolCall] = []
    for msg in payload.get("messages", []):
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        CanonicalToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                            provider_raw=block,
                        )
                    )

    return CanonicalRequest(
        provider="anthropic",
        model=payload.get("model", ""),
        messages=payload.get("messages", []),
        tool_calls=tool_calls,
        stream=payload.get("stream", False),
        max_tokens=payload.get("max_tokens"),
        temperature=payload.get("temperature"),
        system=payload.get("system"),
        raw_payload=payload,
        session_id=session_id,
        request_id=request_id,
    )
