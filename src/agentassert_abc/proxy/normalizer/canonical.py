# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""CanonicalRequest/Response — provider-agnostic proxy request shape.

Ported unchanged from agentassert-typec's `normalizer/canonical.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CanonicalToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    provider_raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalRequest:
    provider: Literal["anthropic", "openai", "gemini", "openrouter"]
    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[CanonicalToolCall] = field(default_factory=list)
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    request_id: str = ""


@dataclass
class CanonicalResponse:
    provider: str
    content: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
