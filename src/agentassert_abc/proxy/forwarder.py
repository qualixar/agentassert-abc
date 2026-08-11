# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Upstream HTTP forwarding — resolves provider URLs and relays requests.

Ported from agentassert-typec's `forwarder.py`. Env var prefix renamed from
the deprecated ``TYPEC_UPSTREAM_*`` to ``AGENTASSERT_UPSTREAM_*`` to match
this package's identity (agentassert-abc, not agentassert-typec).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from fastapi import Request

_PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api",
}

# Env vars checked per provider, in priority order within the env tier.
_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["AGENTASSERT_UPSTREAM_ANTHROPIC", "ANTHROPIC_BASE_URL"],
    "openai": ["AGENTASSERT_UPSTREAM_OPENAI", "OPENAI_BASE_URL"],
    "gemini": ["AGENTASSERT_UPSTREAM_GEMINI"],
    "openrouter": ["AGENTASSERT_UPSTREAM_OPENROUTER"],
}

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            timeout=httpx.Timeout(120.0),
        )
    return _client


def provider_url(
    provider: str,
    upstream_overrides: dict[str, str] | None = None,
) -> str:
    """Resolve the upstream URL for a provider.

    Priority:
      1. Contract ``upstream.*`` passed as `upstream_overrides`.
      2. ``AGENTASSERT_UPSTREAM_{PROVIDER}`` or ``ANTHROPIC_BASE_URL`` /
         ``OPENAI_BASE_URL`` env vars.
      3. Built-in default (api.anthropic.com, api.openai.com, etc.)

    This allows the proxy to chain correctly when the LLM client is configured
    to use a non-default backend — a local model, a compatible third-party
    endpoint, or any OpenAI-compatible provider.
    """
    if upstream_overrides:
        url = upstream_overrides.get(provider, "").strip()
        if url:
            return url.rstrip("/")

    for env_key in _PROVIDER_ENV_VARS.get(provider, []):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val.rstrip("/")

    return _PROVIDER_DEFAULTS.get(provider, "")


_HOP_BY_HOP: frozenset[str] = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    ]
)


def _forward_headers(raw_request: Request, provider: str) -> dict[str, str]:
    return {
        k.lower(): v for k, v in raw_request.headers.items() if k.lower() not in _HOP_BY_HOP
    }


async def forward_request(
    provider: str,
    payload: dict,
    raw_request: Request,
    path: str = "",
    upstream_overrides: dict[str, str] | None = None,
) -> httpx.Response:
    client = get_client()
    base = provider_url(provider, upstream_overrides)
    headers = _forward_headers(raw_request, provider)

    if provider == "anthropic":
        url = f"{base}/v1/messages"
    elif provider == "gemini":
        model = payload.get("model", "gemini-pro")
        url = f"{base}/v1beta/models/{model}:generateContent"
    else:
        url = f"{base}{path}"

    return await client.post(url, json=payload, headers=headers)


async def forward_raw(
    provider: str,
    path: str,
    raw_request: Request,
    method: str = "POST",
    payload: dict | None = None,
    upstream_overrides: dict[str, str] | None = None,
) -> httpx.Response:
    client = get_client()
    base = provider_url(provider, upstream_overrides)
    headers = _forward_headers(raw_request, provider)
    url = f"{base}{path}"

    if method == "GET":
        return await client.get(url, headers=headers)
    return await client.post(url, json=payload, headers=headers)
