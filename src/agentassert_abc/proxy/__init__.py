# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Proxy — FastAPI HTTP forwarding proxy (Type C consolidation, Phase E1).

Ported from `agentassert-typec-proxy` (MIT) into `agentassert_abc.proxy`
(AGPL-3.0-or-later). Sits
between an agent's LLM client and the upstream provider API, routing every
request through :class:`agentassert_abc.gateway.SessionEnforcer` before
forwarding.

Install with the ``proxy`` extra: ``pip install agentassert-abc[proxy]``.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:  # noqa: N807
    """Lazy import so `agentassert_abc.proxy` stays importable even if the
    heavier FastAPI/uvicorn/httpx stack (the `[proxy]` extra) isn't installed
    — only `create_app()` actually needs them.
    """
    if name == "create_app":
        from agentassert_abc.proxy.server import create_app

        return create_app
    raise AttributeError(f"module 'agentassert_abc.proxy' has no attribute {name!r}")
