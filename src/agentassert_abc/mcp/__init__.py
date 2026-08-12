# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""MCP guard — behavioral contracts for any client that speaks MCP.

The other adoption surfaces are tied to something specific: the proxy to an LLM
wire format, the SDK wrappers to a vendor client, the Claude Code hook to one
product. This one is tied only to the tool protocol, so a single artifact
enforces contracts inside Claude Code, Codex, Cursor, VS Code, Antigravity,
Windsurf and anything else that adopts MCP — with no vendor-specific code.

Wire it in by changing the server's launch command in the client's MCP config::

    {"command": "agentassert-abc-mcp-guard",
     "args": ["--contract", "contract.yaml", "--", "npx", "-y", "some-server"]}

Scope, stated plainly: this guards MCP tools. An agent's *native* tools — a
built-in file editor or shell that never crosses the MCP boundary — are not
visible here, and no amount of MCP-level enforcement will see them. Pair the
guard with a vendor hook (Claude Code, Codex, Cursor) where one exists, and with
the HTTP proxy for model-call enforcement.

Note on the name: this package is ``agentassert_abc.mcp`` and is unrelated to
the third-party ``mcp`` distribution, which it deliberately does not depend on
(see :mod:`agentassert_abc.mcp.jsonrpc`).
"""

from __future__ import annotations

from agentassert_abc.mcp.guard import McpGuard, PendingCall, Relay
from agentassert_abc.mcp.interposer import StreamPump, run_guard

__all__ = [
    "McpGuard",
    "PendingCall",
    "Relay",
    "StreamPump",
    "run_guard",
]
