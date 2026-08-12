# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Claude Code hook adapter — enforce contracts on PreToolUse/PostToolUse.

Reads a hook event on stdin and writes an allow / block / modify decision on
stdout. Fail-open by design: any error reading the event, loading the contract,
or evaluating it allows the tool call through rather than blocking the session.

Install with the ``claude-code`` extra: ``pip install agentassert-abc[claude-code]``.
"""

from __future__ import annotations

__all__: list[str] = []
