# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Claude Code hook adapter — enforce contracts on PreToolUse/PostToolUse.

Ported from `agentassert-typec-claude-code` (MIT) into
`agentassert_abc.claude_code` (AGPL-3.0-or-later) per the port-delta
(item #30 — 100% additive). The `case_study/` subdirectory was NOT ported:
it contained fabricated benchmark numbers and personal `~/.claude/*` config
filenames — neither belongs in a published package.

Install with the ``claude-code`` extra: ``pip install agentassert-abc[claude-code]``.
"""

from __future__ import annotations

__all__: list[str] = []
