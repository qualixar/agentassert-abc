# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Framework shims — native hook signature in, native veto convention out.

Each shim is deliberately tiny and deliberately disposable. All the policy lives
in :class:`~agentassert_abc.enforce.EnforcementBridge`; a shim only knows how one
framework spells "block this call".

**No shim imports its framework at module import time.** They are structurally
typed against the objects the framework passes in, so importing this package
costs nothing, installing AgentAssert pulls in no agent framework, and a shim
can be unit-tested against a fake context. Where a framework's own type is
genuinely required to build a return value, it is imported lazily at call time
and can be injected for testing.

The trade-off is stated plainly: structural typing means a framework can rename
a field and the shim will only find out at runtime. That is why every shim
documents the exact API shape it targets, and why the surrounding suite tests
the translation rather than trusting it.

Supported:

===============  =======================================================
Framework        Veto mechanism
===============  =======================================================
CrewAI           ``BeforeToolCallHook`` returns ``False``
LangChain        ``wrap_tool_call`` returns without calling the handler
Microsoft AF     ``FunctionMiddleware`` declines to call ``next``
AgentScope       pre-hook raises to abort the act
===============  =======================================================

DeerFlow is built on LangGraph, so the LangChain shim covers it unchanged.
"""

from __future__ import annotations

from agentassert_abc.enforce.shims.agentscope import register_agentscope_hooks
from agentassert_abc.enforce.shims.crewai import (
    crewai_after_tool_hook,
    crewai_before_tool_hook,
)
from agentassert_abc.enforce.shims.langchain import langchain_tool_middleware
from agentassert_abc.enforce.shims.maf import ContractFunctionMiddleware

__all__ = [
    "ContractFunctionMiddleware",
    "crewai_after_tool_hook",
    "crewai_before_tool_hook",
    "langchain_tool_middleware",
    "register_agentscope_hooks",
]
