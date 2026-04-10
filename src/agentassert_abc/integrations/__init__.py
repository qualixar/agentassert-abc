# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Framework integration adapters for AgentAssert.

Patent §3.3: ContractMiddleware wraps any agent graph and intercepts
state transitions for behavioral contract enforcement.

Available adapters:
- GenericAdapter: Framework-agnostic, works with any dict output
- LangGraphAdapter: LangGraph StateGraph node interception (requires langgraph)
- CrewAIAdapter: CrewAI task guardrails + callbacks (requires crewai)
- OpenAIAgentsAdapter: OpenAI Agents SDK guardrails + hooks (requires openai-agents)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentassert_abc.integrations.base import AgentAdapter
from agentassert_abc.integrations.generic import GenericAdapter

# M-12: TYPE_CHECKING re-exports for IDE autocomplete
if TYPE_CHECKING:
    from agentassert_abc.integrations.crewai import CrewAIAdapter as CrewAIAdapter
    from agentassert_abc.integrations.langgraph import (
        LangGraphAdapter as LangGraphAdapter,
    )
    from agentassert_abc.integrations.openai_agents import (
        OpenAIAgentsAdapter as OpenAIAgentsAdapter,
    )

__all__ = [
    "AgentAdapter",
    "GenericAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]  # noqa: N807
    """Lazy imports for framework-specific adapters.

    These adapters have optional dependencies (langgraph, crewai, openai-agents).
    Importing them eagerly would break installations without those packages.
    """
    _lazy_map = {
        "LangGraphAdapter": "agentassert_abc.integrations.langgraph",
        "CrewAIAdapter": "agentassert_abc.integrations.crewai",
        "OpenAIAgentsAdapter": "agentassert_abc.integrations.openai_agents",
    }
    if name in _lazy_map:
        import importlib

        module = importlib.import_module(_lazy_map[name])
        obj = getattr(module, name)
        globals()[name] = obj  # Cache for next access
        return obj
    raise AttributeError(
        f"module 'agentassert_abc.integrations' has no attribute {name!r}"
    )
