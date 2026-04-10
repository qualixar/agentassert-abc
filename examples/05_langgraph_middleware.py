# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 05: LangGraph Integration — Contract Middleware.

Shows how to add AgentAssert contract monitoring to a LangGraph StateGraph.
Each node's output is evaluated against the contract before state merge.

Requires: pip install langgraph agentassert-abc

NOTE: This example requires langgraph to be installed. If you don't have
langgraph, see 01_basic_monitoring.py for framework-agnostic usage.
"""

# This example demonstrates the PATTERN — it will run if langgraph is installed.
# If not, it shows the code pattern for documentation purposes.

from __future__ import annotations

try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("langgraph not installed — showing pattern only")
    print("Install with: pip install langgraph")
    print()

import agentassert_abc as aa
from agentassert_abc.exceptions import ContractBreachError

# Contract for a customer support agent graph
contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: langgraph-middleware-demo
description: Contract for a LangGraph customer support agent
version: "1.0.0"

invariants:
  hard:
    - name: no-pii-leak
      description: Agent must never expose PII
      check:
        field: output.pii_detected
        equals: false
    - name: no-false-promises
      description: No commitments agent cannot fulfill
      check:
        field: output.false_promise
        equals: false

  soft:
    - name: empathy-score
      description: Maintain empathetic tone
      check:
        field: output.empathy_score
        gte: 0.6
      recovery: improve-empathy
      recovery_window: 2

recovery:
  strategies:
    - name: improve-empathy
      type: inject_correction
      actions:
        - "Rewrite with more empathetic language"
""")


def show_pattern() -> None:
    """Show the LangGraph integration pattern."""
    print("=" * 60)
    print("LangGraph Integration Pattern")
    print("=" * 60)
    print("""
from langgraph.graph import StateGraph, START, END
from agentassert_abc.integrations.langgraph import LangGraphAdapter

# 1. Create adapter
adapter = LangGraphAdapter(contract)

# 2. Define your nodes
def classify(state): ...
def respond(state): ...

# 3. Build graph with wrapped nodes
builder = StateGraph(State)
builder.add_node("classify", adapter.wrap_node(classify))
builder.add_node("respond", adapter.wrap_node(respond))
builder.add_edge(START, "classify")
builder.add_edge("classify", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# 4. Run — hard violations raise ContractBreachError
try:
    result = graph.invoke(initial_state)
except ContractBreachError as e:
    print(f"Contract breach: {e}")

# 5. Get session metrics
summary = adapter.session_summary()
print(f"Theta: {summary.theta}")
""")


if HAS_LANGGRAPH:
    from typing import TypedDict

    from agentassert_abc.integrations.langgraph import LangGraphAdapter

    class State(TypedDict):
        query: str
        category: str
        output_pii_detected: bool
        output_false_promise: bool
        output_empathy_score: float
        response: str

    def classify_node(state: State) -> dict:
        """Classify the customer query."""
        return {
            "category": "billing",
            "output.pii_detected": False,
            "output.false_promise": False,
            "output.empathy_score": 0.8,
        }

    def respond_node(state: State) -> dict:
        """Generate response based on classification."""
        return {
            "response": "I understand your billing concern. Let me help.",
            "output.pii_detected": False,
            "output.false_promise": False,
            "output.empathy_score": 0.85,
        }

    # Create adapter and wrap nodes
    adapter = LangGraphAdapter(contract)

    builder = StateGraph(State)
    builder.add_node("classify", adapter.wrap_node(classify_node))
    builder.add_node("respond", adapter.wrap_node(respond_node))
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile()

    print("=" * 60)
    print("LangGraph Integration — Live Demo")
    print("=" * 60)

    try:
        result = graph.invoke({
            "query": "Why was I charged twice?",
            "category": "",
            "output_pii_detected": False,
            "output_false_promise": False,
            "output_empathy_score": 0.0,
            "response": "",
        })
        print(f"\nGraph completed successfully")
        print(f"Response: {result.get('response', 'N/A')}")
    except ContractBreachError as e:
        print(f"\nContract breach detected: {e}")

    summary = adapter.session_summary()
    print(f"\nSession Theta: {summary.theta:.3f}")
else:
    show_pattern()
