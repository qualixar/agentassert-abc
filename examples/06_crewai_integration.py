# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 06: CrewAI Integration — Task Guardrails.

Shows how to use AgentAssert as a CrewAI task guardrail. When a task
completes, the guardrail validates the output against the contract.
Hard violations cause CrewAI to retry the task.

Requires: pip install crewai agentassert-abc
"""

from __future__ import annotations

import agentassert_abc as aa

contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: crewai-integration-demo
description: Contract for a CrewAI research crew
version: "1.0.0"

invariants:
  hard:
    - name: source-attribution
      description: All claims must cite sources
      check:
        field: output.has_citations
        equals: true
    - name: no-fabrication
      description: No fabricated information
      check:
        field: output.fabrication_detected
        equals: false

  soft:
    - name: depth-score
      description: Research should be thorough
      check:
        field: output.depth_score
        gte: 0.7
      recovery: deepen-research
      recovery_window: 2
    - name: recency
      description: Sources should be recent
      check:
        field: output.source_recency_score
        gte: 0.6
      recovery: update-sources
      recovery_window: 2

recovery:
  strategies:
    - name: deepen-research
      type: inject_correction
      actions:
        - "Search for additional sources"
        - "Expand analysis with more detail"
    - name: update-sources
      type: inject_correction
      actions:
        - "Find more recent sources"
""")


def show_pattern() -> None:
    """Show the CrewAI integration pattern."""
    print("=" * 60)
    print("CrewAI Integration Pattern")
    print("=" * 60)
    print("""
from crewai import Agent, Task, Crew, Process
from agentassert_abc.integrations.crewai import CrewAIAdapter

# 1. Create adapter
adapter = CrewAIAdapter(contract)

# 2. Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Analyze market trends with citations",
    backstory="Expert market researcher",
    llm="gpt-4o",
)

# 3. Create task WITH AgentAssert guardrail
research_task = Task(
    description="Research AI agent frameworks in 2026",
    expected_output="A cited report on top 5 frameworks",
    agent=researcher,
    output_json=ResearchOutput,
    guardrail=adapter.guardrail,      # <-- AgentAssert validates here
    guardrail_max_retries=3,          # CrewAI retries on failure
)

# 4. Or use as callback (monitor only)
review_task = Task(
    description="Review the research report",
    expected_output="Quality assessment",
    agent=reviewer,
    callback=adapter.callback,         # <-- AgentAssert logs metrics
)

# 5. Run crew
crew = Crew(
    agents=[researcher, reviewer],
    tasks=[research_task, review_task],
    process=Process.sequential,
)
result = crew.kickoff()

# 6. Check metrics
summary = adapter.session_summary()
print(f"Theta: {summary.theta}")
""")


# Simulate the guardrail behavior without requiring crewai
print("=" * 60)
print("CrewAI Integration — Simulated Guardrail Demo")
print("=" * 60)

# Simulate what the adapter.guardrail() does with different outputs
from agentassert_abc.integrations.generic import GenericAdapter

adapter = GenericAdapter(contract)

# Scenario 1: Good research output
print("\nScenario 1: Quality research with citations")
good_output = {
    "output.has_citations": True,
    "output.fabrication_detected": False,
    "output.depth_score": 0.85,
    "output.source_recency_score": 0.9,
}
result = adapter.check(good_output)
print(f"  Verdict: {'ACCEPT' if result.hard_violations == 0 else 'REJECT'}")
print(f"  Hard: {result.hard_violations}, Soft: {result.soft_violations}")

# Scenario 2: Missing citations — hard violation
print("\nScenario 2: Research without citations (hard breach)")
no_cite_output = {
    "output.has_citations": False,  # HARD VIOLATION
    "output.fabrication_detected": False,
    "output.depth_score": 0.85,
    "output.source_recency_score": 0.9,
}
result = adapter.check(no_cite_output)
print(f"  Verdict: {'ACCEPT' if result.hard_violations == 0 else 'REJECT + RETRY'}")
print(f"  Hard: {result.hard_violations}, Violated: {result.violated_names}")

# Scenario 3: Shallow research — soft violation
print("\nScenario 3: Shallow research (soft violation, no retry)")
shallow_output = {
    "output.has_citations": True,
    "output.fabrication_detected": False,
    "output.depth_score": 0.4,  # Below threshold
    "output.source_recency_score": 0.5,  # Below threshold
}
result = adapter.check(shallow_output)
print(f"  Verdict: {'ACCEPT (with warnings)' if result.hard_violations == 0 else 'REJECT'}")
print(f"  Soft violations: {result.violated_names}")
print(f"  Recovery needed: {result.recovery_needed}")

summary = adapter.session_summary()
print(f"\nSession Theta: {summary.theta:.3f}")

show_pattern()
