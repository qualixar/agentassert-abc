# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 01: Basic Contract Monitoring.

The simplest way to use AgentAssert — load a contract, monitor agent
output step-by-step, and get a reliability score.

This example uses the GenericAdapter, which works with any agent that
produces dict output. No framework dependency needed.
"""

import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

# 1. Define a contract inline (or load from YAML file)
contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: basic-monitoring-demo
description: Simple contract for demonstrating monitoring
version: "1.0.0"

invariants:
  hard:
    - name: no-pii-leak
      description: Agent must never expose personal information
      check:
        field: output.pii_detected
        equals: false

    - name: response-not-empty
      description: Agent must produce a non-empty response
      check:
        field: output.has_content
        equals: true

  soft:
    - name: tone-quality
      description: Response should maintain professional tone
      check:
        field: output.tone_score
        gte: 0.7
      recovery: fix-tone
      recovery_window: 2

    - name: relevance
      description: Response should be relevant to the query
      check:
        field: output.relevance_score
        gte: 0.6
      recovery: improve-relevance
      recovery_window: 2

recovery:
  strategies:
    - name: fix-tone
      type: inject_correction
      actions:
        - "Rewrite response with professional tone"
    - name: improve-relevance
      type: inject_correction
      actions:
        - "Focus response on the user's actual question"
""")

# 2. Create an adapter
adapter = GenericAdapter(contract)

# 3. Simulate agent outputs (in production, these come from your agent)
turns = [
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.85,
        "output.relevance_score": 0.9,
    },
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.6,  # Below threshold — soft violation
        "output.relevance_score": 0.8,
    },
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.9,  # Recovered
        "output.relevance_score": 0.85,
    },
]

# 4. Monitor each turn
print("=" * 50)
print("AgentAssert — Basic Monitoring Demo")
print("=" * 50)

for i, turn in enumerate(turns):
    result = adapter.check(turn)
    status = "COMPLIANT" if result.hard_violations == 0 and result.soft_violations == 0 else "VIOLATION"
    print(f"\nTurn {i + 1}: {status}")
    print(f"  Hard violations: {result.hard_violations}")
    print(f"  Soft violations: {result.soft_violations}")
    if result.violated_names:
        print(f"  Violated: {result.violated_names}")
    if result.recovery_needed:
        print("  Recovery needed: YES")

# 5. Get session summary
summary = adapter.session_summary()
print("\n" + "=" * 50)
print("Session Summary")
print("=" * 50)
print(f"  Turns: {summary.turn_count}")
print(f"  Hard violations: {summary.total_hard_violations}")
print(f"  Soft violations: {summary.total_soft_violations}")
print(f"  Mean compliance (hard): {summary.mean_c_hard:.2f}")
print(f"  Mean compliance (soft): {summary.mean_c_soft:.2f}")
print(f"  Reliability Index (Θ): {summary.theta:.3f}")
print(f"  Deployment ready (Θ ≥ 0.90): {'YES' if summary.theta >= 0.90 else 'NO'}")
