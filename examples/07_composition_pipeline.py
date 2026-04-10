# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 07: Compositional Pipeline — Multi-Agent Safety Bounds.

Demonstrates AgentAssert's compositional guarantees for multi-agent pipelines.
When agents A and B are composed (A⊕B), the pipeline reliability is bounded:

    p_{A⊕B} ≥ p_A · p_B · p_h

Patent §5.6: Compositional safety proofs for multi-agent systems.
"""

import agentassert_abc as aa
from agentassert_abc.certification.composition import (
    sequential_composition_bound as compose_guarantees,
)
from agentassert_abc.integrations.generic import GenericAdapter

# Agent A: Research agent
contract_a = aa.loads("""
contractspec: "0.1"
kind: agent
name: research-agent
description: Research agent in a pipeline
version: "1.0.0"
invariants:
  hard:
    - name: source-cited
      check:
        field: output.has_citations
        equals: true
  soft:
    - name: depth
      check:
        field: output.depth_score
        gte: 0.6
      recovery: improve
      recovery_window: 2
recovery:
  strategies:
    - name: improve
      type: inject_correction
satisfaction:
  p: 0.95
  delta: 0.1
  k: 3
""")

# Agent B: Writing agent
contract_b = aa.loads("""
contractspec: "0.1"
kind: agent
name: writer-agent
description: Writing agent in a pipeline
version: "1.0.0"
invariants:
  hard:
    - name: no-plagiarism
      check:
        field: output.plagiarism_detected
        equals: false
  soft:
    - name: readability
      check:
        field: output.readability_score
        gte: 0.7
      recovery: simplify
      recovery_window: 2
recovery:
  strategies:
    - name: simplify
      type: inject_correction
satisfaction:
  p: 0.95
  delta: 0.1
  k: 3
""")

# Simulate sessions for both agents
adapter_a = GenericAdapter(contract_a)
adapter_b = GenericAdapter(contract_b)

print("=" * 60)
print("AgentAssert — Compositional Pipeline Demo")
print("=" * 60)

# Agent A: 10 turns of research
print("\nAgent A (Research):")
for i in range(10):
    adapter_a.check({
        "output.has_citations": True,
        "output.depth_score": 0.75 + (i % 3) * 0.05,
    })

summary_a = adapter_a.session_summary()
print(f"  Turns: {summary_a.turn_count}")
print(f"  Theta_A: {summary_a.theta:.3f}")
print(f"  C_hard: {summary_a.mean_c_hard:.3f}")

# Agent B: 10 turns of writing
print("\nAgent B (Writer):")
for i in range(10):
    quality = 0.8 if i != 7 else 0.5  # One soft violation
    adapter_b.check({
        "output.plagiarism_detected": False,
        "output.readability_score": quality,
    })

summary_b = adapter_b.session_summary()
print(f"  Turns: {summary_b.turn_count}")
print(f"  Theta_B: {summary_b.theta:.3f}")
print(f"  C_hard: {summary_b.mean_c_hard:.3f}")

# Compositional bound
print("\n" + "-" * 60)
print("Compositional Guarantee (A⊕B)")
print("-" * 60)

p_a = summary_a.mean_c_hard
p_b = summary_b.mean_c_hard

# Handoff reliability (probability that the interface between agents
# preserves correctness). Typically high for well-defined interfaces.
p_h = 0.99

pipeline_bound = compose_guarantees(p_a, p_b, p_h)

print(f"  p_A (research compliance): {p_a:.3f}")
print(f"  p_B (writer compliance):   {p_b:.3f}")
print(f"  p_h (handoff reliability): {p_h:.3f}")
print(f"  p_{{A⊕B}} ≥ p_A · p_B · p_h = {pipeline_bound:.3f}")
print(f"\n  Pipeline reliability bound: {pipeline_bound:.1%}")

if pipeline_bound >= 0.90:
    print("  Assessment: Pipeline meets 90% reliability threshold")
else:
    print("  Assessment: Pipeline below threshold — improve individual agents")
