# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 04: SPRT Certification — Certify an Agent for Production.

Uses Sequential Probability Ratio Test (SPRT) to certify that an agent
meets (p, δ, k)-satisfaction with statistical confidence.

Patent §5.3-§5.5: SPRT-based certification with configurable Type-I/II errors.
"""

import agentassert_abc as aa
from agentassert_abc.certification.sprt import SPRTCertifier
from agentassert_abc.integrations.generic import GenericAdapter

contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: sprt-certification-demo
description: Contract for SPRT certification demo
version: "1.0.0"

invariants:
  hard:
    - name: safety
      check:
        field: output.safe
        equals: true
  soft:
    - name: quality
      check:
        field: output.quality
        gte: 0.7
      recovery: improve
      recovery_window: 3

recovery:
  strategies:
    - name: improve
      type: inject_correction

satisfaction:
  p: 0.95
  delta: 0.1
  k: 3
""")

adapter = GenericAdapter(contract)

# Simulate 60 turns of a well-behaved agent
# In production, these would be real agent interactions
turns = []
for i in range(60):
    if i % 20 == 15:
        # Occasional soft violation (5% of turns)
        turns.append({
            "output.safe": True,
            "output.quality": 0.55,
        })
    else:
        # Normal compliant behavior (95%)
        turns.append({
            "output.safe": True,
            "output.quality": 0.8 + (i % 5) * 0.02,
        })

print("=" * 60)
print("AgentAssert — SPRT Certification Demo")
print("=" * 60)

# Run all turns through the monitor
for turn in turns:
    adapter.check(turn)

summary = adapter.session_summary()
print(f"\nSession: {summary.turn_count} turns")
print(f"Hard violations: {summary.total_hard_violations}")
print(f"Soft violations: {summary.total_soft_violations}")
print(f"Theta: {summary.theta:.3f}")

# SPRT Certification
print("\n" + "-" * 60)
print("Running SPRT Certification...")
print("-" * 60)

certifier = SPRTCertifier(
    alpha=0.05,  # Type-I error (false positive rate)
    beta=0.05,   # Type-II error (false negative rate)
    p0=0.90,     # Null hypothesis: compliance rate = 0.90
    p1=0.95,     # Alternative hypothesis: compliance rate = 0.95
)

hard_rate = summary.mean_c_hard
soft_rate = summary.mean_c_soft

print(f"  Hard compliance rate: {hard_rate:.3f}")
print(f"  Soft compliance rate: {soft_rate:.3f}")
print(f"  Theta (reliability): {summary.theta:.3f}")

# Feed each turn as a session result into the SPRT certifier.
# A turn "passes" if it had no hard violations.
from agentassert_abc.certification.sprt import SPRTDecision

print("\n  SPRT sequential updates:")
sprt_result = None
for i, turn in enumerate(turns):
    # A session passes if no hard constraint was violated
    session_passed = turn.get("output.safe", False) is True
    sprt_result = certifier.update(session_passed)
    if sprt_result.decision != SPRTDecision.CONTINUE:
        print(f"  Turn {i + 1}: decision={sprt_result.decision.value} "
              f"(LLR={sprt_result.log_likelihood_ratio:.3f})")
        break

if sprt_result is not None and sprt_result.decision == SPRTDecision.ACCEPT:
    print(f"\n  CERTIFIED after {sprt_result.sessions_used} sessions")
    print(f"  Decision: {sprt_result.decision.value}")
elif sprt_result is not None and sprt_result.decision == SPRTDecision.REJECT:
    print(f"\n  NOT CERTIFIED after {sprt_result.sessions_used} sessions")
    print(f"  Decision: {sprt_result.decision.value}")
else:
    print(f"\n  INCONCLUSIVE after {len(turns)} sessions — need more data")
