# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 03: Drift Detection — JSD-Based Behavioral Drift.

Demonstrates AgentAssert's drift detection capability. As an agent's behavior
changes over time, the composite drift score D(t) tracks deviation from
contracted behavior.

D(t) = w_c × D_compliance(t) + w_d × D_distributional(t)

Patent §5.1: Composite drift detection with configurable weights and thresholds.
"""

import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: drift-detection-demo
description: Contract for demonstrating drift detection
version: "1.0.0"

invariants:
  hard:
    - name: safety-check
      description: Safety constraint
      check:
        field: output.safe
        equals: true

  soft:
    - name: quality-score
      description: Output quality
      check:
        field: output.quality
        gte: 0.7
      recovery: improve
      recovery_window: 3

    - name: relevance-score
      description: Response relevance
      check:
        field: output.relevance
        gte: 0.6
      recovery: refocus
      recovery_window: 2

recovery:
  strategies:
    - name: improve
      type: inject_correction
    - name: refocus
      type: inject_correction

drift:
  weights:
    compliance: 0.6
    distributional: 0.4
  window: 10
  thresholds:
    warning: 0.3
    critical: 0.6
""")

adapter = GenericAdapter(contract)

# Simulate 20 turns with gradual quality degradation
# Turns 1-10: Good behavior, high quality
# Turns 11-20: Degrading quality (drift)
print("=" * 60)
print("AgentAssert — Drift Detection Demo")
print("=" * 60)
print(f"{'Turn':>4} {'Quality':>8} {'Relevance':>10} {'Drift':>8} {'Status':>12}")
print("-" * 50)

for i in range(1, 21):
    if i <= 10:
        # Good phase
        quality = 0.85 - (i * 0.005)  # Slight natural variation
        relevance = 0.80 + (i * 0.01)
    else:
        # Degrading phase — drift occurring
        quality = 0.80 - ((i - 10) * 0.04)  # Drops from 0.80 to 0.40
        relevance = 0.90 - ((i - 10) * 0.03)

    state = {
        "output.safe": True,
        "output.quality": round(quality, 2),
        "output.relevance": round(relevance, 2),
    }

    result = adapter.check(state)

    status = "OK"
    if result.hard_violations > 0:
        status = "HARD BREACH"
    elif result.soft_violations > 0:
        status = "SOFT VIOL"

    print(f"{i:>4} {quality:>8.2f} {relevance:>10.2f} {result.drift_score:>8.3f} {status:>12}")

# Final summary
summary = adapter.session_summary()
print("\n" + "=" * 60)
print("Drift Analysis")
print("=" * 60)
print(f"  Total turns: {summary.turn_count}")
print(f"  Mean drift score: {summary.mean_drift:.3f}")
print(f"  Final Θ: {summary.theta:.3f}")

if summary.mean_drift < 0.3:
    print("  Assessment: Stable behavior — within normal bounds")
elif summary.mean_drift < 0.6:
    print("  Assessment: WARNING — Behavioral drift detected, monitor closely")
else:
    print("  Assessment: CRITICAL — Significant drift, intervention required")
