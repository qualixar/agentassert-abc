# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 08: MCP Tool Server Monitoring.

Demonstrates using AgentAssert to monitor MCP (Model Context Protocol)
tool-calling agents. Ensures tool calls comply with schema, rate limits,
and access controls.

This is a 2026-specific use case — MCP has become the standard protocol
for LLM-tool communication.
"""

import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

# Contract for an MCP tool-calling agent
contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: mcp-tool-monitor-demo
description: Contract for monitoring MCP tool-calling agents
version: "1.0.0"

invariants:
  hard:
    - name: tool-schema-compliance
      description: Tool calls must match registered schemas
      check:
        field: tools.schema_valid
        equals: true
    - name: no-unauthorized-access
      description: No access to unauthorized resources
      check:
        field: tools.access_authorized
        equals: true
    - name: input-validated
      description: Tool inputs must pass validation
      check:
        field: tools.input_valid
        equals: true
    - name: rate-limit-ok
      description: Within rate limits
      check:
        field: tools.rate_limit_exceeded
        equals: false
    - name: read-no-side-effects
      description: Read operations must not modify state
      check:
        field: tools.read_side_effect_detected
        equals: false

  soft:
    - name: response-latency
      description: Tool response time
      check:
        field: tools.response_ms
        lte: 2000
      recovery: simplify-request
      recovery_window: 2
    - name: error-helpfulness
      description: Error messages should be actionable
      check:
        field: tools.error_helpfulness_score
        gte: 0.6
      recovery: improve-errors
      recovery_window: 3

recovery:
  strategies:
    - name: simplify-request
      type: reduce_autonomy
      actions:
        - "Break complex tool call into smaller operations"
    - name: improve-errors
      type: inject_correction
      actions:
        - "Add context and suggestions to error messages"
""")

adapter = GenericAdapter(contract)

# Simulate MCP tool calls
tool_calls = [
    # Call 1: Normal read operation — COMPLIANT
    {
        "tools.schema_valid": True,
        "tools.access_authorized": True,
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": False,
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 150,
        "tools.error_helpfulness_score": 0.8,
    },
    # Call 2: Write operation — COMPLIANT
    {
        "tools.schema_valid": True,
        "tools.access_authorized": True,
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": False,
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 500,
        "tools.error_helpfulness_score": 0.9,
    },
    # Call 3: Slow response — SOFT VIOLATION
    {
        "tools.schema_valid": True,
        "tools.access_authorized": True,
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": False,
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 3500,  # Exceeds 2000ms
        "tools.error_helpfulness_score": 0.7,
    },
    # Call 4: Rate limit hit — HARD VIOLATION
    {
        "tools.schema_valid": True,
        "tools.access_authorized": True,
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": True,  # HARD
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 100,
        "tools.error_helpfulness_score": 0.5,
    },
    # Call 5: Unauthorized resource access — HARD VIOLATION
    {
        "tools.schema_valid": True,
        "tools.access_authorized": False,  # HARD
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": False,
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 200,
        "tools.error_helpfulness_score": 0.7,
    },
    # Call 6: Recovery — COMPLIANT
    {
        "tools.schema_valid": True,
        "tools.access_authorized": True,
        "tools.input_valid": True,
        "tools.rate_limit_exceeded": False,
        "tools.read_side_effect_detected": False,
        "tools.response_ms": 300,
        "tools.error_helpfulness_score": 0.85,
    },
]

print("=" * 60)
print("AgentAssert — MCP Tool Server Monitoring Demo")
print("=" * 60)
print(f"{'Call':>4} {'Latency':>8} {'Hard':>6} {'Soft':>6} {'Status':>15}")
print("-" * 45)

for i, call in enumerate(tool_calls):
    result = adapter.check(call)
    latency = call["tools.response_ms"]

    if result.hard_violations > 0:
        status = "BLOCKED"
    elif result.soft_violations > 0:
        status = "WARN"
    else:
        status = "OK"

    print(f"{i + 1:>4} {latency:>7}ms {result.hard_violations:>6} {result.soft_violations:>6} {status:>15}")
    if result.violated_names:
        print(f"      Violated: {result.violated_names}")

summary = adapter.session_summary()
print("\n" + "=" * 60)
print("MCP Monitoring Summary")
print("=" * 60)
print(f"  Tool calls monitored: {summary.turn_count}")
print(f"  Hard violations: {summary.total_hard_violations}")
print(f"  Soft violations: {summary.total_soft_violations}")
print(f"  Theta: {summary.theta:.3f}")
print(f"  Mean drift: {summary.mean_drift:.3f}")

if summary.total_hard_violations > 0:
    print("\n  ACTION REQUIRED: Hard violations detected in MCP tool calls.")
    print("  Review unauthorized access and rate limit handling.")
