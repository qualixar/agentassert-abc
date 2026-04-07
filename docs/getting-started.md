# Getting Started

This guide walks you from zero to a working contract in under 5 minutes. By the end, you will have installed AgentAssert, written a behavioral contract, monitored agent output, and interpreted the results.

---

## Installation

=== "pip"

    ```bash
    pip install agentassert-abc[yaml,math]
    ```

=== "uv"

    ```bash
    uv add agentassert-abc[yaml,math]
    ```

=== "From source"

    ```bash
    git clone https://github.com/qualixar/agentassert-abc.git
    cd agentassert-abc
    uv sync --all-extras
    ```

**Extras:**

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| `yaml` | YAML parsing (ruamel-yaml) | Loading `.yaml` contract files |
| `math` | Drift + Theta computation (scipy, numpy) | Production monitoring with drift detection |
| `llm` | LiteLLM for LLM-based checks | Using expression-based constraints |
| `otel` | OpenTelemetry spans | Tracing contract evaluations |
| `all` | Everything above | Full feature set |

!!! note "Python version"
    AgentAssert requires Python 3.12 or later.

---

## Your First Contract

A contract defines what your agent must and must not do. Create a file called `my-contract.yaml`:

```yaml
contractspec: "0.1"
kind: agent
name: my-first-contract
description: A simple behavioral contract
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

recovery:
  strategies:
    - name: fix-tone
      type: inject_correction
      actions:
        - "Rewrite response with professional tone"

satisfaction:
  p: 0.95
  delta: 0.1
  k: 3
```

**What this contract says:**

- **Hard constraints** (non-negotiable): The agent must never leak PII and must always produce content. Violations halt execution.
- **Soft constraints** (quality goals): The agent should maintain a tone score of 0.7 or higher. If it falls below, a recovery strategy is triggered. The agent has 2 turns to recover before it counts as a persistent violation.
- **Satisfaction target**: 95% of turns should be compliant, with a tolerance of 10%, evaluated over windows of 3 turns.

---

## Your First Check

```python
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

# Load the contract
contract = aa.load("my-contract.yaml")

# Create an adapter (works with any dict-producing agent)
adapter = GenericAdapter(contract)

# Simulate agent output -- in production, this comes from your agent
agent_output = {
    "output.pii_detected": False,
    "output.has_content": True,
    "output.tone_score": 0.85,
}

# Check against the contract
result = adapter.check(agent_output)

print(f"Hard violations: {result.hard_violations}")   # 0
print(f"Soft violations: {result.soft_violations}")    # 0
print(f"Violated constraints: {result.violated_names}")  # []
```

!!! tip "check vs check_and_raise"
    `adapter.check()` returns a `StepResult` without raising exceptions. Use `adapter.check_and_raise()` if you want a `ContractBreachError` thrown on hard violations -- this is the typical production pattern.

---

## Simulating a Session

Agents produce output over multiple turns. AgentAssert tracks compliance across the entire session:

```python
turns = [
    # Turn 1: All compliant
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.85,
    },
    # Turn 2: Soft violation -- tone dropped below 0.7
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.55,
    },
    # Turn 3: Recovered -- tone back above 0.7
    {
        "output.pii_detected": False,
        "output.has_content": True,
        "output.tone_score": 0.90,
    },
]

for i, turn in enumerate(turns):
    result = adapter.check(turn)
    status = "PASS" if result.hard_violations == 0 and result.soft_violations == 0 else "VIOLATION"
    print(f"Turn {i + 1}: {status} | hard={result.hard_violations} soft={result.soft_violations}")
```

---

## Understanding Results

### StepResult

Every call to `adapter.check()` returns a `StepResult`:

| Field | Type | Description |
|-------|------|-------------|
| `hard_violations` | `int` | Count of hard constraint violations this turn |
| `soft_violations` | `int` | Count of soft constraint violations this turn |
| `violated_names` | `list[str]` | Names of all violated constraints |
| `violated_hard_names` | `list[str]` | Names of violated hard constraints only |
| `recovery_needed` | `bool` | Whether recovery strategies should be triggered |
| `drift_score` | `float` | Current behavioral drift score (0.0 = stable) |

### SessionSummary

After processing all turns, get the session summary:

```python
summary = adapter.session_summary()

print(f"Turns processed:   {summary.turn_count}")
print(f"Hard violations:   {summary.total_hard_violations}")
print(f"Soft violations:   {summary.total_soft_violations}")
print(f"Mean compliance:   {summary.mean_c_hard:.2f}")
print(f"Reliability Theta: {summary.theta:.3f}")
print(f"Deploy-ready:      {summary.theta >= 0.90}")
```

### Hard vs Soft Violations

| Property | Hard Constraint | Soft Constraint |
|----------|----------------|-----------------|
| On violation | Raises `ContractBreachError` | Triggers recovery strategy |
| Severity | Non-negotiable safety rule | Quality goal |
| Recovery | None -- execution stops | Allowed within `recovery_window` turns |
| Example | "Never leak PII" | "Maintain tone score above 0.7" |

### Theta Score

Theta is a single number between 0 and 1 that summarizes an agent's overall reliability. It combines four signals: compliance rate, behavioral drift, recovery effectiveness, and stress resilience. A Theta of 0.90 or higher is the recommended threshold for production deployment.

!!! info "Mathematical details"
    The exact formulas for Theta and its components are described in the paper ([arXiv:2602.22302](https://arxiv.org/abs/2602.22302)). This documentation explains what each metric measures and how to interpret it. See [Metrics & Certification](metrics.md) for more detail.

---

## Using check_and_raise in Production

In production, you typically want hard violations to stop the agent immediately:

```python
from agentassert_abc.exceptions import ContractBreachError

try:
    result = adapter.check_and_raise(agent_output)
    # If we get here, no hard violations
    send_response_to_user(result)
except ContractBreachError as e:
    # Hard violation -- do not send this output to the user
    log_violation(e)
    send_fallback_response()
```

---

## Loading Inline Contracts

You can define contracts as YAML strings instead of files:

```python
contract = aa.loads("""
contractspec: "0.1"
kind: agent
name: inline-demo
description: Inline contract
version: "1.0.0"

invariants:
  hard:
    - name: no-pii
      description: No PII leaks
      check:
        field: output.pii_detected
        equals: false
""")
```

---

## Next Steps

- [ContractSpec DSL Reference](contractspec.md) -- All 14 operators, recovery strategies, satisfaction criteria
- [Framework Integrations](integrations.md) -- LangGraph, CrewAI, OpenAI Agents SDK
- [Domain Contracts Catalog](contracts-catalog.md) -- 12 ready-to-use contracts
- [Metrics & Certification](metrics.md) -- Deep dive into Theta, drift, and SPRT
