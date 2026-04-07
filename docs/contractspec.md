# ContractSpec DSL Reference

ContractSpec is a YAML-based domain-specific language for defining behavioral contracts. A contract specifies what an AI agent must do, what it must not do, how to recover from quality drops, and what compliance targets to meet.

---

## Schema Overview

Every contract has this top-level structure:

```yaml
contractspec: "0.1"          # Schema version (required)
kind: agent                   # Contract kind (required)
name: my-contract             # Unique identifier (required)
description: What this does   # Human-readable description (required)
version: "1.0.0"             # Contract version (required)

metadata:                     # Optional metadata
  author: "Team Name"
  domain: ecommerce
  tags: [safety, compliance]

preconditions: [...]          # Checks before agent runs
invariants:                   # Runtime constraints
  hard: [...]                 # Non-negotiable safety rules
  soft: [...]                 # Quality goals with recovery
governance:                   # Operational constraints
  hard: [...]
  soft: [...]
recovery:                     # Recovery strategies
  strategies: [...]
satisfaction:                 # Compliance targets
  p: 0.95
  delta: 0.1
  k: 3
drift:                        # Drift detection config
  weights: {...}
  window: 50
  thresholds: {...}
reliability:                  # Theta weighting config
  weights: {...}
  deployment_threshold: 0.90
```

---

## Preconditions

Preconditions are checked before the agent processes a turn. If a precondition fails, the agent should not proceed.

```yaml
preconditions:
  - name: session-valid
    description: Customer must have a valid session
    check:
      field: session.customer_identified
      equals: true

  - name: catalog-available
    description: Product catalog service must be up
    check:
      field: system.catalog_service_status
      equals: "available"
```

---

## Invariants

Invariants are the core of a contract. They define what must always be true (hard) and what should be true (soft) during agent execution.

### Hard Constraints

Hard constraints are non-negotiable. A single violation raises `ContractBreachError` and halts execution.

```yaml
invariants:
  hard:
    - name: no-pii-leak
      description: Never expose personal information
      category: confidentiality          # Optional categorization
      check:
        field: output.pii_detected
        equals: false

    - name: no-competitor-products
      description: Do not recommend competitor products
      category: role_boundaries
      check:
        field: output.competitor_reference_detected
        equals: false
```

!!! warning "Hard violations cannot be recovered"
    When a hard constraint is violated, there is no recovery window. Execution stops immediately. Use hard constraints only for safety-critical rules.

### Soft Constraints

Soft constraints represent quality goals. Violations trigger a recovery strategy, and the agent has a configurable number of turns to recover.

```yaml
invariants:
  soft:
    - name: tone-quality
      description: Maintain professional tone
      category: role_boundaries
      check:
        field: output.tone_score
        gte: 0.7
      recovery: fix-tone              # Name of recovery strategy
      recovery_window: 2               # Turns allowed to recover
```

### Constraint Categories

Categories are optional labels that help organize constraints:

| Category | Purpose |
|----------|---------|
| `confidentiality` | Data protection and privacy |
| `role_boundaries` | Staying within defined agent role |
| `output_constraints` | Quality and format requirements |
| `action_prohibitions` | Forbidden actions |
| `regulatory_compliance` | Legal and regulatory rules |
| `escalation_rules` | When to hand off to humans |
| `timeout_compliance` | Latency and performance |
| `token_budgets` | Cost and resource limits |
| `spending_limits` | Financial guardrails |

---

## The 14 Operators

Every constraint has a `check` block with a `field` and one operator. The field is a dot-separated path into the agent's output state.

### Equality Operators

```yaml
# Exact match
check:
  field: output.pii_detected
  equals: false

# Negated match
check:
  field: output.status
  not_equals: "error"
```

### Comparison Operators

```yaml
# Greater than
check:
  field: output.confidence
  gt: 0.5

# Greater than or equal
check:
  field: output.tone_score
  gte: 0.7

# Less than
check:
  field: output.error_count
  lt: 5

# Less than or equal
check:
  field: output.latency_ms
  lte: 3000
```

### Range Operator

```yaml
# Value must be within range (inclusive)
check:
  field: output.temperature
  between: [0.0, 1.0]
```

### Membership Operators

```yaml
# Value must be in the allowed set
check:
  field: output.language
  in: ["en", "es", "fr", "de"]

# Value must NOT be in the forbidden set
check:
  field: output.model_used
  not_in: ["gpt-3", "deprecated-model"]
```

### String Operators

```yaml
# Field must contain substring
check:
  field: output.response
  contains: "disclaimer"

# Field must NOT contain substring
check:
  field: output.response
  not_contains: "internal use only"

# Regex match
check:
  field: output.email
  matches: "^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"
```

### Existence Operator

```yaml
# Field must exist and be non-null
check:
  field: output.citation_source
  exists: true
```

### Expression Operator

```yaml
# Arbitrary Python expression (advanced)
check:
  field: output
  expr: "len(value.get('sources', [])) >= 2"
```

!!! tip "When to use expr"
    The `expr` operator is a power tool for complex conditions that the other 13 operators cannot express. Use it sparingly -- simpler operators are easier to audit and understand.

---

## Governance Constraints

Governance constraints are structurally identical to invariants but apply to operational concerns -- tool usage, token budgets, and cost limits.

```yaml
governance:
  hard:
    - name: no-unauthorized-tools
      description: Only use approved tools
      category: action_prohibitions
      check:
        field: tools.all_calls_authorized
        equals: true

  soft:
    - name: token-budget
      description: Stay within token budget
      category: token_budgets
      check:
        field: session.total_tokens
        lte: 30000
      recovery: summarize-and-compress
      recovery_window: 3
```

---

## Recovery Strategies

Recovery strategies define what to do when a soft constraint is violated. Each strategy has a type, a list of actions, and optional retry/fallback configuration.

```yaml
recovery:
  strategies:
    - name: fix-tone
      type: inject_correction
      actions:
        - "Inject system message: maintain professional tone"
        - "Re-score tone before delivery"
      max_attempts: 2
      fallback: escalate-to-support

    - name: escalate-to-support
      type: pause_and_escalate
      actions:
        - "Route to human support representative"
        - "Notify customer of handoff"
      max_attempts: 1
```

### Recovery Types

| Type | Behavior |
|------|----------|
| `inject_correction` | Add a correction instruction to the agent's context |
| `reduce_autonomy` | Limit what the agent can do (e.g., use cached responses) |
| `pause_and_escalate` | Stop and hand off to a human |
| `graceful_shutdown` | Save state and end the session cleanly |

### Recovery Flow

1. Soft constraint is violated
2. Recovery strategy is triggered (actions are executed)
3. Agent has `recovery_window` turns to return to compliance
4. If still violated after the window, the fallback strategy is triggered
5. If no fallback is defined, the violation is recorded as persistent

---

## Satisfaction Criteria

Satisfaction criteria define the compliance target for the contract. They are specified with three parameters:

```yaml
satisfaction:
  p: 0.95     # Target compliance probability
  delta: 0.1  # Tolerance band
  k: 3        # Evaluation window size
```

**Conceptually:**

- **p** -- The target percentage of turns that should be compliant. A value of 0.95 means "95% of turns should pass all constraints."
- **delta** -- The acceptable margin of error around `p`. This controls how strict the compliance check is.
- **k** -- The number of consecutive turns to evaluate together. A window of 3 means compliance is assessed over rolling windows of 3 turns.

The exact statistical formulation is in the paper ([arXiv:2602.22302](https://arxiv.org/abs/2602.22302), Section 4).

---

## Drift Configuration

Drift detection monitors whether agent behavior is shifting over time. Even if individual turns are compliant, the *distribution* of scores might be drifting.

```yaml
drift:
  weights:
    compliance: 0.6        # Weight for compliance-based drift
    distributional: 0.4    # Weight for JSD-based distributional drift
  window: 50               # Number of turns in the drift window
  thresholds:
    warning: 0.3           # Drift score that triggers a warning
    critical: 0.6          # Drift score that triggers an alert
```

---

## Reliability Configuration

The reliability section controls how the Theta score is computed. Theta combines four components, each with a configurable weight:

```yaml
reliability:
  weights:
    compliance: 0.35       # How much compliance rate matters
    drift: 0.25            # How much behavioral stability matters
    recovery: 0.20         # How effectively the agent recovers
    stress: 0.20           # How the agent performs under load
  deployment_threshold: 0.90  # Minimum Theta for production deployment
```

See [Metrics & Certification](metrics.md) for a conceptual explanation of each component.

---

## Pipeline Contracts

For multi-agent pipelines where Agent A hands off to Agent B, you can define contracts for each agent independently and then compose their guarantees:

```python
from agentassert_abc.certification.composition import compose_guarantees

# Agent A has p=0.95, Agent B has p=0.98, handoff reliability is 0.99
pipeline_bound = compose_guarantees(p_a=0.95, p_b=0.98, p_h=0.99)
print(f"Pipeline safety bound: {pipeline_bound:.3f}")
```

The compositional guarantee provides a lower bound on the pipeline's overall compliance rate. See [Metrics & Certification](metrics.md) for details.

---

## Full Annotated Example

Below is a complete production contract with annotations explaining each section:

```yaml
# Schema version -- always "0.1" for the current release
contractspec: "0.1"

# Contract kind -- "agent" for single-agent contracts
kind: agent

# Unique name used in logs and metrics
name: ecommerce-product-recommendation

# Human-readable description
description: >
  Behavioral contract for an AI product recommendation agent.
  Enforces brand guidelines, inventory accuracy, sponsored
  product disclosure, and customer data protection.

# Semantic version of this contract
version: "1.0.0"

# Optional metadata for cataloging
metadata:
  author: "E-commerce Trust & Safety"
  domain: ecommerce
  tags: [ecommerce, recommendations, privacy]

# Preconditions -- checked before the agent runs
preconditions:
  - name: customer-session-valid
    description: Customer must have an active session
    check:
      field: session.customer_identified
      equals: true

# Invariants -- runtime behavioral rules
invariants:
  # Hard constraints -- violations halt execution
  hard:
    - name: no-pii-leak
      description: Never expose customer PII
      category: confidentiality
      check:
        field: output.pii_detected
        equals: false

    - name: sponsored-disclosure
      description: Disclose all sponsored products (FTC compliance)
      category: regulatory_compliance
      check:
        field: output.sponsored_items_disclosed
        equals: true

  # Soft constraints -- violations trigger recovery
  soft:
    - name: brand-tone
      description: Maintain brand-appropriate tone
      category: role_boundaries
      check:
        field: output.brand_tone_score
        gte: 0.7
      recovery: inject-tone-correction
      recovery_window: 2

    - name: recommendation-relevance
      description: Recommendations should match customer preferences
      category: output_constraints
      check:
        field: output.recommendation_relevance_score
        gte: 0.6
      recovery: refine-recommendations
      recovery_window: 2

# Recovery strategies -- what to do on soft violations
recovery:
  strategies:
    - name: inject-tone-correction
      type: inject_correction
      actions:
        - "Inject system message: maintain friendly, helpful tone"
      max_attempts: 2
      fallback: escalate-to-support

    - name: refine-recommendations
      type: inject_correction
      actions:
        - "Re-query recommendation engine with stricter filters"
      max_attempts: 2

    - name: escalate-to-support
      type: pause_and_escalate
      actions:
        - "Route to human support representative"
      max_attempts: 1

# Satisfaction target -- 95% compliance with 10% tolerance over 3-turn windows
satisfaction:
  p: 0.95
  delta: 0.1
  k: 3

# Drift detection -- monitor for behavioral shifts
drift:
  weights:
    compliance: 0.6
    distributional: 0.4
  window: 50
  thresholds:
    warning: 0.3
    critical: 0.6

# Reliability scoring -- weights for Theta computation
reliability:
  weights:
    compliance: 0.35
    drift: 0.25
    recovery: 0.20
    stress: 0.20
  deployment_threshold: 0.90
```
