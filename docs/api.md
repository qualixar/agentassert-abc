# API Reference

!!! note "Full API reference coming soon"
    Auto-generated API documentation with complete type signatures and docstrings will be available in a future release. In the meantime, the source code is fully type-annotated and documented with docstrings. See [src/agentassert_abc/](https://github.com/qualixar/agentassert-abc/tree/main/src/agentassert_abc) for the implementation.

---

## Public Exports

All public symbols are available from the top-level `agentassert_abc` module:

```python
import agentassert_abc as aa
```

### Contract Loading & Parsing

| Symbol | Description |
|--------|-------------|
| `aa.load(path)` | Load a contract from a YAML file path. Returns `ContractSpec`. |
| `aa.loads(yaml_string)` | Load a contract from a YAML string. Returns `ContractSpec`. |
| `aa.parse(path)` | Parse a YAML file and return a `ParseResult` with validation details. |
| `aa.parses(yaml_string)` | Parse a YAML string and return a `ParseResult`. |
| `aa.validate(contract)` | Validate a `ContractSpec` and return validation errors if any. |

### Evaluation

| Symbol | Description |
|--------|-------------|
| `aa.evaluate(contract, state)` | Evaluate a state dict against a contract. Returns `EvaluationResult`. |
| `aa.evaluate_preconditions(contract, state)` | Check preconditions only. Returns `PreconditionCheckResult`. |

### Session Monitoring

| Symbol | Description |
|--------|-------------|
| `aa.SessionMonitor(contract)` | Stateful session monitor. Tracks turns, drift, and computes Theta. |
| `aa.compute_theta(...)` | Compute the Theta reliability index from component scores. |

### Result Types

| Symbol | Description |
|--------|-------------|
| `aa.StepResult` | Result of evaluating a single turn. Contains violation counts, drift score. |
| `aa.SessionSummary` | Aggregated session metrics including Theta, compliance rates, drift. |
| `aa.PreconditionCheckResult` | Result of precondition evaluation. |
| `aa.EvaluationResult` | Detailed evaluation result with per-constraint results. |
| `aa.ConstraintResult` | Result for a single constraint check. |
| `aa.ParseResult` | Result of parsing a YAML contract. |

### Certification

| Symbol | Description |
|--------|-------------|
| `aa.SPRTCertifier(p0, p1, alpha, beta)` | Sequential Probability Ratio Test certifier. |
| `aa.compose_guarantees(p_a, p_b, p_h)` | Compute compositional safety bound for a two-agent pipeline. |

### Adapters

| Symbol | Description |
|--------|-------------|
| `aa.GenericAdapter(contract)` | Framework-agnostic adapter for dict-producing agents. |

Framework-specific adapters are imported from their submodules:

```python
from agentassert_abc.integrations.langgraph import LangGraphAdapter
from agentassert_abc.integrations.crewai import CrewAIAdapter
from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter
```

### Contract Models

| Symbol | Description |
|--------|-------------|
| `aa.ContractSpec` | The top-level contract model. |
| `aa.ContractMetadata` | Optional metadata (author, domain, tags). |
| `aa.Invariants` | Container for hard and soft constraints. |
| `aa.HardConstraint` | A non-negotiable constraint definition. |
| `aa.SoftConstraint` | A quality-goal constraint with recovery. |
| `aa.ConstraintCheck` | The operator and field for a constraint check. |
| `aa.Governance` | Governance constraints (hard and soft). |
| `aa.GovernanceConstraint` | A single governance constraint. |
| `aa.Precondition` | A precondition check. |
| `aa.RecoveryConfig` | Recovery strategies configuration. |
| `aa.RecoveryAction` | A single recovery strategy. |
| `aa.SatisfactionParams` | The (p, delta, k) satisfaction configuration. |
| `aa.DriftConfig` | Drift detection configuration. |
| `aa.DriftWeights` | Weights for compliance vs distributional drift. |
| `aa.DriftThresholds` | Warning and critical drift thresholds. |
| `aa.ReliabilityConfig` | Theta computation configuration. |
| `aa.ReliabilityWeights` | Weights for Theta components. |

### Exceptions

| Symbol | Description |
|--------|-------------|
| `aa.AgentAssertError` | Base exception for all AgentAssert errors. |
| `aa.ContractBreachError` | Raised on hard constraint violations. |
| `aa.ContractParseError` | Raised when a YAML contract cannot be parsed. |
| `aa.ContractValidationError` | Raised when a contract fails validation. |
| `aa.DriftThresholdError` | Raised when drift exceeds the critical threshold. |
| `aa.PreconditionFailedError` | Raised when a precondition check fails. |
| `aa.RecoveryFailedError` | Raised when a recovery strategy fails. |
| `aa.StateExtractionError` | Raised when state cannot be extracted from agent output. |

---

## Lazy Loading

Heavy dependencies (scipy, numpy, ruamel-yaml) are lazy-loaded. The initial `import agentassert_abc` completes in under 10ms. Dependencies are loaded on first access to the functions that need them:

- `aa.load()` / `aa.loads()` -- triggers ruamel-yaml import
- `aa.SessionMonitor` / `aa.compute_theta()` -- triggers scipy/numpy import
- `aa.SPRTCertifier` -- triggers scipy import

This means you can `import agentassert_abc` in any project without paying the import cost for features you do not use.
