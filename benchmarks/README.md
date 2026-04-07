# AgentContract-Bench v2

Benchmark suite for testing AgentAssert contract enforcement accuracy.

## Structure

```
benchmarks/
├── scenarios/
│   ├── ecommerce/           (50 scenarios)
│   ├── financial/            (30 scenarios)
│   ├── healthcare/           (30 scenarios)
│   ├── code-generation/      (20 scenarios)
│   ├── customer-support/     (20 scenarios)
│   ├── mcp-tools/            (25 scenarios)
│   └── rag-agents/           (25 scenarios)
├── runner.py                 # Benchmark runner
└── README.md                 # This file
```

**Total: 200 scenarios across 7 domains**

## Running

```bash
python benchmarks/runner.py                     # All scenarios
python benchmarks/runner.py --domain ecommerce  # One domain
python benchmarks/runner.py --verbose            # Show per-scenario detail
python benchmarks/runner.py --format json        # JSON output to stdout
```

## Scenario Format

```yaml
id: "domain-001-description"
domain: "domain-name"
description: "What this scenario tests"
contract: "contract-name"
input_state:
  output.field_name: value
  session.field_name: value
expected:
  hard_violations: 0
  soft_violations: 0
  verdict: "compliant"  # or "hard_breach" or "soft_violation"
```

## Scenario Types

Each domain includes a mix of:
- **Compliant** — All constraints satisfied
- **Single hard breach** — Exactly one hard constraint violated
- **Multiple hard breaches** — Two or more hard violations
- **Single soft violation** — One soft constraint violated, no hard
- **Multiple soft violations** — Two or more soft violations
- **Mixed** — Both hard and soft violations
- **Edge cases** — Values exactly at thresholds
- **Missing field** — A required constraint field is absent from state
- **Wrong type** — A numeric soft field contains a non-numeric string

## Planned for v2

- **Precondition testing** — Verify that precondition checks gate scenario
  execution correctly (e.g., scenarios where preconditions are NOT met and
  the evaluator should reject them before constraint evaluation). This is
  planned for the next benchmark iteration.
