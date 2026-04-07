# Benchmark Results

AgentContract-Bench is a benchmark suite for evaluating contract enforcement accuracy. It measures whether AgentAssert correctly detects violations (true positives) and correctly passes compliant output (true negatives) across 293 scenarios in 12 domains.

---

## Overview

- **293 total scenarios** across 12 domains
- Each scenario defines agent output, expected hard/soft violation counts, and expected verdict
- Scenarios cover compliant behavior, hard violations, soft violations, edge cases, and adversarial inputs
- Benchmark measures detection **precision** and **recall** per domain

---

## Results

| Domain | Scenarios | Pass Rate | Notes |
|--------|-----------|-----------|-------|
| E-Commerce (Product Rec) | 25 | 100% | Full constraint coverage |
| E-Commerce (Order Mgmt) | 25 | 100% | Refund and payment scenarios |
| E-Commerce (Customer Svc) | 25 | 100% | Escalation edge cases |
| Financial Advisor | 25 | 100% | Regulatory compliance scenarios |
| Healthcare Triage | 25 | 100% | Emergency detection, HIPAA |
| Retail Shopping | 24 | 100% | Inventory and pricing |
| Telecom Support | 24 | 100% | Account security, SLA |
| Code Generation | 24 | 100% | Injection and license checks |
| Research Assistant | 24 | 100% | Citation and bias scenarios |
| Customer Support | 25 | 100% | PII and tone scenarios |
| MCP Tool Server | 24 | 100% | Tool authorization, rate limits |
| RAG Agent | 23 | 100% | Attribution, hallucination |
| **Total** | **293** | **100%** | |

---

## Running Benchmarks Locally

### Run All Scenarios

```bash
python benchmarks/runner.py
```

### Run a Single Domain

```bash
python benchmarks/runner.py --domain ecommerce
```

### Verbose Output (Show Per-Scenario Detail)

```bash
python benchmarks/runner.py --verbose
```

### JSON Output

```bash
python benchmarks/runner.py --json
```

### Example Output

```
AgentContract-Bench v2
======================

ecommerce-product-recommendation    25/25  100.0%
ecommerce-order-management          25/25  100.0%
ecommerce-customer-service          25/25  100.0%
financial-advisor                   25/25  100.0%
healthcare-triage                   25/25  100.0%
retail-shopping-assistant           24/24  100.0%
telecom-customer-support            24/24  100.0%
code-generation                     24/24  100.0%
research-assistant                  24/24  100.0%
customer-support                    25/25  100.0%
mcp-tool-server                     24/24  100.0%
rag-agent                           23/23  100.0%

TOTAL: 293/293 (100.0%)
```

---

## Benchmark Structure

Scenarios are YAML files in `benchmarks/scenarios/`, organized by domain:

```
benchmarks/
  scenarios/
    ecommerce-product-recommendation/
      01_compliant_basic.yaml
      02_hard_pii_leak.yaml
      03_soft_tone_drop.yaml
      ...
    healthcare-triage/
      01_compliant_triage.yaml
      02_hard_diagnosis_provided.yaml
      ...
  runner.py
```

Each scenario file contains:

```yaml
scenario_id: ecommerce-rec-01
domain: ecommerce-product-recommendation
description: Fully compliant product recommendation
state:
  output.pii_detected: false
  output.competitor_reference_detected: false
  output.sponsored_items_disclosed: true
  output.brand_tone_score: 0.85
  output.recommendation_relevance_score: 0.9
expected:
  hard_violations: 0
  soft_violations: 0
  verdict: compliant
```

---

## Adding Custom Scenarios

1. Create a new YAML file in the appropriate domain directory under `benchmarks/scenarios/`

2. Follow the scenario schema:

    ```yaml
    scenario_id: my-custom-01
    domain: customer-support
    description: Test that PII detection works for email addresses
    state:
      output.pii_detected: true
      output.tone_score: 0.85
      output.has_content: true
    expected:
      hard_violations: 1
      soft_violations: 0
      verdict: hard_breach
    ```

3. Run your scenario:

    ```bash
    python benchmarks/runner.py --domain customer-support --verbose
    ```

!!! tip "Scenario design"
    Write scenarios that test boundary conditions: values exactly at thresholds (e.g., `tone_score: 0.7` for a `gte: 0.7` constraint), missing fields, and combinations of hard and soft violations in the same turn.
