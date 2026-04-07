# AgentAssert

**Formal behavioral specification and runtime enforcement for autonomous AI agents.**

AgentAssert implements Agent Behavioral Contracts (ABC) -- a framework for defining what an AI agent *must* do, what it *must not* do, and how to measure whether it is behaving correctly over time. You write a contract in YAML, plug it into your agent framework, and AgentAssert monitors every turn for compliance, drift, and reliability.

Unlike ad-hoc prompt guardrails or regex filters, AgentAssert provides mathematically grounded governance. Contracts separate hard constraints (non-negotiable safety rules) from soft constraints (quality goals with recovery windows), track behavioral drift across sessions, and produce a single reliability score -- Theta -- that tells you whether an agent is production-ready.

---

## Key Features

- **ContractSpec DSL** -- Define behavioral contracts in YAML with 14 operators. No code changes required to update constraints.
- **Hard/Soft Constraints** -- Hard violations halt execution immediately. Soft violations trigger recovery strategies within configurable windows.
- **Drift Detection** -- Detect when agent behavior shifts over time using distributional analysis (Jensen-Shannon Divergence).
- **Satisfaction Guarantees** -- Prove that an agent meets compliance targets with statistical confidence.
- **Compositional Safety** -- Calculate safety bounds for multi-agent pipelines where agents hand off to each other.
- **SPRT Certification** -- Certify agents for production deployment using 50-80% fewer test sessions than fixed-sample testing.

---

## Quick Install

```bash
pip install agentassert-abc[yaml,math]
```

Requires Python 3.12+. Licensed under [Elastic License 2.0](https://github.com/qualixar/agentassert-abc/blob/main/LICENSE).

---

## 5-Minute Quick Start

```python
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

# 1. Load a domain contract (12 included out of the box)
contract = aa.load("contracts/examples/ecommerce-product-recommendation.yaml")

# 2. Create an adapter
adapter = GenericAdapter(contract)

# 3. Monitor agent output on every turn
result = adapter.check({
    "output.pii_detected": False,
    "output.competitor_reference_detected": False,
    "output.sponsored_items_disclosed": True,
    "output.brand_tone_score": 0.85,
    "output.recommendation_relevance_score": 0.9,
})

print(f"Hard violations: {result.hard_violations}")
print(f"Soft violations: {result.soft_violations}")

# 4. Get session reliability score
summary = adapter.session_summary()
print(f"Reliability (Theta): {summary.theta:.3f}")
print(f"Deploy: {summary.theta >= 0.90}")
```

---

## How AgentAssert Differs

| Dimension | AgentAssert | Guardrails AI | NeMo Guardrails | Microsoft AGT |
|-----------|-------------|---------------|-----------------|---------------|
| Formal math (Theta, SPRT) | Yes | No | No | No |
| Drift detection (JSD) | Yes | No | No | No |
| Compositional proofs | Yes | No | No | No |
| Framework integrations | 10 adapters | 3 | 1 (LangChain) | 2 |

---

## Next Steps

- [Getting Started](getting-started.md) -- Step-by-step setup and first contract
- [ContractSpec DSL](contractspec.md) -- Full YAML schema reference
- [Framework Integrations](integrations.md) -- LangGraph, CrewAI, OpenAI Agents SDK
- [Metrics & Certification](metrics.md) -- Understanding Theta, drift, and SPRT

---

## Links

- **Paper:** [arXiv:2602.22302](https://arxiv.org/abs/2602.22302)
- **Website:** [agentassert.com](https://agentassert.com)
- **Part of:** [Qualixar](https://qualixar.com) -- AI Agent Reliability Engineering
