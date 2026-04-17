<p align="center">
  <strong>AgentAssert</strong><br>
  <em>Formal Behavioral Contracts for AI Agents</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentassert-abc/"><img src="https://img.shields.io/pypi/v/agentassert-abc?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/agentassert-abc/"><img src="https://img.shields.io/pypi/pyversions/agentassert-abc?style=flat-square" alt="Python"></a>
  <a href="https://arxiv.org/abs/2602.22302"><img src="https://img.shields.io/badge/arXiv-2602.22302-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/qualixar/agentassert-abc/actions"><img src="https://img.shields.io/github/actions/workflow/status/qualixar/agentassert-abc/ci.yml?style=flat-square&label=tests" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=flat-square" alt="AGPL v3"></a>
</p>

---

AgentAssert is the **formal behavioral specification and runtime enforcement engine** for autonomous AI agents. Define what your agent must and must not do in a YAML contract, then enforce those rules at runtime with mathematical guarantees.

It is the only framework combining all **6 pillars** of rigorous agent governance:

1. **ContractSpec DSL** -- YAML-based behavioral specification with 14 operators
2. **Hard/Soft Constraints** -- Formal separation with graduated enforcement and recovery
3. **Drift Detection** -- Jensen-Shannon Divergence for distributional behavioral analysis
4. **(p, delta, k)-Satisfaction** -- Probabilistic compliance guarantees with statistical bounds
5. **Compositional Safety Proofs** -- Formal bounds for multi-agent pipelines
6. **Mathematical Stability** -- Ornstein-Uhlenbeck dynamics with Lyapunov stability proof

> **Paper:** Bhardwaj, V.P. (2026). *AgentAssert: Formal Behavioral Contracts for Autonomous AI Agents.* [arXiv:2602.22302](https://arxiv.org/abs/2602.22302)

---

## Install

```bash
pip install agentassert-abc[yaml,math]
```

Requires **Python 3.12+**. Licensed under [AGPL-3.0](LICENSE).

Optional extras:

| Extra | What it adds |
|-------|-------------|
| `yaml` | YAML contract parsing (ruamel.yaml) |
| `math` | Drift detection, Theta computation (scipy, numpy) |
| `llm` | Recovery re-prompting (LiteLLM) |
| `otel` | OpenTelemetry metric export |
| `all` | Everything above |

---

## Quick Start -- 5 Minutes to Behavioral Contracts

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

# 4. Raise on critical violations
adapter.check_and_raise({
    "output.pii_detected": False,
    "output.competitor_reference_detected": False,
    "output.sponsored_items_disclosed": True,
    "output.brand_tone_score": 0.85,
    "output.recommendation_relevance_score": 0.9,
})

# 5. Get session reliability score (Theta)
summary = adapter.session_summary()
print(f"Reliability (Theta): {summary.theta:.3f}")
print(f"Deploy-ready: {summary.theta >= 0.90}")
```

---

## Framework Integration

AgentAssert is **plug-and-play** with the major 2026 agent frameworks.

### LangGraph -- Node Interception

```python
from langgraph.graph import StateGraph, START, END
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.integrations.langgraph import LangGraphAdapter

contract = aa.load("contracts/examples/customer-support.yaml")
adapter = LangGraphAdapter(contract)

builder = StateGraph(State)
builder.add_node("classify", adapter.wrap_node(classify_fn))
builder.add_node("respond", adapter.wrap_node(respond_fn))
builder.add_edge(START, "classify")
builder.add_edge("classify", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

try:
    result = graph.invoke(initial_state)
except ContractBreachError as e:
    print(f"Hard violation blocked: {e}")

print(f"Session Theta: {adapter.session_summary().theta:.3f}")
```

### CrewAI -- Task Guardrails

```python
from crewai import Agent, Task, Crew
from agentassert_abc.integrations.crewai import CrewAIAdapter

contract = aa.load("contracts/examples/research-assistant.yaml")
adapter = CrewAIAdapter(contract)

# Guardrail rejects output on hard violations -- CrewAI retries automatically
research_task = Task(
    description="Research AI agent frameworks in 2026",
    expected_output="Cited report on top 5 frameworks",
    agent=researcher,
    guardrail=adapter.guardrail,
    guardrail_max_retries=3,
)
```

### OpenAI Agents SDK -- Output Guardrails

```python
from agents import Agent, Runner
from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter

contract = aa.load("contracts/examples/healthcare-triage.yaml")
adapter = OpenAIAgentsAdapter(contract)

agent = Agent(
    name="triage-agent",
    instructions="You are a medical triage assistant.",
    output_guardrails=[adapter.output_guardrail],
    output_type=TriageOutput,
)

result = await Runner.run(agent, "I have chest pain", hooks=adapter.run_hooks)
print(f"Theta: {adapter.session_summary().theta:.3f}")
```

---

## AgentContract-Bench -- 293 Scenarios, 12 Domains

AgentAssert ships with **AgentContract-Bench**, a benchmark suite of 293 scenarios across 12 real-world domains for testing contract enforcement accuracy.

### Benchmark Results (v0.1.0)

| Domain | Scenarios | Pass Rate | Hard P/R/F1 | Soft P/R/F1 |
|--------|-----------|-----------|-------------|-------------|
| E-Commerce (Product) | 50 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Financial Advisor | 33 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Healthcare Triage | 33 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| MCP Tool Server | 28 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| RAG Agent | 28 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Code Generation | 23 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Customer Support | 23 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| E-Commerce (CS) | 15 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| E-Commerce (Order) | 15 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Research Assistant | 15 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Retail Shopping | 15 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Telecom Support | 15 | 100% | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| **Total** | **293** | **100%** | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** |

```bash
# Run benchmarks locally
python benchmarks/runner.py                     # All 293 scenarios
python benchmarks/runner.py --domain ecommerce  # Single domain
python benchmarks/runner.py --verbose           # Show details
```

### Live LLM Benchmark -- Real Models, Real Contracts

We tested AgentAssert against **3 production LLMs** on a 10-16 turn e-commerce session using the `retail-shopping-assistant` contract with real Azure AI Foundry endpoints:

| Model | Turns | Hard Violations | Soft Violations | Theta | Mean Drift |
|-------|-------|----------------|-----------------|-------|------------|
| **GPT-5.3** (OpenAI) | 16 | **0** | 11 | 0.688 | 0.034 |
| **Claude Sonnet 4.6** (Anthropic) | 10 | 4 | 0 | 0.823 | 0.020 |
| **Mistral-Large-3** (Mistral) | 10 | 5 | 0 | 0.813 | 0.025 |

**Key findings:**
- **GPT-5.3** achieved zero hard violations but exhibited soft quality drift (response completeness and latency)
- **Claude Sonnet 4.6** and **Mistral-Large-3** triggered `no-false-availability` hard violations -- fabricating product availability without catalog access
- All three models scored below the 0.90 Theta threshold for autonomous deployment, demonstrating why runtime behavioral contracts are essential

> These results are consistent with the findings reported in [arXiv:2602.22302](https://arxiv.org/abs/2602.22302). AgentAssert catches violations that traditional guardrails miss because it tracks behavioral drift over entire sessions, not just individual outputs.

---

## Domain Contracts -- Ready to Use

12 production-ready contracts ship with AgentAssert in `contracts/examples/`:

| Contract | Domain | Hard | Soft | Key Checks |
|----------|--------|------|------|------------|
| `ecommerce-product-recommendation` | E-Commerce | 7 | 8 | PII, competitor mentions, sponsored disclosure |
| `ecommerce-order-management` | E-Commerce | 7 | 8 | Payment data, order accuracy, refund policy |
| `ecommerce-customer-service` | E-Commerce | 7 | 8 | Escalation, SLA, customer sentiment |
| `financial-advisor` | Finance | 7 | 8 | Regulatory compliance, risk disclosure, suitability |
| `healthcare-triage` | Healthcare | 9 | 7 | Medical safety, urgency detection, no diagnosis |
| `retail-shopping-assistant` | Retail | 7 | 9 | Availability, pricing accuracy, upsell limits |
| `telecom-customer-support` | Telecom | 7 | 9 | Plan accuracy, billing, cancellation handling |
| `code-generation` | Dev Tools | 7 | 7 | License compliance, security, test coverage |
| `research-assistant` | Research | 6 | 7 | Citation accuracy, source attribution, bias |
| `customer-support` | General | 6 | 5 | Tone, escalation, resolution quality |
| `mcp-tool-server` | MCP (2026) | 6 | 5 | Tool authorization, rate limits, output bounds |
| `rag-agent` | RAG (2026) | 7 | 7 | Hallucination, source grounding, retrieval quality |

---

## ContractSpec DSL

Define behavioral contracts in YAML:

```yaml
contractspec: "0.1"
kind: agent
name: my-agent-contract
description: Behavioral contract for my agent
version: "1.0.0"

invariants:
  hard:
    - name: no-pii-leak
      description: Never expose personal information
      check:
        field: output.pii_detected
        equals: false

  soft:
    - name: tone-quality
      description: Maintain professional tone
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
        - "Rewrite with professional tone"

satisfaction:
  p: 0.95
  delta: 0.1
  k: 3
```

**14 operators:** `equals`, `not_equals`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `contains`, `not_contains`, `matches`, `exists`, `expr`, `between`

---

## Writing Your Own Contract

1. **Identify fields** -- Examine your agent's output and list the fields that matter for safety and quality
2. **Map to flat dict** -- AgentAssert uses `output.field_name` as keys (e.g., `{"output.safe": True}`)
3. **Choose constraint type** -- **Hard** for non-negotiable safety (violations halt execution), **Soft** for quality goals (violations trigger recovery)
4. **Set satisfaction** -- `p` = target compliance rate, `delta` = tolerance, `k` = max violations before alert

---

## SPRT Certification

Certify agents for production with **50-80% fewer test sessions** using Sequential Probability Ratio Testing:

```python
from agentassert_abc.certification.sprt import SPRTCertifier, SPRTDecision

certifier = SPRTCertifier(p0=0.85, p1=0.95, alpha=0.05, beta=0.10)
for session_passed in session_results:
    result = certifier.update(session_passed)
    if result.decision != SPRTDecision.CONTINUE:
        print(f"Decision: {result.decision.value} after {result.sessions_used} sessions")
        break
```

## Compositional Guarantees

Prove safety bounds for multi-agent pipelines:

```python
from agentassert_abc.certification.composition import compose_guarantees

# Agent A (p=0.95) -> Agent B (p=0.98), handoff reliability 0.99
bound = compose_guarantees(p_a=0.95, p_b=0.98, p_h=0.99)
print(f"Pipeline bound: {bound:.3f}")  # p_{A+B} >= 0.921
```

---

## How AgentAssert Differs

| Dimension | AgentAssert | Guardrails AI | NeMo Guardrails | Microsoft AGT |
|-----------|-------------|---------------|-----------------|---------------|
| Formal math (Theta, SPRT) | Yes | No | No | No |
| Session drift detection (JSD) | Yes | No | No | No |
| Compositional safety proofs | Yes | No | No | No |
| Hard/Soft constraint separation | Yes | Partial | No | No |
| Recovery re-prompting | Yes | Yes | Yes | No |
| Framework integrations | 10 adapters | 3 | 1 (LangChain) | 2 |
| Statistical certification (SPRT) | Yes | No | No | No |
| Benchmark suite | 293 scenarios | No | No | No |
| Academic paper | [arXiv:2602.22302](https://arxiv.org/abs/2602.22302) | No | No | No |

---

## Examples

See `examples/` for runnable demos:

| Example | What It Shows |
|---------|---------------|
| `01_basic_monitoring.py` | Simplest usage -- load, monitor, get Theta |
| `02_ecommerce_session.py` | Full e-commerce session from the paper |
| `03_drift_detection.py` | JSD-based behavioral drift over 20 turns |
| `04_sprt_certification.py` | SPRT statistical certification |
| `05_langgraph_middleware.py` | LangGraph StateGraph integration |
| `06_crewai_integration.py` | CrewAI task guardrails |
| `07_composition_pipeline.py` | Multi-agent compositional bounds |
| `08_mcp_tool_monitoring.py` | MCP tool server monitoring |

---

## Research Paper

**"AgentAssert: Formal Behavioral Contracts for Autonomous AI Agents"**

The theoretical foundations, formal proofs, and experimental validation are published in a peer-reviewed paper covering all 6 pillars of the framework, with full mathematical treatment of the Reliability Index, drift dynamics, compositional guarantees, and SPRT certification.

**[Read the paper on arXiv](https://arxiv.org/abs/2602.22302)** (cs.AI + cs.SE)

### Cite This Work

```bibtex
@article{bhardwaj2026agentassert,
  title={AgentAssert: Formal Behavioral Contracts for Autonomous AI Agents},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2602.22302},
  year={2026},
  url={https://arxiv.org/abs/2602.22302}
}
```

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and submission guidelines.

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

For commercial licensing (closed-source, proprietary, or hosted use), see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or contact varun.pratap.bhardwaj@gmail.com.

Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar.

---

<p align="center">
  <strong>Part of <a href="https://qualixar.com">Qualixar</a></strong> -- AI Agent Reliability Engineering<br>
  A research initiative by <a href="https://varunpratap.com">Varun Pratap Bhardwaj</a><br><br>
  <a href="https://qualixar.com">qualixar.com</a> &middot; <a href="https://varunpratap.com">varunpratap.com</a> &middot; <a href="https://arxiv.org/abs/2602.22302">arXiv:2602.22302</a> &middot; <a href="https://agentassert.com">agentassert.com</a>
</p>

---

## ⭐ Support This Project

If this project solves a real problem for you, **please star the repo** — it helps other developers discover Qualixar and signals that the AI agent reliability community is growing. Every star matters.

[![Star History Chart](https://api.star-history.com/svg?repos=qualixar/agentassert-abc&type=Date)](https://star-history.com/#qualixar/agentassert-abc&Date)

---

## Part of the Qualixar AI Agent Reliability Platform

Qualixar is building the open-source infrastructure for AI agent reliability engineering. Seven products, seven peer-reviewed papers, one coherent platform. Each tool solves one reliability pillar:

| Product | Purpose | Install | Paper |
|---------|---------|---------|-------|
| **[SuperLocalMemory](https://github.com/qualixar/superlocalmemory)** | Persistent memory + learning for AI agents | `npx superlocalmemory` | [arXiv:2604.04514](https://arxiv.org/abs/2604.04514) |
| **[Qualixar OS](https://github.com/qualixar/qualixar-os)** | Universal agent runtime (13 execution topologies) | `npx qualixar-os` | [arXiv:2604.06392](https://arxiv.org/abs/2604.06392) |
| **[SLM Mesh](https://github.com/qualixar/slm-mesh)** | P2P coordination across AI agent sessions | `npm i slm-mesh` | — |
| **[SLM MCP Hub](https://github.com/qualixar/slm-mcp-hub)** | Federate 430+ MCP tools through one gateway | `pip install slm-mcp-hub` | — |
| **[AgentAssay](https://github.com/qualixar/agentassay)** | Token-efficient AI agent testing | `pip install agentassay` | [arXiv:2603.02601](https://arxiv.org/abs/2603.02601) |
| **[AgentAssert](https://github.com/qualixar/agentassert-abc)** | Behavioral contracts + drift detection |  `pip install agentassert-abc` | [arXiv:2602.22302](https://arxiv.org/abs/2602.22302) |
| **[SkillFortify](https://github.com/qualixar/skillfortify)** | Formal verification for AI agent skills | `pip install skillfortify` | [arXiv:2603.00195](https://arxiv.org/abs/2603.00195) |

**Zero cloud dependency. Local-first. EU AI Act compliant.**

Start here → **[qualixar.com](https://qualixar.com)** · [All papers on Qualixar HuggingFace](https://huggingface.co/Qualixar)

---
