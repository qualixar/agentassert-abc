# Domain Contracts Catalog

AgentAssert ships with 12 production-ready contracts covering common agent domains. Each contract defines hard constraints (safety rules that halt execution), soft constraints (quality goals with recovery), and governance rules (operational guardrails).

---

## Included Contracts

All contracts are in `contracts/examples/` and can be loaded directly:

```python
import agentassert_abc as aa

contract = aa.load("contracts/examples/ecommerce-product-recommendation.yaml")
```

| Contract File | Domain | Hard | Soft | Gov | Key Checks |
|--------------|--------|------|------|-----|------------|
| `ecommerce-product-recommendation.yaml` | E-Commerce | 6 | 6 | 2 | PII, competitor filtering, sponsored disclosure, price manipulation, brand tone, relevance |
| `ecommerce-order-management.yaml` | E-Commerce | 7 | 8 | -- | Order accuracy, refund authorization, payment data, fulfillment status |
| `ecommerce-customer-service.yaml` | E-Commerce | 7 | 8 | -- | PII protection, escalation triggers, response quality, wait time |
| `financial-advisor.yaml` | Finance | 7 | 8 | -- | Suitability rules, disclaimer requirements, unauthorized trading, risk disclosure |
| `healthcare-triage.yaml` | Healthcare | 9 | 7 | -- | Diagnosis prohibition, emergency detection, medication safety, HIPAA compliance |
| `retail-shopping-assistant.yaml` | Retail | 7 | 9 | -- | Inventory accuracy, pricing transparency, return policy compliance, upsell limits |
| `telecom-customer-support.yaml` | Telecom | 7 | 9 | -- | Account security, plan change authorization, billing accuracy, SLA compliance |
| `code-generation.yaml` | Dev Tools | 7 | 7 | -- | Injection prevention, license compliance, test coverage, code quality scores |
| `research-assistant.yaml` | Research | 6 | 7 | -- | Citation accuracy, source verification, bias detection, hallucination prevention |
| `customer-support.yaml` | General | 6 | 5 | -- | PII, tone, escalation, response completeness, latency |
| `mcp-tool-server.yaml` | MCP (2026) | 6 | 5 | -- | Tool authorization, input validation, rate limiting, schema compliance |
| `rag-agent.yaml` | RAG (2026) | 7 | 7 | -- | Source attribution, hallucination detection, context relevance, chunk quality |

---

## Contract Details

### E-Commerce Product Recommendation

**Use case:** An AI agent that suggests products based on browsing history and preferences.

**Hard constraints:**

- No competitor product recommendations
- No false availability claims
- No unauthorized discounts
- No PII exposure (GDPR/CCPA)
- Sponsored products must be disclosed (FTC)
- No undisclosed dynamic pricing

**Soft constraints with recovery:**

- Brand tone score above 0.7 (recovery: inject tone correction)
- Recommendation relevance above 0.6 (recovery: re-query with stricter filters)
- Response completeness above 0.6 (recovery: enrich product details)
- Upsell count limited to 2 per interaction
- Response latency under 3000ms
- Customer satisfaction above 0.4 (recovery: escalate to human)

---

### Healthcare Triage

**Use case:** A medical triage assistant that assesses symptoms and recommends next steps.

**Hard constraints:**

- Never provide a definitive diagnosis
- Always detect emergency symptoms and escalate
- Never recommend specific medications without physician oversight
- HIPAA-compliant data handling
- No contradicting established medical guidelines
- Always include disclaimers for medical advice
- No sharing patient data across sessions
- Emergency symptoms must trigger immediate escalation
- Never dismiss reported symptoms

!!! warning "Healthcare contracts require additional validation"
    The healthcare triage contract is a starting point. Consult your compliance team and legal counsel before deploying medical AI agents. Regulatory requirements vary by jurisdiction.

---

### Financial Advisor

**Use case:** An AI agent providing investment guidance and financial planning.

**Hard constraints:**

- Suitability rules enforcement (recommendations must match risk profile)
- Required disclaimers on all financial advice
- No unauthorized trading or transactions
- Risk disclosure on every recommendation
- Regulatory compliance (SEC, FINRA)
- No guarantees of returns
- Client data confidentiality

---

### MCP Tool Server

**Use case:** A Model Context Protocol tool server that exposes tools to AI agents.

**Hard constraints:**

- Tool calls must be authorized against a registry
- Input validation on all tool parameters
- Rate limiting enforcement
- Schema compliance for tool responses
- No execution of unauthorized system commands
- Audit logging for all tool invocations

---

## Using a Pre-Built Contract

```python
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

# Load any included contract by path
contract = aa.load("contracts/examples/healthcare-triage.yaml")
adapter = GenericAdapter(contract)

# Check agent output
result = adapter.check({
    "output.diagnosis_provided": False,
    "output.emergency_detected": True,
    "output.emergency_escalated": True,
    "output.disclaimer_included": True,
    "output.hipaa_compliant": True,
})

print(f"Violations: {result.hard_violations} hard, {result.soft_violations} soft")
```

---

## Customizing a Contract

Start from an included contract and modify it for your needs:

1. Copy the contract file:

    ```bash
    cp contracts/examples/customer-support.yaml my-support-contract.yaml
    ```

2. Edit the YAML to add, remove, or modify constraints:

    ```yaml
    # Add a new hard constraint
    invariants:
      hard:
        # ... existing constraints ...
        - name: no-competitor-mention
          description: Never mention competitor products
          check:
            field: output.competitor_mentioned
            equals: false
    ```

3. Update the satisfaction targets if needed:

    ```yaml
    satisfaction:
      p: 0.98     # Stricter: 98% compliance required
      delta: 0.05  # Tighter tolerance
      k: 5         # Larger evaluation window
    ```

4. Load your custom contract:

    ```python
    contract = aa.load("my-support-contract.yaml")
    ```

!!! tip "Contract versioning"
    Use semantic versioning in the `version` field. When you tighten constraints, bump the minor version. When you add new hard constraints, bump the major version. This helps track which contract version an agent was certified against.
