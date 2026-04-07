# Examples Walkthrough

AgentAssert ships with 8 runnable example scripts in the `examples/` directory. Each demonstrates a specific capability, from basic monitoring to multi-agent pipelines.

All examples are self-contained and can be run directly:

```bash
python examples/01_basic_monitoring.py
```

---

## 01 -- Basic Monitoring

**File:** `examples/01_basic_monitoring.py`

The simplest way to use AgentAssert. Defines a contract inline, monitors three turns of simulated agent output, and prints the session summary with Theta.

**What you learn:**

- Loading a contract from a YAML string with `aa.loads()`
- Creating a `GenericAdapter`
- Calling `adapter.check()` on each turn
- Reading `StepResult` fields (hard/soft violations, violated names)
- Getting a `SessionSummary` with Theta

---

## 02 -- E-Commerce Session

**File:** `examples/02_ecommerce_session.py`

A full e-commerce product recommendation session with realistic data. Simulates a multi-turn conversation where the agent makes recommendations, encounters a soft violation (low tone score), recovers, and then hits a hard violation (PII leak).

**What you learn:**

- Loading a domain contract from a YAML file
- Multi-turn session tracking
- Soft violation recovery within the recovery window
- Hard violation detection and `ContractBreachError`
- Session summary with compliance rates and Theta

---

## 03 -- Drift Detection

**File:** `examples/03_drift_detection.py`

Demonstrates behavioral drift detection over 20 turns. The agent's tone score gradually decreases -- each turn passes the constraint, but the distribution is shifting. AgentAssert detects this drift using JSD.

**What you learn:**

- Drift score tracking per turn
- Warning and critical drift thresholds
- How drift accumulates even when individual turns are compliant

---

## 04 -- SPRT Certification

**File:** `examples/04_sprt_certification.py`

Shows how to certify an agent for production using Sequential Probability Ratio Testing. Runs multiple simulated sessions and feeds pass/fail results into the `SPRTCertifier` until a statistical decision is reached.

**What you learn:**

- Creating an `SPRTCertifier` with p0, p1, alpha, beta parameters
- Feeding session results with `certifier.update()`
- Interpreting ACCEPT, REJECT, and CONTINUE decisions
- How SPRT reaches decisions in fewer sessions than fixed-sample testing

---

## 05 -- LangGraph Middleware

**File:** `examples/05_langgraph_middleware.py`

Integrates AgentAssert with a LangGraph `StateGraph`. Wraps node functions with `adapter.wrap_node()` so contract evaluation happens after each node executes.

**What you learn:**

- Creating a `LangGraphAdapter`
- Wrapping sync and async node functions
- How `ContractBreachError` interrupts graph execution on hard violations
- Getting session metrics after graph execution

!!! note "Requires langgraph"
    Install with `pip install langgraph` before running this example.

---

## 06 -- CrewAI Integration

**File:** `examples/06_crewai_integration.py`

Integrates AgentAssert with CrewAI using the guardrail and callback interfaces. Shows both blocking mode (guardrail rejects on hard violations) and monitoring mode (callback logs metrics).

**What you learn:**

- Creating a `CrewAIAdapter`
- Using `adapter.guardrail` as a task guardrail (CrewAI retries on rejection)
- Using `adapter.callback` for non-blocking monitoring
- How structured output (Pydantic/JSON) is preserved through the guardrail

!!! note "Requires crewai"
    Install with `pip install crewai` before running this example.

---

## 07 -- Composition Pipeline

**File:** `examples/07_composition_pipeline.py`

Demonstrates compositional safety guarantees for multi-agent pipelines. Computes the lower bound on pipeline compliance given individual agent compliance rates and handoff reliability.

**What you learn:**

- Using `compose_guarantees()` for two-agent pipelines
- Chaining guarantees for three or more agents
- How handoff reliability affects pipeline bounds
- Interpreting conservative (lower bound) guarantees

---

## 08 -- MCP Tool Monitoring

**File:** `examples/08_mcp_tool_monitoring.py`

Monitors a Model Context Protocol (MCP) tool server against a behavioral contract. Validates tool authorization, input schemas, rate limiting, and response quality.

**What you learn:**

- Loading the `mcp-tool-server.yaml` contract
- Monitoring tool calls as agent turns
- Governance constraints (tool authorization, token budgets)
- Session metrics for tool servers

---

## Running All Examples

```bash
# Run each example
for i in 01 02 03 04 05 06 07 08; do
    echo "=== Example $i ==="
    python examples/${i}_*.py
    echo
done
```

!!! tip "Start with 01 and 02"
    If you are new to AgentAssert, start with examples 01 (basic monitoring) and 02 (e-commerce session). These cover the core workflow. Move to the framework-specific examples (05, 06) and certification (04) once you are comfortable with contracts and adapters.
