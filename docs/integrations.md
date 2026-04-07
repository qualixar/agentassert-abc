# Framework Integration Guide

AgentAssert is plug-and-play with the major 2026 agent frameworks. Each adapter translates the framework's native output format into a flat dict that the contract engine can evaluate.

All adapters share a common interface:

- `check(output)` -- Evaluate agent output, return `StepResult`
- `session_summary()` -- Get aggregated metrics including Theta
- `extract_state(output)` -- Normalize framework output to flat dict

Framework-specific methods (like `wrap_node`, `guardrail`, `output_guardrail`) are added on top.

---

## GenericAdapter -- Framework-Agnostic

The simplest adapter. Works with any agent that produces a Python dict. No framework dependency required.

```python
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

contract = aa.load("contracts/examples/customer-support.yaml")
adapter = GenericAdapter(contract)

# Check without raising
result = adapter.check({
    "output.pii_detected": False,
    "output.tone_score": 0.85,
})

# Check and raise on hard violations
from agentassert_abc.exceptions import ContractBreachError

try:
    result = adapter.check_and_raise({
        "output.pii_detected": True,  # Hard violation!
    })
except ContractBreachError as e:
    print(f"Blocked: {e}")

# Session metrics
summary = adapter.session_summary()
print(f"Theta: {summary.theta:.3f}")
```

!!! tip "When to use GenericAdapter"
    Use `GenericAdapter` when your agent is a plain function that returns a dict, when you are prototyping, or when no framework-specific adapter exists for your setup. It is also the right choice for testing contracts in isolation.

---

## LangGraph Integration

The LangGraph adapter intercepts StateGraph node outputs. It wraps individual node functions or entire compiled graphs.

**Install:** `pip install langgraph`

### Wrapping Individual Nodes

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

import agentassert_abc as aa
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.integrations.langgraph import LangGraphAdapter


# Define your state schema
class State(TypedDict):
    query: str
    response: str
    pii_detected: bool
    tone_score: float


# Load contract and create adapter
contract = aa.load("contracts/examples/customer-support.yaml")
adapter = LangGraphAdapter(contract)


# Define your node functions
def classify(state: State) -> dict:
    return {
        "output.pii_detected": False,
        "output.tone_score": 0.9,
    }


def respond(state: State) -> dict:
    return {
        "response": "Here is your answer.",
        "output.pii_detected": False,
        "output.tone_score": 0.85,
    }


# Build the graph with wrapped nodes
builder = StateGraph(State)
builder.add_node("classify", adapter.wrap_node(classify))
builder.add_node("respond", adapter.wrap_node(respond))
builder.add_edge(START, "classify")
builder.add_edge("classify", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# Run with contract enforcement
try:
    result = graph.invoke({"query": "What is my order status?"})
except ContractBreachError as e:
    print(f"Hard violation blocked execution: {e}")

# Session metrics
print(f"Session Theta: {adapter.session_summary().theta:.3f}")
```

### Wrapping the Entire Graph

If you want to evaluate the final state after graph execution (instead of per-node), wrap the compiled graph:

```python
monitored = adapter.wrap_graph(graph)

# invoke() now evaluates the final state against the contract
result = monitored.invoke({"query": "Help me"})

# Async support
result = await monitored.ainvoke({"query": "Help me"})
```

### Sync and Async Nodes

The adapter auto-detects async node functions. Both work transparently:

```python
# Sync node -- works
builder.add_node("sync_node", adapter.wrap_node(my_sync_fn))

# Async node -- also works
builder.add_node("async_node", adapter.wrap_node(my_async_fn))
```

!!! note "Streaming"
    When using `wrap_graph()`, the `stream()` method is proxied to the underlying graph without monitoring. For per-node monitoring during streaming, use `wrap_node()` instead.

---

## CrewAI Integration

The CrewAI adapter provides two integration modes: a guardrail that rejects output on hard violations (triggering CrewAI's built-in retry), and a callback for non-blocking monitoring.

**Install:** `pip install crewai`

### Guardrail Mode (Blocking)

```python
from crewai import Agent, Task, Crew
import agentassert_abc as aa
from agentassert_abc.integrations.crewai import CrewAIAdapter

contract = aa.load("contracts/examples/research-assistant.yaml")
adapter = CrewAIAdapter(contract)

# Define your agents
researcher = Agent(
    role="Senior Researcher",
    goal="Produce a cited research report",
    backstory="Expert researcher with 10 years experience",
)

reviewer = Agent(
    role="Quality Reviewer",
    goal="Review research for accuracy and citations",
    backstory="Meticulous editor",
)

# Guardrail mode: rejects on hard violations, CrewAI retries automatically
research_task = Task(
    description="Research AI agent frameworks in 2026",
    expected_output="Cited report on top 5 frameworks",
    agent=researcher,
    guardrail=adapter.guardrail,
    guardrail_max_retries=3,
)

# Callback mode: logs metrics without blocking
review_task = Task(
    description="Review the research report for accuracy",
    expected_output="Approved report with corrections",
    agent=reviewer,
    callback=adapter.callback,
)

# Run the crew
crew = Crew(agents=[researcher, reviewer], tasks=[research_task, review_task])
result = crew.kickoff()

# Session metrics
summary = adapter.session_summary()
print(f"Theta: {summary.theta:.3f}")
```

### How the Guardrail Works

1. A task completes and produces `TaskOutput`
2. The guardrail calls `adapter.extract_state()` to normalize the output
3. The contract engine evaluates all constraints
4. If hard violations exist: returns `(False, error_message)` -- CrewAI retries
5. If only soft violations: returns `(True, task_output)` -- execution continues

!!! tip "Structured output preservation"
    The guardrail returns the original `TaskOutput` on success, preserving any Pydantic or JSON structured output. This means downstream tasks receive the full typed object, not a raw string.

### State Extraction

The CrewAI adapter extracts state from `TaskOutput` in priority order:

1. **Pydantic model** -- `output.pydantic.model_dump()` with `output.` prefix
2. **JSON dict** -- `output.json_dict` with `output.` prefix
3. **Raw string** -- `{"output.raw": text}`

Metadata like agent name and message count is also extracted when available.

---

## OpenAI Agents SDK Integration

The OpenAI Agents SDK adapter provides output guardrails, input guardrails, and lifecycle hooks.

**Install:** `pip install openai-agents`

### Output Guardrail (Blocking)

```python
from agents import Agent, Runner
from pydantic import BaseModel

import agentassert_abc as aa
from agentassert_abc.integrations.openai_agents import OpenAIAgentsAdapter


class TriageOutput(BaseModel):
    urgency: str
    recommendation: str
    pii_detected: bool
    confidence: float


contract = aa.load("contracts/examples/healthcare-triage.yaml")
adapter = OpenAIAgentsAdapter(contract)

# Output guardrail triggers tripwire on hard violations
agent = Agent(
    name="triage-agent",
    instructions="You are a medical triage assistant.",
    output_guardrails=[adapter.output_guardrail],
    output_type=TriageOutput,
)

result = await Runner.run(agent, "I have chest pain and shortness of breath")

summary = adapter.session_summary()
print(f"Theta: {summary.theta:.3f}")
```

### Input Guardrail (Precondition Check)

```python
# Input guardrail validates preconditions before the agent processes input
agent = Agent(
    name="triage-agent",
    instructions="You are a medical triage assistant.",
    input_guardrails=[adapter.input_guardrail],
    output_guardrails=[adapter.output_guardrail],
    output_type=TriageOutput,
)
```

### Run Hooks (Non-Blocking Monitoring)

```python
# Hooks monitor agent lifecycle without stopping execution
result = await Runner.run(
    agent,
    "I have a headache",
    hooks=adapter.run_hooks,
)

summary = adapter.session_summary()
print(f"Theta: {summary.theta:.3f}")
```

### How the Output Guardrail Works

1. The agent produces output (Pydantic model, dict, or string)
2. `extract_state()` normalizes it to a flat dict with `output.` prefix
3. The contract engine evaluates all constraints
4. Returns `GuardrailFunctionOutput` with `tripwire_triggered=True` on hard violations
5. The Agents SDK raises `OutputGuardrailTripwireTriggered`, stopping the agent

---

## Writing Your Own Adapter

To integrate AgentAssert with a framework not listed above, implement three methods:

```python
from agentassert_abc.monitor.session import SessionMonitor
from agentassert_abc.models import ContractSpec


class MyFrameworkAdapter:
    def __init__(self, contract: ContractSpec) -> None:
        self._monitor = SessionMonitor(contract)

    def extract_state(self, output):
        """Convert your framework's output to a flat dict.

        Keys should use dot notation: "output.field_name"
        Values should be primitives (str, int, float, bool).
        """
        # Example: extract from your framework's output object
        return {
            "output.pii_detected": output.pii_detected,
            "output.tone_score": output.tone_score,
            # ... map all fields your contract checks
        }

    def check(self, agent_output):
        """Evaluate output against the contract."""
        state = self.extract_state(agent_output)
        return self._monitor.step(state)

    def session_summary(self):
        """Get aggregated session metrics."""
        return self._monitor.session_summary()
```

!!! warning "Thread safety"
    If your framework uses multiple threads, wrap `self._monitor.step()` and `self._monitor.session_summary()` calls with a `threading.Lock`. All built-in adapters do this.

### Key Design Rules

1. **Flat dicts only.** The contract engine evaluates flat `{"output.field": value}` dicts. Your `extract_state()` must flatten nested output.
2. **Immutable copies.** Always copy the output dict before passing it to the monitor. Do not let the contract engine mutate the original state.
3. **Prefix convention.** Use `output.` for agent output fields, `session.` for session metadata, `tools.` for tool-related fields, `system.` for system state.
4. **Lazy imports.** Import your framework's types inside a try/except block so AgentAssert does not fail to import when the framework is not installed.
