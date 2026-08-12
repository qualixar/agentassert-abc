# Enforcement coverage — what AgentAssert can and cannot stop

Measurement scores a session after it happened. Enforcement decides, before a
tool runs, whether it may run at all. This page states exactly where the second
one is possible, and where it is not.

There are three interception planes. Every supported target lands in at least
one.

| Plane | Mechanism | Power |
|---|---|---|
| **P1 — in-process** | the framework's own tool hook | full pre-execution veto |
| **P2 — tool protocol** | [MCP guard](./mcp-guard.md) | full veto on MCP tools |
| **P3 — transport** | HTTP proxy (`base_url`) | veto the model call |

## Agent frameworks (P1)

You control the process, so enforcement is complete: a denied tool never runs.

| Framework | Hook | How a denial is spelled |
|---|---|---|
| CrewAI | `BeforeToolCallHook` | return `False` |
| LangChain / LangGraph | `wrap_tool_call` | never call the handler |
| **DeerFlow** | *(built on LangGraph)* | covered by the LangChain shim |
| Microsoft Agent Framework | `FunctionMiddleware` | never `await next(...)` |
| AgentScope | `pre__acting` hook | raise `ContractBreachError` |
| Any other | `EnforcementBridge` directly | your call — see below |

```python
from agentassert_abc.enforce import bridge_from_yaml
from agentassert_abc.enforce.shims import crewai_before_tool_hook

guard = bridge_from_yaml("contract.yaml", surface="crewai")
crew = Crew(agents=[...], before_tool_call_hooks=[crewai_before_tool_hook(guard)])
```

Nothing is installed for you. The shims are structurally typed against each
framework's hook objects and import nothing at module level, so
`agentassert-abc[enforce]` pulls in **no agent framework**.

### Anything not on that list

The bridge is the whole API, and it is deliberately small enough to wire by hand
in a few lines:

```python
guard = bridge_from_yaml("contract.yaml", surface="my-framework")

decision = guard.before_tool(name, args)
if not decision:
    return f"blocked: {decision.reason}"       # do not run the tool
result = run_tool(decision.arguments)          # note: possibly rewritten

outcome = guard.after_tool(name, decision.arguments, result)
if not outcome:
    return f"output withheld: {outcome.reason}"
return outcome.redacted_text if outcome.redacted else result
```

That is the entire contract a shim satisfies. Writing one for a new framework is
a ~40-line file.

## Coding agents (P2 / P3)

You do **not** control the process. Coverage depends on what the vendor exposes.

| Agent | MCP tools | Native tools (built-in edit/shell) | Model calls |
|---|---|---|---|
| Claude Code | ✅ guard | ✅ hook (`PreToolUse`) | ✅ proxy |
| Codex CLI | ✅ guard | ✅ hook (`PreToolUse`) | ✅ proxy |
| Cursor | ✅ guard | ✅ hook (`beforeShellExecution`) | ✅ proxy |
| VS Code / Copilot | ✅ guard | ❌ **no tool veto hook** | ⚠️ proxy via bridge |
| Antigravity | ✅ guard | ❌ **no tool veto hook** | ⚠️ proxy via bridge |
| Windsurf | ✅ guard | ❌ **no tool veto hook** | ⚠️ proxy via bridge |

**The honest limit.** In Copilot, Antigravity and Windsurf the built-in
file-edit and shell tools route through neither MCP nor any registrable hook.
There, AgentAssert enforces on MCP tools and model calls and *measures* the
rest. No amount of MCP-level enforcement will see those native tools. This table
is the statement of that boundary, not a footnote to it.

## Pre- versus post-execution

Two decision points, and they are not equivalent:

- **`before_tool` DENY** — returns before the tool runs. The tool is never
  executed.
- **`after_tool` DENY** — the tool has already run. Its output is withheld from
  the model, which still keeps the data out of the context window, but the side
  effect has happened.

Every surface reports these with different wording ("the tool did not run" vs
"the tool executed, but its result was not returned") so an audit trail never
overstates what enforcement prevented.

## Failure behaviour

Default is **fail-open** everywhere: if enforcement itself errors, the agent
proceeds. A contract bug must not take a production agent down. Pass
`fail_closed=True` (or `--fail-closed` to the MCP guard) where an unevaluable
call should not run.

One asymmetry is deliberate and applies to every surface: a scoring failure on
the **result** path always fails open, even under `fail_closed`. The tool has
already executed, so withholding its output would punish the agent for our bug
without preventing any side effect.

A call whose contract could not be evaluated is marked `evaluated=False` and its
result is **not** scored — otherwise the agent would be recorded as violating an
invariant that never actually ran.
