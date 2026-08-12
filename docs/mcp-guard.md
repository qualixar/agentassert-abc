# MCP Guard — contracts for any agent that speaks MCP

The other adoption surfaces are each tied to something specific: the proxy to an
LLM wire format, the SDK wrappers to a vendor client, the Claude Code hook to
one product. The MCP guard is tied only to the tool protocol, so a single
artifact enforces contracts inside **Claude Code, Codex, Cursor, VS Code,
Antigravity, Windsurf** and anything else that adopts MCP — with no
vendor-specific code.

## Install

```bash
pip install "agentassert-abc[mcp]"
```

## Wire it in

Change the server's launch command in your client's MCP config. Before:

```json
{ "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] }
```

After:

```json
{
  "command": "agentassert-abc-mcp-guard",
  "args": [
    "--contract", "contract.yaml",
    "--server-label", "github",
    "--", "npx", "-y", "@modelcontextprotocol/server-github"
  ]
}
```

That is the entire integration. The guard launches the real server as a child
process and relays JSON-RPC in both directions.

## What it does

On every `tools/call`:

| Decision | Effect |
|---|---|
| `ALLOW` | forwarded unchanged |
| `DENY` | **never forwarded**; the client receives an `isError` result explaining the block |
| `MODIFY` | forwarded with rewritten arguments |
| `REDACT` | forwarded; matched patterns masked in the response |
| `WARN` | forwarded; recorded against Θ |

A `DENY` is returned as an `isError` *result* rather than a JSON-RPC *error* on
purpose: the model reads it as tool output and can choose a different action,
where a protocol error reads to most clients as a transport fault. Enforcement
should redirect the agent, not break its session.

Everything that is not a `tools/call` — initialisation, `tools/list`, resource
reads, sampling, notifications, and any method a future spec revision adds — is
relayed byte-for-byte. The guard does not model MCP; it relays JSON-RPC and
recognises one method, which is why it does not break when the protocol grows.

## Contract state

Tool results are flattened into the `output.*` convention the evaluator uses, so
a contract written against `output.*` works here unchanged. The guard also
always supplies:

- `tool.name` — the tool being called
- `tool.server` — the `--server-label`, so one contract can scope invariants to
  a single server when several are guarded

A contract referencing state the guard can never observe is **refused at
start-up** with a non-zero exit, which surfaces in the client's MCP server log.
It is not silently disabled, and it does not run on scoring the agent as
violating an invariant it has no way to satisfy.

## Failure behaviour

Default is **fail-open**: if the guard itself errors, the call proceeds. A
contract bug must not take an agent down. Pass `--fail-closed` for
security-critical deployments, where a call that cannot be evaluated should not
run.

One asymmetry is deliberate. A scoring failure on the *response* path always
fails open, even under `--fail-closed`: the tool has already executed, so
withholding its output would punish the agent for the guard's own bug without
preventing any side effect.

## Coverage — read this before relying on it

The guard sees **MCP tools**. It does not see an agent's *native* tools.

| Agent | MCP tools | Native tools (built-in edit/shell) | Model calls |
|---|---|---|---|
| Claude Code | ✅ guard | ✅ [hook](./integrations.md) | ✅ proxy |
| Codex CLI | ✅ guard | ✅ hook | ✅ proxy |
| Cursor | ✅ guard | ✅ hook (`beforeShellExecution`) | ✅ proxy |
| VS Code / Copilot | ✅ guard | ❌ no tool veto hook | ⚠️ proxy via bridge |
| Antigravity | ✅ guard | ❌ no tool veto hook | ⚠️ proxy via bridge |

In Copilot and Antigravity the built-in file-edit and shell tools route through
neither MCP nor any registrable hook. There, the guard enforces on MCP tools and
model calls and *measures* the rest. No MCP-level enforcement will see those
native tools, and this table is the honest statement of that boundary rather
than a footnote.

Pair the guard with a vendor hook where one exists, and with the HTTP proxy for
model-call enforcement.

## Pre- vs post-execution

Two decision points, and they are not equivalent:

- A **`PreAction` DENY** returns before the request reaches the server. The tool
  is never executed.
- A **`PostAction` DENY** happens after the server ran the tool. It withholds the
  output from the model — which still keeps the data out of the context window —
  but it cannot un-execute the call.

The guard reports these differently ("the tool did not run" vs "the tool
executed, but its result was not returned") so an audit log never overstates
what was prevented.
