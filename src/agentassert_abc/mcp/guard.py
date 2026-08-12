# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Enforcement policy for MCP ``tools/call`` traffic.

Pure and I/O-free on purpose. Every decision this module makes is a function of
the message plus enforcer state, so the whole policy is testable without
spawning a subprocess or touching a pipe. :mod:`agentassert_abc.mcp.interposer`
owns all the I/O and does no policy.

That split is deliberate: the Claude Code hook put its policy inline in
``main()``, which is why it shipped at 0% coverage. The same mistake is not
repeated here.

The contract decisions themselves come from
:class:`~agentassert_abc.enforce.EnforcementBridge`, which every framework
integration also uses. This module only translates between MCP's JSON-RPC
vocabulary and the bridge's neutral one — it carries no policy of its own, so
a change to how denials or redaction behave lands in one place, not two.

**What this surface can and cannot enforce.** A ``PreAction`` DENY returns
before the request reaches the downstream server, so a denied tool is never
executed. A ``PostAction`` DENY happens after the server has already run the
tool — it withholds the *output* from the model (which still blocks
exfiltration into the context window) but cannot un-execute the call. The two
are not equivalent and are not reported as if they were.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentassert_abc.enforce.bridge import EnforcementBridge
from agentassert_abc.mcp import jsonrpc

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer

__all__ = ["McpGuard", "PendingCall", "Relay"]


@dataclass(frozen=True)
class Relay:
    """What the pump should do with one message.

    Exactly one of the two fields is normally set. ``forward`` continues the
    message on its original path; ``reply`` short-circuits it straight back to
    the sender, which is how a DENY is delivered without the downstream server
    ever seeing the request.
    """

    forward: dict[str, Any] | None = None
    reply: dict[str, Any] | None = None


@dataclass(frozen=True)
class PendingCall:
    """A ``tools/call`` in flight, awaiting its response."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    redact_on_return: bool = False


class McpGuard:
    """Applies a contract to MCP ``tools/call`` traffic in both directions.

    Args:
        enforcer: loaded contract enforcer.
        server_label: name for the downstream server, recorded as
            ``tool.server`` so a contract can scope invariants to one server
            when several are guarded.
        session_id: stable id for this session; generated when omitted.
        fail_closed: when the guard itself errors, deny instead of allowing.
            Defaults to ``False`` to match the other adoption surfaces — a
            contract bug must not take an agent down. Set it for
            security-critical deployments, where an unevaluable call should not
            proceed.
    """

    def __init__(
        self,
        enforcer: SessionEnforcer,
        *,
        server_label: str = "mcp",
        session_id: str | None = None,
        fail_closed: bool = False,
    ) -> None:
        self._bridge = EnforcementBridge(
            enforcer,
            surface="mcp",
            session_id=session_id,
            fail_closed=fail_closed,
            base_state={"tool.server": server_label},
        )
        self._server_label = server_label
        self._pending: dict[Any, PendingCall] = {}
        self._lock = threading.Lock()

    # -- properties ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._bridge.session_id

    @property
    def deny_count(self) -> int:
        """Calls this guard blocked or whose output it withheld."""
        return self._bridge.deny_count

    @property
    def pending_count(self) -> int:
        """Tool calls forwarded but not yet answered."""
        with self._lock:
            return len(self._pending)

    # -- client -> server ---------------------------------------------------

    def on_client_message(self, message: dict[str, Any]) -> Relay:
        """Screen one message travelling from the MCP client to the server.

        Anything that is not a ``tools/call`` request is relayed untouched —
        initialisation, ``tools/list``, resource reads, notifications and any
        method a future spec revision adds.
        """
        if not jsonrpc.is_tool_call_request(message):
            return Relay(forward=message)

        req_id = jsonrpc.request_id(message)
        tool = jsonrpc.tool_call_name(message)
        args = jsonrpc.tool_call_arguments(message)

        decision = self._bridge.before_tool(tool, args)

        if not decision.allowed:
            return Relay(
                reply=jsonrpc.tool_error_result(
                    req_id, _deny_text(tool, decision.reason, decision.violation)
                )
            )

        forward = message
        if decision.modified:
            forward = jsonrpc.with_tool_arguments(message, decision.arguments)

        # A call the bridge could not evaluate is deliberately left untracked:
        # scoring its response would report a violation caused by our own fault.
        if decision.evaluated:
            self._track(
                req_id,
                PendingCall(
                    tool=tool,
                    args=decision.arguments,
                    redact_on_return=decision.redact_result,
                ),
            )
        return Relay(forward=forward)

    # -- server -> client ---------------------------------------------------

    def on_server_message(self, message: dict[str, Any]) -> Relay:
        """Screen one message travelling from the MCP server back to the client.

        Only responses to tool calls this guard forwarded are inspected. Server
        -initiated requests (sampling, elicitation, roots) pass through: they are
        the server asking the *client* for something, not a tool executing.
        """
        req_id = jsonrpc.request_id(message)
        if req_id is None or "result" not in message:
            return Relay(forward=message)

        pending = self._take(req_id)
        if pending is None:
            return Relay(forward=message)

        outcome = self._bridge.after_tool(
            pending.tool,
            pending.args,
            message.get("result"),
            text=jsonrpc.result_text(message),
            force_redact=pending.redact_on_return,
        )

        if not outcome.allowed:
            # The tool already ran. Withholding its output still keeps the data
            # out of the model's context, which is the only thing left to
            # protect at this point.
            return Relay(
                forward=jsonrpc.tool_error_result(
                    req_id, _withheld_text(pending.tool, outcome.reason, outcome.violation)
                )
            )

        if outcome.redacted:
            return Relay(forward=jsonrpc.with_result_text(message, outcome.redacted_text or ""))

        return Relay(forward=message)

    # -- internals ----------------------------------------------------------

    def _track(self, req_id: Any, call: PendingCall) -> None:
        with self._lock:
            self._pending[req_id] = call

    def _take(self, req_id: Any) -> PendingCall | None:
        with self._lock:
            return self._pending.pop(req_id, None)


def _deny_text(tool: str, reason: str, violation: str) -> str:
    detail = reason or "the active behavioral contract forbids this action"
    suffix = f" [{violation}]" if violation else ""
    return (
        f"AgentAssert denied '{tool}' before execution: {detail}{suffix}. "
        "The tool did not run. Choose a different action that satisfies the contract."
    )


def _withheld_text(tool: str, reason: str, violation: str) -> str:
    detail = reason or "the active behavioral contract forbids returning this output"
    suffix = f" [{violation}]" if violation else ""
    return (
        f"AgentAssert withheld the output of '{tool}': {detail}{suffix}. "
        "The tool executed, but its result was not returned."
    )
