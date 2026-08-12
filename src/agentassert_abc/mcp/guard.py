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

**What this surface can and cannot enforce.** A ``PreAction`` DENY returns
before the request reaches the downstream server, so a denied tool is never
executed. A ``PostAction`` DENY happens after the server has already run the
tool — it withholds the *output* from the model (which still blocks
exfiltration into the context window) but cannot un-execute the call. The two
are not equivalent and are not reported as if they were.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentassert_abc.gateway.content.pii import apply_pii_redaction, evaluate_pii_filter
from agentassert_abc.gateway.events import PostAction, PreAction
from agentassert_abc.gateway.state import flatten_output
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
        self._enforcer = enforcer
        self._server_label = server_label
        self._session_id = session_id or f"mcp-{uuid.uuid4().hex[:12]}"
        self._fail_closed = fail_closed
        self._pending: dict[Any, PendingCall] = {}
        self._lock = threading.Lock()
        self._denied = 0
        self._contract_id = getattr(enforcer._contract, "name", "unknown")

    # -- properties ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def deny_count(self) -> int:
        """Calls this guard blocked or whose output it withheld."""
        return self._denied

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

        try:
            result = self._enforcer.evaluate(
                PreAction(
                    session_id=self._session_id,
                    contract_id=self._contract_id,
                    tool=tool,
                    args=args,
                )
            )
        except Exception as exc:  # noqa: BLE001 — policy failure must not corrupt the stream.
            return self._on_internal_error(message, req_id, tool, exc)

        if result.is_deny():
            self._denied += 1
            return Relay(
                reply=jsonrpc.tool_error_result(
                    req_id, _deny_text(tool, result.reason, result.violation_name)
                )
            )

        forward = message
        args_used = args
        if result.is_modify() and result.modified_args is not None:
            forward = jsonrpc.with_tool_arguments(message, result.modified_args)
            args_used = result.modified_args

        self._track(
            req_id,
            PendingCall(tool=tool, args=args_used, redact_on_return=result.is_redact()),
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

        try:
            return Relay(forward=self._screen_result(message, pending))
        except Exception:  # noqa: BLE001 — a scoring failure must not eat the response.
            # Deliberately fail-open even under `fail_closed`: the tool has
            # already run, so withholding output here punishes the agent for the
            # guard's own bug without preventing any side effect.
            return Relay(forward=message)

    def _screen_result(self, message: dict[str, Any], pending: PendingCall) -> dict[str, Any]:
        result_payload = message.get("result")
        text = jsonrpc.result_text(message)

        state: dict[str, Any] = {
            "tool.name": pending.tool,
            "tool.server": self._server_label,
        }
        state.update(flatten_output(result_payload))
        if text:
            state.setdefault("output.text", text)

        decision = self._enforcer.evaluate(
            PostAction(
                session_id=self._session_id,
                contract_id=self._contract_id,
                tool=pending.tool,
                args=pending.args,
                state=state,
                result=result_payload,
            )
        )

        if decision.is_deny():
            # The tool already ran. Withholding its output still keeps the data
            # out of the model's context, which is the only thing left to
            # protect at this point.
            self._denied += 1
            return jsonrpc.tool_error_result(
                jsonrpc.request_id(message),
                _withheld_text(pending.tool, decision.reason, decision.violation_name),
            )

        redact = pending.redact_on_return or decision.is_redact()
        if text:
            pii = evaluate_pii_filter(
                text, self._enforcer._compiled, self._enforcer._violations, is_streaming=False
            )
            if pii is not None and pii.is_deny():
                self._denied += 1
                return jsonrpc.tool_error_result(
                    jsonrpc.request_id(message),
                    _withheld_text(pending.tool, pii.reason, pii.violation_name),
                )
            redact = redact or (pii is not None and pii.is_redact())

        if redact and text:
            masked = apply_pii_redaction(text, self._enforcer._compiled.pii_compiled_patterns)
            return jsonrpc.with_result_text(message, masked)

        return message

    # -- internals ----------------------------------------------------------

    def _on_internal_error(
        self, message: dict[str, Any], req_id: Any, tool: str, exc: Exception
    ) -> Relay:
        if self._fail_closed:
            self._denied += 1
            return Relay(
                reply=jsonrpc.tool_error_result(
                    req_id,
                    f"AgentAssert blocked '{tool}': the contract could not be "
                    f"evaluated and this guard runs fail-closed ({type(exc).__name__}).",
                )
            )
        # Fail-open relays the ORIGINAL request. The call is deliberately left
        # untracked: evaluation is already known to be broken for it, so scoring
        # its response would report a violation caused by the guard's own fault.
        return Relay(forward=message)

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
