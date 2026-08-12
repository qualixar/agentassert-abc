# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""The framework-neutral enforcement bridge.

Every agent framework worth integrating exposes the same two extension points:
a pre-tool hook that can veto, and a post-tool hook that sees the result. They
differ only in how the veto is spelled — CrewAI returns ``False``, LangChain
short-circuits a handler, Microsoft Agent Framework declines to call
``call_next``, AgentScope rewrites kwargs in a pre-hook.

This module is that shape, once. :class:`EnforcementBridge` turns a tool call
into a :class:`ToolDecision` and a tool result into a :class:`ToolOutcome`;
a framework shim's only job is to translate its native hook signature in and its
native veto convention out. Shims stay at roughly forty lines and stay
disposable, which is what keeps a framework's breaking release from becoming an
AgentAssert breaking release.

This is also the single source of policy truth: the MCP guard delegates here
rather than carrying its own copy of the deny/redact/PII logic.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.gateway.content.pii import apply_pii_redaction, evaluate_pii_filter
from agentassert_abc.gateway.events import PostAction, PreAction, TurnEnd, TurnStart
from agentassert_abc.gateway.state import flatten_output

if TYPE_CHECKING:
    from agentassert_abc.gateway.enforcer import SessionEnforcer

__all__ = ["EnforcementBridge", "ToolDecision", "ToolOutcome"]


@dataclass(frozen=True)
class ToolDecision:
    """The verdict on a tool call that has not run yet.

    ``allowed is False`` means the tool must not execute. ``arguments`` is what
    the tool should run with when it is allowed — identical to what was passed
    in unless the contract rewrote it.
    """

    allowed: bool
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    violation: str = ""
    modified: bool = False
    #: False when the contract could not be evaluated at all. A caller that
    #: pairs pre- with post-hooks should skip the post-hook for these: scoring
    #: the result would report a violation caused by the bridge's own failure.
    evaluated: bool = True
    #: The contract asked for the *result* to be redacted when it comes back.
    #: Pass this to ``after_tool(force_redact=...)``, since the decision is made
    #: before the output that it applies to exists.
    redact_result: bool = False
    tool: str = ""
    session_id: str = ""
    contract_id: str = ""

    def raise_if_denied(self) -> None:
        """Raise :class:`ContractBreachError` when the call was denied.

        For frameworks whose hooks signal refusal by raising rather than by
        returning a sentinel.
        """
        if self.allowed:
            return
        raise ContractBreachError(
            violation_name=self.violation,
            reason=self.reason,
            tool=self.tool,
            session_id=self.session_id,
            contract_id=self.contract_id,
        )

    def __bool__(self) -> bool:
        """Truthy when allowed, so a shim can ``return bool(decision)``."""
        return self.allowed


@dataclass(frozen=True)
class ToolOutcome:
    """The verdict on a tool result, after the tool has run.

    ``allowed is False`` means the output must be withheld from the model. It
    does **not** mean the tool did not execute — by this point it has. The two
    are reported separately so an audit trail never overstates what enforcement
    prevented.
    """

    allowed: bool
    result: Any = None
    redacted_text: str | None = None
    reason: str = ""
    violation: str = ""
    tool: str = ""
    session_id: str = ""
    contract_id: str = ""

    @property
    def redacted(self) -> bool:
        return self.redacted_text is not None

    def raise_if_denied(self) -> None:
        if self.allowed:
            return
        raise ContractBreachError(
            violation_name=self.violation,
            reason=self.reason,
            tool=self.tool,
            session_id=self.session_id,
            contract_id=self.contract_id,
        )

    def __bool__(self) -> bool:
        return self.allowed


class EnforcementBridge:
    """Framework-neutral enforcement over a loaded contract.

    Args:
        enforcer: the loaded contract enforcer.
        surface: short name for the integration, recorded on events and used in
            error messages (``"crewai"``, ``"langchain"``, ``"mcp"``, ...).
        session_id: stable id for this session; generated when omitted.
        fail_closed: when the bridge itself errors, deny instead of allowing.
            Defaults to ``False`` — a contract bug must not take an agent down.
        base_state: fields merged into every ``PostAction`` state, for a surface
            that always knows something extra (the MCP guard supplies
            ``tool.server``).
    """

    def __init__(
        self,
        enforcer: SessionEnforcer,
        *,
        surface: str = "bridge",
        session_id: str | None = None,
        fail_closed: bool = False,
        base_state: dict[str, Any] | None = None,
    ) -> None:
        self._enforcer = enforcer
        self._surface = surface
        self._session_id = session_id or f"{surface}-{uuid.uuid4().hex[:12]}"
        self._fail_closed = fail_closed
        self._base_state = dict(base_state or {})
        self._contract_id = getattr(enforcer._contract, "name", "unknown")
        self._denied = 0
        self._lock = threading.Lock()

    # -- properties ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def contract_id(self) -> str:
        return self._contract_id

    @property
    def surface(self) -> str:
        return self._surface

    @property
    def deny_count(self) -> int:
        """Calls blocked, plus results whose output was withheld."""
        return self._denied

    @property
    def enforcer(self) -> SessionEnforcer:
        return self._enforcer

    # -- tool boundary ------------------------------------------------------

    def before_tool(self, tool: str, args: dict[str, Any] | None = None) -> ToolDecision:
        """Screen a tool call that has not executed yet.

        Args:
            tool: the tool's name.
            args: the arguments it would run with.

        Returns:
            A :class:`ToolDecision`. When ``allowed`` is False the caller must
            not execute the tool.
        """
        arguments = dict(args or {})
        try:
            result = self._enforcer.evaluate(
                PreAction(
                    session_id=self._session_id,
                    contract_id=self._contract_id,
                    tool=tool,
                    args=arguments,
                )
            )
        except Exception as exc:  # noqa: BLE001 — a contract bug must not crash the agent.
            return self._decision_on_error(tool, arguments, exc)

        if result.is_deny():
            self._count_deny()
            return ToolDecision(
                allowed=False,
                arguments=arguments,
                reason=result.reason or "the active behavioral contract forbids this action",
                violation=result.violation_name,
                **self._ids(tool),
            )

        if result.is_modify() and result.modified_args is not None:
            return ToolDecision(
                allowed=True,
                arguments=dict(result.modified_args),
                modified=True,
                reason=result.reason,
                violation=result.violation_name,
                **self._ids(tool),
            )

        return ToolDecision(
            allowed=True,
            arguments=arguments,
            reason=result.reason,
            violation=result.violation_name,
            redact_result=result.is_redact(),
            **self._ids(tool),
        )

    def after_tool(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
        *,
        text: str | None = None,
        force_redact: bool = False,
    ) -> ToolOutcome:
        """Score a tool result that has already been produced.

        Args:
            tool: the tool's name.
            args: the arguments it actually ran with — the rewritten ones when
                :meth:`before_tool` returned a modified decision, so the record
                describes the call that happened rather than the one requested.
            result: whatever the tool returned.
            text: the human-readable text of the result, when the caller can
                extract it more accurately than flattening can.
            force_redact: redact regardless of this result's own verdict. Pass
                ``decision.redact_result`` here, so a redaction the contract
                asked for before the tool ran still applies to its output.

        Returns:
            A :class:`ToolOutcome`. ``allowed is False`` means withhold the
            output; ``redacted_text`` is set when the text must be replaced.
        """
        arguments = dict(args or {})
        try:
            return self._score_result(tool, arguments, result, text, force_redact)
        except Exception:  # noqa: BLE001
            # Always fail open here, even under `fail_closed`. The tool has
            # already run, so withholding its output punishes the agent for the
            # bridge's own bug without preventing any side effect.
            return ToolOutcome(allowed=True, result=result, **self._ids(tool))

    def _score_result(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
        text: str | None,
        force_redact: bool = False,
    ) -> ToolOutcome:
        state: dict[str, Any] = {"tool.name": tool, **self._base_state}
        state.update(flatten_output(result))

        body = text if text is not None else _as_text(state.get("output.text"))
        if body:
            state.setdefault("output.text", body)

        decision = self._enforcer.evaluate(
            PostAction(
                session_id=self._session_id,
                contract_id=self._contract_id,
                tool=tool,
                args=arguments,
                state=state,
                result=result,
            )
        )

        if decision.is_deny():
            self._count_deny()
            return ToolOutcome(
                allowed=False,
                result=result,
                reason=decision.reason
                or "the active behavioral contract forbids returning this output",
                violation=decision.violation_name,
                **self._ids(tool),
            )

        redact = decision.is_redact() or force_redact
        if body:
            pii = evaluate_pii_filter(
                body, self._enforcer._compiled, self._enforcer._violations, is_streaming=False
            )
            if pii is not None and pii.is_deny():
                self._count_deny()
                return ToolOutcome(
                    allowed=False,
                    result=result,
                    reason=pii.reason,
                    violation=pii.violation_name,
                    **self._ids(tool),
                )
            redact = redact or (pii is not None and pii.is_redact())

        if redact and body:
            masked = apply_pii_redaction(body, self._enforcer._compiled.pii_compiled_patterns)
            return ToolOutcome(
                allowed=True, result=result, redacted_text=masked, **self._ids(tool)
            )

        return ToolOutcome(allowed=True, result=result, **self._ids(tool))

    # -- turn boundary ------------------------------------------------------

    def start_turn(self, user_input: str = "") -> None:
        """Record the start of a turn. Never raises."""
        self._safe_evaluate(
            TurnStart(
                session_id=self._session_id,
                contract_id=self._contract_id,
                user_input=user_input,
            )
        )

    def end_turn(self, assistant_output: str = "") -> None:
        """Record the end of a turn and queue any sampled judge evaluation."""
        self._safe_evaluate(
            TurnEnd(
                session_id=self._session_id,
                contract_id=self._contract_id,
                assistant_output=assistant_output,
            )
        )
        try:
            self._enforcer.schedule_judge_evaluation(assistant_output, self._session_id)
        except Exception:  # noqa: BLE001 — an optional judge must not break a turn.
            return

    def close(self) -> Any:
        """End the session and return the enforcer's ``SessionEnd``."""
        try:
            return self._enforcer.close()
        except Exception:  # noqa: BLE001 — teardown must not raise into a framework.
            return None

    # -- internals ----------------------------------------------------------

    def _safe_evaluate(self, event: Any) -> None:
        try:
            self._enforcer.evaluate(event)
        except Exception:  # noqa: BLE001
            return

    def _decision_on_error(
        self, tool: str, arguments: dict[str, Any], exc: Exception
    ) -> ToolDecision:
        if self._fail_closed:
            self._count_deny()
            return ToolDecision(
                allowed=False,
                arguments=arguments,
                reason=(
                    f"the contract could not be evaluated and this {self._surface} "
                    f"integration runs fail-closed ({type(exc).__name__})"
                ),
                violation="enforcement-error",
                evaluated=False,
                **self._ids(tool),
            )
        return ToolDecision(allowed=True, arguments=arguments, evaluated=False, **self._ids(tool))

    def _count_deny(self) -> None:
        with self._lock:
            self._denied += 1

    def _ids(self, tool: str) -> dict[str, str]:
        return {
            "tool": tool,
            "session_id": self._session_id,
            "contract_id": self._contract_id,
        }


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)
