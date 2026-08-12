# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Shared fixtures for the MCP guard suite."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.process.models import DecisionResult, TypeCDecision

CONTRACTS = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


def tool_call(req_id: Any, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """A well-formed ``tools/call`` request."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments if arguments is not None else {}},
    }


def tool_result(req_id: Any, text: str = "ok", **extra: Any) -> dict[str, Any]:
    """A well-formed ``tools/call`` result."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"content": [{"type": "text", "text": text}], **extra},
    }


class StubEnforcer:
    """Enforcer stand-in that returns scripted decisions and records events.

    Lets a test drive REDACT/WARN/DENY on either event type without needing a
    contract that happens to produce them, and without coupling the assertion to
    the compiler's internals.
    """

    def __init__(
        self,
        decisions: list[DecisionResult] | None = None,
        *,
        raises: Exception | None = None,
        name: str = "stub",
    ) -> None:
        self.events: list[Any] = []
        self._decisions = list(decisions or [])
        self._raises = raises
        self._contract = SimpleNamespace(name=name)
        # No pii_filter configured -> evaluate_pii_filter short-circuits to None.
        self._compiled = SimpleNamespace(pii_filter_config=None, pii_compiled_patterns=[])
        self._violations: list[Any] = []

    def evaluate(self, event: Any) -> DecisionResult:
        if self._raises is not None:
            raise self._raises
        self.events.append(event)
        if self._decisions:
            return self._decisions.pop(0)
        return DecisionResult(decision=TypeCDecision.ALLOW)


def allow() -> DecisionResult:
    return DecisionResult(decision=TypeCDecision.ALLOW)


def deny(reason: str = "nope", violation: str = "v") -> DecisionResult:
    return DecisionResult(decision=TypeCDecision.DENY, reason=reason, violation_name=violation)


def modify(args: dict[str, Any]) -> DecisionResult:
    return DecisionResult(decision=TypeCDecision.MODIFY, modified_args=args)


def redact() -> DecisionResult:
    return DecisionResult(decision=TypeCDecision.REDACT)


@pytest.fixture
def safety_enforcer() -> SessionEnforcer:
    """Real enforcer with a tool blocklist — for end-to-end DENY."""
    return SessionEnforcer.from_yaml(str(CONTRACTS / "safety-minimal.yaml"))


@pytest.fixture
def governance_enforcer() -> SessionEnforcer:
    """Real enforcer whose hard invariant reads ``output.pii_detected``."""
    return SessionEnforcer.from_yaml(str(CONTRACTS / "full-governance.yaml"))
