# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Typed EventBus for contract monitoring events — Phase 4.

Thread-safe publish/subscribe with strongly-typed event dataclasses.
Subscribers register per-event-type callbacks. Events are immutable
and carry enough context for downstream consumers (OTEL exporter,
dashboard, alerting webhooks, etc.).

Patent reference: TECHNICAL-ATTACHMENT.md §5.2-5.4.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable  # noqa: TC003

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    VIOLATION = "violation"
    RECOVERY = "recovery"
    DRIFT_WARNING = "drift_warning"
    SESSION_SUMMARY = "session_summary"


# ---------------------------------------------------------------------------
# Event data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViolationEvent:
    """Emitted when a constraint violation is detected.

    Attributes:
        kind: "violation"
        constraint_name: Name from ContractSpec.
        constraint_type: "hard" or "soft".
        field: The field that was checked.
        expected: Human-readable expectation.
        actual: Human-readable actual value.
        turn: Turn number within the session.
        timestamp: Unix epoch float (monotonic for within-session).
    """

    # Ledger 3e: Literal kind narrows the type and prevents mis-routing events
    kind: Literal["violation"] = "violation"
    constraint_name: str = ""
    constraint_type: str = ""
    field: str = ""
    expected: str = ""
    actual: str = ""
    turn: int = 0
    timestamp: float = 0.0


@dataclass(frozen=True)
class RecoveryEvent:
    """Emitted when a recovery is attempted.

    Attributes:
        kind: "recovery"
        constraint_name: Name of the violated soft constraint.
        strategy: Recovery strategy name (inject_correction, etc.).
        succeeded: Whether the LLM re-prompt corrected the violation.
        turn: Turn number within the session.
        elapsed_ms: Recovery latency in milliseconds.
        timestamp: Unix epoch float.
    """

    # Ledger 3e: Literal kind narrows the type and prevents mis-routing events
    kind: Literal["recovery"] = "recovery"
    constraint_name: str = ""
    strategy: str = ""
    succeeded: bool = False
    turn: int = 0
    elapsed_ms: float = 0.0
    timestamp: float = 0.0


@dataclass(frozen=True)
class DriftWarningEvent:
    """Emitted when drift crosses a configured threshold.

    Attributes:
        kind: "drift_warning"
        level: "warning" or "critical".
        drift_score: The computed D(t) at this turn.
        threshold: The threshold that was crossed.
        turn: Turn number.
        timestamp: Unix epoch float.
    """

    # Ledger 3e: Literal kind narrows the type and prevents mis-routing events
    kind: Literal["drift_warning"] = "drift_warning"
    level: str = ""
    drift_score: float = 0.0
    threshold: float = 0.0
    turn: int = 0
    timestamp: float = 0.0


@dataclass(frozen=True)
class SessionSummaryEvent:
    """Emitted at session end with aggregate metrics.

    Attributes:
        kind: "session_summary"
        theta: Reliability Index Θ.
        c_bar: Mean compliance.
        d_bar: Mean drift.
        total_events: Violation event count.
        recovery_rate: Fraction of recoveries that succeeded.
        turn_count: Total turns in session.
        timestamp: Unix epoch float.
    """

    # Ledger 3e: Literal kind narrows the type and prevents mis-routing events
    kind: Literal["session_summary"] = "session_summary"
    theta: float = 0.0
    c_bar: float = 0.0
    d_bar: float = 0.0
    total_events: int = 0
    recovery_rate: float = 0.0
    turn_count: int = 0
    timestamp: float = 0.0


# Union type for any event
MonitorEvent = ViolationEvent | RecoveryEvent | DriftWarningEvent | SessionSummaryEvent


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Thread-safe typed event bus.

    Subscribers register per-EventKind callbacks. All callbacks receive
    the strongly-typed event dataclass. The bus is synchronous by default
    (callbacks execute on the publisher's thread).

    Usage:
        bus = EventBus()

        @bus.on(EventKind.VIOLATION)
        def log_violation(event: ViolationEvent) -> None:
            print(f"Violation: {event.constraint_name}")

        bus.emit(ViolationEvent(
            constraint_name="no-secrets",
            constraint_type="hard",
            field="output.secrets_detected",
            expected="False",
            actual="True",
            turn=3,
            timestamp=time.monotonic(),
        ))
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventKind, list[Callable[[Any], None]]] = {
            k: [] for k in EventKind
        }
        self._lock = threading.Lock()

    def on(self, event_kind: EventKind) -> Callable:
        """Decorator: register callback for event_kind.

        Returns the callable unchanged so it can still be called directly.
        """

        def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            self.subscribe(event_kind, fn)
            return fn

        return decorator

    def subscribe(self, event_kind: EventKind, callback: Callable[[Any], None]) -> None:
        """Register callback for event_kind. Thread-safe."""
        with self._lock:
            self._subscribers[event_kind].append(callback)

    def unsubscribe(self, event_kind: EventKind, callback: Callable[[Any], None]) -> None:
        """Remove a previously registered callback. Thread-safe."""
        with self._lock, contextlib.suppress(ValueError):
            self._subscribers[event_kind].remove(callback)

    def emit(self, event: MonitorEvent) -> None:
        """Publish event to all subscribers of its kind. Thread-safe.

        Callbacks execute synchronously. Exceptions in callbacks are
        caught and logged but do not propagate.
        """
        with self._lock:
            callbacks = list(self._subscribers.get(EventKind(event.kind), []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                # Ledger 5a: log instead of silently swallowing; one subscriber
                # crash must not break others, but it should be observable.
                _logger.exception(
                    "EventBus subscriber %r raised while handling %r",
                    cb,
                    type(event).__name__,
                )

    def clear(self) -> None:
        """Remove all subscribers. Thread-safe."""
        with self._lock:
            for k in self._subscribers:
                self._subscribers[k].clear()
