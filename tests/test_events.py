# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for typed EventBus."""

import pytest

from agentassert_abc.monitor.events import (
    DriftWarningEvent,
    EventBus,
    EventKind,
    RecoveryEvent,
    SessionSummaryEvent,
    ViolationEvent,
)


class TestEventDataclasses:
    """Ensure event types are immutable and correctly defaulted."""

    def test_violation_event_defaults(self) -> None:
        ev = ViolationEvent()
        assert ev.kind == EventKind.VIOLATION

    def test_recovery_event_defaults(self) -> None:
        ev = RecoveryEvent()
        assert ev.kind == EventKind.RECOVERY

    def test_drift_warning_event_defaults(self) -> None:
        ev = DriftWarningEvent()
        assert ev.kind == EventKind.DRIFT_WARNING

    def test_session_summary_event_defaults(self) -> None:
        ev = SessionSummaryEvent()
        assert ev.kind == EventKind.SESSION_SUMMARY

    def test_events_are_frozen(self) -> None:
        ev = ViolationEvent(constraint_name="test")
        with pytest.raises(Exception):
            ev.constraint_name = "changed"  # type: ignore[misc]  # noqa: B017


class TestEventBus:
    """Thread-safe publish/subscribe."""

    def test_subscribe_and_emit(self) -> None:
        seen: list[ViolationEvent] = []
        bus = EventBus()

        @bus.on(EventKind.VIOLATION)
        def handler(ev: ViolationEvent) -> None:
            seen.append(ev)

        ev = ViolationEvent(constraint_name="no-secrets", turn=3)
        bus.emit(ev)
        assert len(seen) == 1
        assert seen[0].constraint_name == "no-secrets"
        assert seen[0].turn == 3

    def test_multiple_subscribers(self) -> None:
        hits: list[str] = []
        bus = EventBus()

        bus.subscribe(EventKind.RECOVERY, lambda e: hits.append("a"))
        bus.subscribe(EventKind.RECOVERY, lambda e: hits.append("b"))
        bus.emit(RecoveryEvent())
        assert hits == ["a", "b"]

    def test_unsubscribe(self) -> None:
        hits: list[str] = []
        bus = EventBus()

        def cb(e: RecoveryEvent) -> None:
            hits.append("x")

        bus.subscribe(EventKind.RECOVERY, cb)
        bus.emit(RecoveryEvent())
        assert hits == ["x"]

        bus.unsubscribe(EventKind.RECOVERY, cb)
        bus.emit(RecoveryEvent())
        assert hits == ["x"]  # no second call

    def test_unsubscribe_nonexistent_noop(self) -> None:
        bus = EventBus()
        bus.unsubscribe(EventKind.VIOLATION, lambda e: None)  # no error

    def test_emit_wrong_kind_not_delivered(self) -> None:
        hits: list[str] = []
        bus = EventBus()

        @bus.on(EventKind.DRIFT_WARNING)
        def handler(ev: DriftWarningEvent) -> None:
            hits.append("got")

        bus.emit(ViolationEvent())  # different kind
        assert hits == []

    def test_callback_exception_does_not_crash_bus(self) -> None:
        bus = EventBus()
        hits: list[str] = []

        bus.subscribe(EventKind.VIOLATION, lambda e: (_ for _ in ()).throw(ValueError("boom")))
        bus.subscribe(EventKind.VIOLATION, lambda e: hits.append("still called"))
        bus.emit(ViolationEvent())
        assert hits == ["still called"]

    def test_clear_removes_all(self) -> None:
        hits: list[str] = []
        bus = EventBus()

        @bus.on(EventKind.VIOLATION)
        def handler(ev: ViolationEvent) -> None:
            hits.append("got")

        bus.clear()
        bus.emit(ViolationEvent())
        assert hits == []

    def test_emit_all_kinds(self) -> None:
        bus = EventBus()
        received: list[str] = []

        bus.subscribe(EventKind.VIOLATION, lambda e: received.append("V"))
        bus.subscribe(EventKind.RECOVERY, lambda e: received.append("R"))
        bus.subscribe(EventKind.DRIFT_WARNING, lambda e: received.append("D"))
        bus.subscribe(EventKind.SESSION_SUMMARY, lambda e: received.append("S"))

        bus.emit(ViolationEvent())
        bus.emit(RecoveryEvent())
        bus.emit(DriftWarningEvent())
        bus.emit(SessionSummaryEvent())

        assert received == ["V", "R", "D", "S"]
