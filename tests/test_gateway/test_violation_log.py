# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_violation_log.py`."""

from __future__ import annotations

import threading

from agentassert_abc.gateway.violation_log import ViolationLog


class TestViolationLog:
    def test_record_hard(self) -> None:
        vl = ViolationLog()
        vl.record("tool_blocklist", "PreAction", "rm", "dangerous tool")
        assert len(vl.all_violations()) == 1
        v = vl.all_violations()[0]
        assert v["name"] == "tool_blocklist"
        assert v["kind"] == "hard"

    def test_record_soft(self) -> None:
        vl = ViolationLog()
        vl.record_soft("context_budget", "ContextWindow", "context", "too many tokens")
        assert len(vl.all_violations()) == 1
        v = vl.all_violations()[0]
        assert v["kind"] == "soft"

    def test_maxlen_enforced(self) -> None:
        vl = ViolationLog(maxlen=5)
        for i in range(10):
            vl.record(f"test_{i}", "PreAction", "tool", "reason")
        assert len(vl.all_violations()) == 5

    def test_thread_safe(self) -> None:
        vl = ViolationLog(maxlen=1000)
        errors: list[Exception] = []

        def record_many() -> None:
            try:
                for i in range(100):
                    vl.record(f"test_{i}", "PreAction", "tool", "reason")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(vl.all_violations()) == 1000

    def test_hard_count(self) -> None:
        """hard_count() — new helper added during the gateway port for DriftReport."""
        vl = ViolationLog()
        vl.record("tool_blocklist", "PreAction", "rm", "reason")
        vl.record_soft("pii_filter", "PostAction", "response", "reason")
        assert vl.hard_count() == 1
