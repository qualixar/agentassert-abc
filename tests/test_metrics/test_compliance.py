# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for ComplianceTracker — running C_hard and C_soft averages."""


class TestComplianceTracker:
    def test_initial_state(self) -> None:
        from agentassert_abc.metrics.compliance import ComplianceTracker

        tracker = ComplianceTracker()
        assert tracker.mean_c_hard == 1.0
        assert tracker.mean_c_soft == 1.0
        assert tracker.turn_count == 0

    def test_record_perfect_turn(self) -> None:
        from agentassert_abc.metrics.compliance import ComplianceTracker

        tracker = ComplianceTracker()
        tracker.record(c_hard=1.0, c_soft=1.0)
        assert tracker.mean_c_hard == 1.0
        assert tracker.mean_c_soft == 1.0
        assert tracker.turn_count == 1

    def test_record_multiple_turns(self) -> None:
        from agentassert_abc.metrics.compliance import ComplianceTracker

        tracker = ComplianceTracker()
        tracker.record(c_hard=1.0, c_soft=1.0)
        tracker.record(c_hard=0.5, c_soft=0.5)
        assert tracker.mean_c_hard == 0.75
        assert tracker.mean_c_soft == 0.75
        assert tracker.turn_count == 2

    def test_all_hard_violated(self) -> None:
        from agentassert_abc.metrics.compliance import ComplianceTracker

        tracker = ComplianceTracker()
        tracker.record(c_hard=0.0, c_soft=1.0)
        assert tracker.mean_c_hard == 0.0
