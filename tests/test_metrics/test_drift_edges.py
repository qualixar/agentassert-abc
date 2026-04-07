# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Edge case tests for DriftTracker — audit Finding 5."""


class TestDriftTrackerProperties:
    """Finding 5: mean_drift and history properties."""

    def test_mean_drift_empty(self) -> None:
        from agentassert_abc.metrics.drift import DriftTracker

        tracker = DriftTracker()
        assert tracker.mean_drift == 0.0

    def test_mean_drift_after_turns(self) -> None:
        from agentassert_abc.metrics.drift import DriftTracker

        tracker = DriftTracker()
        tracker.compute_drift(c_total=1.0, action_dist=None)  # D=0
        tracker.compute_drift(c_total=0.0, action_dist=None)  # D=0.6
        assert abs(tracker.mean_drift - 0.3) < 0.01

    def test_history_tracks_all_turns(self) -> None:
        from agentassert_abc.metrics.drift import DriftTracker

        tracker = DriftTracker()
        tracker.compute_drift(c_total=1.0, action_dist=None)
        tracker.compute_drift(c_total=0.5, action_dist=None)
        tracker.compute_drift(c_total=0.0, action_dist=None)
        assert len(tracker.history) == 3

    def test_history_is_copy(self) -> None:
        """history property returns a copy, not a reference."""
        from agentassert_abc.metrics.drift import DriftTracker

        tracker = DriftTracker()
        tracker.compute_drift(c_total=1.0, action_dist=None)
        h = tracker.history
        h.append(999.0)
        assert len(tracker.history) == 1  # Original unchanged
