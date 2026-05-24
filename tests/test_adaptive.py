# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for Adaptive Threshold Engine."""

import numpy as np

from agentassert_abc.metrics.adaptive import AdaptiveConfig, AdaptiveThresholdEngine


class TestAdaptiveThresholdEngine:
    """Adaptive threshold learning tests."""

    def test_fallback_defaults_uncalibrated(self) -> None:
        engine = AdaptiveThresholdEngine()
        t = engine.current()
        assert t.warning == 0.3
        assert t.critical == 0.6

    def test_is_calibrated_false_initially(self) -> None:
        engine = AdaptiveThresholdEngine()
        assert engine.is_calibrated() is False

    def test_calibrates_after_min_samples(self) -> None:
        rng = np.random.default_rng(42)
        drift = rng.normal(0.15, 0.05, 60).tolist()
        engine = AdaptiveThresholdEngine(
            AdaptiveConfig(warning_percentile=90, critical_percentile=99, min_samples=50),
        )
        engine.update(drift)
        assert engine.is_calibrated() is True
        t = engine.current()
        assert t.warning > 0.0
        assert t.critical > t.warning

    def test_not_calibrated_below_min_samples(self) -> None:
        drift = [0.1, 0.2, 0.15]
        engine = AdaptiveThresholdEngine(
            AdaptiveConfig(min_samples=10),
        )
        engine.update(drift)
        assert engine.is_calibrated() is False
        assert engine.current().warning == 0.3  # fallback

    def test_frozen_locks_after_calibration(self) -> None:
        rng = np.random.default_rng(42)
        drift1 = rng.normal(0.1, 0.02, 60).tolist()
        engine = AdaptiveThresholdEngine(AdaptiveConfig(min_samples=50, frozen=True))
        engine.update(drift1)
        first = engine.current()
        # Second update with different distribution — should be ignored
        drift2 = rng.normal(0.5, 0.1, 60).tolist()
        engine.update(drift2)
        second = engine.current()
        assert first.warning == second.warning
        assert first.critical == second.critical

    def test_not_frozen_updates_continuously(self) -> None:
        rng = np.random.default_rng(42)
        drift1 = rng.normal(0.1, 0.02, 60).tolist()
        engine = AdaptiveThresholdEngine(AdaptiveConfig(min_samples=50, frozen=False))
        engine.update(drift1)
        first_warning = engine.current().warning
        drift2 = rng.normal(0.6, 0.1, 60).tolist()
        engine.update(drift2)
        second_warning = engine.current().warning
        assert second_warning > first_warning  # higher drift → higher thresholds

    def test_reset_clears_state(self) -> None:
        rng = np.random.default_rng(42)
        drift = rng.normal(0.15, 0.05, 60).tolist()
        engine = AdaptiveThresholdEngine(AdaptiveConfig(min_samples=50))
        engine.update(drift)
        assert engine.is_calibrated() is True
        engine.reset()
        assert engine.is_calibrated() is False
        assert engine.current().warning == 0.3

    def test_thresholds_clamped_to_range(self) -> None:
        drift = [0.0] * 50 + [2.0] * 10
        engine = AdaptiveThresholdEngine(
            AdaptiveConfig(warning_percentile=50, critical_percentile=95, min_samples=50),
        )
        engine.update(drift)
        t = engine.current()
        assert 0.0 <= t.warning <= 1.0
        assert 0.0 <= t.critical <= 1.0

    def test_config_defaults(self) -> None:
        cfg = AdaptiveConfig()
        assert cfg.warning_percentile == 90.0
        assert cfg.critical_percentile == 99.0
        assert cfg.min_samples == 50
        assert cfg.frozen is True
