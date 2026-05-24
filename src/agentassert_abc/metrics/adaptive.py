# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Adaptive Threshold Engine — learns optimal drift thresholds from calibration data.

Replaces static warning=0.3 / critical=0.6 with percentile-based thresholds
computed from the agent's observed drift distribution during a calibration
session. Thresholds can be frozen after calibration or updated continuously.

Patent reference: TECHNICAL-ATTACHMENT.md §5.1 (Drift Metric), Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentassert_abc.models import DriftThresholds


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive threshold learning.

    Attributes:
        warning_percentile: Percentile for warning threshold (default 90th).
        critical_percentile: Percentile for critical threshold (default 99th).
        min_samples: Minimum calibration observations before thresholds activate.
        frozen: If True, thresholds are locked after first calibration.
            If False, update continuously (sliding window by the tracker).
    """

    warning_percentile: float = 90.0
    critical_percentile: float = 99.0
    min_samples: int = 50
    frozen: bool = True


class AdaptiveThresholdEngine:
    """Learns drift warning/critical thresholds from observed drift distribution.

    Usage:
        engine = AdaptiveThresholdEngine()
        engine.update(drift_sequence)  # calibrate from calibration run
        thresholds = engine.current()   # get learned thresholds
        # Pass thresholds to DriftTracker config
    """

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self._config = config or AdaptiveConfig()
        self._observations: list[float] = []
        self._locked: bool = False
        self._current: DriftThresholds | None = None

    def update(self, drift_values: list[float]) -> None:
        """Ingest drift observations and recompute thresholds.

        If frozen and already locked, subsequent calls are no-ops.

        Args:
            drift_values: Drift scores from calibration or operational turns.
        """
        if self._locked and self._config.frozen:
            return

        self._observations.extend(drift_values)

        if len(self._observations) >= self._config.min_samples:
            arr = np.array(self._observations, dtype=float)
            warning = float(np.percentile(arr, self._config.warning_percentile))
            critical = float(np.percentile(arr, self._config.critical_percentile))
            # Floor at 0, ceiling at 1
            self._current = DriftThresholds(
                warning=max(0.0, min(warning, 1.0)),
                critical=max(0.0, min(critical, 1.0)),
            )
            if self._config.frozen:
                self._locked = True

    def current(self) -> DriftThresholds:
        """Return current thresholds, falling back to patent defaults."""
        if self._current is not None:
            return self._current
        return DriftThresholds()  # warning=0.3, critical=0.6

    def is_calibrated(self) -> bool:
        """Whether enough data has been observed for reliable thresholds."""
        return self._current is not None

    def reset(self) -> None:
        """Clear all observations and reset to uncalibrated state."""
        self._observations.clear()
        self._locked = False
        self._current = None
