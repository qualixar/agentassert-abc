# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""DriftTracker — patent §5.1.

D(t) = w_c × D_compliance(t) + w_d × D_distributional(t)
D_compliance(t) = 1 - C(t)
D_distributional(t) = JSD(P_t || P_ref)

Patent reference: TECHNICAL-ATTACHMENT.md §5.1, §5.4.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.distance import jensenshannon

from agentassert_abc.models import DriftConfig


def compute_jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon Divergence between distributions p and q.

    Returns JSD in [0, ln(2)]. Uses scipy for numerical stability.
    Handles zero probabilities and zero-sum distributions gracefully.

    Fix C-04: Returns 0.0 for identical zero-sum distributions.
    """
    p_arr = np.array(p, dtype=np.float64)
    q_arr = np.array(q, dtype=np.float64)

    # C-04: Handle zero-sum distributions — both empty = no divergence
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum == 0.0 and q_sum == 0.0:
        return 0.0

    # Normalize to ensure valid distributions
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum

    # scipy's jensenshannon returns sqrt(JSD), we want JSD itself
    js_distance = jensenshannon(p_arr, q_arr, base=np.e)

    # Guard against NaN/Inf from edge cases (denormalized floats, etc.)
    if np.isnan(js_distance) or np.isinf(js_distance):
        return 0.0

    result = float(js_distance ** 2)  # Square to get divergence, not distance

    # Clamp to valid JSD range [0, ln(2)]
    return min(result, float(np.log(2)))


class DriftTracker:
    """Tracks behavioral drift using composite D(t) metric.

    D(t) = w_c × D_compliance(t) + w_d × D_distributional(t)

    Where:
    - D_compliance(t) = 1 - C(t) (constraint violation rate)
    - D_distributional(t) = JSD(P_window || P_ref) (distributional shift)

    Fix C-10: Uses windowed empirical distribution over recent turns
    instead of single-turn degenerate distributions.
    Fix M-09/M-17/L-07: Uses deque(maxlen=window) for bounded history.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self._config = config or DriftConfig()
        self._reference: dict[str, float] | None = None
        window = self._config.window
        self._history: deque[float] = deque(maxlen=window)
        # C-10: Windowed action distribution for empirical P_t
        self._action_window: deque[str] = deque(maxlen=window)

    def set_reference(self, distribution: dict[str, float]) -> None:
        """Set the reference action distribution (established during calibration)."""
        self._reference = dict(distribution)

    def compute_drift(
        self,
        c_total: float,
        action_dist: dict[str, float] | None = None,
    ) -> float:
        """Compute D(t) for a single turn.

        Args:
            c_total: Overall compliance score C(t).
            action_dist: Current action distribution (label → frequency).
                If a single action label is passed as {label: 1.0}, it is
                accumulated into the windowed empirical distribution.

        Returns:
            D(t) composite drift score.
        """
        w_c = self._config.weights.compliance
        w_d = self._config.weights.distributional

        # D_compliance = 1 - C(t)
        d_compliance = 1.0 - c_total

        # D_distributional = JSD(P_window || P_ref)
        d_distributional = 0.0
        if action_dist is not None and self._reference is not None:
            # C-10: Accumulate into windowed empirical distribution
            for label, freq in action_dist.items():
                for _ in range(max(1, int(freq))):
                    self._action_window.append(label)

            # Build empirical distribution from window
            if self._action_window:
                empirical = self._build_empirical_distribution()
                d_distributional = self._compute_jsd_from_dicts(
                    empirical, self._reference
                )

        d_t = w_c * d_compliance + w_d * d_distributional
        self._history.append(d_t)
        return d_t

    def is_warning(self, d: float) -> bool:
        """Check if drift exceeds warning threshold."""
        return d >= self._config.thresholds.warning

    def is_critical(self, d: float) -> bool:
        """Check if drift exceeds critical threshold."""
        return d >= self._config.thresholds.critical

    @property
    def mean_drift(self) -> float:
        """Mean drift D̄ across all recorded turns."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def history(self) -> list[float]:
        """Drift history (bounded by window size)."""
        return list(self._history)

    def _build_empirical_distribution(self) -> dict[str, float]:
        """Build empirical distribution from windowed action labels."""
        counts: dict[str, int] = {}
        for label in self._action_window:
            counts[label] = counts.get(label, 0) + 1
        total = len(self._action_window)
        return {k: v / total for k, v in counts.items()}

    def _compute_jsd_from_dicts(
        self,
        current: dict[str, float],
        reference: dict[str, float],
    ) -> float:
        """Compute JSD from label→frequency dicts, aligning keys."""
        all_keys = sorted(set(current) | set(reference))
        p = [current.get(k, 0.0) for k in all_keys]
        q = [reference.get(k, 0.0) for k in all_keys]
        return compute_jsd(p, q)
