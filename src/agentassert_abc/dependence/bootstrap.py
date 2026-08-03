# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Scenario-cluster bootstrap for dependence statistics (LLD-B Sec 6.11, LLD-E Sec 8).

Missions that share a scenario are dependent, so a naive per-mission bootstrap
destroys the very correlation being estimated. This resamples whole *clusters*
(scenarios) with replacement, keeping every mission inside a selected cluster
together, then recomputes the statistic — the authoritative interval per LLD-B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from agentassert_abc.dependence.estimators import CoFailureTable
from agentassert_abc.exceptions import DependenceError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["BootstrapCI", "cluster_bootstrap"]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Immutable percentile confidence interval from a cluster bootstrap."""

    point: float
    lower: float
    upper: float
    alpha: float
    n_boot: int
    n_clusters: int
    n_valid: int

    @property
    def width(self) -> float:
        """Interval width upper - lower."""
        return self.upper - self.lower

    def excludes(self, value: float) -> bool:
        """True if ``value`` lies outside the closed interval."""
        return value < self.lower or value > self.upper


def cluster_bootstrap(
    fail_a: Sequence[int | bool],
    fail_b: Sequence[int | bool],
    cluster_ids: Sequence[object],
    statistic: Callable[[CoFailureTable], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapCI:
    """Percentile CI for a co-failure ``statistic`` via a scenario-cluster bootstrap.

    Args:
        fail_a, fail_b: paired binary failure outcomes (1 = failed).
        cluster_ids: scenario/cluster label per mission (same length).
        statistic: maps a :class:`CoFailureTable` to a scalar (e.g. ``kendall_tau_a``).
        n_boot: number of bootstrap resamples.
        alpha: two-sided level; the interval is [alpha/2, 1-alpha/2] quantiles.
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`BootstrapCI`. Degenerate resamples (a statistic that raises
        :class:`DependenceError`) are dropped and counted via ``n_valid``.
    """
    n = len(fail_a)
    if len(fail_b) != n or len(cluster_ids) != n:
        raise DependenceError("fail_a, fail_b, and cluster_ids must be equal length")
    if n == 0:
        raise DependenceError("no data")
    if not (0.0 < alpha < 1.0):
        raise DependenceError("alpha must be in (0, 1)")
    if n_boot < 1:
        raise DependenceError("n_boot must be positive")

    a = list(fail_a)
    b = list(fail_b)
    groups: dict[object, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        groups.setdefault(cid, []).append(idx)
    keys = list(groups)
    k = len(keys)

    point = statistic(CoFailureTable.from_pairs(a, b))

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        idxs: list[int] = []
        for c in chosen:
            idxs.extend(groups[keys[int(c)]])
        try:
            table = CoFailureTable.from_pairs(
                [a[j] for j in idxs], [b[j] for j in idxs]
            )
            draws.append(statistic(table))
        except DependenceError:
            continue  # degenerate resample; drop it

    if not draws:
        raise DependenceError("all bootstrap resamples were degenerate")
    arr = np.asarray(draws, dtype=float)
    lower = float(np.quantile(arr, alpha / 2.0))
    upper = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point=float(point),
        lower=lower,
        upper=upper,
        alpha=alpha,
        n_boot=n_boot,
        n_clusters=k,
        n_valid=len(draws),
    )
