"""Tests for the scenario-cluster bootstrap (LLD-B Sec 6.11, LLD-E Sec 8)."""
from __future__ import annotations

import numpy as np
import pytest

from agentassert_abc.dependence.bootstrap import cluster_bootstrap
from agentassert_abc.dependence.estimators import kendall_tau_a
from agentassert_abc.exceptions import DependenceError


def _clustered(rng, k_clusters: int, per: int, couple: bool) -> tuple[list, list, list]:
    a: list[int] = []
    b: list[int] = []
    cid: list[int] = []
    for c in range(k_clusters):
        for _ in range(per):
            av = int(rng.integers(0, 2))
            bv = av if couple else int(rng.integers(0, 2))
            a.append(av)
            b.append(bv)
            cid.append(c)
    return a, b, cid


def test_independent_data_ci_includes_zero() -> None:
    rng = np.random.default_rng(0)
    a, b, cid = _clustered(rng, k_clusters=40, per=10, couple=False)
    ci = cluster_bootstrap(a, b, cid, kendall_tau_a, n_boot=600, seed=1)
    assert ci.lower <= 0.0 <= ci.upper
    assert ci.width > 0.0
    assert ci.n_clusters == 40


def test_perfect_coupling_ci_excludes_zero() -> None:
    rng = np.random.default_rng(0)
    a, b, cid = _clustered(rng, k_clusters=40, per=10, couple=True)
    ci = cluster_bootstrap(a, b, cid, kendall_tau_a, n_boot=600, seed=1)
    # a == b -> tau_a = 2 p(1-p) > 0; interval must sit strictly above 0.
    assert ci.lower > 0.0
    assert ci.excludes(0.0)


def test_bootstrap_is_deterministic_with_seed() -> None:
    rng = np.random.default_rng(2)
    a, b, cid = _clustered(rng, k_clusters=20, per=8, couple=True)
    ci1 = cluster_bootstrap(a, b, cid, kendall_tau_a, n_boot=300, seed=7)
    ci2 = cluster_bootstrap(a, b, cid, kendall_tau_a, n_boot=300, seed=7)
    assert (ci1.lower, ci1.upper, ci1.point) == (ci2.lower, ci2.upper, ci2.point)


def test_length_mismatch_raises() -> None:
    with pytest.raises(DependenceError):
        cluster_bootstrap([1, 0], [1, 0], [0], kendall_tau_a)
