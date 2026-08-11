# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Tests for the copula-agnostic LP all-success bounds (LLD-B Thm B.8).

The load-bearing invariants are theorems, so we test them directly:
  * the empirical / independent / comonotone joints are *feasible points*, so
    each is sandwiched by the LP bounds;
  * adding pairwise rows to the Fréchet program only shrinks the feasible set,
    so ``frechet_lo ≤ LP_lo ≤ LP_hi ≤ frechet_hi``;
  * pairwise independence does NOT collapse to the product (the whole point of
    the copula-agnostic floor) — XOR is the witness;
  * the returned minimiser is an actual distribution achieving ``lower`` (an
    independent optimality witness, not a re-run of the same solver).
"""

from __future__ import annotations

import numpy as np
import pytest

from agentassert_abc.certification.lp_bound import (
    cell_patterns,
    empirical_moments,
    pairwise_cp_box_floor,
    pairwise_lp_all_success_bounds,
)
from agentassert_abc.exceptions import DependenceError


def _pw(m: int, p: np.ndarray, offdiag: dict[tuple[int, int], float]) -> np.ndarray:
    """Assemble an m×m pairwise matrix (diagonal = p) from an off-diagonal dict."""
    pw = np.diag(np.asarray(p, float))
    for (i, j), v in offdiag.items():
        pw[i, j] = pw[j, i] = v
    return pw


# --------------------------------------------------------------------------
# Sharp bounds — closed-form anchors (independent of the implementation)
# --------------------------------------------------------------------------


def test_m2_pins_the_pairwise_cell():
    # For m = 2 the all-success cell IS the pairwise moment, so both LP bounds
    # collapse onto P_12 exactly.
    p = np.array([0.7, 0.6])
    res = pairwise_lp_all_success_bounds(p, _pw(2, p, {(0, 1): 0.5}))
    assert res.feasible
    assert res.lower == pytest.approx(0.5, abs=1e-9)
    assert res.upper == pytest.approx(0.5, abs=1e-9)


def test_comonotone_pairwise_gives_min_marginal():
    # P_ij = min(p_i, p_j) forces the nested/comonotone joint S_1 ⊆ S_2 ⊆ S_3,
    # whose all-success is exactly min_i p_i — a unique feasible point, so lower
    # == upper == min(p). Independent closed form.
    p = np.array([0.5, 0.6, 0.7])
    pw = _pw(3, p, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.6})
    res = pairwise_lp_all_success_bounds(p, pw)
    assert res.feasible
    assert res.lower == pytest.approx(0.5, abs=1e-8)
    assert res.upper == pytest.approx(0.5, abs=1e-8)


def test_pairwise_independence_is_not_mutual_independence():
    # XOR: three fair coins with S_3 = S_1 ⊕ S_2 are pairwise independent
    # (P_ij = 0.25 = 0.5·0.5) yet never all succeed. The sharp lower bound is 0,
    # strictly below the independence product 0.125 — this gap is precisely what
    # the copula-agnostic floor refuses to assume away.
    p = np.array([0.5, 0.5, 0.5])
    pw = _pw(3, p, {(0, 1): 0.25, (0, 2): 0.25, (1, 2): 0.25})
    res = pairwise_lp_all_success_bounds(p, pw)
    assert res.feasible
    assert res.lower == pytest.approx(0.0, abs=1e-8)
    prod = float(np.prod(p))
    assert res.lower <= prod <= res.upper + 1e-9  # product is merely feasible


def test_pairwise_lifts_above_a_vacuous_frechet_lower():
    # p = 0.6 each ⇒ Fréchet lower = max(0, 1.8 − 2) = 0 (says nothing). Knowing
    # the pairwise co-success is 0.5 forces a strictly positive all-success floor
    # WITHOUT any copula — the headline value of the LP bound.
    p = np.array([0.6, 0.6, 0.6])
    pw = _pw(3, p, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5})
    res = pairwise_lp_all_success_bounds(p, pw)
    assert res.feasible
    assert res.lower > 1e-6                      # informative, unlike Fréchet
    assert res.lower <= 0.6 + 1e-9               # ≤ min(p) = Fréchet upper


def test_bounds_sandwich_the_generating_joint():
    # Draw pairwise moments FROM a real random joint over the 8 cells (so they
    # are jointly feasible — drawing each P_ij inside its own 2-var Fréchet range
    # does NOT guarantee a compatible 3-way joint; the LP rightly rejects those).
    # Then both the Fréchet sandwich and — more sharply — the generating joint's
    # own all-success must lie inside [lower, upper].
    from agentassert_abc.certification.factor_reliability import (
        frechet_all_success_bounds,
    )

    pat = cell_patterns(3)
    rng = np.random.default_rng(3)
    for _ in range(25):
        x = rng.dirichlet(np.ones(8))            # a genuine 8-cell distribution
        p = x @ pat
        pw = np.diag(p)
        for i in range(3):
            for j in range(i + 1, 3):
                pw[i, j] = pw[j, i] = float(x @ (pat[:, i] * pat[:, j]))
        res = pairwise_lp_all_success_bounds(p, pw)
        fl_lo, fl_hi = frechet_all_success_bounds(p)
        assert res.feasible
        assert fl_lo - 1e-9 <= res.lower <= res.upper + 1e-9 <= fl_hi + 1e-9
        assert res.lower - 1e-9 <= x[-1] <= res.upper + 1e-9   # generating joint


def test_minimizer_is_a_feasible_witness_achieving_lower():
    # Independent optimality check: the returned minimiser must be a genuine
    # distribution (≥0, sums to 1) that reproduces every moment and whose
    # all-success cell equals `lower`. That proves `lower` is ATTAINABLE.
    p = np.array([0.55, 0.62, 0.7])
    pw = _pw(3, p, {(0, 1): 0.4, (0, 2): 0.45, (1, 2): 0.5})
    res = pairwise_lp_all_success_bounds(p, pw)
    assert res.feasible
    x = np.array(res.minimizer)
    pat = cell_patterns(3)
    assert np.all(x >= -1e-9)
    assert x.sum() == pytest.approx(1.0, abs=1e-8)
    for i in range(3):                                    # marginals reproduced
        assert float(x @ pat[:, i]) == pytest.approx(p[i], abs=1e-7)
    for i in range(3):                                    # pairwise reproduced
        for j in range(i + 1, 3):
            assert float(x @ (pat[:, i] * pat[:, j])) == pytest.approx(pw[i, j], abs=1e-7)
    assert x[-1] == pytest.approx(res.lower, abs=1e-7)    # all-ones cell = lower


def test_infeasible_moments_fall_back_to_frechet():
    # P_12 > min(p) is impossible for any joint ⇒ LP infeasible ⇒ Fréchet sandwich.
    p = np.array([0.5, 0.5])
    res = pairwise_lp_all_success_bounds(p, _pw(2, p, {(0, 1): 0.9}))
    assert not res.feasible
    assert res.minimizer is None
    assert (res.lower, res.upper) == (0.0, 0.5)  # Fréchet lo/hi for [.5,.5]


# --------------------------------------------------------------------------
# Series certified floor
# --------------------------------------------------------------------------


def _one_factor_data(marginals, loadings, n, seed):
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    a = norm.ppf(marginals)
    lam = np.asarray(loadings, float)
    d = np.sqrt(1 - lam**2)
    factor = rng.standard_normal(n)
    m = len(marginals)
    passes = np.empty((m, n), dtype=int)
    for j in range(m):
        u = lam[j] * factor + d[j] * rng.standard_normal(n)
        passes[j] = (u <= a[j]).astype(int)
    return passes


def _adversary_law(p0=0.6, lam0=0.8):
    # the pairwise-indistinguishable worst-case joint (audit F1): identical
    # marginals + pairwise co-success to a Gaussian one-factor model, but the
    # minimum 3-way all-success cell.
    import itertools

    from scipy.optimize import linprog
    from scipy.stats import multivariate_normal, norm

    a = norm.ppf(p0)
    pij = float(multivariate_normal.cdf([a, a], mean=[0, 0], cov=[[1, lam0**2], [lam0**2, 1]]))
    cells = list(itertools.product([0, 1], repeat=3))
    c = np.array([1.0 if all(b) else 0.0 for b in cells])
    a_eq = [np.ones(8)] + [[1.0 if b[j] else 0.0 for b in cells] for j in range(3)]
    a_eq += [[1.0 if (b[i] and b[j]) else 0.0 for b in cells]
             for i, j in itertools.combinations(range(3), 2)]
    b_eq = [1.0] + [p0] * 3 + [pij] * 3
    law = linprog(c, A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                  bounds=[(0, 1)] * 8, method="highs").x
    return np.clip(law, 0, None), float(c @ law), np.array(cells)


def test_cp_box_floor_is_a_valid_lower_bound():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=4000, seed=1)
    r = pairwise_cp_box_floor(passes, eta_conf=0.05)
    assert 0.0 <= r.floor <= r.upper <= 1.0
    assert r.floor <= r.observed + 1e-9          # the safety invariant
    assert r.feasible
    assert r.k_functionals == 3 + 3


def test_cp_box_floor_beats_vacuous_frechet_under_dependence():
    # strong shared factor ⇒ positive pairwise co-success ⇒ box floor well above
    # the (vacuous, = 0) Fréchet lower bound, with no copula assumption.
    passes = _one_factor_data([0.6, 0.6, 0.6], [0.85, 0.85, 0.85], n=5000, seed=5)
    r = pairwise_cp_box_floor(passes, eta_conf=0.05)
    assert r.floor > 0.05
    assert r.floor <= r.observed + 1e-9


def test_cp_box_floor_is_deterministic():
    # no bootstrap ⇒ a pure function of the data (unlike the old LP bootstrap).
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=2000, seed=2)
    r1 = pairwise_cp_box_floor(passes, eta_conf=0.05)
    r2 = pairwise_cp_box_floor(passes, eta_conf=0.05)
    assert r1.floor == r2.floor and r1.upper == r2.upper


def test_cp_box_floor_covers_true_r_under_adversary():
    # THE point of Tier 1: finite-sample coverage of the TRUE all-success under
    # the pairwise-indistinguishable worst case that collapses the Gaussian floor
    # to 0. The copula-agnostic CP-box floor must hold at >= nominal.
    law, true_r, cells = _adversary_law()
    rng = np.random.default_rng(2026)
    cover = 0
    trials = 60
    for _ in range(trials):
        idx = rng.choice(8, size=1000, p=law / law.sum())
        data = np.array([[cells[i][j] for i in idx] for j in range(3)], dtype=int)
        if pairwise_cp_box_floor(data, eta_conf=0.05).floor <= true_r + 1e-12:
            cover += 1
    assert cover / trials >= 0.95, (cover / trials, true_r)


def test_empirical_moments_match_direct_recompute():
    passes = _one_factor_data([0.8, 0.7], [0.5, 0.5], n=3000, seed=7)
    p, pw = empirical_moments(passes)
    assert p[0] == pytest.approx(passes[0].mean())
    assert pw[0, 1] == pytest.approx(np.mean(passes[0] * passes[1]))


# --------------------------------------------------------------------------
# Input validation (fail loud, never silently)
# --------------------------------------------------------------------------


def test_moment_validation():
    with pytest.raises(DependenceError):
        pairwise_lp_all_success_bounds([0.5, 1.2], np.eye(2))   # marginal OOR
    with pytest.raises(DependenceError):
        pairwise_lp_all_success_bounds([0.5, 0.5], np.eye(3))   # shape mismatch
    with pytest.raises(DependenceError):
        cell_patterns(0)


def test_floor_param_validation():
    passes = _one_factor_data([0.7, 0.7], [0.5, 0.5], n=500, seed=1)
    with pytest.raises(DependenceError):
        pairwise_cp_box_floor(passes, eta_conf=0.0)
    with pytest.raises(DependenceError):
        pairwise_cp_box_floor(np.array([0, 1, 1]))              # 1-D not allowed
