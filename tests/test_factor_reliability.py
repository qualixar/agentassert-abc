# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Tests for the dependence-aware compositional bounds (LLD-B B.1/B.5/B.6/B.7)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from agentassert_abc.certification.factor_reliability import (
    factor_all_success,
    frechet_all_success_bounds,
    gaussian_copula_all_success,
    series_reliability_floor,
    shared_factor_all_success,
)
from agentassert_abc.exceptions import DependenceError


def _one_factor_data(marginals, loadings, n, seed):
    """Draw n paired outcomes from the shared-factor model (ground truth)."""
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


# --------------------------------------------------------------------------
# Thm B.1 — Fréchet sandwich
# --------------------------------------------------------------------------


def test_frechet_known_values():
    lo, hi = frechet_all_success_bounds([0.8, 0.9])
    assert lo == pytest.approx(0.7)  # 0.8 + 0.9 - 1
    assert hi == pytest.approx(0.8)  # min


def test_frechet_vacuous_when_sum_small():
    lo, hi = frechet_all_success_bounds([0.6, 0.6, 0.6])  # sum 1.8 < 2
    assert lo == 0.0
    assert hi == pytest.approx(0.6)


def test_frechet_single():
    lo, hi = frechet_all_success_bounds([0.42])
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


# --------------------------------------------------------------------------
# Independence special cases (λ = 0 / identity correlation)
# --------------------------------------------------------------------------


def test_shared_factor_zero_loadings_is_independence():
    p = [0.7, 0.6, 0.8]
    r = shared_factor_all_success(p, [0.0, 0.0, 0.0], q=64)
    assert r == pytest.approx(np.prod(p), abs=1e-6)


def test_gaussian_copula_identity_is_independence():
    p = [0.7, 0.6, 0.8]
    r = gaussian_copula_all_success(p, np.eye(3))
    assert r == pytest.approx(np.prod(p), abs=1e-4)


def test_bivariate_copula_extremes():
    # rho = 0 -> product; rho -> 1 -> min (comonotone)
    assert gaussian_copula_all_success([0.7, 0.6], np.eye(2)) == pytest.approx(0.42, abs=1e-4)
    hi = gaussian_copula_all_success([0.7, 0.6], np.array([[1.0, 0.999999], [0.999999, 1.0]]))
    assert hi == pytest.approx(0.6, abs=1e-3)


def test_psd_retract_never_inflates_correlations():
    # audit F8: the PSD projection must never raise a correlation (that would
    # raise the orthant and be anti-conservative). Retraction shrinks toward 0.
    from agentassert_abc.certification.factor_reliability import _psd_retract_corr

    r_indef = np.array([[1.0, -0.6, -0.6], [-0.6, 1.0, -0.6], [-0.6, -0.6, 1.0]])
    assert np.linalg.eigvalsh(r_indef).min() < 0.0            # genuinely indefinite
    rr = _psd_retract_corr(r_indef)
    assert np.linalg.eigvalsh(rr).min() >= -1e-9              # now PSD
    # every off-diagonal shrank toward 0 (never inflated in magnitude)
    assert np.all(np.abs(rr - np.eye(3)) <= np.abs(r_indef - np.eye(3)) + 1e-12)
    # a valid PD correlation is returned unchanged (no-op path)
    r_pd = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    assert np.allclose(_psd_retract_corr(r_pd), r_pd)


# --------------------------------------------------------------------------
# Thm B.6 vs the general orthant (they must agree under one-factor structure)
# --------------------------------------------------------------------------


def test_b6_matches_general_orthant_under_one_factor():
    p = [0.6, 0.55, 0.62]
    lam = np.array([0.7, 0.65, 0.72])
    corr = np.outer(lam, lam)
    np.fill_diagonal(corr, 1.0)
    b6, _err = factor_all_success(p, lam)
    gen = gaussian_copula_all_success(p, corr)
    assert b6 == pytest.approx(gen, abs=3e-3)


def test_b6_monotone_in_loading():
    p = [0.6, 0.6, 0.6]
    low, _ = factor_all_success(p, [0.2, 0.2, 0.2])
    high, _ = factor_all_success(p, [0.9, 0.9, 0.9])
    # stronger positive common factor => more co-success => higher all-success
    assert high > low > np.prod(p) - 1e-9


def test_factor_all_success_accurate_in_sharp_regime():
    # lambda -> 1 is where fixed-node GH over/undershoots (audit Q7a); the
    # adaptive-quad evaluator must match scipy.quad ground truth and stay inside
    # the Frechet sandwich.
    from scipy.integrate import quad as _quad

    p = np.array([0.6, 0.6, 0.6])
    lam = np.array([0.98, 0.98, 0.98])
    val, err = factor_all_success(p, lam)
    a = norm.ppf(p)
    d = np.sqrt(1 - lam**2)
    truth, _ = _quad(
        lambda xi: norm.pdf(xi) * np.prod(norm.cdf((a - lam * xi) / d)),
        -40, 40, limit=200,
    )
    assert val == pytest.approx(truth, abs=2e-3)
    assert err < 1e-3


def test_factor_all_success_never_exceeds_frechet_upper():
    # audit Q7a/F4: R <= min_j p_j for every joint; the evaluator must never
    # return a value above the Frechet upper bound, across the sharp regime.
    for p in ([0.4, 0.4, 0.4], [0.61, 0.61, 0.61], [0.3, 0.5, 0.7]):
        for lam_val in (0.9, 0.99, 0.999, 1.0 - 1e-7):
            val, _ = factor_all_success(p, [lam_val] * 3)
            _lo, hi = frechet_all_success_bounds(p)
            assert val <= hi + 1e-9, (p, lam_val, val, hi)


def test_nested_cofailure_does_not_force_rho_one():
    # audit Q8/C3: a nested co-failure pattern (one agent's failures subset of
    # another's) must NOT drive the fit to lambda~1 and a floor above observed.
    rng = np.random.default_rng(11)
    n = 3000
    # agent0 fails on a subset of agent1's failures (nested); agent2 independent-ish
    f1 = (rng.random(n) < 0.35).astype(int)
    f0 = ((f1 == 1) & (rng.random(n) < 0.4)).astype(int)  # subset of f1
    f2 = (rng.random(n) < 0.30).astype(int)
    passes = 1 - np.vstack([f0, f1, f2])
    r = series_reliability_floor(passes, eta_conf=0.05, n_boot=250, seed=0)
    observed = float(passes.prod(axis=0).mean())
    assert r.model_floor <= observed + 1e-9   # model floor must not exceed the truth
    assert 0.0 <= r.model_floor <= r.point <= 1.0


# --------------------------------------------------------------------------
# Thm B.7 — floor invariants + calibration
# --------------------------------------------------------------------------


def test_floor_invariants_hold():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=4000, seed=1)
    r = series_reliability_floor(passes, eta_conf=0.05, n_boot=300, seed=0)
    assert 0.0 <= r.model_floor <= r.point <= 1.0
    assert r.frechet_lower <= r.point + 1e-9
    observed = float(passes.prod(axis=0).mean())
    assert r.model_floor <= observed + 1e-9  # model floor must not exceed the truth
    assert r.method == "one-factor-quad"


def test_floor_recovers_independence_when_no_common_factor():
    # lambda = 0 => independent; all-success ~ product; floor a bit below it
    passes = _one_factor_data([0.8, 0.75], [0.0, 0.0], n=6000, seed=3)
    r = series_reliability_floor(passes, eta_conf=0.05, n_boot=300, seed=0)
    assert r.point == pytest.approx(r.independence_product, abs=0.03)


def test_floor_beats_independence_under_positive_dependence():
    # positive common factor => AND-reliability >> independence product
    passes = _one_factor_data([0.7, 0.7, 0.7], [0.85, 0.85, 0.85], n=5000, seed=5)
    r = series_reliability_floor(passes, eta_conf=0.05, n_boot=300, seed=0)
    assert r.point > r.independence_product + 0.1
    assert r.model_floor > r.independence_product  # tighter model floor


def test_floor_is_deterministic_for_fixed_seed():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=2000, seed=2)
    r1 = series_reliability_floor(passes, eta_conf=0.05, n_boot=200, seed=42)
    r2 = series_reliability_floor(passes, eta_conf=0.05, n_boot=200, seed=42)
    assert r1.model_floor == r2.model_floor
    assert r1.point == r2.point


def test_saturated_branch_raises_not_silently_overcertifies():
    # audit F5: a saturated stage (p̂ = 1) gives the mission bootstrap zero
    # variance, so the model floor cannot price the boundary and was
    # anti-conservative (coverage 0.56, floor -> 1.0). It must now fail loud and
    # redirect to the exact Tier-0 / finite-sample Tier-2 floors.
    rng = np.random.default_rng(9)
    passes = np.vstack([
        np.ones((1, 2000), dtype=int),
        (rng.random((2, 2000)) < 0.6).astype(int),
    ])
    with pytest.raises(DependenceError, match="saturated"):
        series_reliability_floor(passes, eta_conf=0.05, n_boot=200, seed=0)


# --------------------------------------------------------------------------
# Input validation (fail loud, never silently)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [0.5, 1.2],          # out of range
        [],                  # empty
    ],
)
def test_marginals_validation(bad):
    with pytest.raises(DependenceError):
        frechet_all_success_bounds(bad)


def test_pass_matrix_validation():
    with pytest.raises(DependenceError):
        series_reliability_floor(np.array([0, 1, 1]))  # 1-D not allowed
    with pytest.raises(DependenceError):
        series_reliability_floor(np.array([[0, 2], [1, 0]]))  # non-binary


def test_floor_param_validation():
    passes = _one_factor_data([0.7, 0.7], [0.5, 0.5], n=500, seed=1)
    with pytest.raises(DependenceError):
        series_reliability_floor(passes, eta_conf=0.0)
    with pytest.raises(DependenceError):
        series_reliability_floor(passes, n_boot=0)
    with pytest.raises(DependenceError):
        shared_factor_all_success([0.6, 0.6], [0.5, 0.5], q=1)
