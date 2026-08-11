# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Tests for the Tier 0 exact Clopper–Pearson all-success floor.

The load-bearing property is EXACT finite-sample coverage: over repeated
Binomial(n, R) draws the lower bound sits at or below the true R at least
(1 − η) of the time, for every R — including the saturating boundary R → 1
where the bootstrap model floor degenerates (audit F5).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta

from agentassert_abc.certification.observed_floor import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    design_effect_adjusted_floor,
    observed_all_success_floor,
    observed_atleast_k_floor,
)
from agentassert_abc.exceptions import DependenceError

# --------------------------------------------------------------------------
# Closed-form anchors
# --------------------------------------------------------------------------


def test_cp_lower_boundaries():
    assert clopper_pearson_lower(0, 100) == 0.0                 # no successes
    # all successes: L = eta^(1/n), strictly below 1 (prices the boundary; F5)
    assert clopper_pearson_lower(200, 200, 0.05) == pytest.approx(0.05 ** (1 / 200))
    assert clopper_pearson_lower(200, 200, 0.05) < 1.0


def test_cp_upper_boundaries():
    assert clopper_pearson_upper(100, 100) == 1.0
    assert clopper_pearson_upper(0, 200, 0.05) == pytest.approx(1.0 - 0.05 ** (1 / 200))


def test_cp_lower_matches_beta_quantile():
    # exact Beta-quantile identity for an interior case
    assert clopper_pearson_lower(709, 1281, 0.05) == pytest.approx(
        float(beta.ppf(0.05, 709, 1281 - 709 + 1))
    )


def test_cp_lower_below_mle_and_upper_above():
    k, n = 709, 1281
    lo = clopper_pearson_lower(k, n)
    hi = clopper_pearson_upper(k, n)
    assert lo < k / n < hi


# --------------------------------------------------------------------------
# EXACT COVERAGE — the reason this is the certificate value
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r_true", [0.05, 0.3, 0.55, 0.9, 0.99])
def test_cp_lower_has_at_least_nominal_coverage(r_true):
    rng = np.random.default_rng(0)
    n, trials, eta = 400, 20000, 0.05
    k = rng.binomial(n, r_true, size=trials)
    lo = beta.ppf(eta, np.maximum(k, 1), n - k + 1)
    lo = np.where(k == 0, 0.0, lo)
    coverage = float(np.mean(lo <= r_true))
    assert coverage >= 1 - eta - 0.01, (r_true, coverage)  # exact => at/above nominal


def test_saturated_all_pass_is_priced_not_certified_as_one():
    # F5: an all-pass sample must NOT certify perfect reliability. CP prices the
    # boundary uncertainty; the floor is eta^(1/n) < 1.
    passes = np.ones((3, 250), dtype=int)
    r = observed_all_success_floor(passes, eta_conf=0.05)
    assert r.observed == 1.0
    assert r.floor == pytest.approx(0.05 ** (1 / 250))
    assert r.floor < 1.0


# --------------------------------------------------------------------------
# observed_all_success_floor wiring
# --------------------------------------------------------------------------


def test_observed_floor_counts_all_success_correctly():
    # all three stages pass on exactly the first 3 of 5 missions
    passes = np.array([
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ])
    r = observed_all_success_floor(passes)
    assert r.k == 3 and r.n == 5
    assert r.observed == pytest.approx(0.6)
    assert 0.0 <= r.floor <= r.observed <= r.upper <= 1.0


def test_observed_floor_dominates_independence_under_positive_dependence():
    # co-failure clustering => joint all-success >> product of marginals; the
    # observed floor captures it directly, no model needed.
    rng = np.random.default_rng(2)
    n = 3000
    common = rng.random(n) < 0.2                 # shared failure event
    passes = np.array([
        np.where(common, 0, 1),
        np.where(common, 0, (rng.random(n) < 0.95).astype(int)),
        np.where(common, 0, (rng.random(n) < 0.95).astype(int)),
    ])
    r = observed_all_success_floor(passes)
    indep = float(np.prod(passes.mean(axis=1)))
    assert r.floor > indep                       # dependence-aware, for free


def test_validation():
    with pytest.raises(DependenceError):
        clopper_pearson_lower(5, 3)              # k > n
    with pytest.raises(DependenceError):
        clopper_pearson_lower(1, 10, eta=0.0)    # eta out of range
    with pytest.raises(DependenceError):
        observed_all_success_floor(np.array([0, 1, 1]))   # 1-D
    with pytest.raises(DependenceError):
        observed_all_success_floor(np.array([[0, 2], [1, 0]]))  # non-binary


# --------------------------------------------------------------------------
# k-of-n quorum floor (audit: certify the functional the system actually uses)
# --------------------------------------------------------------------------


def test_atleast_k_equals_all_success_at_k_equals_m():
    passes = np.array([[1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    rk = observed_atleast_k_floor(passes, k=3)          # k = m => all-success
    rall = observed_all_success_floor(passes)
    assert rk.k == rall.k
    assert rk.floor == pytest.approx(rall.floor)


def test_atleast_k_lower_threshold_is_higher_reliability():
    # >=2 of 3 is easier than 3 of 3, so its floor is at least as high — the
    # 8.5-point slack the audit flagged when a quorum is certified with the AND floor.
    passes = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]])
    r2 = observed_atleast_k_floor(passes, k=2)
    r3 = observed_atleast_k_floor(passes, k=3)
    assert r2.observed >= r3.observed
    assert r2.floor >= r3.floor - 1e-9


def test_atleast_k_validation():
    passes = np.array([[1, 0, 1], [0, 1, 1]])
    with pytest.raises(DependenceError):
        observed_atleast_k_floor(passes, k=0)            # k < 1
    with pytest.raises(DependenceError):
        observed_atleast_k_floor(passes, k=3)            # k > m


# --------------------------------------------------------------------------
# Design-effect-adjusted floor (i.i.d.-missions due diligence)
# --------------------------------------------------------------------------


def test_design_effect_is_neutral_on_iid_series():
    # i.i.d. Bernoulli series => DEFF ≈ 1 => adjusted floor ≈ plain CP floor.
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.55).astype(int)
    r = design_effect_adjusted_floor(y, eta_conf=0.05, n_boot=400)
    plain = clopper_pearson_lower(int(y.sum()), y.size)
    assert r.n >= 0.9 * y.size                            # n_eff ≈ n
    assert r.floor == pytest.approx(plain, abs=0.01)


def test_design_effect_lowers_floor_on_autocorrelated_series():
    # a strongly serially-correlated series (long runs) => DEFF > 1 => n_eff < n
    # => strictly more conservative floor than the naive CP.
    rng = np.random.default_rng(1)
    blocks = (rng.random(200) < 0.55).astype(int)
    y = np.repeat(blocks, 10)                             # runs of length 10
    r = design_effect_adjusted_floor(y, eta_conf=0.05, n_boot=400)
    plain = clopper_pearson_lower(int(y.sum()), y.size)
    assert r.n < y.size                                  # effective n shrank
    assert r.floor < plain                               # more conservative
    assert "DEFF" in r.basis


def test_design_effect_validation():
    with pytest.raises(DependenceError):
        design_effect_adjusted_floor(np.array([[0, 1], [1, 0]]))   # 2-D
    with pytest.raises(DependenceError):
        design_effect_adjusted_floor(np.array([0, 2, 1]))          # non-binary
