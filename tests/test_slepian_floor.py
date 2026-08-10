# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Tests for the Tier 2 Slepian monotone-corner model floor + Bonferroni CP box.

Tier 2 is the correct finite-sample Thm B.7 floor for the Gaussian-copula MODEL
functional. It must (a) sit below that functional under correct specification
(it is a valid LCB on it), (b) price the saturated boundary via CP instead of
returning 1.0 (audit F5), and (c) be honestly labelled as a model bound — it is
NOT a guarantee on true reliability (audit F1).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from agentassert_abc.certification.factor_reliability import factor_all_success
from agentassert_abc.certification.observed_floor import bonferroni_cp_cells
from agentassert_abc.certification.slepian_floor import (
    _rho_from_failure_cells,
    slepian_model_floor,
)
from agentassert_abc.exceptions import DependenceError


def _one_factor_data(marginals, loadings, n, seed):
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
# Bonferroni CP box
# --------------------------------------------------------------------------


def test_bonferroni_box_brackets_point_estimates():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=4000, seed=1)
    box = bonferroni_cp_cells(passes, eta=0.05)
    assert box.k_functionals == 3 + 3          # m + C(m,2)
    p = passes.mean(axis=1)
    for j in range(3):
        assert box.p_lo[j] <= p[j] <= box.p_hi[j]
    fails = 1 - passes
    for i in range(3):
        for j in range(i + 1, 3):
            s = float(np.mean(passes[i] * passes[j]))
            f = float(np.mean(fails[i] * fails[j]))
            assert box.cosuccess_lo[i][j] <= s <= box.cosuccess_hi[i][j]
            assert box.cofailure_lo[i][j] <= f <= box.cofailure_hi[i][j]


# --------------------------------------------------------------------------
# tetrachoric-from-cells inverse
# --------------------------------------------------------------------------


def test_rho_from_failure_cells_recovers_known_rho():
    rho = 0.5
    qa = qb = 0.3
    za, zb = norm.ppf(qa), norm.ppf(qb)
    f11 = float(multivariate_normal.cdf([za, zb], mean=[0, 0], cov=[[1, rho], [rho, 1]]))
    assert _rho_from_failure_cells(qa, qb, f11) == pytest.approx(rho, abs=2e-3)


def test_rho_from_failure_cells_frechet_clamps():
    # clamps to +/-_RCLIP (= +/-(1 - 1e-6)) at the Frechet-infeasible extremes
    assert _rho_from_failure_cells(0.3, 0.3, 0.0) == pytest.approx(-1.0, abs=1e-5)   # min
    assert _rho_from_failure_cells(0.3, 0.3, 0.3) == pytest.approx(1.0, abs=1e-5)    # comonotone


# --------------------------------------------------------------------------
# Slepian model floor
# --------------------------------------------------------------------------


def test_slepian_floor_below_model_functional_and_observed():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=8000, seed=2)
    r = slepian_model_floor(passes, eta_conf=0.05)
    fit_p = passes.mean(axis=1)
    # the functional it bounds, at the empirical params (approx the true one)
    r_star, _ = factor_all_success(fit_p, [0.6, 0.55, 0.62])
    assert 0.0 < r.floor <= r_star + 0.02          # valid LCB on the model functional
    assert r.floor <= r.observed + 1e-9
    assert r.is_model_bound is True


def test_slepian_floor_prices_saturated_branch_not_one():
    # audit F5: an all-pass stage must NOT push the floor to 1.0 — CP gives
    # p_lo < 1 so the orthant stays strictly below 1.
    rng = np.random.default_rng(9)
    passes = np.vstack([
        np.ones((1, 2000), dtype=int),
        (rng.random((2, 2000)) < 0.6).astype(int),
    ])
    r = slepian_model_floor(passes, eta_conf=0.05)
    assert r.floor < 1.0
    assert 0.0 <= r.floor <= r.observed + 1e-9


def test_slepian_floor_is_labelled_a_model_bound():
    passes = _one_factor_data([0.7, 0.7, 0.7], [0.5, 0.5, 0.5], n=2000, seed=3)
    r = slepian_model_floor(passes)
    assert r.is_model_bound is True
    joined = " ".join(r.assumptions).lower()
    assert "gaussian copula" in joined
    assert "not true reliability" in joined


def test_dominated_psd_never_inflates_a_correlation():
    # Muse+Grok audit F-Slepian(3): the PSD projection for the lower corner must
    # never RAISE any off-diagonal (scaling a negative ρ toward 0 would inflate
    # the orthant and break the lower bound). It must return a matrix elementwise
    # <= the input off-diagonal (and PSD), or signal degeneracy.
    from agentassert_abc.certification.slepian_floor import _dominated_psd

    corr = np.array([[1.0, 0.9, -0.8], [0.9, 1.0, 0.9], [-0.8, 0.9, 1.0]])
    assert np.linalg.eigvalsh(corr).min() < 0.0            # genuinely indefinite
    try:
        out = _dominated_psd(corr)
    except DependenceError:
        return                                              # degenerate: acceptable
    off = ~np.eye(3, dtype=bool)
    assert np.all(out[off] <= corr[off] + 1e-12)            # never inflated
    assert np.linalg.eigvalsh(out).min() >= -1e-9           # PSD


def test_slepian_floor_validation():
    passes = _one_factor_data([0.7, 0.7], [0.5, 0.5], n=500, seed=1)
    with pytest.raises(DependenceError):
        slepian_model_floor(passes, eta_conf=0.0)
    with pytest.raises(DependenceError):
        slepian_model_floor(np.array([0, 1, 1]))          # 1-D
