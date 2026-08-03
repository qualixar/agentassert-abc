"""RED tests for the dependence-measurement estimators (LLD-B / LLD-E).

Every expected value is derived analytically so the tests pin the math, not the
implementation's own output:

* Binary Kendall tau_a = 2(p11*p00 - p10*p01)  (LLD-B Sec 6.9).
* Sample-size floor floor(n/2) >= (2/eps^2) ln(2/alpha) -> n = 5904 at
  eps = alpha = 0.05 (LLD-B Thm B.9).
* Tetrachoric: a symmetric 2x2 built from a latent bivariate normal with
  correlation rho and zero thresholds has P(both>0) = 1/4 + arcsin(rho)/(2*pi);
  at rho = 0.5 that is exactly 1/3, so the table (2000,1000,1000,2000) must
  invert to rho = 0.5.
* One-factor loadings are identified only with >= 3 indicators
  (lambda_i = sqrt(R_ij R_ik / R_jk)); with 2 indicators the model is
  underidentified and must raise (grok pre-lock fix, LLD-E).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agentassert_abc.dependence.estimators import (
    CoFailureTable,
    kendall_tau_a,
    one_factor_loadings,
    phi_coefficient,
    tau_a_min_samples,
    tetrachoric,
)
from agentassert_abc.exceptions import DependenceError


def test_cofailure_table_from_pairs_counts_and_marginals() -> None:
    a = [1, 1, 1, 0, 0]  # fails
    b = [1, 1, 0, 0, 0]
    t = CoFailureTable.from_pairs(a, b)
    assert (t.n11, t.n10, t.n01, t.n00) == (2, 1, 0, 2)
    assert t.n == 5
    assert t.p_a == pytest.approx(3 / 5)
    assert t.p_b == pytest.approx(2 / 5)
    assert t.p11 == pytest.approx(2 / 5)


def test_cofailure_table_is_immutable() -> None:
    t = CoFailureTable(n11=1, n10=2, n01=3, n00=4)
    with pytest.raises((AttributeError, TypeError)):
        t.n11 = 99  # type: ignore[misc]


def test_from_pairs_rejects_length_mismatch() -> None:
    with pytest.raises(DependenceError):
        CoFailureTable.from_pairs([1, 0], [1])


def test_tau_a_perfect_coupling_fair_coins_is_half() -> None:
    # a == b at rate 0.5 -> table (N/2, 0, 0, N/2); tau_a = 2*(0.25) = 0.5.
    t = CoFailureTable(n11=50, n10=0, n01=0, n00=50)
    assert kendall_tau_a(t) == pytest.approx(0.5)


def test_tau_a_independence_is_zero() -> None:
    # Independent, both rate 0.5 -> all cells equal -> tau_a = 0.
    t = CoFailureTable(n11=25, n10=25, n01=25, n00=25)
    assert kendall_tau_a(t) == pytest.approx(0.0)


def test_tau_a_matches_closed_form_on_asymmetric_table() -> None:
    t = CoFailureTable(n11=40, n10=10, n01=20, n00=30)  # N=100
    n = 100
    p11, p10, p01, p00 = 0.40, 0.10, 0.20, 0.30
    expected = 2 * (p11 * p00 - p10 * p01)
    assert kendall_tau_a(t) == pytest.approx(expected)
    assert n == t.n


def test_tau_a_min_samples_is_5904_at_005_005() -> None:
    assert tau_a_min_samples(eps=0.05, alpha=0.05) == 5904


def test_tau_a_min_samples_tighter_eps_needs_more() -> None:
    assert tau_a_min_samples(0.025, 0.05) > tau_a_min_samples(0.05, 0.05)


def test_tau_a_min_samples_rejects_bad_args() -> None:
    with pytest.raises(DependenceError):
        tau_a_min_samples(eps=0.0, alpha=0.05)
    with pytest.raises(DependenceError):
        tau_a_min_samples(eps=0.05, alpha=1.5)


def test_phi_perfect_coupling_is_one() -> None:
    t = CoFailureTable(n11=50, n10=0, n01=0, n00=50)
    assert phi_coefficient(t) == pytest.approx(1.0)


def test_phi_independence_is_zero() -> None:
    t = CoFailureTable(n11=25, n10=25, n01=25, n00=25)
    assert phi_coefficient(t) == pytest.approx(0.0)


def test_tetrachoric_recovers_rho_half() -> None:
    # rho=0.5, zero thresholds -> P(both fail)=1/3 -> (2000,1000,1000,2000).
    t = CoFailureTable(n11=2000, n10=1000, n01=1000, n00=2000)
    assert tetrachoric(t) == pytest.approx(0.5, abs=1e-3)


def test_tetrachoric_independence_is_zero() -> None:
    t = CoFailureTable(n11=1500, n10=1500, n01=1500, n00=1500)
    assert tetrachoric(t) == pytest.approx(0.0, abs=1e-3)


def test_one_factor_loadings_recovers_known_lambdas() -> None:
    lam = np.array([0.7, 0.6, 0.5, 0.8])
    R = np.outer(lam, lam)
    np.fill_diagonal(R, 1.0)  # R = lam lam' + Psi, Psi diagonal
    est = one_factor_loadings(R)
    # Loadings identified up to a global sign; align then compare.
    if np.dot(est, lam) < 0:
        est = -est
    assert est == pytest.approx(lam, abs=1e-6)


def test_one_factor_requires_three_indicators() -> None:
    R = np.array([[1.0, 0.42], [0.42, 1.0]])
    with pytest.raises(DependenceError):
        one_factor_loadings(R)
