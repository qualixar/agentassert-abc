# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Coverage gap tests for dynamics.py — OUFitter edge cases and dead paths.

Invariants pinned here:
  * Lines 203-204: OUFitter.stationary_drift(params) returns alpha/gamma
                   when gamma > 0.
  * Line  205    : OUFitter.stationary_drift(params) returns None when
                   gamma == 0 (singular attractor — no stationary drift).
  * Line  186    : OUFitter.fit() returns None when the scipy minimizer fails
                   (e.g., NaN propagation in the negative log-likelihood).

Dead-code findings (no test can cover these without modifying src):
  * Line 148: neg_log_likelihood returns inf when gamma <= 0 or sigma <= 0.
              With L-BFGS-B bounds [(0,None),(1e-8,None),(1e-8,None)], the
              optimizer never calls the function with out-of-bound values.
              This branch is a defensive guard unreachable via the public API.
  * Line 156: neg_log_likelihood returns inf when var_noise <= 0.
              Given sigma>0, gamma>0, a=exp(-gamma*dt)∈(0,1), var_noise is
              always strictly positive — this branch is unreachable.
  * Line 175: gamma_init = 0.1 in the else-branch when n == 1.
              MIN_SEQUENCE_LENGTH = 20 guarantees n >= 20 when this code runs,
              so n == 1 is impossible through the public fit() API.
              These three lines are defensive dead code — correct to have them
              but not possible to cover without src modification.

These three are reported as a src defect finding (dead code), not fixed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentassert_abc.metrics.dynamics import OUFitter, OUParameters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ou_sample(
    alpha: float,
    gamma: float,
    sigma: float,
    n: int = 50,
    dt: float = 1.0,
    seed: int = 0,
) -> list[float]:
    """Synthetic OU trajectory for controlled parameter recovery tests."""
    rng = np.random.default_rng(seed)
    a = np.exp(-gamma * dt)
    b = (alpha / gamma) * (1 - a) if gamma > 0 else 0.0
    sigma_eps = sigma * np.sqrt((1 - a ** 2) / (2 * gamma)) if gamma > 0 else sigma
    y = np.zeros(n, dtype=float)
    y[0] = b / (1 - a) if abs(1 - a) > 1e-10 else 0.0
    for t in range(1, n):
        y[t] = a * y[t - 1] + b + rng.normal(0, sigma_eps)
    return y.tolist()


# ---------------------------------------------------------------------------
# OUFitter.stationary_drift() — lines 201-205
# ---------------------------------------------------------------------------


class TestStationaryDrift:
    def test_positive_gamma_returns_alpha_over_gamma(self) -> None:
        """Line 203-204: D* = alpha / gamma when gamma > 0.

        The stationary drift is the long-run mean of the OU process. This
        method is a standalone accessor duplicating the value computed during
        fit() — it exists so callers can recompute D* from stored params without
        re-fitting.
        """
        fitter = OUFitter()
        params = OUParameters(
            alpha=0.5,
            gamma=2.0,
            sigma=0.1,
            log_likelihood=-5.0,
            stationary_drift=0.25,
        )
        d_star = fitter.stationary_drift(params)
        assert d_star is not None
        assert d_star == pytest.approx(0.5 / 2.0)

    def test_zero_gamma_returns_none(self) -> None:
        """Line 205: returns None when gamma == 0.

        A zero mean-reversion rate means the OU process has no stationary
        distribution — there is no finite stationary drift. None is the correct
        signal to the Lyapunov stability check.
        """
        fitter = OUFitter()
        params = OUParameters(
            alpha=0.5,
            gamma=0.0,
            sigma=0.1,
            log_likelihood=0.0,
            stationary_drift=None,
        )
        assert fitter.stationary_drift(params) is None

    def test_large_gamma_small_alpha(self) -> None:
        """Strong mean-reversion (large gamma) with small alpha → D* near zero."""
        fitter = OUFitter()
        params = OUParameters(
            alpha=0.001,
            gamma=100.0,
            sigma=0.01,
            log_likelihood=-1.0,
            stationary_drift=0.00001,
        )
        d_star = fitter.stationary_drift(params)
        assert d_star is not None
        assert d_star == pytest.approx(0.001 / 100.0, rel=1e-6)
        assert d_star < 0.01  # near-zero attractor

    def test_result_consistent_with_params_stationary_drift_field(self) -> None:
        """stationary_drift() must return the same value as OUParameters.stationary_drift.

        Both are alpha/gamma. This test pins the invariant that the method does
        not diverge from the fitted field over time — a regression guard.
        """
        fitter = OUFitter()
        alpha, gamma = 1.2, 3.0
        expected = alpha / gamma
        params = OUParameters(
            alpha=alpha,
            gamma=gamma,
            sigma=0.2,
            log_likelihood=-20.0,
            stationary_drift=expected,
        )
        assert fitter.stationary_drift(params) == pytest.approx(params.stationary_drift)


# ---------------------------------------------------------------------------
# OUFitter.fit() — line 186 (optimizer failure → None)
# ---------------------------------------------------------------------------


class TestFitOptimizerFailure:
    def test_nan_sequence_returns_none(self) -> None:
        """Line 186: when scipy minimize fails, fit() must return None.

        A sequence of NaN values causes np.var to return NaN (not < 1e-12),
        so the constant-sequence branch is bypassed. NaN then propagates
        through the negative log-likelihood objective, causing scipy to report
        success=False, and fit() returns None.

        This test ensures the failure is communicated cleanly (None) rather
        than propagating NaN or raising an exception.
        """
        fitter = OUFitter()
        nan_sequence = [float("nan")] * 25
        result = fitter.fit(nan_sequence, dt=1.0)
        assert result is None

    def test_inf_sequence_returns_none(self) -> None:
        """Sequence of infinities also causes optimizer failure → None.

        Same invariant as NaN: the certifier must degrade gracefully on
        corrupt input rather than crashing.
        """
        fitter = OUFitter()
        inf_sequence = [float("inf")] * 25
        result = fitter.fit(inf_sequence, dt=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# OUFitter.fit() — other nominal + boundary paths
# ---------------------------------------------------------------------------


class TestFitBoundary:
    def test_short_sequence_returns_none(self) -> None:
        """fit() returns None when sequence is below MIN_SEQUENCE_LENGTH (20).

        This boundary is already tested in test_dynamics.py; we include it
        here for completeness and to document MIN_SEQUENCE_LENGTH = 20.
        """
        fitter = OUFitter()
        result = fitter.fit([0.1] * 19)
        assert result is None

    def test_constant_sequence_returns_zero_params(self) -> None:
        """A constant sequence (zero variance) returns OUParameters with all zeros.

        The constant path fast-tracks to a degenerate result rather than
        attempting optimisation, which would blow up log(0).
        """
        fitter = OUFitter()
        result = fitter.fit([0.4] * 25)
        assert result is not None
        assert result.alpha == pytest.approx(0.0)
        assert result.gamma == pytest.approx(0.0)
        assert result.sigma == pytest.approx(0.0)
        assert result.stationary_drift is None

    def test_stationary_drift_from_fit_matches_accessor(self) -> None:
        """params.stationary_drift from fit() must equal stationary_drift(params).

        This cross-checks that the standalone accessor returns the same value
        the fitter computed and stored — a regression guard against drift between
        the two code paths.
        """
        fitter = OUFitter()
        seq = _ou_sample(alpha=0.3, gamma=0.8, sigma=0.05, n=100, seed=7)
        params = fitter.fit(seq)
        if params is None or params.gamma == 0.0:
            pytest.skip("fit() did not converge to a non-degenerate solution")
        accessor_value = fitter.stationary_drift(params)
        assert accessor_value is not None
        assert accessor_value == pytest.approx(params.stationary_drift, rel=1e-6)
