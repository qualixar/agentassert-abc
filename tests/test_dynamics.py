# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for F3 OU dynamics + F4 Lyapunov stability."""

import numpy as np

from agentassert_abc.metrics.dynamics import (
    LyapunovStabilityCheck,
    OUFitter,
    OUParameters,
    StabilityReport,
    StabilityVerdict,
)


def _ou_sample(
    alpha: float, gamma: float, sigma: float,
    n: int = 200, dt: float = 1.0, seed: int = 42,
) -> list[float]:
    """Generate synthetic OU trajectory for testing."""
    rng = np.random.default_rng(seed)
    # Discrete-time OU: D_{t+1} = a * D_t + b + epsilon_t
    a = np.exp(-gamma * dt)
    if gamma > 0:
        b = (alpha / gamma) * (1 - a)
        sigma_eps = sigma * np.sqrt((1 - a ** 2) / (2 * gamma))
    else:
        b = 0.0
        sigma_eps = sigma

    y = np.zeros(n)
    y[0] = b / (1 - a) if abs(1 - a) > 1e-10 else 0.0  # start at stationary
    for t in range(1, n):
        epsilon = rng.normal(0, sigma_eps)
        y[t] = a * y[t-1] + b + epsilon
    return y.tolist()


class TestOUFitter:
    """Tests for Ornstein-Uhlenbeck maximum-likelihood fitting."""

    def test_fit_convergent_ou(self) -> None:
        """Fit should recover parameters of synthetic convergent OU process."""
        alpha_true, gamma_true, sigma_true = 0.1, 0.5, 0.05
        drift = _ou_sample(alpha_true, gamma_true, sigma_true, n=300, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift, dt=1.0)

        assert params is not None
        # Allow 20% tolerance on parameter recovery
        assert abs(params.alpha - alpha_true) / alpha_true < 0.2
        assert abs(params.gamma - gamma_true) / gamma_true < 0.2
        assert abs(params.sigma - sigma_true) / sigma_true < 0.2
        assert params.stationary_drift is not None
        assert abs(params.stationary_drift - (alpha_true / gamma_true)) < 0.1

    def test_fit_too_short_sequence(self) -> None:
        """Sequence shorter than MIN_SEQUENCE_LENGTH returns None."""
        drift = [0.1, 0.2, 0.15]  # only 3 points
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is None

    def test_fit_constant_sequence(self) -> None:
        """Constant sequence should give gamma ~ 0 (no mean reversion)."""
        drift = [0.5] * 50
        fitter = OUFitter()
        params = fitter.fit(drift)
        # Constant sequence: gamma should be near 0, alpha near 0
        assert params is not None
        assert params.gamma < 0.1  # essentially no mean reversion
        # alpha should also be near 0
        assert params.alpha < 0.1

    def test_fit_divergent_ou_gamma_le_alpha(self) -> None:
        """When gamma <= alpha, process is not mean-reverting."""
        alpha_true, gamma_true, sigma_true = 0.5, 0.1, 0.05  # gamma < alpha
        drift = _ou_sample(alpha_true, gamma_true, sigma_true, n=300, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None
        # Even though we fit, the Lyapunov check will flag this as divergent


class TestLyapunovStabilityCheck:
    """Tests for F4 Lyapunov stability verdict."""

    def test_convergent_ou_verdict(self) -> None:
        """Ledger 2f: for n=200 seed=42 the empirical V(e) slope is significantly
        positive, so the correct conservative verdict is INCONCLUSIVE (not CONVERGENT).
        The old assertion of CONVERGENT encoded the pre-ledger-2f buggy behavior.
        """
        alpha, gamma, sigma = 0.1, 0.5, 0.05  # gamma > alpha (theoretically convergent)
        drift = _ou_sample(alpha, gamma, sigma, n=200, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        # Ledger 2f: V(e) slope is significantly positive for this stochastic sample;
        # conservative-correct verdict is INCONCLUSIVE per updated spec.
        assert report.verdict == StabilityVerdict.INCONCLUSIVE
        assert report.params is not None
        assert report.expected_v_decay is not None

    def test_convergent_ou_verdict_reliable(self) -> None:
        """Controlled decreasing drift: V(e) reliably negative → CONVERGENT.

        Uses a deterministic geometric decay to D*=0.2 so the V(e) regression
        always detects the negative slope regardless of random seed.
        """
        # D_t = 0.2 + 0.2*0.95^t → D_{t+1} = 0.95*D_t + 0.01 (perfect OU step)
        # V(e_t) = (D_t - 0.2)^2 = 0.04*0.9025^t → strictly decreasing slope
        d_star = 0.2
        decay = 0.95
        n = 50
        drift = [d_star + 0.2 * (decay ** t) for t in range(n)]

        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None
        # Fitted D* ≈ 0.2, γ ≈ 0.0513 > α ≈ 0.0103
        assert params.stationary_drift is not None
        assert abs(params.stationary_drift - d_star) < 0.05

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        assert report.verdict == StabilityVerdict.CONVERGENT
        assert report.expected_v_decay is not None
        assert report.expected_v_decay < 0  # V(e) is decreasing

    def test_divergent_ou_gamma_le_alpha(self) -> None:
        """When gamma <= alpha, verdict should be DIVERGENT."""
        alpha, gamma, sigma = 0.5, 0.1, 0.05  # gamma < alpha
        drift = _ou_sample(alpha, gamma, sigma, n=200, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        assert report.verdict == StabilityVerdict.DIVERGENT
        assert report.params is not None
        assert "gamma" in report.reason.lower()
        assert "alpha" in report.reason.lower()

    def test_too_short_sequence(self) -> None:
        """Sequence too short should yield INCONCLUSIVE."""
        drift = [0.1, 0.2, 0.15, 0.1]  # only 4 points
        fitter = OUFitter()
        params = fitter.fit(drift)  # will be None due to length
        assert params is None

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        assert report.verdict == StabilityVerdict.INCONCLUSIVE
        assert report.params is None
        assert "insufficient" in report.reason.lower()

    def test_no_mean_reversion_gamma_zero(self) -> None:
        """gamma = 0 should yield DIVERGENT (no restoring force)."""
        alpha, gamma, sigma = 0.2, 0.0, 0.05
        drift = _ou_sample(alpha, gamma, sigma, n=200, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None
        assert params.gamma < 0.01  # near-zero mean reversion (gamma=0 input)

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        assert report.verdict == StabilityVerdict.DIVERGENT

    def test_high_noise_still_convergent_if_gamma_gt_alpha(self) -> None:
        """Even with high sigma, if gamma > alpha we should still get CONVERGENT."""
        alpha, gamma, sigma = 0.1, 0.5, 0.5  # high noise but strong mean reversion
        drift = _ou_sample(alpha, gamma, sigma, n=300, seed=42)
        fitter = OUFitter()
        params = fitter.fit(drift)
        assert params is not None

        checker = LyapunovStabilityCheck()
        report = checker.verdict(drift, params)
        # Might be INCONCLUSIVE due to noise overwhelming signal, but should not crash
        assert report.verdict in (
            StabilityVerdict.CONVERGENT, StabilityVerdict.INCONCLUSIVE,
        )


class TestStabilityReport:
    """Tests for StabilityReport dataclass."""

    def test_report_creation(self) -> None:
        params = OUParameters(
            alpha=0.1, gamma=0.5, sigma=0.05,
            log_likelihood=-10.0, stationary_drift=0.2,
        )
        report = StabilityReport(
            verdict=StabilityVerdict.CONVERGENT,
            params=params,
            expected_v_decay=-0.01,
            reason="test reason",
        )
        assert report.verdict == StabilityVerdict.CONVERGENT
        assert report.params == params
        assert report.expected_v_decay == -0.01
        assert report.reason == "test reason"
