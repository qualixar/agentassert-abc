# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""F3 OU dynamics + F4 Lyapunov stability verdict — Patent §5.3-5.4.

Implements Ornstein-Uhlenbeck maximum-likelihood fitting and Lyapunov-derived
stability classification on observed drift sequences.

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §5.3 (F3),
§5.4 (F4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress


class StabilityVerdict(StrEnum):
    """Lyapunov stability verdict on an observed drift sequence."""

    CONVERGENT = "convergent"    # Matches OU process with γ > α
    DIVERGENT = "divergent"      # γ ≤ α or empirical divergence
    INCONCLUSIVE = "inconclusive" # Too short, noisy, or fit failed


@dataclass(frozen=True)
class OUParameters:
    """Maximum-likelihood estimate of Ornstein-Uhlenbeck process parameters.

    For discrete-time process: D_{t+1} = D_t * e^{-γ·dt} + (α/γ)*(1 - e^{-γ·dt}) + ε_t
    where ε_t ~ Normal(0, σ²·(1 - e^{-2γ·dt}) / (2γ))

    Attributes:
        alpha: Natural drift rate (α ≥ 0).
        gamma: Mean-reversion rate (γ ≥ 0).
        sigma: Noise magnitude (σ ≥ 0).
        log_likelihood: Log-likelihood of the fitted parameters.
        stationary_drift: Long-run mean D* = α/γ if γ > 0 else None.
    """

    alpha: float
    gamma: float
    sigma: float
    log_likelihood: float
    stationary_drift: float | None


@dataclass(frozen=True)
class StabilityReport:
    """Result of Lyapunov stability check on drift sequence.

    Attributes:
        verdict: CONVERGENT, DIVERGENT, or INCONCLUSIVE.
        params: OUParameters if verdict != INCONCLUSIVE else None.
        expected_v_decay: Slope of V(e_t) = (D_t - D*)² over t (negative = converging).
        reason: Human-readable explanation.
    """

    verdict: StabilityVerdict
    params: OUParameters | None
    expected_v_decay: float | None
    reason: str


class OUFitter:
    """Fit Ornstein-Uhlenbeck parameters to observed drift sequence via MLE.

    Uses discrete-time approximation and maximum likelihood estimation.
    Not a simulator — given observed D(t), infer (α, γ, σ).
    """

    MIN_SEQUENCE_LENGTH = 20

    def fit(self, drift_sequence: list[float], dt: float = 1.0) -> OUParameters | None:
        """Fit OU parameters (α, γ, σ) to drift sequence via maximum likelihood.

        Args:
            drift_sequence: Observed drift scores D(t), one per turn.
            dt: Time step between observations (default 1.0).

        Returns:
            OUParameters if fit succeeds and sequence long enough, else None.
        """
        if len(drift_sequence) < self.MIN_SEQUENCE_LENGTH:
            return None

        y = np.array(drift_sequence, dtype=float)
        n = len(y)

        # Handle constant (zero-variance) sequences: no mean reversion, no drift
        if np.var(y) < 1e-12:
            return OUParameters(
                alpha=0.0,
                gamma=0.0,
                sigma=0.0,
                log_likelihood=0.0,
                stationary_drift=None,
            )

        # Define negative log-likelihood for minimization
        def neg_log_likelihood(params):
            alpha, gamma, sigma = params
            if gamma <= 0 or sigma <= 0:
                return np.inf  # Invalid parameters

            # Discrete-time OU parameters
            a = np.exp(-gamma * dt)  # autocorrelation factor
            b = (alpha / gamma) * (1 - a)  # drift term
            var_noise = (sigma ** 2) * (1 - a ** 2) / (2 * gamma)  # variance of ε_t

            if var_noise <= 0:
                return np.inf

            # Build prediction errors: y_{t+1} - (a * y_t + b)
            errors = y[1:] - (a * y[:-1] + b)
            # Ledger 2e: the errors vector has n-1 transitions, not n observations.
            # Using n as the normalizer biases σ² ~5% low at n=20 and inflates log_lik.
            n_err = len(errors)  # = n - 1
            sse = np.sum(errors ** 2)
            log_lik = -0.5 * n_err * np.log(2 * np.pi * var_noise) - 0.5 * sse / var_noise
            return -log_lik  # Minimize negative log-likelihood

        # Initial guess: method of moments
        y_mean = np.mean(y)
        y_var = np.var(y)
        # Rough estimate: gamma ≈ -log(lag-1 autocorr) / dt
        if n > 1:
            lag1_corr = np.corrcoef(y[:-1], y[1:])[0, 1]
            gamma_init = max(-np.log(max(0.001, abs(lag1_corr))) / dt, 0.01)
        else:
            gamma_init = 0.1
        alpha_init = gamma_init * y_mean if gamma_init > 0 else 0.1
        sigma_init = np.sqrt(max(y_var * 2 * gamma_init, 0.01)) if gamma_init > 0 else 0.1

        # Bounds: alpha ≥ 0, gamma > 0, sigma > 0
        bounds = [(0, None), (1e-8, None), (1e-8, None)]
        x0 = [alpha_init, gamma_init, sigma_init]

        # Optimize
        res = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        if not res.success:
            return None

        alpha_opt, gamma_opt, sigma_opt = res.x
        # Recompute log-likelihood for the optimal parameters
        log_lik = -neg_log_likelihood(res.x)
        stationary = alpha_opt / gamma_opt if gamma_opt > 0 else None

        return OUParameters(
            alpha=max(alpha_opt, 0.0),
            gamma=max(gamma_opt, 0.0),
            sigma=max(sigma_opt, 0.0),
            log_likelihood=float(log_lik),
            stationary_drift=float(stationary) if stationary is not None else None,
        )

    def stationary_drift(self, params: OUParameters) -> float | None:
        """Return stationary drift D* = α/γ if γ > 0 else None."""
        if params.gamma > 0:
            return params.alpha / params.gamma
        return None


class LyapunovStabilityCheck:
    """F4-derived stability verdict using Lyapunov function V(e) = e².

    Where e_t = D_t - D* (deviation from stationary mean).
    """

    def verdict(
        self,
        drift_sequence: list[float],
        fitted: OUParameters | None,
    ) -> StabilityReport:
        """Return stability verdict combining F3 fit quality and F4 V(e) decay.

        Args:
            drift_sequence: Observed drift scores D(t).
            fitted: OUParameters from OUFitter.fit() (may be None).

        Returns:
            StabilityReport with verdict, parameters, and explanation.
        """
        # Guard: insufficient data or failed fit
        if fitted is None or len(drift_sequence) < OUFitter.MIN_SEQUENCE_LENGTH:
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=None,
                expected_v_decay=None,
                reason="Insufficient data for OU fit (need ≥20 turns) or fit failed to converge",
            )

        # Guard: no mean reversion (γ ≤ 0) -> divergent
        if fitted.gamma <= 0:
            return StabilityReport(
                verdict=StabilityVerdict.DIVERGENT,
                params=fitted,
                expected_v_decay=None,
                reason=f"Mean-reversion rate gamma={fitted.gamma:.4f} ≤ 0 (no restoring force)",
            )

        # Compute stationary mean and Lyapunov variable
        d_star = fitted.stationary_drift
        if d_star is None:
            return StabilityReport(
                verdict=StabilityVerdict.DIVERGENT,
                params=fitted,
                expected_v_decay=None,
                reason="Stationary mean undefined (gamma ≤ 0)",
            )

        # e_t = D_t - D*
        e = np.array(drift_sequence) - d_star
        # V(e_t) = e_t²
        v = e ** 2

        # Regression of V(e_t) on time t -> slope indicates convergence/divergence
        t = np.arange(len(v))
        res = linregress(t, v)
        slope = float(res.slope)  # type: ignore[union-attr]
        p_value = float(res.pvalue)  # type: ignore[union-attr]

        # Verdict logic:
        # - If gamma <= alpha: no strong enough restoring force -> divergent
        # - Else if V(t) shows significant negative slope -> convergent
        # - Else if slope not significantly different from zero -> inconclusive
        # - Else (positive slope) -> divergent
        if fitted.gamma <= fitted.alpha:
            return StabilityReport(
                verdict=StabilityVerdict.DIVERGENT,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    "No strong restoring force: "
                    f"gamma ({fitted.gamma:.4f}) ≤ alpha ({fitted.alpha:.4f})"
                ),
            )

        if p_value < 0.05 and slope < 0:
            return StabilityReport(
                verdict=StabilityVerdict.CONVERGENT,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    "Lyapunov function V(e) significantly decreasing "
                    f"(slope={slope:.6f}, p={p_value:.4f})"
                ),
            )

        if p_value >= 0.05 or abs(slope) < 1e-6:  # Not significant or essentially zero
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    "No significant trend in V(e) "
                    f"(slope={slope:.6f}, p={p_value:.4f})"
                ),
            )

        # Ledger 2f: a significant POSITIVE V(e) slope contradicts the comment "→INCONCLUSIVE".
        # Returning CONVERGENT here while V(e) is significantly *increasing* is wrong; the
        # conservative and correct verdict is INCONCLUSIVE (theory says convergent but
        # empirical evidence shows the opposite — we cannot certify stability).
        if fitted.gamma > fitted.alpha and fitted.gamma >= 0.05:
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    "Theoretically convergent "
                    f"(gamma={fitted.gamma:.4f} > alpha={fitted.alpha:.4f}) "
                    "but empirical V(e) slope is significantly positive "
                    f"(slope={slope:.6f}, p={p_value:.4f}) — conservative INCONCLUSIVE"
                ),
            )

        # Positive significant slope with gamma <= alpha or gamma too small: divergent
        return StabilityReport(
            verdict=StabilityVerdict.DIVERGENT,
            params=fitted,
            expected_v_decay=float(slope),
            reason=(
                "No restoring force: "
                f"gamma ({fitted.gamma:.4f}) too small vs alpha ({fitted.alpha:.4f}); "
                f"V(e) significantly increasing (slope={slope:.6f}, p={p_value:.4f})"
            ),
        )
