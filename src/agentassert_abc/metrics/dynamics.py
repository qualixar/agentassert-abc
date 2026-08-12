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

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

DEFAULT_D_CRIT = 0.6
"""Critical stationary-drift threshold for the admissibility gate (paper A.4).

Sourced from ``models.DriftThresholds.critical`` (the v1 drift alert threshold,
"warning=0.3, critical=0.6"): the attractor is admissible exactly when it sits
below the level at which drift would raise a critical alert. Both sides are in
drift-score units.

NOTE for the paper: A.4 attributes this to "A.3", but A.3 defines the drift
*weights* (ν_c = 0.6, ν_d = 0.4), not a threshold. The value is right and has a
real source; the cross-reference is wrong and should point at the drift
thresholds instead.
"""


class StabilityVerdict(StrEnum):
    """Verdict on an observed drift sequence (paper A.4: two separate gates).

    A.4 separates two properties that the v1 verdict conflated:

    * **Stability** — the OU process is mean-reverting for *every* ``γ > 0``;
      ``α`` does not enter. The v1 test ``γ > α`` is not scale-invariant (under
      ``D ↦ cD`` we get ``α ↦ cα`` while ``γ`` is unchanged, so the comparison
      flips on an unchanged process) and is therefore not used.
    * **Admissibility** — the attractor ``D* = α/γ`` must lie below
      ``D_crit``. Both sides are in drift units.

    A process can be perfectly stable yet settle on an unacceptably high drift
    level; that is ``INADMISSIBLE``, not ``DIVERGENT``.
    """

    CONVERGENT = "convergent"      # γ > 0 and D* < D_crit: stable, admissible
    INADMISSIBLE = "inadmissible"  # γ > 0 (stable) but D* = α/γ ≥ D_crit
    DIVERGENT = "divergent"        # γ ≤ 0: no restoring force at all
    INCONCLUSIVE = "inconclusive"  # too short, fit failed, or data contradicts fit


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
        verdict: CONVERGENT, INADMISSIBLE, DIVERGENT, or INCONCLUSIVE.
        params: OUParameters if verdict != INCONCLUSIVE else None.
        expected_v_decay: Slope of V(e_t) = (D_t - D*)² over t (negative = converging).
        reason: Human-readable explanation.
        stable: A.4 gate (i) — mean-reverting (γ > 0). None if not assessable.
        admissible: A.4 gate (ii) — D* < D_crit. None if not assessable.
        d_star: Stationary drift D* = α/γ, the attractor the process settles on.
        d_crit: The admissibility threshold applied.
    """

    verdict: StabilityVerdict
    params: OUParameters | None
    expected_v_decay: float | None
    reason: str
    stable: bool | None = None
    admissible: bool | None = None
    d_star: float | None = None
    d_crit: float = DEFAULT_D_CRIT


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
        d_crit: float = DEFAULT_D_CRIT,
    ) -> StabilityReport:
        """Return the A.4 two-gate verdict on an observed drift sequence.

        Gate (i) stability: mean-reverting for every ``γ > 0`` (``α`` does not
        enter). Gate (ii) admissibility: ``D* = α/γ < d_crit``.

        Order of evaluation: stability, then admissibility, then the empirical
        check. If the observed Lyapunov trajectory ``V(e_t)`` has a significantly
        *positive* slope the fit is contradicted by the data and the verdict is
        ``INCONCLUSIVE`` — an honest refusal (A.4) — but that override applies
        only to processes that already cleared both gates. A marginally
        significant slope is not evidence that a well-fitted ``D*`` is wrong, and
        an inadmissible attractor stays inadmissible either way (see the inline
        note on the admissibility gate for the measured case behind this order).

        ``stable``, ``admissible``, ``d_star`` and ``d_crit`` are populated on
        every report that can compute them, so both gates remain readable
        regardless of which one determined ``verdict``.

        Args:
            drift_sequence: Observed drift scores D(t).
            fitted: OUParameters from OUFitter.fit() (may be None).
            d_crit: Admissibility threshold on the attractor. Default 0.6.

        Returns:
            StabilityReport with the verdict, both gate outcomes, and D*.
        """
        # Guard: insufficient data or failed fit
        if fitted is None or len(drift_sequence) < OUFitter.MIN_SEQUENCE_LENGTH:
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=None,
                expected_v_decay=None,
                reason="Insufficient data for OU fit (need ≥20 turns) or fit failed to converge",
                stable=None,
                admissible=None,
                d_star=None,
                d_crit=d_crit,
            )

        # Gate (i) — stability. No restoring force at all: genuinely divergent.
        # NOTE: this is γ ≤ 0, NOT the v1 γ ≤ α test, which A.4 disavows.
        if fitted.gamma <= 0:
            return StabilityReport(
                verdict=StabilityVerdict.DIVERGENT,
                params=fitted,
                expected_v_decay=None,
                reason=f"Mean-reversion rate gamma={fitted.gamma:.4f} ≤ 0 (no restoring force)",
                stable=False,
                admissible=None,
                d_star=None,
                d_crit=d_crit,
            )

        # Compute stationary mean and Lyapunov variable
        # gamma > 0 but no attractor supplied. Not reachable from OUFitter (which
        # always sets stationary_drift when gamma > 0), but a caller can hand us a
        # hand-built OUParameters. Gate (i) has passed, so this is NOT divergence —
        # we simply cannot run gate (ii), so we decline to rule.
        d_star = fitted.stationary_drift
        if d_star is None:
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=fitted,
                expected_v_decay=None,
                reason=(
                    f"Mean-reverting (gamma={fitted.gamma:.4f} > 0) but no stationary "
                    "drift was supplied, so admissibility cannot be assessed"
                ),
                stable=True,
                admissible=None,
                d_star=None,
                d_crit=d_crit,
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

        # A non-finite regression (NaN/inf in the drift series, or a degenerate
        # fit) must not leak a NaN into the report: NaN comparisons are all False,
        # so the branches below would silently fall through to a verdict while
        # `expected_v_decay=nan` poisons serialisation and any downstream compare.
        if not (math.isfinite(slope) and math.isfinite(p_value)):
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=fitted,
                expected_v_decay=None,
                reason=(
                    "Lyapunov regression is not finite (non-finite drift values or a "
                    "degenerate series) — cannot assess convergence"
                ),
                stable=True,
                admissible=bool(d_star < d_crit),
                d_star=float(d_star),
                d_crit=d_crit,
            )

        # gamma > 0 here, so gate (i) has passed: the process IS mean-reverting.
        admissible = d_star < d_crit
        significant = p_value < 0.05

        # Gate (ii) — admissibility, checked BEFORE the empirical override.
        #
        # The tempting alternative is to let a contradicted fit suppress this gate,
        # on the theory that D* = alpha/gamma is untrustworthy when the data
        # disagrees with the model. Measured against the real case A.4 was written
        # for, that is wrong: an OU series with alpha=0.5, gamma=0.1 fits D*=4.985
        # against a true 5.0 — an excellent fit — yet its V(e) slope is 2.6e-5 with
        # p=0.025, "significant" only because n=200. Ordering the override first
        # would convert a correct INADMISSIBLE (attractor 8x above D_crit) into
        # INCONCLUSIVE on the strength of a practically-zero slope.
        #
        # So: a marginally significant slope is not evidence the fit is wrong, and
        # INADMISSIBLE is a negative verdict that can never over-certify. The gate
        # runs first and the empirical override remains available for cases that
        # clear it.
        if not admissible:
            return StabilityReport(
                verdict=StabilityVerdict.INADMISSIBLE,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    f"Mean-reverting (gamma={fitted.gamma:.4f} > 0) but the attractor "
                    f"D*=alpha/gamma={d_star:.4f} is not below D_crit={d_crit:.4f}"
                ),
                stable=True,
                admissible=False,
                d_star=float(d_star),
                d_crit=d_crit,
            )

        # Empirical override (A.4): among processes that pass BOTH gates, a
        # significantly positive V(e) slope means the data contradicts the fitted
        # model — refuse rather than certify convergence.
        if significant and slope > 0:
            return StabilityReport(
                verdict=StabilityVerdict.INCONCLUSIVE,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    f"Theoretically stable (gamma={fitted.gamma:.4f} > 0) and admissible "
                    f"(D*={d_star:.4f} < D_crit={d_crit:.4f}) but empirical V(e) slope is "
                    f"significantly positive (slope={slope:.6f}, p={p_value:.4f}) — "
                    "fit contradicted, conservative INCONCLUSIVE"
                ),
                stable=True,
                admissible=True,
                d_star=float(d_star),
                d_crit=d_crit,
            )

        if significant and slope < 0:
            return StabilityReport(
                verdict=StabilityVerdict.CONVERGENT,
                params=fitted,
                expected_v_decay=float(slope),
                reason=(
                    "Lyapunov function V(e) significantly decreasing "
                    f"(slope={slope:.6f}, p={p_value:.4f}); "
                    f"attractor D*={d_star:.4f} < D_crit={d_crit:.4f}"
                ),
                stable=True,
                admissible=True,
                d_star=float(d_star),
                d_crit=d_crit,
            )

        # Stable and admissible in theory, but the observed trajectory shows no
        # significant trend either way — no empirical corroboration, so do not
        # certify convergence.
        return StabilityReport(
            verdict=StabilityVerdict.INCONCLUSIVE,
            params=fitted,
            expected_v_decay=float(slope),
            reason=(
                f"No significant trend in V(e) (slope={slope:.6f}, p={p_value:.4f}); "
                f"stable and admissible in theory (D*={d_star:.4f} < D_crit={d_crit:.4f}) "
                "but unconfirmed empirically"
            ),
            stable=True,
            admissible=True,
            d_star=float(d_star),
            d_crit=d_crit,
        )
