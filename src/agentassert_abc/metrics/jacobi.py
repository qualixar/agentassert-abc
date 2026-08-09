# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Jacobi/Wright-Fisher bounded-support drift dynamics (LLD-D).

Implements the Jacobi SDE
    dX = kappa*(theta - X)*dt + sigma*sqrt(X*(1-X))*dW
on [0,1] with method-of-moments fitting and an identifiability gate.

The gate (Theorem D.4) refuses any boundary-confinement claim when the
bootstrap confidence region straddles the Feller threshold g_j = 0.

Patent reference: arXiv:2602.22302, LLD-D §D.0–D.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import chi2

from agentassert_abc.exceptions import AgentAssertError

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Module-local exception
# ---------------------------------------------------------------------------

class JacobiFitError(AgentAssertError):
    """Raised when Jacobi fitting is inadmissible or numerically degenerate.

    See LLD-D §D.4.1 Eq. D.25 for the admissibility conditions.
    """


# ---------------------------------------------------------------------------
# Core parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class JacobiParams:
    """Fitted parameters for the Jacobi/Wright-Fisher diffusion (LLD-D §D.0).

    SDE: dX = kappa*(theta - X)*dt + sigma*sqrt(X*(1-X))*dW

    Attributes:
        kappa: Mean-reversion rate (kappa > 0).
        theta: Long-run mean, i.e. the stationary mean (0 < theta < 1).
        sigma2: Squared diffusion intensity (sigma2 > 0).
    """

    kappa: float
    theta: float
    sigma2: float

    def __post_init__(self) -> None:
        """Validate parameter constraints (LLD-D §D.0, Eq. D.1)."""
        if self.kappa <= 0.0:
            raise JacobiFitError(f"kappa must be > 0, got {self.kappa!r}")
        if not (0.0 < self.theta < 1.0):
            raise JacobiFitError(f"theta must be in (0,1), got {self.theta!r}")
        if self.sigma2 <= 0.0:
            raise JacobiFitError(f"sigma2 must be > 0, got {self.sigma2!r}")

    @property
    def a(self) -> float:
        """Shape parameter a = 2*kappa*theta/sigma2 (Eq. D.2, LLD-D §D.0)."""
        return 2.0 * self.kappa * self.theta / self.sigma2

    @property
    def b(self) -> float:
        """Shape parameter b = 2*kappa*(1-theta)/sigma2 (Eq. D.2, LLD-D §D.0)."""
        return 2.0 * self.kappa * (1.0 - self.theta) / self.sigma2

    @property
    def stationary_mean(self) -> float:
        """Stationary mean E[X] = theta (Theorem D.2, Eq. D.12)."""
        return self.theta

    @property
    def stationary_var(self) -> float:
        """Stationary variance Var(X) = theta*(1-theta)*sigma2/(2*kappa+sigma2).

        Derived from Beta(a,b) stationary distribution — Theorem D.2, Eq. D.12.
        """
        return (
            self.theta * (1.0 - self.theta) * self.sigma2
            / (2.0 * self.kappa + self.sigma2)
        )


# ---------------------------------------------------------------------------
# Feller boundary classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FellerClassification:
    """Per-boundary Feller class for the Jacobi diffusion (Theorem D.1, LLD-D §D.1).

    Attributes:
        boundary_0: "unattainable" if a >= 1, else "attainable".
        boundary_1: "unattainable" if b >= 1, else "attainable".
        a: Shape parameter a = 2*kappa*theta/sigma2.
        b: Shape parameter b = 2*kappa*(1-theta)/sigma2.
    """

    boundary_0: str
    boundary_1: str
    a: float
    b: float


def feller_classification(params: JacobiParams) -> FellerClassification:
    """Classify each boundary as unattainable (entrance) or attainable (regular).

    Theorem D.1: boundary 0 is unattainable iff a >= 1 (2*kappa*theta >= sigma2);
    boundary 1 is unattainable iff b >= 1 (2*kappa*(1-theta) >= sigma2).

    Args:
        params: Fitted Jacobi parameters.

    Returns:
        FellerClassification with per-boundary labels and shape parameters.
    """
    a_val = params.a
    b_val = params.b
    b0 = "unattainable" if a_val >= 1.0 else "attainable"
    b1 = "unattainable" if b_val >= 1.0 else "attainable"
    return FellerClassification(boundary_0=b0, boundary_1=b1, a=a_val, b=b_val)


# ---------------------------------------------------------------------------
# Method-of-moments fitting (LLD-D §D.4.1, Eqs. D.22–D.24)
# ---------------------------------------------------------------------------

def fit_jacobi_mom(
    series: Sequence[float] | np.ndarray,
    dt: float,
) -> JacobiParams:
    """Fit Jacobi parameters by method of moments (Eqs. D.22–D.24, LLD-D §D.4.1).

    Estimating equations:
        theta_hat = sample mean           (Eq. D.22)
        kappa_hat = -ln(rho1_hat) / dt    (Eq. D.23)
        sigma2_hat = 2*kappa*S2 / (theta*(1-theta) - S2)  (Eq. D.24)

    where S2 is the sample variance (ddof=1) and rho1_hat is the lag-1
    autocorrelation.  Admissibility conditions (Eq. D.25) are checked before
    any inversion; violations raise JacobiFitError rather than producing
    silent numerical garbage.

    Args:
        series: Observed drift values in [0, 1], length >= 3.
        dt: Positive time step between consecutive observations.

    Returns:
        JacobiParams with fitted (kappa, theta, sigma2).

    Raises:
        JacobiFitError: Any admissibility condition from Eq. D.25 is violated,
            or the moment inversion is numerically degenerate.
    """
    arr = np.asarray(series, dtype=float)

    _validate_inputs(arr, dt)

    theta_hat = float(np.mean(arr))
    if not (0.0 < theta_hat < 1.0):
        raise JacobiFitError(
            f"Sample mean {theta_hat:.6f} not in (0,1) — Jacobi model inadmissible"
        )

    rho1_hat = _lag1_autocorr(arr)
    if not (0.0 < rho1_hat < 1.0):
        raise JacobiFitError(
            f"Lag-1 autocorrelation {rho1_hat:.4f} not in (0,1) — "
            "use constrained likelihood or reject the Jacobi model (Eq. D.25)"
        )

    s2 = float(np.var(arr, ddof=1))
    max_var = theta_hat * (1.0 - theta_hat)
    if not (0.0 < s2 < max_var):
        raise JacobiFitError(
            f"Sample variance {s2:.6f} not in (0, {max_var:.6f}) — "
            "stationary variance must be strictly less than theta*(1-theta) (Eq. D.25)"
        )

    kappa_hat = -float(np.log(rho1_hat)) / dt

    denom = max_var - s2
    if denom <= 0.0:
        raise JacobiFitError(
            f"Degenerate denominator in sigma2 inversion: "
            f"theta*(1-theta) - S2 = {denom:.6e} (Eq. D.24)"
        )
    sigma2_hat = 2.0 * kappa_hat * s2 / denom

    return JacobiParams(kappa=kappa_hat, theta=theta_hat, sigma2=sigma2_hat)


def _validate_inputs(arr: np.ndarray, dt: float) -> None:
    """Validate series array and time step before fitting.

    Args:
        arr: Drift observations.
        dt: Time step.

    Raises:
        JacobiFitError: If len(arr) < 3, dt <= 0, or any value outside [0, 1].
    """
    if len(arr) < 3:
        raise JacobiFitError(
            f"Need >= 3 observations for Jacobi MoM fit, got {len(arr)}"
        )
    if dt <= 0.0:
        raise JacobiFitError(f"dt must be > 0, got {dt!r}")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise JacobiFitError("All values must be in [0, 1]")


def _lag1_autocorr(arr: np.ndarray) -> float:
    """Compute centered lag-1 autocorrelation.

    Uses the global sample mean for centering, normalised by the full sum of
    squares.  Consistent with the moment-equation derivation in LLD-D §D.4.1.

    Args:
        arr: Observations array.

    Returns:
        Lag-1 autocorrelation in [-1, 1].
    """
    X = arr - float(np.mean(arr))  # noqa: N806
    denom = float(np.dot(X, X))
    if denom < 1e-15:
        return 0.0
    return float(np.dot(X[:-1], X[1:])) / denom


# ---------------------------------------------------------------------------
# Identifiability gate (Theorem D.4, LLD-D §D.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of the Theorem D.4 identifiability gate (LLD-D §D.5).

    Conditions CHECKED by this implementation (LLD-D §D.5 pass rule):
        1. Exploration (Eq. D.33): independent excursions at each boundary.
        (pre) Bootstrap stability: >= 50% of resamples produce admissible fits.
        4. Information (Eq. D.34): lambda_min(J_N/N) >= iota_star and
           cond(J_N/N) <= k_star, where J_N^{-1} is estimated by the bootstrap
           parameter covariance in zeta=(log kappa, logit theta, log sigma2).
        5. Precision (Eq. D.35): C_{1-alpha} is bounded and
           diam_zeta(C_{1-alpha}) <= r_star.
        3. Classification separation (Eq. D.36): (1-alpha) percentile CI for
           g_j lies strictly on one side of 0.

    Because J_N is estimated from the moving-block bootstrap (LLD-D §D.5's
    preferred finite-sample calibration), all coverage is ASYMPTOTIC.

    gate_passed is True iff all four LLD-D §D.5 conditions pass.

    Attributes:
        params: Fitted JacobiParams from the base MoM fit, or None if the
            base fit itself failed (degenerate/too-short series).
        gate_passed: True iff all checked gate conditions pass.
        boundary_0_class: Feller class for boundary 0, or None if gate refused.
        boundary_1_class: Feller class for boundary 1, or None if gate refused.
        g0_min: Lower bound of the (1-alpha) bootstrap CI for
            g0 = 2*kappa*theta - sigma2 (alpha/2 percentile).
        g0_max: Upper bound of the (1-alpha) bootstrap CI for g0
            (1 - alpha/2 percentile).
        g1_min: Lower bound of the (1-alpha) bootstrap CI for
            g1 = 2*kappa*(1-theta) - sigma2 (alpha/2 percentile).
        g1_max: Upper bound of the (1-alpha) bootstrap CI for g1.
        reason: Human-readable gate verdict, explicit about which conditions
            were and were not checked.

    Note:
        The g0_*/g1_* CI bounds are ``float("nan")`` when the gate refuses
        BEFORE computing the bootstrap CI (base-fit failure, exploration
        failure, or bootstrap instability). Callers MUST check ``gate_passed``
        (and ``params is not None``) before reading the CI bounds.
    """

    params: JacobiParams | None
    gate_passed: bool
    boundary_0_class: str | None
    boundary_1_class: str | None
    g0_min: float
    g0_max: float
    g1_min: float
    g1_max: float
    reason: str
    info_lambda_min: float = float("nan")
    info_cond: float = float("nan")
    region_diameter: float = float("nan")


def identifiability_gate(
    series: Sequence[float] | np.ndarray,
    dt: float,
    alpha: float = 0.05,
    *,
    n_bootstrap: int = 500,
    eps: float = 0.1,
    m0: int = 1,
    m1: int = 1,
    iota_star: float = 1e-10,
    k_star: float = 1e6,
    r_star: float = 10.0,
) -> GateResult:
    """Run the Theorem D.4 identifiability gate; refuse confinement when it fails.

    Conditions checked (LLD-D §D.5 operational pass rule):
        1. Exploration (Eq. D.33): E0(eps) >= m0 and E1(eps) >= m1, where
           E_j is the count of INDEPENDENT EXCURSIONS (separated blocks) into
           the eps-band near boundary j, not a raw per-observation count.
        2. Bootstrap stability: fraction of successful moving-block bootstrap
           fits >= 50%.  Block length is derived from the lag-1 ACF to preserve
           the Markov dependence structure — i.i.d. resampling is NOT used.
        3. Bootstrap classification separation (Eq. D.36): the (1-alpha)
           percentile CI for g_j lies strictly on one side of 0.  alpha=0.05
           uses a 95% CI (wider → stricter); alpha=0.50 uses a 50% CI
           (narrower → more lenient).

    Also checked (LLD-D §D.5 conditions 4-5, bootstrap-calibrated):
        4. Information (Eq. D.34): lambda_min(J_N/N) >= iota_star and
           cond(J_N/N) <= k_star (J_N^{-1} ~ bootstrap covariance in zeta).
        5. Precision (Eq. D.35): diam_zeta(C_{1-alpha}) <= r_star and bounded.

    If ANY checked condition fails, boundary_0_class and/or boundary_1_class
    are None and gate_passed is False.  No confinement claim may be made
    (LLD-D §D.5 mandatory refusal).

    A degenerate or too-short series that causes the base fit to fail returns
    GateResult(gate_passed=False, params=None) — it does NOT raise.

    Args:
        series: Observed drift values in [0, 1].
        dt: Time step between observations.
        alpha: Significance level for the bootstrap CI.  Smaller alpha →
            wider CI → stricter classification separation.
        n_bootstrap: Number of moving-block bootstrap resamples.
        eps: Boundary band width for exploration check (Eq. D.33).
        m0: Minimum independent-excursion count for boundary 0.
        m1: Minimum independent-excursion count for boundary 1.
        iota_star: Min-eigenvalue floor for J_N/N (Eq. D.34; default 1e-10 ~
            positive-definiteness). Preregister for a real study.
        k_star: Condition-number ceiling for J_N/N (Eq. D.34; default 1e6,
            matching LLD-E §8.4 criterion 5).
        r_star: Max (1-alpha) confidence-region diameter in zeta (Eq. D.35;
            default 10.0 — wide; preregister tighter for a real study).

    Returns:
        GateResult with fitted params (or None), gate verdict, (1-alpha) CI
        bounds for g0/g1, and a reason string listing which conditions were
        checked.
    """
    arr = np.asarray(series, dtype=float)

    # Base fit — graceful refusal on degenerate / too-short series
    try:
        params = fit_jacobi_mom(arr, dt)
    except JacobiFitError as exc:
        return GateResult(
            params=None,
            gate_passed=False,
            boundary_0_class=None,
            boundary_1_class=None,
            g0_min=float("nan"),
            g0_max=float("nan"),
            g1_min=float("nan"),
            g1_max=float("nan"),
            reason=(
                f"Base fit failed (degenerate or inadmissible series): {exc}. "
                "Confinement claim refused."
            ),
        )

    # Gate condition 1: boundary exploration — count INDEPENDENT EXCURSIONS (Eq. D.33)
    e0 = _count_excursions(arr, eps, side="lower")
    e1 = _count_excursions(arr, eps, side="upper")
    if e0 < m0 or e1 < m1:
        return GateResult(
            params=params,
            gate_passed=False,
            boundary_0_class=None,
            boundary_1_class=None,
            g0_min=float("nan"),
            g0_max=float("nan"),
            g1_min=float("nan"),
            g1_max=float("nan"),
            reason=(
                f"Exploration gate failed (condition 1, Eq. D.33): "
                f"E0({eps:.2f})={e0} independent excursions (need {m0}), "
                f"E1({1.0 - eps:.2f})={e1} (need {m1}). "
                "Confinement claim refused — series does not independently "
                "explore both boundaries. "
                "(Conditions NOT checked: information D.34, diameter D.35.)"
            ),
        )

    # Pre-check: moving-block bootstrap stability
    g0_samples, g1_samples, zeta_samples = _bootstrap_samples(arr, dt, n_bootstrap)

    if len(g0_samples) < n_bootstrap // 2:
        return GateResult(
            params=params,
            gate_passed=False,
            boundary_0_class=None,
            boundary_1_class=None,
            g0_min=float("nan"),
            g0_max=float("nan"),
            g1_min=float("nan"),
            g1_max=float("nan"),
            reason=(
                f"Bootstrap stability gate failed: only "
                f"{len(g0_samples)}/{n_bootstrap} moving-block resamples "
                "produced admissible fits. "
                "Confinement claim refused — fit is numerically unstable. "
                "(Refused before D.34 information / D.35 diameter checks.)"
            ),
        )

    # Gate conditions 4 & 5 — information (Eq. D.34) and precision/diameter
    # (Eq. D.35), from the bootstrap parameter distribution in
    # zeta = (log kappa, logit theta, log sigma2).  The bootstrap covariance is
    # the finite-sample stand-in for J_N^{-1} (LLD-D §D.5 preferred calibration);
    # coverage is therefore ASYMPTOTIC.
    zeta = np.asarray(zeta_samples, dtype=float)
    cov = np.cov(zeta, rowvar=False)
    cov_evals = np.linalg.eigvalsh(cov)
    lam_min_cov = float(cov_evals[0])
    lam_max_cov = float(cov_evals[-1])
    n_trans = max(1, len(arr) - 1)
    if lam_min_cov <= 0.0 or not np.isfinite(lam_max_cov):
        info_lambda_min = 0.0
        info_cond = float("inf")
        region_diameter = float("inf")
    else:
        info_lambda_min = 1.0 / (n_trans * lam_max_cov)  # lambda_min(J_N/N)
        info_cond = lam_max_cov / lam_min_cov            # cond(J_N/N) = cond(cov)
        c_1a = float(chi2.ppf(1.0 - alpha, df=3))
        region_diameter = 2.0 * math.sqrt(c_1a * lam_max_cov)

    info_ok = info_lambda_min >= iota_star and info_cond <= k_star
    diam_ok = math.isfinite(region_diameter) and region_diameter <= r_star
    diag = (
        f"[info lambda_min(J/N)={info_lambda_min:.2e} (>= {iota_star:.0e}), "
        f"cond={info_cond:.1f} (<= {k_star:.0e}); "
        f"diam_zeta={region_diameter:.3f} (<= {r_star})]"
    )
    if not info_ok or not diam_ok:
        return GateResult(
            params=params,
            gate_passed=False,
            boundary_0_class=None,
            boundary_1_class=None,
            g0_min=float("nan"),
            g0_max=float("nan"),
            g1_min=float("nan"),
            g1_max=float("nan"),
            info_lambda_min=info_lambda_min,
            info_cond=info_cond,
            region_diameter=region_diameter,
            reason=(
                "Information/precision gate failed (conditions 4-5, "
                f"Eqs. D.34-D.35): {diag}. Confinement claim refused — the fit "
                "is under-identified or its confidence region is too wide."
            ),
        )

    # Gate condition 3 (Eq. D.36): (1-alpha) percentile CI strictly one side of 0
    g0_arr = np.array(g0_samples)
    g1_arr = np.array(g1_samples)
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    g0_lo = float(np.percentile(g0_arr, lo_pct))
    g0_hi = float(np.percentile(g0_arr, hi_pct))
    g1_lo = float(np.percentile(g1_arr, lo_pct))
    g1_hi = float(np.percentile(g1_arr, hi_pct))

    b0_sep = (g0_lo > 0.0) or (g0_hi < 0.0)
    b1_sep = (g1_lo > 0.0) or (g1_hi < 0.0)
    ci_pct = int(round(100.0 * (1.0 - alpha)))

    if not b0_sep or not b1_sep:
        b0_cls: str | None = _class_from_range(g0_lo, g0_hi) if b0_sep else None
        b1_cls: str | None = _class_from_range(g1_lo, g1_hi) if b1_sep else None
        return GateResult(
            params=params,
            gate_passed=False,
            boundary_0_class=b0_cls,
            boundary_1_class=b1_cls,
            g0_min=g0_lo,
            g0_max=g0_hi,
            g1_min=g1_lo,
            g1_max=g1_hi,
            info_lambda_min=info_lambda_min,
            info_cond=info_cond,
            region_diameter=region_diameter,
            reason=(
                f"Classification separation gate failed (condition 3, Eq. D.36, "
                f"{ci_pct}% CI): bootstrap CI straddles the Feller threshold for "
                + _straddled_boundaries(b0_sep, b1_sep)
                + f". g0 CI [{g0_lo:.3f}, {g0_hi:.3f}], "
                f"g1 CI [{g1_lo:.3f}, {g1_hi:.3f}]. "
                f"Confinement claim refused. {diag}"
            ),
        )

    b0_class = _class_from_range(g0_lo, g0_hi)
    b1_class = _class_from_range(g1_lo, g1_hi)
    return GateResult(
        params=params,
        gate_passed=True,
        boundary_0_class=b0_class,
        boundary_1_class=b1_class,
        g0_min=g0_lo,
        g0_max=g0_hi,
        g1_min=g1_lo,
        g1_max=g1_hi,
        info_lambda_min=info_lambda_min,
        info_cond=info_cond,
        region_diameter=region_diameter,
        reason=(
            "Gate PASSED — all four LLD-D §D.5 conditions hold: exploration "
            f"(D.33), information (D.34), diameter (D.35), classification "
            f"separation (D.36) at {ci_pct}% CI. "
            f"Boundary 0: {b0_class}. Boundary 1: {b1_class}. "
            f"g0 CI [{g0_lo:.3f}, {g0_hi:.3f}], g1 CI [{g1_lo:.3f}, {g1_hi:.3f}]. "
            f"{diag} (coverage asymptotic — bootstrap-calibrated)."
        ),
    )


def _count_excursions(arr: np.ndarray, eps: float, side: str) -> int:
    """Count independent excursions into the boundary band (LLD-D Eq. D.33).

    An excursion is a maximal connected run of consecutive observations that
    lie within eps of the specified boundary.  Two runs separated by at least
    one observation outside the band are counted as two independent excursions.
    This is the correct operationalisation of D.33 — raw observation counts
    (which inflate on one long run) are NOT used.

    Args:
        arr: Observed series array.
        eps: Half-width of the boundary band (same as the gate's eps parameter).
        side: "lower" for boundary 0 (arr <= eps); "upper" for boundary 1
            (arr >= 1 - eps).

    Returns:
        Number of independent excursions.
    """
    in_band = arr <= eps if side == "lower" else arr >= 1.0 - eps
    excursions = 0
    in_excursion = False
    for inside in in_band:
        if inside and not in_excursion:
            excursions += 1
            in_excursion = True
        elif not inside:
            in_excursion = False
    return excursions


def _moving_block_resample(
    arr: np.ndarray,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample arr using the moving-block bootstrap.

    Draws overlapping blocks of length block_len with replacement and
    concatenates them to length len(arr).  Preserves local autocorrelation
    structure, unlike i.i.d. index resampling.

    Args:
        arr: Original time series of length n.
        block_len: Block length (>= 1, <= n).
        rng: NumPy random generator.

    Returns:
        Resampled array of the same length as arr.
    """
    n = len(arr)
    n_blocks = int(np.ceil(n / block_len))
    # Valid start positions: 0 to n - block_len (inclusive)
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    resample = np.concatenate([arr[s : s + block_len] for s in starts])
    return resample[:n]


def _bootstrap_samples(
    arr: np.ndarray,
    dt: float,
    n_bootstrap: int,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Moving-block bootstrap of g0, g1, and zeta=(log k, logit theta, log s2).

    Block length is derived from the lag-1 ACF to preserve the Markov
    dependence structure — i.i.d. index resampling is NOT used (it would
    destroy autocorrelation, causing systematic under-coverage for the
    Feller boundary classification test).  The zeta samples are the basis for
    the D.34 information and D.35 diameter checks: their covariance is the
    finite-sample stand-in for J_N^{-1} (LLD-D §D.5).

    Args:
        arr: Observed series.
        dt: Time step.
        n_bootstrap: Number of resamples.

    Returns:
        (g0_samples, g1_samples, zeta_samples) from successful bootstrap fits,
        where zeta_samples[i] = [log(kappa), logit(theta), log(sigma2)].
    """
    rng = np.random.default_rng(20260725)  # fixed seed for reproducibility
    n = len(arr)

    # Block length ~ 2 * integrated autocorrelation time for AR(1)
    # tau = -1 / log(rho1) for AR(1) with lag-1 autocorrelation rho1
    rho1_est = _lag1_autocorr(arr)
    if 0.0 < rho1_est < 1.0:
        tau = max(1.0, -1.0 / float(np.log(rho1_est)))
        block_len = max(3, int(np.ceil(2.0 * tau)))
    else:
        block_len = max(3, int(np.ceil(n ** 0.25)))
    # Cap at n//3 to guarantee at least 3 independent blocks
    block_len = min(block_len, max(3, n // 3))

    g0_samples: list[float] = []
    g1_samples: list[float] = []
    zeta_samples: list[list[float]] = []

    for _ in range(n_bootstrap):
        resample = _moving_block_resample(arr, block_len, rng)
        try:
            bp = fit_jacobi_mom(resample, dt)
        except JacobiFitError:
            continue  # inadmissible resample — skip, count failure implicitly
        g0_samples.append(2.0 * bp.kappa * bp.theta - bp.sigma2)
        g1_samples.append(2.0 * bp.kappa * (1.0 - bp.theta) - bp.sigma2)
        zeta_samples.append(
            [
                math.log(bp.kappa),
                math.log(bp.theta / (1.0 - bp.theta)),
                math.log(bp.sigma2),
            ]
        )

    return g0_samples, g1_samples, zeta_samples


def _class_from_range(g_min: float, g_max: float) -> str:
    """Translate a bootstrap g-range to a Feller class string.

    Args:
        g_min: Minimum g value across bootstrap samples.
        g_max: Maximum g value across bootstrap samples.

    Returns:
        "unattainable" if g_min > 0, "attainable" if g_max < 0.
    """
    if g_min > 0.0:
        return "unattainable"
    return "attainable"


def _straddled_boundaries(b0_sep: bool, b1_sep: bool) -> str:
    """Return a human-readable list of boundaries that straddle the threshold."""
    if not b0_sep and not b1_sep:
        return "boundaries 0 and 1"
    if not b0_sep:
        return "boundary 0"
    return "boundary 1"
