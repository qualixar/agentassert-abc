# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Dependence-aware compositional reliability (LLD-B Thm B.1 / B.5 / B.6 / B.7).

The v1 compositional guarantee multiplies per-agent reliabilities
(``sequential_composition_bound`` in ``composition.py``). That product is valid
only under conditional independence (condition C5). When agents share a model
they fail the *same* missions, C5 is violated, and the product can be badly
miscalibrated — empirically off by 30+ points for shared/similar models, and
*anti-conservative* (over-certifying) for redundant topologies.

This module provides the dependence-aware family that stays valid across that
whole range, from correlated (shared) to independent (diverse) pipelines:

* :func:`frechet_all_success_bounds`   — Thm B.1: assumption-free sandwich.
  Valid under *any* dependence; often vacuous (lower = 0) in practice, which is
  exactly why a model-based bound is needed.
* :func:`gaussian_copula_all_success`  — Thm B.5 all-success case: the exact
  orthant probability under a Gaussian copula (used for the m = 2 series pair).
* :func:`shared_factor_all_success`    — Thm B.6: the one-factor O(mQ)
  Gauss–Hermite reduction (the paper's analytic formula; **reference only** —
  fixed-node GH is inaccurate as λ → 1, so certificates do not use it).
* :func:`factor_all_success`           — the **certificate-grade** adaptive-
  quadrature evaluator of the same 1-D reduction; accurate as λ → 1 and clipped
  to the Fréchet sandwich with a hard non-convergence guard.
* :func:`series_reliability_floor`     — bootstrap LCB on the *model*
  functional. **DIAGNOSTIC only, NOT the shipped guarantee** (audit F1: coverage
  of the true reliability collapses under pairwise-indistinguishable
  misspecification). The certificate is assembled by
  :mod:`agentassert_abc.certification.certificate` — exact Tier-0 Clopper–Pearson
  (:mod:`agentassert_abc.certification.observed_floor`) for executed
  compositions, or the finite-sample copula-agnostic Tier-1 LP
  (:mod:`agentassert_abc.certification.lp_bound`) for extrapolation. The correct
  finite-sample model floor is Tier-2 Slepian
  (:mod:`agentassert_abc.certification.slepian_floor`).

"All-success" = every stage of a series / AND pipeline meets its contract
(:math:`Y_G = 1`). Inputs are per-mission pass indicators; nothing here calls a
model or touches the network — pure, offline-testable statistics.

Sign convention: each stage has a latent :math:`U_j \\sim \\mathcal N(0, 1)`;
success is :math:`\\{U_j \\le a_j\\}` with threshold :math:`a_j = \\Phi^{-1}(p_j)`,
so :math:`\\Pr(\\text{success}) = \\Phi(a_j) = p_j`. This is *equivalent* to
LLD-B §4.6's :math:`L_j \\sim \\mathcal N(\\eta_j, 1)` with success
:math:`\\{L_j \\ge 0\\}` and :math:`\\eta_j = a_j` — the latent's sign is absorbed
(``U = a − L``), and both yield :math:`p_j` (audit F2: the ≤/≥ are two
conventions for one model, not a contradiction). Positive latent correlation
``Corr(U_i, U_j) = λ_i λ_j`` ⇒ co-failure.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal, norm

from agentassert_abc.dependence.estimators import CoFailureTable, tetrachoric
from agentassert_abc.exceptions import DependenceError

# Probability clip: keeps Φ⁻¹(p) finite for degenerate (all-pass / all-fail)
# marginals without perturbing realistic rates.
_PCLIP = 1e-6
# Loading clip: keeps d_j = sqrt(1 − λ²) strictly positive (and bounds the fit).
_LCLIP = 1.0 - 1e-6
# Correlation clip: keeps the 2×2 latent covariance strictly PSD.
_RCLIP = 1.0 - 1e-6
# Gross Fréchet-violation tolerance. The one-factor reliability must lie in the
# Fréchet sandwich (Thm B.1) for every joint; a value outside by more than this
# is a numerical non-convergence, not a probability, and is raised (Q7a fix).
_FR_TOL = 1e-3
# Max Gauss–Hermite nodes for the reference formula ``shared_factor_all_success``
# (numpy's ``hermgauss`` overflows for q >= ~384). The certificate path does NOT
# use fixed-node GH — it uses adaptive quadrature (``factor_all_success``), which
# stays accurate in the sharp λ → 1 regime where fixed-node GH overshoots.
_QMAX = 320
# Finite integration half-width for the 1-D factor integral: φ(ξ) is < 1e-300
# beyond |ξ| ≈ 38, so [-40, 40] captures the whole mass and lets us pass the
# integrand's kink locations to the adaptive quadrature.
_QUAD_HALFWIDTH = 40.0


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _as_marginals(marginals: object) -> np.ndarray:
    """Validate and return a 1-D float array of marginal pass probabilities."""
    p = np.asarray(marginals, dtype=float).ravel()
    if p.size == 0:
        raise DependenceError("marginals must be non-empty")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise DependenceError("marginals must lie in [0, 1]")
    return p


def _as_pass_matrix(passes: object) -> np.ndarray:
    """Validate an m×n binary pass matrix (rows = branches, cols = missions)."""
    a = np.asarray(passes)
    if a.ndim != 2:
        raise DependenceError("pass matrix must be 2-D (m branches × n missions)")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise DependenceError("pass matrix must have >=1 branch and >=1 mission")
    uniq = set(np.unique(a).tolist())
    if not uniq <= {0, 1}:  # 0.0/1.0/False/True compare equal to 0/1
        raise DependenceError("pass matrix entries must be binary (0/1)")
    return a.astype(int)


# ---------------------------------------------------------------------------
# Thm B.1 — assumption-free Fréchet–Hoeffding sandwich (all-success)
# ---------------------------------------------------------------------------


def frechet_all_success_bounds(marginals: object) -> tuple[float, float]:
    """Sharp all-success bounds valid under **any** dependence (LLD-B Thm B.1).

    For success indicators with marginals :math:`p_1, …, p_m`,

        lower = max(0, Σ p_j − (m − 1)),   upper = min_j p_j,

    which are **pointwise sharp**: for the given threshold vector each is attained
    by some coupling, so no assumption-free bound is tighter. (For m ≥ 3 the
    Fréchet lower bound ``W`` is not itself a copula, so no *single* coupling
    attains it at all threshold vectors simultaneously — audit F12.) The lower
    bound is frequently **vacuous** (0) whenever Σ p_j < m − 1 — the empirical
    norm, and the motivation for the model-based bounds below.

    Returns:
        ``(lower, upper)`` on :math:`\\Pr(\\text{all }m\\text{ succeed})`.
    """
    p = _as_marginals(marginals)
    m = p.size
    lower = max(0.0, float(p.sum()) - (m - 1))
    upper = float(p.min())
    return lower, upper


# ---------------------------------------------------------------------------
# Thm B.5 (all-success) — exact Gaussian-copula orthant probability
# ---------------------------------------------------------------------------


def gaussian_copula_all_success(
    marginals: object, corr: object, _assume_psd: bool = False
) -> float:
    """All-success probability under a Gaussian copula (LLD-B Thm B.5).

    Computes :math:`\\Pr(U_1 \\le a_1, …, U_m \\le a_m)` where
    :math:`U \\sim \\mathcal N(0, R)`, :math:`R` is the latent **success**
    correlation matrix and :math:`a_j = \\Phi^{-1}(p_j)`. Exact for any ``m``
    (evaluated via the multivariate-normal CDF); for ``m = 1`` it is ``p_1``.

    This is the general orthant form. For ``m ≥ 3`` under a one-factor
    structure prefer :func:`shared_factor_all_success` — the O(mQ) reduction
    that the paper contributes and that avoids the m-dimensional MVN integral.

    Args:
        marginals: per-branch success probabilities ``p_j``.
        corr: ``m × m`` latent success-correlation matrix (unit diagonal).
        _assume_psd: when ``True``, skip the internal :func:`_psd_retract_corr`
            projection because the caller has *already* produced a PD, elementwise
            monotone-safe matrix (e.g. :func:`slepian_floor._dominated_psd`).
            Prevents a **double projection** that would silently re-apply the
            *unsafe* scale-toward-0 retraction inside a lower-bound guarantee path
            (Opus 5 audit, 2026-08-11). Default ``False`` for the general orthant.

    Returns:
        The all-success probability in ``[0, 1]``.
    """
    p = _clip_marginals(_as_marginals(marginals))
    m = p.size
    a = norm.ppf(p)
    if m == 1:
        return float(p[0])
    R = np.array(corr, dtype=float)  # noqa: N806
    if R.shape != (m, m):
        raise DependenceError(f"corr must be {m}×{m} to match {m} marginals")
    if m == 2:
        rho = float(np.clip(R[0, 1], -_RCLIP, _RCLIP))
        cov = [[1.0, rho], [rho, 1.0]]
        return float(multivariate_normal.cdf(a, mean=[0.0, 0.0], cov=cov))
    if not _assume_psd:
        R = _psd_retract_corr(R)  # noqa: N806
    return float(multivariate_normal.cdf(a, mean=np.zeros(m), cov=R, allow_singular=True))


# ---------------------------------------------------------------------------
# Thm B.6 — one-factor all-success reduction (Gauss–Hermite, O(mQ))
# ---------------------------------------------------------------------------


def shared_factor_all_success(
    marginals: object, loadings: object, q: int = 64
) -> float:
    """One-factor all-success reliability via Gauss–Hermite (LLD-B Thm B.6).

    Under the shared-factor model :math:`U_j = λ_j Ξ + \\sqrt{1 − λ_j^2}\\,ε_j`
    with a common factor :math:`Ξ \\sim \\mathcal N(0,1)`, the all-success
    probability reduces from an m-dimensional integral to a single one:

    .. math::
        \\widetilde R_Q = \\frac{1}{\\sqrt\\pi}\\sum_{q=1}^Q w_q
        \\prod_{j=1}^m \\Phi\\!\\left(\\frac{a_j - \\sqrt2\\,λ_j t_q}{d_j}\\right),

    with :math:`a_j = \\Phi^{-1}(p_j)`, :math:`d_j = \\sqrt{1 − λ_j^2}`, and
    :math:`(t_q, w_q)` the Q-point Gauss–Hermite nodes/weights
    (substitution :math:`ξ = \\sqrt2\\,t`). Products are formed in log space for
    stability. Cost is O(mQ) after the O(1) threshold precompute.

    Args:
        marginals: per-branch success probabilities ``p_j`` (length m).
        loadings: shared-factor loadings ``λ_j`` (length m), ``|λ_j| < 1``.
        q: number of Gauss–Hermite nodes (default 64; raise if λ → 1 or m large).

    Returns:
        The all-success probability in ``[0, 1]``.
    """
    if q < 2:
        raise DependenceError("q (quadrature nodes) must be >= 2")
    p = _clip_marginals(_as_marginals(marginals))
    lam = np.asarray(loadings, dtype=float).ravel()
    if lam.size != p.size:
        raise DependenceError("loadings and marginals must have equal length")
    lam = np.clip(lam, -_LCLIP, _LCLIP)
    a = norm.ppf(p)
    d = np.sqrt(1.0 - lam**2)
    t, w = np.polynomial.hermite.hermgauss(q)
    total = 0.0
    for tq, wq in zip(t, w, strict=True):
        z = (a - np.sqrt(2.0) * lam * tq) / d
        total += wq * np.exp(np.sum(norm.logcdf(z)))
    return float(np.clip(total / np.sqrt(np.pi), 0.0, 1.0))


def factor_all_success(marginals: object, loadings: object) -> tuple[float, float]:
    """One-factor all-success reliability via ADAPTIVE quadrature (Thm B.6).

    The **certificate-grade** evaluator. It integrates the 1-D reduction

        R = ∫ φ(ξ) Π_j Φ((a_j − λ_j ξ)/d_j) dξ,    d_j = √(1 − λ_j²),

    with adaptive Gauss–Kronrod quadrature (``scipy.integrate.quad``), subdividing
    at the integrand kinks ξ*_j = a_j/λ_j. Unlike fixed-node Gauss–Hermite
    (:func:`shared_factor_all_success`), this stays accurate as any ``λ_j → 1``
    (where ``d_j → 0`` and the integrand becomes a near-step) — the regime where
    fixed-node GH over- or under-shoots with uncontrolled sign (LLD-B §4.5; the
    external audit's Q7a). It is validated against ``scipy.quad`` ground truth to
    ~1e-8.

    The result is clipped to the Fréchet sandwich (Thm B.1). A raw value outside
    it by more than ``_FR_TOL`` is a genuine non-convergence — not a probability —
    and is **raised**: a free, sound safety net against over-certification
    (audit Q7a / F4; ``R ≤ min_j p_j`` holds for every joint).

    Returns:
        ``(value, abserr)`` — reliability in ``[frechet_lo, frechet_hi]`` and the
        quadrature's own absolute-error estimate (a valid ε_Q, unlike the
        fixed-GH successive-difference heuristic that the quantisation plateau
        fools — audit Q7b).
    """
    p = _clip_marginals(_as_marginals(marginals))
    lam = np.asarray(loadings, dtype=float).ravel()
    if lam.size != p.size:
        raise DependenceError("loadings and marginals must have equal length")
    lam = np.clip(lam, -_LCLIP, _LCLIP)
    a = norm.ppf(p)
    d = np.sqrt(1.0 - lam**2)

    def _integrand(xi: float) -> float:
        log_phi = -0.5 * xi * xi - 0.5 * np.log(2.0 * np.pi)
        z = (a - lam * xi) / d
        return float(np.exp(log_phi + np.sum(norm.logcdf(z))))

    kinks = sorted(
        float(a[j] / lam[j])
        for j in range(a.size)
        if abs(lam[j]) > 1e-9 and abs(a[j] / lam[j]) < _QUAD_HALFWIDTH
    )
    value, abserr = quad(
        _integrand, -_QUAD_HALFWIDTH, _QUAD_HALFWIDTH,
        points=kinks or None, limit=200,
    )
    lo, hi = frechet_all_success_bounds(p)
    if value < lo - _FR_TOL or value > hi + _FR_TOL:
        raise DependenceError(
            f"factor reliability {value:.6f} fell outside the Fréchet sandwich "
            f"[{lo:.6f}, {hi:.6f}] — quadrature non-convergence; refusing to certify"
        )
    return float(np.clip(value, lo, hi)), float(abserr)


# ---------------------------------------------------------------------------
# Fitting the series factor model from per-mission pass indicators
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SeriesFactorFit:
    """Fitted shared-factor parameters for a series (all-success) pipeline.

    Attributes:
        marginals: per-branch success probabilities ``p_j``.
        corr: latent success-correlation matrix (tetrachoric).
        loadings: one-factor loadings ``λ_j`` (``None`` when m < 3, where the
            one-factor split is underidentified and the m = 2 copula uses
            ``corr`` directly).
        n_missions: number of paired missions the fit is based on.
    """

    marginals: tuple[float, ...]
    corr: tuple[tuple[float, ...], ...]
    loadings: tuple[float, ...] | None
    n_missions: int


def _clip_marginals(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _PCLIP, 1.0 - _PCLIP)


def _psd_retract_corr(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """Shrink off-diagonals toward 0 until the correlation matrix is PD.

    Monotone-**safe** projection (audit F8). By Slepian's inequality the
    all-success orthant probability is nondecreasing in every latent
    correlation, so scaling all off-diagonals toward independence can only
    *lower* the orthant — never inflate a floor. The previous projection
    (eigenvalue-clip + diagonal renormalisation) could push individual ``ρ_ij``
    *up*, raising the orthant, which is anti-conservative for a lower bound.

    A matrix that is already PD (the common case — a one-factor ``λλᵀ`` plus
    positive diagonal, or any valid tetrachoric fit) is returned unchanged, so
    this is a no-op except on genuinely indefinite inputs.
    """
    R = 0.5 * (R + R.T)  # noqa: N806
    m = R.shape[0]
    np.fill_diagonal(R, 1.0)
    if np.linalg.eigvalsh(R).min() >= 1e-8:
        return R
    off = R - np.eye(m)  # bisection on the largest shrink factor that stays PD
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(np.eye(m) + mid * off).min() >= 1e-8:
            lo = mid
        else:
            hi = mid
    return np.eye(m) + lo * off


def _tetrachoric_success_corr(passes: np.ndarray) -> np.ndarray:
    """Tetrachoric latent correlation of the success indicators (m×m).

    Built from FAILURE cells (``CoFailureTable`` convention); the latent
    correlation is flip-invariant so it equals the success-indicator latent
    correlation. Degenerate pairs (a branch always/never passing) fall back to
    the Pearson correlation, clamped to keep the matrix well-conditioned.
    """
    m = passes.shape[0]
    R = np.eye(m)  # noqa: N806
    fails = 1 - passes
    for i in range(m):
        for j in range(i + 1, m):
            fi, fj = fails[i], fails[j]
            n11 = float(np.sum((fi == 1) & (fj == 1)))
            n10 = float(np.sum((fi == 1) & (fj == 0)))
            n01 = float(np.sum((fi == 0) & (fj == 1)))
            n00 = float(np.sum((fi == 0) & (fj == 0)))
            # Haldane 0.5 continuity correction when any cell is empty. Fixes the
            # nested-cell non-identifiability (audit Q8/C3): with an empty cell,
            # p11 = min(p_a, p_b) exactly, so Φ₂(τ_a,τ_b;ρ)=p11 is FLAT over an
            # interval of ρ and the root-finder returns the most-dependent
            # endpoint ρ→1 — which drives λ→1 and over-certifies. Adding ½ to
            # every cell shifts p11 strictly inside its range, restoring a unique
            # finite root; it also stabilises sparse tables. Untouched otherwise.
            if min(n11, n10, n01, n00) == 0.0:
                n11 += 0.5
                n10 += 0.5
                n01 += 0.5
                n00 += 0.5
            try:
                rho = tetrachoric(CoFailureTable(n11=n11, n10=n10, n01=n01, n00=n00))
            except (DependenceError, ValueError):
                with np.errstate(invalid="ignore"):
                    pear = np.corrcoef(passes[i], passes[j])[0, 1]
                rho = float(pear) if np.isfinite(pear) else 0.0
            rho = float(np.clip(rho, -_RCLIP, _RCLIP))
            R[i, j] = R[j, i] = rho
    return R


def _robust_loadings(corr: np.ndarray) -> np.ndarray:
    """One-factor loadings by bounded least-squares (never raises; audit M2/F13).

    Minimises Σ_{i<j}(R_ij − λ_i λ_j)² subject to ``|λ_j| ≤ 1 − 1e-6``, instead of
    the triad identity λ_i² = R_ij R_ik / R_jk. At m = 3 the triad estimator is
    all-or-nothing — a single out-of-range triad *raises*, forcing the bootstrap
    to silently switch functionals (audit F7) and biasing λ̂ by a data-dependent
    filter (F13). The constrained fit degrades continuously: it returns the best
    rank-one approximation to the off-diagonals even for infeasible triads, and
    always returns. Loadings are identified up to a global sign (irrelevant for
    all-success). Initialised from the principal eigenvector (Eckart–Young
    rank-one), which fixes the relative signs.
    """
    R = np.asarray(corr, dtype=float)  # noqa: N806
    m = R.shape[0]
    iu = np.triu_indices(m, k=1)
    target = R[iu]
    vals, vecs = np.linalg.eigh(R)
    x0 = np.clip(vecs[:, -1] * np.sqrt(max(float(vals[-1]), 1e-6)), -_LCLIP, _LCLIP)

    def _resid(lam: np.ndarray) -> np.ndarray:
        return np.outer(lam, lam)[iu] - target

    sol = least_squares(_resid, x0, bounds=(-_LCLIP, _LCLIP), method="trf")
    return np.clip(sol.x, -_LCLIP, _LCLIP)


def fit_series_factor(passes: object) -> SeriesFactorFit:
    """Fit the shared-factor series model from an m×n pass matrix.

    Args:
        passes: ``m × n`` binary matrix, rows = branches/stages, columns =
            paired missions (entry 1 = that stage met its contract on that
            mission). Missions must be aligned across rows (same column = same
            mission), the paired-outcome requirement of LLD-B §6.1.

    Returns:
        A :class:`SeriesFactorFit`. ``loadings`` is populated only when m ≥ 3.
    """
    a = _as_pass_matrix(passes)
    marg = a.mean(axis=1)
    corr = _tetrachoric_success_corr(a)
    loadings: tuple[float, ...] | None = None
    if a.shape[0] >= 3:
        # Robust bounded-LS fit: always identifies (never raises), so the
        # bootstrap never switches functionals (audit M1/M2/F7).
        loadings = tuple(float(x) for x in _robust_loadings(corr))
    return SeriesFactorFit(
        marginals=tuple(float(x) for x in marg),
        corr=tuple(tuple(float(x) for x in row) for row in corr),
        loadings=loadings,
        n_missions=int(a.shape[1]),
    )


def series_all_success_point(fit: SeriesFactorFit) -> float:
    """Point estimate of series all-success reliability from a fit.

    Dispatches to the exact bivariate copula (m ≤ 2) or the adaptive-quadrature
    one-factor evaluator (:func:`factor_all_success`, m ≥ 3). This is a *point
    estimate* of the model functional — not a guarantee; use
    :func:`series_reliability_floor` for the certified floor.
    """
    m = len(fit.marginals)
    if m <= 2 or fit.loadings is None:
        return gaussian_copula_all_success(fit.marginals, np.array(fit.corr))
    value, _abserr = factor_all_success(fit.marginals, fit.loadings)
    return value


# ---------------------------------------------------------------------------
# Thm B.7 — confidence-region lower bound (the shipped certificate value)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SeriesReliabilityFloor:
    """Bootstrap LCB on the Gaussian-copula MODEL functional — a **diagnostic**.

    .. warning::
        ``model_floor`` is **NOT the certificate value.** It is a bootstrap lower
        confidence bound on the *fitted Gaussian one-factor / copula model
        functional*, not on the true reliability. Under a dependence structure
        indistinguishable from the model on marginals + pairwise cells, its
        coverage of the true reliability collapses to ~0 as ``n`` grows (audit
        F1: the identification gap is O(1), the bootstrap haircut O(n^{-1/2})).
        Ship :func:`~agentassert_abc.certification.certificate.certify` instead —
        the exact Tier-0 Clopper–Pearson floor (executed compositions) or the
        finite-sample copula-agnostic Tier-1 LP floor (extrapolation). This
        object is retained for diagnostics and to exhibit the failure mode.

    Attributes:
        point: dependence-aware point estimate (model functional at the fit).
        model_floor: bootstrap (1 − η_conf) LCB on the **model functional**
            (``min(model_floor_percentile, model_floor_basic) − quad_eps``,
            clipped). A DIAGNOSTIC, not a guarantee (see warning).
        model_floor_percentile: percentile-bootstrap endpoint ``Q_{η}(reps)``.
        model_floor_basic: basic-bootstrap endpoint ``2·point − Q_{1−η}(reps)``.
        independence_product: the naive v1 bound Π p_j (for comparison).
        frechet_lower: assumption-free Thm B.1 lower bound (often 0).
        frechet_upper: assumption-free Thm B.1 upper bound.
        eta_conf: one-sided miscoverage of ``model_floor``.
        n_boot: bootstrap replicate count.
        n_missions: paired missions in the fit.
        m: number of branches/stages.
        quad_eps: numerical error allowance (max quadrature abserr over the point
            fit and all replicates) subtracted per LLD-B §5.2.
        method: ``"bivariate-copula"`` (m ≤ 2) or ``"one-factor-quad"`` (m ≥ 3).
        is_guarantee: always ``False`` — a hard flag that this is a model
            diagnostic, never the shipped guarantee.
    """

    point: float
    model_floor: float
    model_floor_percentile: float
    model_floor_basic: float
    independence_product: float
    frechet_lower: float
    frechet_upper: float
    eta_conf: float
    n_boot: int
    n_missions: int
    m: int
    quad_eps: float
    method: str
    is_guarantee: bool = False


def series_reliability_floor(
    passes: object,
    eta_conf: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> SeriesReliabilityFloor:
    """Certified dependence-aware series reliability floor (LLD-B Thm B.7).

    Implements the confidence-region image: the floor is the ``η_conf`` lower
    quantile of the dependence-aware all-success functional over a mission-level
    (cluster) bootstrap of the paired outcomes, minus the quadrature-error
    allowance ``ε_Q`` (§5.2) and clipped to ``[0, 1]``. On the coverage event
    the true model functional lies at or above ``floor`` with probability
    ≥ 1 − η_conf (Thm B.7, one-sided).

    **DIAGNOSTIC — not the certificate (audit F1/F2).** ``model_floor`` bounds
    the *fitted Gaussian one-factor / copula model* functional, not the true
    reliability, and its coverage of the truth collapses to ~0 under a
    pairwise-indistinguishable misspecification (the identification gap is O(1),
    the bootstrap haircut O(n^{-1/2})). Ship
    :func:`~agentassert_abc.certification.certificate.certify` (exact Tier-0
    Clopper–Pearson, or the finite-sample Tier-1 LP). This function is retained
    to compute the model point estimate and to exhibit the failure mode.

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions).
        eta_conf: one-sided miscoverage (default 0.05 → 95% floor).
        n_boot: bootstrap replicates (default 1000).
        seed: RNG seed for reproducible resampling.

    Returns:
        A :class:`SeriesReliabilityFloor`.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    if n_boot < 1:
        raise DependenceError("n_boot must be >= 1")
    a = _as_pass_matrix(passes)
    m, n = a.shape

    # F5: a saturated stage (p̂ ∈ {0, 1}) gives the nonparametric mission
    # bootstrap ZERO variance for that stage — every resample of a constant row
    # is that same constant — so this model floor cannot price the boundary
    # uncertainty and is anti-conservative (measured coverage 0.56 at nominal
    # 0.95, floor reaching 1.0). Fail loud: the shipped certificate value comes
    # from the exact Tier-0 ``observed_all_success_floor`` (Clopper–Pearson
    # prices the boundary as η^{1/n} < 1) or the finite-sample Tier-2
    # ``slepian_model_floor`` — not from this bootstrap diagnostic.
    sat = [int(j) for j in range(m) if a[j].min() == a[j].max()]
    if sat:
        raise DependenceError(
            f"stages {sat} are saturated (p̂ ∈ {{0, 1}}); the bootstrap model "
            "floor would be anti-conservative. Use observed_all_success_floor "
            "(Tier 0, exact Clopper–Pearson) or slepian_model_floor (Tier 2, "
            "finite-sample) which price the boundary."
        )

    fit = fit_series_factor(a)
    independence = float(np.prod(_clip_marginals(np.array(fit.marginals))))
    fh_lo, fh_hi = frechet_all_success_bounds(fit.marginals)
    method = "bivariate-copula" if m <= 2 else "one-factor-quad"

    # ONE functional for the point and every replicate (audit M1/F7): the exact
    # bivariate Gaussian copula for m ≤ 2, the adaptive-quadrature one-factor
    # evaluator for m ≥ 3. A replicate that cannot be evaluated (a genuine
    # quadrature non-convergence / Fréchet violation) contributes the
    # assumption-free worst case frechet_lo — never a different functional.
    def _eval(fb: SeriesFactorFit) -> tuple[float, float]:
        if m <= 2 or fb.loadings is None:
            return gaussian_copula_all_success(fb.marginals, np.array(fb.corr)), 0.0
        try:
            return factor_all_success(fb.marginals, fb.loadings)
        except DependenceError:
            lo, _hi = frechet_all_success_bounds(fb.marginals)
            return lo, 0.0

    point, point_err = _eval(fit)

    # Mission-level (cluster) bootstrap of the single all-success functional.
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    errs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[b], errs[b] = _eval(fit_series_factor(a[:, idx]))

    # Numerical allowance = max quadrature abserr over the fit AND all replicates
    # (the sup over the confidence region, per §5.2 — not the fit-point value,
    # which understates it near the λ→1 boundary; audit Q7b/F5).
    quad_eps = float(max(point_err, float(errs.max()) if n_boot else 0.0))

    # Two first-order-valid one-sided lower endpoints; ship the min to remove the
    # percentile bootstrap's bias-direction risk (audit Q4b/F9).
    q_percentile = float(np.quantile(reps, eta_conf))
    q_basic = float(2.0 * point - np.quantile(reps, 1.0 - eta_conf))
    floor = float(np.clip(min(q_percentile, q_basic) - quad_eps, 0.0, 1.0))

    return SeriesReliabilityFloor(
        point=point,
        model_floor=floor,
        model_floor_percentile=q_percentile,
        model_floor_basic=q_basic,
        independence_product=independence,
        frechet_lower=fh_lo,
        frechet_upper=fh_hi,
        eta_conf=eta_conf,
        n_boot=n_boot,
        n_missions=n,
        m=m,
        quad_eps=quad_eps,
        method=method,
    )
