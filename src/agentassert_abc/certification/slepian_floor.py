# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tier 2 — the correct finite-sample Thm B.7 floor for the Gaussian model.

Theorem B.7 asks for ``inf_{θ ∈ C} R(θ)`` over a simultaneous confidence region
``C`` of the model parameters. The shipped v1/early-v2 code approximated that
infimum with a *percentile bootstrap*, which (a) is only asymptotically valid and
(b) degenerates at a saturated stage (zero resample variance → floor → 1.0,
audit F5). Two monotonicity facts make the infimum **closed form** instead — no
bootstrap, finite-sample valid:

* ``R`` is nondecreasing in each latent threshold ``a_j`` (hence in each success
  marginal ``p_j``);
* ``R`` is nondecreasing in each off-diagonal latent correlation ``ρ_ij`` — this
  is **Slepian's inequality** for the Gaussian orthant.

So over a rectangle in ``(p, ρ)`` the infimum sits at the componentwise **lower
corner** ``(p_lo, ρ_lo)``. Building the rectangle from Bonferroni Clopper–Pearson
intervals (:func:`~agentassert_abc.certification.observed_floor.bonferroni_cp_cells`)
makes it finite-sample valid, and the boundary (``p̂ = 1``) is priced correctly
because ``p_lo < 1`` always. ``ρ_lo`` is the smallest tetrachoric correlation
consistent with the box (evaluated over its corners; the PSD retraction only
lowers it further, staying conservative — audit F8).

**This is still a bound on the Gaussian one-factor / copula MODEL functional, not
on the true reliability.** Under a dependence structure indistinguishable from
the model on marginals + pairwise cells, the true all-success can lie *below*
this floor (audit F1: the identification gap does not shrink with ``n``). It is a
diagnostic; the shipped guarantee is the assumption-free Tier 0
(:func:`~agentassert_abc.certification.observed_floor.observed_all_success_floor`)
or the copula-agnostic Tier 1 (:mod:`~agentassert_abc.certification.lp_bound`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import multivariate_normal, norm

from agentassert_abc.certification.factor_reliability import (
    _RCLIP,
    _as_pass_matrix,
    gaussian_copula_all_success,
)
from agentassert_abc.certification.observed_floor import bonferroni_cp_cells
from agentassert_abc.exceptions import DependenceError


def _dominated_psd(corr: np.ndarray) -> np.ndarray:  # noqa: N803
    """A PD correlation matrix that is elementwise ≤ ``corr`` off the diagonal.

    Slepian gives a valid *lower* bound only for a correlation matrix that does
    not exceed ``ρ_lo`` in **any** off-diagonal entry (Muse adversarial audit,
    2026-08-11). Scaling toward 0 (the earlier retraction) fails this when an
    entry is negative — ``−0.4 → −0.2`` *raises* it, which by Slepian can *raise*
    the orthant and break the guarantee. This projection instead:

    * returns ``corr`` unchanged if it is already PD (the exact, tightest corner);
    * else drops the positive dependence (sets positive off-diagonals to 0,
      keeps the negatives at ``corr``) — still ``≤ corr`` elementwise, hence
      still a valid Slepian lower bound, merely looser;
    * else (indefinite even with positive dependence removed) raises, because no
      elementwise-dominated PSD correlation exists — the honest signal to fall
      back to Tier 0 / Tier 1.
    """
    m = corr.shape[0]
    if np.linalg.eigvalsh(corr).min() >= 1e-8:
        return corr
    off = ~np.eye(m, dtype=bool)
    base = corr.copy()
    base[off & (corr > 0.0)] = 0.0
    if np.linalg.eigvalsh(base).min() >= 1e-8:
        return base
    raise DependenceError(
        "Slepian floor: the lower-corner correlation is indefinite even with "
        "positive dependence removed; no elementwise-dominated PSD matrix "
        "exists. Use observed_all_success_floor (Tier 0) or the pairwise LP "
        "(Tier 1)."
    )


def _rho_from_failure_cells(qa: float, qb: float, f11: float) -> float:
    """Tetrachoric ρ solving Φ₂(Φ⁻¹(qa), Φ⁻¹(qb); ρ) = f11 (failure scale).

    ``qa, qb`` are failure marginals and ``f11`` the co-failure probability. The
    bivariate normal CDF is strictly increasing in ρ, so a plain bisection
    converges; the value is clamped to the 2×2 Fréchet-feasible range.
    """
    qa = float(np.clip(qa, 1e-9, 1.0 - 1e-9))
    qb = float(np.clip(qb, 1e-9, 1.0 - 1e-9))
    lo_feas = max(0.0, qa + qb - 1.0)
    hi_feas = min(qa, qb)
    if f11 <= lo_feas:
        return -_RCLIP
    if f11 >= hi_feas:
        return _RCLIP
    za, zb = norm.ppf(qa), norm.ppf(qb)
    lo, hi = -_RCLIP, _RCLIP
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cov = [[1.0, mid], [mid, 1.0]]
        val = multivariate_normal.cdf([za, zb], mean=[0.0, 0.0], cov=cov, allow_singular=True)
        if float(val) < f11:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rho_lower_over_box(
    qa_lo: float, qa_hi: float, qb_lo: float, qb_hi: float, f_lo: float, f_hi: float
) -> float:
    """Smallest latent correlation over the CP box for one pair (8 corners).

    The tetrachoric map is monotone in each of ``(q_i, q_j, f_ij)``, so its
    minimum over the rectangle is attained at a corner; all eight are evaluated
    for safety. ``f11`` is clamped to ``min(f11, qa, qb)`` (co-failure cannot
    exceed either failure marginal).
    """
    best = _RCLIP
    for qa in (qa_lo, qa_hi):
        for qb in (qb_lo, qb_hi):
            for f11 in (f_lo, f_hi):
                best = min(best, _rho_from_failure_cells(qa, qb, min(f11, qa, qb)))
    return best


@dataclasses.dataclass(frozen=True, slots=True)
class SlepianModelFloor:
    """Finite-sample Thm B.7 floor for the Gaussian-copula MODEL functional.

    Attributes:
        floor: the model-functional lower bound (orthant at the ``(p_lo, ρ_lo)``
            corner of the Bonferroni-CP box). A valid (1 − η_conf) LCB **on the
            model functional**, not on true reliability.
        observed: empirical all-success rate (reference).
        eta_conf: one-sided family-wise miscoverage.
        m: number of stages.
        n: number of missions.
        rho_lower: the lower-corner latent correlation matrix used (retracted).
        p_lo: lower-corner success marginals used for the thresholds.
        is_model_bound: always ``True`` — a flag that this is NOT a guarantee on
            true reliability (audit F1); do not print it on a certificate alone.
        basis: human-readable basis string.
        assumptions: assumptions the bound rests on (includes the Gaussian copula).
    """

    floor: float
    observed: float
    eta_conf: float
    m: int
    n: int
    rho_lower: tuple[tuple[float, ...], ...]
    p_lo: tuple[float, ...]
    is_model_bound: bool
    basis: str
    assumptions: tuple[str, ...]


def slepian_model_floor(passes: object, eta_conf: float = 0.05) -> SlepianModelFloor:
    """Finite-sample Thm B.7 floor for the Gaussian model functional (Slepian).

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions).
        eta_conf: one-sided family-wise miscoverage (default 0.05).

    Returns:
        A :class:`SlepianModelFloor`. Valid **(1 − 2·η_conf)** LCB on the Gaussian
        one-factor / copula model functional — **not** on true reliability.

    Confidence-level note. This consumes ``2K`` one-sided tails (two-sided
    marginal *and* co-failure boxes) from a Bonferroni family sized ``K``, so the
    family-wise miscoverage is ``2·η_conf``: at the default this is a 90% bound,
    not 95%. Tier 1's :func:`~agentassert_abc.certification.lp_bound.pairwise_cp_box_floor`
    compensates by halving (it passes ``η_conf/2``); Tier 2 deliberately does not,
    because it is a **diagnostic** that is never selected as the certificate
    (``is_model_bound=True``, ``is_guarantee=False``) and because the published
    Tier-2 figures were computed at this level. Do not "fix" this by halving
    ``eta_conf`` without also restating those numbers.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    # NOTE: eta_conf (not eta_conf/2) — see the confidence-level note above.
    box = bonferroni_cp_cells(a, eta_conf)
    p_lo = np.array(box.p_lo)
    p_hi = np.array(box.p_hi)
    observed = float(a.prod(axis=0).mean())
    assumptions = (
        "missions i.i.d. from the certified mission distribution",
        "Gaussian copula for the latent stage scores",
        "bound is on the MODEL functional, NOT true reliability (audit F1)",
    )
    if m == 1:
        return SlepianModelFloor(
            floor=float(p_lo[0]), observed=observed, eta_conf=eta_conf, m=m, n=n,
            rho_lower=((1.0,),), p_lo=(float(p_lo[0]),), is_model_bound=True,
            basis="single stage (Slepian floor vacuous at m=1)", assumptions=assumptions,
        )
    # Failure-marginal boxes: q = 1 − p, so q_lo = 1 − p_hi and q_hi = 1 − p_lo.
    q_lo = 1.0 - p_hi
    q_hi = 1.0 - p_lo
    f_lo = np.array(box.cofailure_lo)
    f_hi = np.array(box.cofailure_hi)
    corr = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            r = _rho_lower_over_box(q_lo[i], q_hi[i], q_lo[j], q_hi[j], f_lo[i, j], f_hi[i, j])
            corr[i, j] = corr[j, i] = float(np.clip(r, -_RCLIP, _RCLIP))
    try:
        corr_used = _dominated_psd(corr)  # elementwise ≤ ρ_lo ⇒ Slepian LB stays valid
        # _assume_psd=True: corr_used is already PD and monotone-safe; do NOT let
        # gaussian_copula_all_success re-apply the unsafe scale-toward-0 retraction
        # (double-projection hazard, Opus 5 audit 2026-08-11).
        floor = float(np.clip(
            gaussian_copula_all_success(p_lo, corr_used, _assume_psd=True), 0.0, 1.0
        ))
        basis = "Thm B.7 exact: Gaussian orthant at the monotone (p_lo, ρ_lo) corner (Slepian)"
    except DependenceError:
        # Grok CRIT#1: an indefinite lower corner admits no elementwise-dominated
        # PSD matrix, so no sound Slepian orthant exists. The only measurable
        # valid lower bound is 0 — the model floor is simply inapplicable here
        # (use Tier 0 / Tier 1). Return 0 rather than crash.
        corr_used = corr
        floor = 0.0
        basis = "Slepian floor degenerate (indefinite lower corner) → conservative 0"
    return SlepianModelFloor(
        floor=floor,
        observed=observed,
        eta_conf=eta_conf,
        m=m,
        n=n,
        rho_lower=tuple(tuple(float(x) for x in row) for row in corr_used),
        p_lo=tuple(float(x) for x in p_lo),
        is_model_bound=True,
        basis=basis,
        assumptions=assumptions,
    )
