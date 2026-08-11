# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tier 0 certificate floor — exact Clopper–Pearson on observed all-success.

The **primary** reliability floor for a composition that was executed
end-to-end — free of any *inter-stage dependence* assumption (no copula, no
factor model), though **not** assumption-free: it rests on i.i.d. missions and a
fixed, pre-specified scoring rule (see :class:`ObservedFloor`). When every stage
is scored on the *same* missions, the
all-success event ``Y_i = Π_j passes[j, i]`` is **directly observed**, so
``Σ_i Y_i ~ Binomial(n, R)`` with ``R`` the true all-success reliability, and a
one-sided Clopper–Pearson lower bound is an **exact, finite-sample (1 − η) lower
confidence bound on R** under the single assumption that missions are i.i.d.
draws from the certified mission distribution. No copula, no factor model, no
bootstrap, no asymptotics — arbitrary inter-stage dependence is already realised
in the joint outcomes, so nothing needs to be modelled.

This is the value a certificate should print. It dominates the Gaussian
one-factor *model floor* (:mod:`factor_reliability`): at the paper's real sample
sizes it matches it on tightness, and it stays valid under **any** dependence —
including the pairwise-indistinguishable misspecification under which the model
floor's coverage of the true reliability collapses to zero (LLD-B audit F1). It
also prices the boundary correctly: an all-pass sample (``k = n``) yields
``η^{1/n} < 1``, never a spurious ``1.0`` (audit F5).

Use the copula-agnostic pairwise LP floor (:mod:`lp_bound`) instead only when the
composition was **never run end-to-end**, so the joint all-success rate was not
observed and must be bounded from per-stage / pairwise data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import beta

from agentassert_abc.certification.factor_reliability import _as_pass_matrix
from agentassert_abc.exceptions import DependenceError


def clopper_pearson_lower(k: int, n: int, eta: float = 0.05) -> float:
    """Exact one-sided (1 − η) Clopper–Pearson **lower** bound on a Binomial p.

    Beta-quantile form ``L = B^{-1}(η; k, n − k + 1)`` (``L = 0`` when ``k = 0``).
    Coverage is ``≥ 1 − η`` for **every** true ``p`` — the discreteness of the
    binomial makes it conservative, never anti-conservative.
    """
    if n < 1:
        raise DependenceError("n must be >= 1")
    if not 0 <= k <= n:
        raise DependenceError("need 0 <= k <= n")
    if not 0.0 < eta < 1.0:
        raise DependenceError("eta must be in (0, 1)")
    if k == 0:
        return 0.0
    return float(beta.ppf(eta, k, n - k + 1))


def clopper_pearson_upper(k: int, n: int, eta: float = 0.05) -> float:
    """Exact one-sided (1 − η) Clopper–Pearson **upper** bound (diagnostic)."""
    if n < 1:
        raise DependenceError("n must be >= 1")
    if not 0 <= k <= n:
        raise DependenceError("need 0 <= k <= n")
    if not 0.0 < eta < 1.0:
        raise DependenceError("eta must be in (0, 1)")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - eta, k + 1, n - k))


@dataclasses.dataclass(frozen=True, slots=True)
class ObservedFloor:
    """Tier 0 exact all-success floor from directly observed joint outcomes.

    Attributes:
        floor: **the certificate value** — exact (1 − η) Clopper–Pearson lower
            bound on the true all-success reliability ``R``.
        observed: empirical all-success rate ``k / n``.
        upper: (1 − η) Clopper–Pearson upper bound (diagnostic, one-sided).
        k: number of missions on which every stage met its contract.
        n: number of missions.
        eta_conf: one-sided miscoverage (``floor`` is a 1 − η_conf LCB).
        basis: human-readable basis string for the certificate.
        assumptions: the (single) assumption the guarantee rests on.
    """

    floor: float
    observed: float
    upper: float
    k: int
    n: int
    eta_conf: float
    basis: str
    assumptions: tuple[str, ...]


def observed_all_success_floor(passes: object, eta_conf: float = 0.05) -> ObservedFloor:
    """Exact (1 − η_conf) LCB on P(all stages meet contract), assumption-free.

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions);
            entry 1 = that stage met its contract on that mission. Every stage
            must be scored on the same missions (aligned columns).
        eta_conf: one-sided miscoverage (default 0.05 → 95% floor).

    Returns:
        An :class:`ObservedFloor`. ``floor`` is exact and finite-sample valid
        under the sole assumption that missions are i.i.d. draws from the
        certified mission distribution.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    y = a.all(axis=0).astype(int)
    k, n = int(y.sum()), int(y.size)
    return ObservedFloor(
        floor=clopper_pearson_lower(k, n, eta_conf),
        observed=k / n,
        upper=clopper_pearson_upper(k, n, eta_conf),
        k=k,
        n=n,
        eta_conf=eta_conf,
        basis="exact Clopper–Pearson lower bound on the observed all-success rate",
        # The Beta quantile is exact; the ONLY substantive risk is these
        # assumptions (Muse adversarial audit, 2026-08-11): the certificate is
        # void if missions are not i.i.d. draws from the deployment distribution
        # (drift, batch effects, cross-mission correlation), if a stage was
        # missing/unscored on some mission, or if Y_i is measured with error.
        assumptions=(
            "missions i.i.d. from the certified mission distribution",
            "every stage scored on the same missions (aligned columns)",
            "no missing/unscored stage on any counted mission",
            "the all-success indicator Y_i observed without error",
            "scoring thresholds pre-specified, not tuned on these missions",
        ),
    )


def observed_atleast_k_floor(
    passes: object, k: int, eta_conf: float = 0.05
) -> ObservedFloor:
    """Exact (1 − η_conf) CP lower bound on P(≥ k stages meet contract).

    The **quorum functional**. For a k-of-m voter the reliability is
    ``R_k = Pr(Σ_j S_j ≥ k)``, and ``Y_i = 1[Σ_j passes[j, i] ≥ k]`` is directly
    observed, so ``Σ_i Y_i ~ Binomial(n, R_k)`` and Clopper–Pearson is exact and
    assumption-light — exactly the Tier-0 argument, of which all-success is the
    special case ``k = m``. Use this to certify a k-of-m quorum **system**: the
    all-success (``k = m``) floor is a *valid but looser* bound on it (since
    ``{all succeed} ⊆ {≥ k succeed}``), so certifying a 2-of-3 quorum with the
    3-of-3 floor leaves reliability on the table (audit, Opus 5 2026-08-11).

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions).
        k: quorum threshold, ``1 ≤ k ≤ m``.
        eta_conf: one-sided miscoverage (default 0.05 → 95% floor).

    Returns:
        An :class:`ObservedFloor` (``k`` field = the observed success **count**;
        the ``basis`` string records the ``≥ k of m`` threshold).
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    if not 1 <= k <= m:
        raise DependenceError(f"need 1 <= k <= m (got k={k}, m={m})")
    y = (a.sum(axis=0) >= k).astype(int)
    kc = int(y.sum())
    return ObservedFloor(
        floor=clopper_pearson_lower(kc, n, eta_conf),
        observed=kc / n,
        upper=clopper_pearson_upper(kc, n, eta_conf),
        k=kc,
        n=n,
        eta_conf=eta_conf,
        basis=f"exact Clopper–Pearson lower bound on P(>= {k} of {m} stages meet contract)",
        assumptions=(
            "missions i.i.d. from the certified mission distribution",
            "every stage scored on the same missions (aligned columns)",
            "no missing/unscored stage on any counted mission",
            "the quorum indicator Y_i observed without error",
            "scoring thresholds pre-specified, not tuned on these missions",
        ),
    )


def design_effect_adjusted_floor(
    all_success: object,
    eta_conf: float = 0.05,
    block_lengths: tuple[int, ...] = (5, 10, 25),
    n_boot: int = 1000,
    seed: int = 0,
) -> ObservedFloor:
    """Design-effect-corrected Tier-0 floor for a **time-ordered** all-success series.

    Tier 0 is exact *given i.i.d. missions* — the sole load-bearing assumption, and
    the one Clopper–Pearson cannot itself test. If the ordered series carries
    serial or batch structure (drift over the collection window, shared prompts),
    the effective sample is smaller than the nominal ``n`` and the plain CP floor is
    optimistic. This estimates the design effect
    ``DEFF = Var_block(Ȳ) / Var_iid(Ȳ)`` by a circular **moving-block bootstrap**
    (taking the max over ``block_lengths`` — the conservative short-range estimate),
    sets ``n_eff = n / max(1, DEFF)``, and returns the exact CP lower bound at
    ``n_eff`` — a floor robust to the observed dependence.

    Supply ``all_success`` as the length-n binary indicator **in collection order**
    (e.g. sorted by mission timestamp). On our arms DEFF ≈ 1.16–1.50 (n_eff 67–86%
    of nominal) yet the floor moves ≤ 0.3 points — the guarantee is robust.

    Args:
        all_success: 1-D binary all-success indicator, in collection order.
        eta_conf: one-sided miscoverage (default 0.05).
        block_lengths: moving-block lengths for the DEFF estimate.
        n_boot: block-bootstrap replicates. seed: RNG seed.

    Returns:
        An :class:`ObservedFloor` at the effective sample size (``n`` field = n_eff).
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    y = np.asarray(all_success)
    if y.ndim != 1 or y.size < 1:
        raise DependenceError("all_success must be a non-empty 1-D series")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise DependenceError("all_success must be binary (0/1)")
    n = int(y.size)
    ybar = float(y.mean())
    v_iid = ybar * (1.0 - ybar) / n
    deff = 1.0
    if v_iid > 0.0:
        rng = np.random.default_rng(seed)
        for length in block_lengths:
            nb = int(np.ceil(n / length))
            means = np.empty(n_boot)
            offs = np.arange(length)
            for t in range(n_boot):
                idx = ((rng.integers(0, n, size=nb)[:, None] + offs[None, :]).ravel()[:n]) % n
                means[t] = y[idx].mean()
            deff = max(deff, float(np.var(means, ddof=1) / v_iid))
    n_eff = n / deff
    n_eff_int = max(1, int(round(n_eff)))
    k_eff = int(round(ybar * n_eff_int))
    return ObservedFloor(
        floor=clopper_pearson_lower(k_eff, n_eff_int, eta_conf),
        observed=ybar,
        upper=clopper_pearson_upper(k_eff, n_eff_int, eta_conf),
        k=k_eff,
        n=n_eff_int,
        eta_conf=eta_conf,
        basis=(f"design-effect-adjusted Clopper–Pearson (DEFF={deff:.2f}, "
               f"n_eff={n_eff_int} of {n})"),
        assumptions=(
            "missions from the certified distribution (serial/batch dependence "
            "priced via a moving-block bootstrap, not assumed away)",
            "the observed collection order reflects the deployment mission stream",
            "scoring thresholds pre-specified, not tuned on these missions",
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CpCellBox:
    """Simultaneous (1 − η) Clopper–Pearson intervals on the first two moments.

    A Bonferroni family over ``K = m + C(m, 2)`` functionals — ``m`` success
    marginals and ``C(m, 2)`` pairwise cells — each at level ``η / K``, so the
    whole box holds simultaneously with probability ``≥ 1 − η``. This is the
    finite-sample confidence region that the copula-agnostic LP floor
    (:mod:`lp_bound`) and the Slepian model floor (:mod:`slepian_floor`) both
    minimise over — replacing the asymptotic bootstrap.

    All arrays are length ``m`` / ``m × m`` (pairwise symmetric, diagonal
    unused). ``cosuccess`` cells are ``Pr(S_i=1, S_j=1)``; ``cofailure`` cells
    are ``Pr(S_i=0, S_j=0)``.
    """

    p_lo: tuple[float, ...]
    p_hi: tuple[float, ...]
    cosuccess_lo: tuple[tuple[float, ...], ...]
    cosuccess_hi: tuple[tuple[float, ...], ...]
    cofailure_lo: tuple[tuple[float, ...], ...]
    cofailure_hi: tuple[tuple[float, ...], ...]
    k_functionals: int
    eta_per: float


def bonferroni_cp_cells(passes: object, eta: float = 0.05) -> CpCellBox:
    """Simultaneous CP box on marginals + pairwise co-success/co-failure cells.

    Args:
        passes: ``m × n`` binary pass matrix.
        eta: family-wise one-sided miscoverage (split Bonferroni across the
            ``K = m + C(m, 2)`` functionals).

    Returns:
        A :class:`CpCellBox` valid simultaneously with probability ``≥ 1 − η``.
    """
    if not 0.0 < eta < 1.0:
        raise DependenceError("eta must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    k_func = m + m * (m - 1) // 2
    e = eta / k_func
    p_lo = np.empty(m)
    p_hi = np.empty(m)
    for j in range(m):
        k = int(a[j].sum())
        p_lo[j] = clopper_pearson_lower(k, n, e)
        p_hi[j] = clopper_pearson_upper(k, n, e)
    s_lo = np.zeros((m, m))
    s_hi = np.zeros((m, m))
    f_lo = np.zeros((m, m))
    f_hi = np.zeros((m, m))
    fails = 1 - a
    for i in range(m):
        for j in range(i + 1, m):
            ks = int(np.sum((a[i] == 1) & (a[j] == 1)))
            kf = int(np.sum((fails[i] == 1) & (fails[j] == 1)))
            s_lo[i, j] = s_lo[j, i] = clopper_pearson_lower(ks, n, e)
            s_hi[i, j] = s_hi[j, i] = clopper_pearson_upper(ks, n, e)
            f_lo[i, j] = f_lo[j, i] = clopper_pearson_lower(kf, n, e)
            f_hi[i, j] = f_hi[j, i] = clopper_pearson_upper(kf, n, e)
    _t = lambda mtx: tuple(tuple(float(x) for x in row) for row in mtx)  # noqa: E731
    return CpCellBox(
        p_lo=tuple(float(x) for x in p_lo),
        p_hi=tuple(float(x) for x in p_hi),
        cosuccess_lo=_t(s_lo),
        cosuccess_hi=_t(s_hi),
        cofailure_lo=_t(f_lo),
        cofailure_hi=_t(f_hi),
        k_functionals=k_func,
        eta_per=e,
    )
