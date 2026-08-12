# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Copula-agnostic all-success bounds via linear programming (LLD-B Thm B.8).

The one-factor floor (:mod:`factor_reliability`) is a lower confidence bound on
the fitted *Gaussian-copula* functional. It is only as trustworthy as that
copula: a Gaussian copula has **zero tail dependence**, so it can understate the
clustered co-failure in exactly the tail a safety certificate is bought for
(external audit F10). This module removes the copula assumption entirely.

Given only the per-branch success marginals ``p_i`` and the pairwise co-success
probabilities ``P_ij = Pr(S_i = 1, S_j = 1)`` — both estimable directly from the
paired pass data with **no** distributional assumption — the sharp lower bound on
the all-success probability is a small linear program over the ``2^m`` joint
cells ``x_c = Pr(S = c)``:

.. math::
    \\min\\; x_{1\\dots1}\\quad\\text{s.t.}\\quad
    \\sum_c x_c = 1,\\;
    \\sum_{c:\\,c_i=1} x_c = p_i\\ \\forall i,\\;
    \\sum_{c:\\,c_i=1,c_j=1} x_c = P_{ij}\\ \\forall i<j,\\;
    x_c \\ge 0.

The minimiser is a genuine joint distribution matching every constraint, so the
bound is **attained** (sharp) and **valid under any dependence** consistent with
the observed pairwise structure — no Gaussian, no tail assumption. Maximising the
same cell gives the sharp upper bound. Because this program only *adds* the
pairwise-moment rows to the marginal-only Fréchet program, its feasible set is a
subset of Fréchet's, hence

    frechet_lower ≤ LP_lower ≤ LP_upper ≤ frechet_upper,

and the pairwise rows routinely lift ``LP_lower`` strictly above the (usually
vacuous, = 0) Fréchet lower bound — an informative floor bought with no copula.

The LP minimiser is also the **adversarial data-generating process**: the joint
with the lowest all-success rate that still reproduces the observed marginals and
pairwise co-success. Sampling from it stress-tests the Gaussian floor's coverage
(the misspecification gap the audit asked us to quantify — see
:func:`cell_patterns` for the cell→pattern mapping used to sample it).

Everything here is pure offline statistics; nothing calls a model or the network.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from scipy.optimize import linprog

from agentassert_abc.certification.factor_reliability import (
    _as_pass_matrix,
    frechet_all_success_bounds,
)
from agentassert_abc.certification.observed_floor import (
    bonferroni_cp_cells,
    clopper_pearson_lower,
    clopper_pearson_upper,
)
from agentassert_abc.exceptions import DependenceError

# 2^m joint cells; caps the LP at 4096 variables. Composed pipelines in the
# paper are m = 3; the guard keeps a mistaken large m from exploding memory.
_LP_MAX_M = 12
# Solver feasibility tolerance subtracted from the certified floor so a HiGHS
# result sitting a hair above the true optimum can never inflate the guarantee.
_LP_TOL = 1e-7


def cell_patterns(m: int) -> np.ndarray:
    """All ``2^m`` binary cells as a ``(2^m, m)`` 0/1 matrix.

    Row ``k`` is the big-endian bit expansion of ``k``; row ``2^m − 1`` is the
    all-success cell ``(1, …, 1)``. The adversarial-DGP sampler uses this to map
    a minimiser weight vector back to its success pattern.
    """
    if m < 1:
        raise DependenceError("m must be >= 1")
    idx = np.arange(2**m)
    bits = (idx[:, None] >> np.arange(m - 1, -1, -1)) & 1
    return bits.astype(float)


def _validate_moments(marginals: object, pairwise: object) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate marginals ``p`` (length m) and an ``m×m`` pairwise matrix ``P``."""
    p = np.asarray(marginals, dtype=float).ravel()
    m = p.size
    if m == 0:
        raise DependenceError("marginals must be non-empty")
    if m > _LP_MAX_M:
        raise DependenceError(f"LP bound supports m <= {_LP_MAX_M} branches (got {m})")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise DependenceError("marginals must lie in [0, 1]")
    pw = np.asarray(pairwise, dtype=float)
    if pw.shape != (m, m):
        raise DependenceError(f"pairwise must be {m}×{m} to match {m} marginals")
    return p, pw, m


# ---------------------------------------------------------------------------
# Thm B.8 — sharp copula-agnostic all-success bounds (a small LP)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PairwiseLPBounds:
    """Sharp all-success bounds using only marginal + pairwise moments.

    Attributes:
        lower: sharp lower bound ``min x_{1…1}`` — the copula-agnostic worst case.
        upper: sharp upper bound ``max x_{1…1}``.
        minimizer: the ``2^m`` cell probabilities attaining ``lower`` — a valid
            joint distribution and the adversarial DGP (``None`` on the Fréchet
            fallback). Map cells to patterns with :func:`cell_patterns`.
        feasible: ``True`` if the LP solved; ``False`` if the supplied moments
            were inconsistent and the assumption-free Fréchet sandwich was
            returned instead.
        m: number of branches.
    """

    lower: float
    upper: float
    minimizer: tuple[float, ...] | None
    feasible: bool
    m: int


def pairwise_lp_all_success_bounds(marginals: object, pairwise: object) -> PairwiseLPBounds:
    """Sharp all-success bounds from marginals + pairwise co-success (Thm B.8).

    Solves the two linear programs (minimise / maximise the all-success cell)
    described in the module docstring via ``scipy.optimize.linprog`` (HiGHS).

    Args:
        marginals: per-branch success probabilities ``p_i`` (length m).
        pairwise: ``m × m`` matrix with ``P_ij = Pr(S_i=1, S_j=1)`` off-diagonal
            (diagonal ignored). Must be consistent with ``marginals``
            (``max(0, p_i+p_j−1) ≤ P_ij ≤ min(p_i, p_j)``); inconsistent input
            makes the LP infeasible and triggers the Fréchet fallback.

    Returns:
        A :class:`PairwiseLPBounds`. On any coupling consistent with the given
        moments, the true all-success probability lies in ``[lower, upper]``.
    """
    p, pw, m = _validate_moments(marginals, pairwise)
    fl_lo, fl_hi = frechet_all_success_bounds(p)
    patterns = cell_patterns(m)
    n_cells = patterns.shape[0]
    all_ones = n_cells - 1

    # Equality system: normalisation, then m marginal moments, then C(m,2)
    # pairwise moments. Rows are indicator combinations over the joint cells.
    rows: list[np.ndarray] = [np.ones(n_cells)]
    rhs: list[float] = [1.0]
    for i in range(m):
        rows.append(patterns[:, i])
        rhs.append(float(p[i]))
    for i in range(m):
        for j in range(i + 1, m):
            rows.append(patterns[:, i] * patterns[:, j])
            rhs.append(float(pw[i, j]))
    a_eq = np.array(rows)
    b_eq = np.array(rhs)

    c = np.zeros(n_cells)
    c[all_ones] = 1.0
    res_lo = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")
    res_hi = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")

    if not (res_lo.success and res_hi.success):
        # Inconsistent moments (only reachable from externally supplied p, P —
        # empirical p, P read off one joint are always consistent). Degrade to
        # the assumption-free Fréchet sandwich rather than crash (audit-style
        # "fail to the valid worst case, never a different object").
        return PairwiseLPBounds(
            lower=fl_lo, upper=fl_hi, minimizer=None, feasible=False, m=m
        )

    # Clip only against [0, 1] and the Fréchet upper (LP_lo ≥ frechet_lo and
    # LP_hi ≤ frechet_hi are theorems; we never clip the lower bound *upward*
    # toward frechet_lo, which would make the floor less conservative).
    lower = float(min(max(res_lo.fun, 0.0), fl_hi))
    upper = float(min(max(-res_hi.fun, lower), 1.0))
    minimizer = tuple(float(x) for x in res_lo.x)
    return PairwiseLPBounds(
        lower=lower, upper=upper, minimizer=minimizer, feasible=True, m=m
    )


# ---------------------------------------------------------------------------
# General moment sets: the tunable Tier-1 hierarchy (paper Sec 6.2 / 7.1)
# ---------------------------------------------------------------------------
#
# Marginals + pairwise is the order-<=2 member of a hierarchy. Supplying
# higher-order co-execution moments (triples, quadruples, ...) adds equality
# rows to the SAME joint-cell LP, shrinking the feasible set and weakly raising
# the floor. The moment set is identified by the subsets S of stages for which
# Pr(AND_{i in S} S_i = 1) is supplied; J = |moment set| is the paper's box
# functional count.
#
# HONEST CAVEAT (finite samples): richer moment sets tighten the *identification*
# set but widen every Clopper-Pearson interval, because the Bonferroni family
# grows from J to J'. The net effect on the CERTIFIED floor is therefore an
# empirical question, not a theorem -- extrapolation can lose to the penalty at
# small n. Only the point-moment bounds are monotone by construction.


def moment_subsets(m: int, orders: object = (1, 2)) -> tuple[tuple[int, ...], ...]:
    """Every subset of ``{0..m-1}`` whose size appears in ``orders``.

    ``orders=(1, 2)`` reproduces the classic marginals + pairwise moment set
    (``J = m + C(m,2)``); adding ``3`` appends the triple co-success moments.

    Args:
        m: number of stages.
        orders: iterable of subset sizes to include, e.g. ``(1, 2, 3)``.

    Returns:
        Subsets as sorted index tuples, ordered by size then lexicographically,
        so the ordering is deterministic across calls (the LP row order and the
        Bonferroni family both depend on it).
    """
    if m < 1:
        raise DependenceError("m must be >= 1")
    if m > _LP_MAX_M:
        raise DependenceError(f"LP bound supports m <= {_LP_MAX_M} branches (got {m})")
    try:
        wanted = sorted({int(o) for o in orders})  # type: ignore[union-attr]
    except TypeError as exc:
        raise DependenceError("orders must be an iterable of subset sizes") from exc
    if not wanted:
        raise DependenceError("orders must name at least one subset size")
    if any(o < 1 or o > m for o in wanted):
        raise DependenceError(f"orders must lie in [1, m]; got {wanted} for m={m}")
    return tuple(
        subset
        for order in wanted
        for subset in itertools.combinations(range(m), order)
    )


def _subset_rows(patterns: np.ndarray, subsets: object) -> np.ndarray:
    """Indicator row per subset: 1 on cells where every stage in the subset succeeds."""
    rows = []
    for subset in subsets:  # type: ignore[union-attr]
        col = np.ones(patterns.shape[0])
        for i in subset:
            col = col * patterns[:, i]
        rows.append(col)
    return np.array(rows)


def empirical_subset_moments(
    passes: object, subsets: object
) -> tuple[float, ...]:
    """``Pr(all stages in S succeed)`` for each subset, read off one joint.

    Every moment comes from the same empirical joint, so the values are mutually
    consistent by construction and the point-moment LP is always feasible (the
    empirical joint is itself a feasible point). No smoothing — see
    :func:`empirical_moments`.
    """
    a = _as_pass_matrix(passes)
    out: list[float] = []
    for subset in subsets:  # type: ignore[union-attr]
        col = np.ones(a.shape[1])
        for i in subset:
            col = col * a[i]
        out.append(float(col.mean()))
    return tuple(out)


@dataclasses.dataclass(frozen=True, slots=True)
class MomentLPBounds:
    """Sharp all-success bounds over an arbitrary supplied moment set.

    Attributes:
        lower: sharp minimum all-success probability over the moment set.
        upper: sharp maximum.
        minimizer: the adversarial joint attaining ``lower`` (cell order matches
            :func:`cell_patterns`), or ``None`` when infeasible.
        feasible: ``False`` if the supplied moments admit no joint law.
        m: number of stages.
        subsets: the moment set used, in LP row order.
        j_functionals: ``J = len(subsets)``.
    """

    lower: float
    upper: float
    minimizer: tuple[float, ...] | None
    feasible: bool
    m: int
    subsets: tuple[tuple[int, ...], ...]
    j_functionals: int


def moment_lp_all_success_bounds(
    m: int, subsets: object, values: object
) -> MomentLPBounds:
    """Sharp all-success bounds from an **arbitrary** moment set (general Thm B.8).

    Generalises :func:`pairwise_lp_all_success_bounds` from the fixed
    {marginals, pairwise} set to any collection of co-execution moments. Each
    subset ``S`` contributes one equality row
    ``sum_{c : c_i = 1 for all i in S} x_c = value_S``.

    Because extra rows only *remove* feasible joints, a superset moment family
    weakly raises ``lower`` and weakly lowers ``upper`` — the hierarchy is
    monotone at the point-moment level.

    Args:
        m: number of stages.
        subsets: moment subsets, e.g. from :func:`moment_subsets`.
        values: matching ``Pr(AND_{i in S} S_i = 1)`` per subset.

    Returns:
        A :class:`MomentLPBounds`; on infeasible input it degrades to the
        assumption-free Fréchet sandwich rather than raising.
    """
    subs = tuple(tuple(int(i) for i in s) for s in subsets)  # type: ignore[union-attr]
    vals = np.asarray(list(values), dtype=float)  # type: ignore[call-overload]
    if len(subs) != vals.size:
        raise DependenceError(
            f"subsets and values differ in length: {len(subs)} vs {vals.size}"
        )
    if not subs:
        raise DependenceError("at least one moment subset is required")
    for s in subs:
        if not s:
            raise DependenceError("moment subsets must be non-empty")
        if any(i < 0 or i >= m for i in s):
            raise DependenceError(f"subset {s} has an index outside [0, {m - 1}]")
    if np.any(vals < 0.0) or np.any(vals > 1.0):
        raise DependenceError("moment values must lie in [0, 1]")

    patterns = cell_patterns(m)
    n_cells = patterns.shape[0]
    # Marginals, where supplied, pin the Fréchet fallback; otherwise stay vacuous.
    singles = {s[0]: v for s, v in zip(subs, vals, strict=True) if len(s) == 1}
    if len(singles) == m:
        fl_lo, fl_hi = frechet_all_success_bounds(
            np.array([singles[i] for i in range(m)])
        )
    else:
        fl_lo, fl_hi = 0.0, 1.0

    a_eq = np.vstack([np.ones(n_cells), _subset_rows(patterns, subs)])
    b_eq = np.concatenate([[1.0], vals])
    c = np.zeros(n_cells)
    c[n_cells - 1] = 1.0
    res_lo = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")
    res_hi = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")

    if not (res_lo.success and res_hi.success):
        return MomentLPBounds(
            lower=fl_lo, upper=fl_hi, minimizer=None, feasible=False, m=m,
            subsets=subs, j_functionals=len(subs),
        )
    lower = float(min(max(res_lo.fun, 0.0), fl_hi))
    upper = float(min(max(-res_hi.fun, lower), 1.0))
    return MomentLPBounds(
        lower=lower, upper=upper,
        minimizer=tuple(float(x) for x in res_lo.x),
        feasible=True, m=m, subsets=subs, j_functionals=len(subs),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class MomentBoxFloor:
    """Certified finite-sample floor over a general Bonferroni-CP moment box.

    Same guarantee as :class:`PairwiseRobustFloor` — an exact ``(1 − η_conf)``
    LCB assuming only i.i.d. missions, no copula — but over an arbitrary moment
    family. ``J`` functionals give ``2J`` one-sided tails, each at
    ``η_conf / (2J)``, so the box holds jointly with probability ``≥ 1 − η_conf``.
    """

    floor: float
    upper: float
    observed: float
    eta_conf: float
    m: int
    n: int
    orders: tuple[int, ...]
    j_functionals: int
    j_budget: int
    feasible: bool
    basis: str
    assumptions: tuple[str, ...]


def moment_cp_box_floor(
    passes: object,
    eta_conf: float = 0.05,
    orders: object = (1, 2),
    budget_orders: object = None,
) -> MomentBoxFloor:
    """Certified copula-agnostic floor over an arbitrary moment set + CP box.

    The tunable Tier-1 certificate: ``orders=(1,2)`` is exactly
    :func:`pairwise_cp_box_floor`; ``orders=(1,2,3)`` adds triple co-success
    moments, the extrapolation the paper's moment hierarchy describes.

    Soundness: on the event that all Clopper–Pearson intervals cover
    (probability ``≥ 1 − η_conf`` by the union bound over the one-sided tails),
    the true joint law is LP-feasible, so the LP minimum is ``≤`` the true
    all-success probability.

    **Two Bonferroni allocations** (both in the paper):

    * *used-set* (default, ``budget_orders=None``) — spend ``η_conf`` over the
      ``J`` moments actually supplied, each tail at ``η_conf/(2J)``. Tightest
      for a given moment set, but adding a moment widens every interval, so
      monotonicity in the moment set is empirical, not guaranteed.
    * *pre-allocated* (``budget_orders`` given) — spend the budget over the
      larger family ``J_max`` induced by ``budget_orders`` while constraining
      only the ``orders`` moments. Every interval then has a width that does not
      depend on how many moments are used, so enriching the moment set can only
      add constraints and the certified floor is monotone **by construction**,
      at a small stated width cost.

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages).
        eta_conf: family-wise one-sided miscoverage (default 0.05).
        orders: subset sizes forming the moment set actually constrained.
        budget_orders: if given, subset sizes defining the Bonferroni budget
            family ``J_max``. Must be a superset family of ``orders``.

    Returns:
        A :class:`MomentBoxFloor`.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    subs = moment_subsets(m, orders)
    j = len(subs)
    if budget_orders is None:
        j_budget = j
    else:
        budget_subs = set(moment_subsets(m, budget_orders))
        if not set(subs) <= budget_subs:
            raise DependenceError(
                "budget_orders must induce a superset of the constrained moment "
                f"family (got {len(budget_subs)} budget moments vs {j} used)"
            )
        j_budget = len(budget_subs)
    # Two-sided box: 2J one-sided tails each at η/(2J) ⇒ family-wise η.
    e = eta_conf / (2.0 * j_budget)

    lo = np.empty(j)
    hi = np.empty(j)
    for idx, subset in enumerate(subs):
        col = np.ones(n)
        for i in subset:
            col = col * a[i]
        k = int(col.sum())
        lo[idx] = clopper_pearson_lower(k, n, e)
        hi[idx] = clopper_pearson_upper(k, n, e)

    patterns = cell_patterns(m)
    rows = _subset_rows(patterns, subs)
    n_cells = patterns.shape[0]
    c = np.zeros(n_cells)
    c[n_cells - 1] = 1.0
    a_ub = np.vstack([rows, -rows])
    b_ub = np.concatenate([hi, -lo])
    a_eq = np.ones((1, n_cells))
    b_eq = np.array([1.0])

    def _solve(maximize: bool) -> float | None:
        res = linprog(
            -c if maximize else c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=(0.0, 1.0), method="highs",
        )
        if not res.success:
            return None
        return -float(res.fun) if maximize else float(res.fun)

    lp_lo = _solve(maximize=False)
    lp_hi = _solve(maximize=True)
    feasible = lp_lo is not None
    floor = 0.0 if lp_lo is None else float(np.clip(lp_lo - _LP_TOL, 0.0, 1.0))
    upper = 1.0 if lp_hi is None else float(np.clip(lp_hi, floor, 1.0))
    return MomentBoxFloor(
        floor=floor,
        upper=upper,
        observed=float(a.prod(axis=0).mean()),
        eta_conf=eta_conf,
        m=m,
        n=n,
        orders=tuple(sorted({int(o) for o in orders})),  # type: ignore[union-attr]
        j_functionals=j,
        j_budget=j_budget,
        feasible=feasible,
        basis=(
            "sharp LP over the Bonferroni-CP moment box on subsets of order "
            f"{tuple(sorted({int(o) for o in orders}))}"  # type: ignore[union-attr]
            + (f", Bonferroni budget pre-allocated over J_max={j_budget}"
               if j_budget != j else "")
        ),
        assumptions=(
            "missions i.i.d. from the certified mission distribution",
            "NO copula / factor / tail-dependence assumption",
        ),
    )


# ---------------------------------------------------------------------------
# Series certified floor from paired pass data (copula-agnostic)
# ---------------------------------------------------------------------------


def empirical_moments(passes: object) -> tuple[np.ndarray, np.ndarray]:
    """Marginals ``p_i`` and pairwise co-success ``P_ij`` from an m×n pass matrix.

    Both are read off the **same** empirical joint, so they are always mutually
    consistent — the LP is therefore guaranteed feasible (the empirical joint is
    itself a feasible point). No continuity correction is applied: unlike the
    tetrachoric fit, the LP has no ρ→±1 pole to regularise, and smoothing ``p_i``
    and ``P_ij`` from different tables would desynchronise them and induce
    spurious infeasibility.
    """
    a = _as_pass_matrix(passes)
    m = a.shape[0]
    p = a.mean(axis=1)
    pw = np.eye(m)
    for i in range(m):
        pw[i, i] = p[i]
        for j in range(i + 1, m):
            pw[i, j] = pw[j, i] = float(np.mean(a[i] * a[j]))
    return p, pw


@dataclasses.dataclass(frozen=True, slots=True)
class PairwiseRobustFloor:
    """Finite-sample copula-agnostic all-success floor over a CP moment box.

    **The certified Tier-1 floor.** It is the minimum all-success probability over
    *every* joint law whose success marginals and pairwise co-success lie in a
    simultaneous Bonferroni Clopper–Pearson box built at level ``η_conf / 2`` (so
    the ``2K`` one-sided tails, ``K = m + C(m,2)``, are each at ``η_conf/(2K)`` and
    hold jointly with probability ``≥ 1 − η_conf``). ``floor`` is therefore an
    **exact** (1 − η_conf) LCB. (Building the box at ``η`` would be only 1 − 2η —
    audit, Opus 5 2026-08-11 — hence the ``/2``.) Valid under **any** dependence
    (no copula, no factor model, no tail-dependence assumption) **and**
    finite-sample — no bootstrap. It supersedes the earlier bootstrap LP floor:
    it removes the percentile bootstrap's validity caveat (audit F9) and prices the
    ``p̂ = 1`` boundary through Clopper–Pearson (audit F5, ``p_lo < 1``).

    Use this when the composition was **not** executed end-to-end (only per-stage
    / pairwise data available). When it *was* executed, the exact Tier-0
    :func:`~agentassert_abc.certification.observed_floor.observed_all_success_floor`
    is tighter and should be the certificate.

    Attributes:
        floor: certificate value — (1 − η_conf) LCB on the all-success
            probability, assuming only i.i.d. missions.
        upper: matching upper bound over the box (diagnostic).
        observed: empirical all-success rate (reference).
        eta_conf: family-wise one-sided miscoverage.
        m: number of stages.
        n: number of missions.
        k_functionals: Bonferroni family size ``m + C(m, 2)``.
        feasible: ``True`` if the LP solved (always, for empirical inputs).
        basis: human-readable basis string.
        assumptions: assumptions the guarantee rests on.
    """

    floor: float
    upper: float
    observed: float
    eta_conf: float
    m: int
    n: int
    k_functionals: int
    feasible: bool
    basis: str
    assumptions: tuple[str, ...]


def _box_lp_all_success(
    patterns: np.ndarray,
    p_lo: np.ndarray,
    p_hi: np.ndarray,
    s_lo: np.ndarray,
    s_hi: np.ndarray,
    maximize: bool = False,
) -> float | None:
    """min / max of the all-success cell over the CP moment box (``None`` = infeasible).

    Same joint-cell LP as :func:`pairwise_lp_all_success_bounds` but with the
    marginal and pairwise moments constrained to *intervals* (the confidence box)
    rather than point equalities — the finite-sample copula-agnostic bound.
    """
    m = patterns.shape[1]
    n_cells = patterns.shape[0]
    c = np.zeros(n_cells)
    c[n_cells - 1] = 1.0
    a_eq = np.ones((1, n_cells))
    b_eq = np.array([1.0])
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for j in range(m):
        col = patterns[:, j]
        a_ub.append(col)
        b_ub.append(float(p_hi[j]))
        a_ub.append(-col)
        b_ub.append(-float(p_lo[j]))
    for i in range(m):
        for j in range(i + 1, m):
            col = patterns[:, i] * patterns[:, j]
            a_ub.append(col)
            b_ub.append(float(s_hi[i, j]))
            a_ub.append(-col)
            b_ub.append(-float(s_lo[i, j]))
    obj = -c if maximize else c
    res = linprog(
        obj, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
        A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs",
    )
    if not res.success:
        return None
    return -float(res.fun) if maximize else float(res.fun)


def pairwise_cp_box_floor(passes: object, eta_conf: float = 0.05) -> PairwiseRobustFloor:
    """Certified finite-sample copula-agnostic all-success floor (Thm B.8 + CP box).

    Minimises the all-success cell over every joint whose marginals and pairwise
    co-success lie in a simultaneous Bonferroni Clopper–Pearson **box** built at
    level ``η_conf / 2``: each of the ``K = m + C(m,2)`` functionals gets a
    two-sided interval whose two one-sided tails are each at ``η_conf / (2K)``, so
    the ``2K`` one-sided events hold simultaneously with probability
    ``≥ 1 − η_conf`` by the union bound. On that event the true joint is
    LP-feasible, so the LP minimum ``≤`` the true all-success — an **exact**
    (1 − η_conf) lower confidence bound assuming **only** i.i.d. missions (no copula).

    Audit note (Opus 5, 2026-08-11): building the box at level ``η`` (each tail
    ``η/K``) gives family-wise miscoverage ≤ ``2η`` — it proves a (1 − 2η) floor,
    not (1 − η). Halving to ``η/2`` restores the honest (1 − η) level while keeping
    the (binding, materially tightening) upper constraints — 12–18 points tighter
    on the real quorum arms than dropping them. Practical ceiling: the LP has
    ``2^m`` cells and ``K`` Bonferroni functionals, so Tier 1 is informative to
    about ``m ≈ 5–6`` and returns ~0 by ``m = 12`` (documented, not a bug).

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions).
        eta_conf: family-wise one-sided miscoverage (default 0.05 → 95% floor).

    Returns:
        A :class:`PairwiseRobustFloor`.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    if m > _LP_MAX_M:
        raise DependenceError(f"LP bound supports m <= {_LP_MAX_M} branches (got {m})")
    # Two-sided box at η/2: 2K one-sided tails each at η/(2K) ⇒ family-wise η.
    box = bonferroni_cp_cells(a, eta_conf / 2.0)
    p_lo = np.array(box.p_lo)
    p_hi = np.array(box.p_hi)
    s_lo = np.array(box.cosuccess_lo)
    s_hi = np.array(box.cosuccess_hi)
    patterns = cell_patterns(m)
    observed = float(a.prod(axis=0).mean())

    lo = _box_lp_all_success(patterns, p_lo, p_hi, s_lo, s_hi, maximize=False)
    hi = _box_lp_all_success(patterns, p_lo, p_hi, s_lo, s_hi, maximize=True)
    feasible = lo is not None
    floor = 0.0 if lo is None else float(np.clip(lo - _LP_TOL, 0.0, 1.0))
    upper = 1.0 if hi is None else float(np.clip(hi, floor, 1.0))
    return PairwiseRobustFloor(
        floor=floor,
        upper=upper,
        observed=observed,
        eta_conf=eta_conf,
        m=m,
        n=n,
        k_functionals=box.k_functionals,
        feasible=feasible,
        basis="sharp LP over the Bonferroni-CP pairwise-consistent moment box",
        assumptions=(
            "missions i.i.d. from the certified mission distribution",
            "NO copula / factor / tail-dependence assumption",
        ),
    )
