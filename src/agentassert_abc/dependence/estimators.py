# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Co-failure dependence estimators (LLD-B Sec 4/6, LLD-E Sec 8).

Given paired binary *failure* outcomes for two agents (1 = contract failed),
these estimate how strongly their failures co-occur:

* :func:`kendall_tau_a`  — binary Kendall tau_a = 2(p11*p00 - p10*p01).
* :func:`phi_coefficient` — the phi / point Pearson correlation (= tau_b binary).
* :func:`tetrachoric`     — latent bivariate-normal correlation behind the 2x2.
* :func:`one_factor_loadings` — shared-factor loadings from a correlation matrix
  (identified only with >= 3 indicators).
* :func:`tau_a_min_samples` — the distribution-free sample floor for a tau_a CI.

Design notes:
* Failure convention: ``1`` means the contract failed. Positive dependence
  therefore means "fail together".
* All estimators are immutable and side-effect free; inputs are never mutated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from agentassert_abc.exceptions import DependenceError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CoFailureTable",
    "kendall_tau_a",
    "one_factor_loadings",
    "phi_coefficient",
    "tau_a_min_samples",
    "tetrachoric",
]


@dataclass(frozen=True, slots=True)
class CoFailureTable:
    """Immutable 2x2 contingency of two agents' failure indicators.

    Cells count missions by ``(a_failed, b_failed)``:
    ``n11`` both failed, ``n10`` only a, ``n01`` only b, ``n00`` neither.
    """

    n11: int
    n10: int
    n01: int
    n00: int

    def __post_init__(self) -> None:
        for name in ("n11", "n10", "n01", "n00"):
            if getattr(self, name) < 0:
                raise DependenceError(f"cell {name} must be non-negative")
        if self.n == 0:
            raise DependenceError("contingency table is empty")

    @classmethod
    def from_pairs(
        cls, fail_a: Sequence[int | bool], fail_b: Sequence[int | bool]
    ) -> CoFailureTable:
        """Build the table from two equal-length paired failure sequences."""
        if len(fail_a) != len(fail_b):
            raise DependenceError(
                f"paired sequences differ in length: {len(fail_a)} vs {len(fail_b)}"
            )
        n11 = n10 = n01 = n00 = 0
        for a, b in zip(fail_a, fail_b, strict=True):
            af, bf = bool(a), bool(b)
            if af and bf:
                n11 += 1
            elif af:
                n10 += 1
            elif bf:
                n01 += 1
            else:
                n00 += 1
        return cls(n11=n11, n10=n10, n01=n01, n00=n00)

    @property
    def n(self) -> int:
        """Total mission count."""
        return self.n11 + self.n10 + self.n01 + self.n00

    @property
    def p_a(self) -> float:
        """Marginal failure rate of agent a."""
        return (self.n11 + self.n10) / self.n

    @property
    def p_b(self) -> float:
        """Marginal failure rate of agent b."""
        return (self.n11 + self.n01) / self.n

    @property
    def p11(self) -> float:
        """Joint failure rate."""
        return self.n11 / self.n

    def _proportions(self) -> tuple[float, float, float, float]:
        n = self.n
        return self.n11 / n, self.n10 / n, self.n01 / n, self.n00 / n


def kendall_tau_a(table: CoFailureTable) -> float:
    """Binary Kendall tau_a = 2*(p11*p00 - p10*p01)  (LLD-B Sec 6.9).

    This is 2x the covariance of the two failure indicators. It is bounded by
    the marginals (tau_a can never reach +/-1 unless marginals are 0.5), which
    is why LLD-B keeps tau_a distinct from the tie-normalized tau_b.
    """
    p11, p10, p01, p00 = table._proportions()
    return 2.0 * (p11 * p00 - p10 * p01)


def phi_coefficient(table: CoFailureTable) -> float:
    """Phi coefficient = point Pearson correlation of the two indicators.

    Equals ``(p11*p00 - p10*p01) / sqrt(p_a(1-p_a) p_b(1-p_b))``. Returns 0.0
    when either marginal is degenerate (0 or 1), where phi is undefined.
    """
    p11, p10, p01, p00 = table._proportions()
    pa, pb = table.p_a, table.p_b
    denom = math.sqrt(pa * (1.0 - pa) * pb * (1.0 - pb))
    if denom == 0.0:
        return 0.0
    return (p11 * p00 - p10 * p01) / denom


def tetrachoric(table: CoFailureTable) -> float:
    """Latent bivariate-normal correlation behind the 2x2 table (LLD-B Sec 5).

    Models each failure indicator as a threshold on a standard normal; returns
    the correlation ``rho`` of the latent pair that reproduces the joint failure
    rate. Solved by root-finding on ``Phi_2(tau_a, tau_b; rho) = p11``.

    Raises:
        DependenceError: if a marginal is degenerate (0 or 1), where the latent
            threshold is +/- infinity and rho is not identified.
    """
    pa, pb, p11 = table.p_a, table.p_b, table.p11
    if not (0.0 < pa < 1.0) or not (0.0 < pb < 1.0):
        raise DependenceError(
            "tetrachoric undefined for a degenerate marginal (0 or 1)"
        )
    tau_a = float(norm.ppf(pa))
    tau_b = float(norm.ppf(pb))

    def joint(rho: float) -> float:
        cov = [[1.0, rho], [rho, 1.0]]
        return float(multivariate_normal.cdf([tau_a, tau_b], mean=[0.0, 0.0], cov=cov))

    # Frechet feasibility: p11 must lie within the joint's attainable range.
    lo, hi = -0.999999, 0.999999
    f_lo, f_hi = joint(lo) - p11, joint(hi) - p11
    if f_lo > 0:  # even at rho=-1 the joint exceeds p11 -> clamp
        return -1.0
    if f_hi < 0:  # even at rho=+1 the joint is below p11 -> clamp
        return 1.0
    return float(brentq(lambda r: joint(r) - p11, lo, hi, xtol=1e-10))


def tau_a_min_samples(eps: float, alpha: float) -> int:
    """Smallest n with a distribution-free (1-alpha) tau_a CI of half-width eps.

    From LLD-B Thm B.9: ``floor(n/2) >= (2/eps^2) ln(2/alpha)``. The minimal n is
    ``2 * ceil((2/eps^2) ln(2/alpha))``. At eps = alpha = 0.05 this is 5904.
    """
    if not (0.0 < eps <= 1.0):
        raise DependenceError("eps must be in (0, 1]")
    if not (0.0 < alpha < 1.0):
        raise DependenceError("alpha must be in (0, 1)")
    k_min = math.ceil((2.0 / (eps * eps)) * math.log(2.0 / alpha))
    return 2 * k_min


def one_factor_loadings(corr: np.ndarray) -> np.ndarray:
    """Shared-factor loadings from an m x m correlation matrix (m >= 3).

    Under a one-factor model ``R = lambda lambda^T + Psi`` (Psi diagonal), every
    off-diagonal is ``R_ij = lambda_i lambda_j``. With three or more indicators
    the loadings are identified via the triad
    ``lambda_i^2 = R_ij R_ik / R_jk`` (averaged over available (j, k) pairs).
    Loadings are returned up to a single global sign.

    Raises:
        DependenceError: if fewer than 3 indicators (the model is
            underidentified — with 2 indicators only the product
            ``lambda_1 lambda_2`` is known, not each loading).
    """
    R = np.asarray(corr, dtype=float)  # noqa: N806
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise DependenceError("correlation matrix must be square")
    m = R.shape[0]
    if m < 3:
        raise DependenceError(
            "one-factor loadings are underidentified with fewer than 3 indicators"
        )

    lam_abs = np.empty(m)
    for i in range(m):
        ratios: list[float] = []
        others = [j for j in range(m) if j != i]
        for a in range(len(others)):
            for b in range(a + 1, len(others)):
                j, k = others[a], others[b]
                if abs(R[j, k]) < 1e-12:
                    continue
                val = R[i, j] * R[i, k] / R[j, k]
                if val > 0.0:
                    ratios.append(val)
        if not ratios:
            raise DependenceError(
                f"indicator {i} has no valid triad; one-factor model not identified"
            )
        lam_abs[i] = math.sqrt(sum(ratios) / len(ratios))

    # Fix relative signs against indicator 0 (taken positive): sign(lambda_i)
    # follows sign(R_0i) since R_0i = lambda_0 lambda_i with lambda_0 > 0.
    signs = np.ones(m)
    for i in range(1, m):
        if R[0, i] < 0:
            signs[i] = -1.0
    return signs * lam_abs
