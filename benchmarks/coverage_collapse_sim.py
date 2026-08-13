# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents

"""Coverage-collapse simulation for the paper's §6.4 identification result.

The paper reports that a bootstrap lower confidence bound on the *fitted
Gaussian model functional* loses coverage of the TRUE all-success probability as
``n`` grows, while the copula-agnostic Tier-1 floor holds at or above nominal.
That is not a bug in the bootstrap: the model floor targets the wrong estimand.
The identification gap is ``O(1)`` in ``n`` while the bootstrap haircut shrinks
like ``n^{-1/2}``, so past some ``n`` the interval sits entirely above the truth
and coverage goes to zero.

The paper described this as "reproducible from the released simulation" but no
such script shipped. This is that script.

The published table is produced by exactly two invocations::

    python benchmarks/coverage_collapse_sim.py --witness adversarial
    python benchmarks/coverage_collapse_sim.py --witness gaussian

The defaults (reps=200, n_boot=500) are the published settings, so a bare run
reproduces the paper. ``--reps``/``--n-boot`` trade fidelity for runtime; every
run prints the exact parameters, so a reduced-fidelity result can never be
mistaken for the published one.

Two defects in an earlier version of this script are worth recording, because
both produced plausible output rather than an error:

1. It read ``model.floor``; the result type exposes ``model_floor``. A broad
   ``except Exception`` converted the resulting ``AttributeError`` into a
   counted miss, reporting "coverage 0.00" for runs in which the estimator
   never executed.
2. It sampled the Gaussian witness in every replication. The one-factor family
   is CORRECTLY SPECIFIED for that law, so the identification gap is zero and
   no collapse is possible; the experiment could not exhibit the effect it
   claimed. The adversarial arm below fixes this.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agentassert_abc.certification.factor_reliability import series_reliability_floor
from agentassert_abc.certification.lp_bound import (
    cell_patterns,
    empirical_subset_moments,
    moment_lp_all_success_bounds,
    moment_subsets,
    pairwise_cp_box_floor,
)
from agentassert_abc.exceptions import DependenceError

# §6.4 witness law: the equicorrelated Gaussian-copula joint whose LP-identified
# interval is [0.3305, 0.4652] while the Gaussian model functional sits at
# 0.39214 — the gap the model floor cannot see.
WITNESS_P = 0.6
WITNESS_LAMBDA = 0.8
WITNESS_M = 3


def sample_witness(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``m × n`` pass matrix from the §6.4 one-factor witness law."""
    from scipy.stats import norm

    threshold = norm.ppf(WITNESS_P)
    loading = WITNESS_LAMBDA
    idio = np.sqrt(1.0 - loading**2)
    factor = rng.standard_normal(n)
    out = np.empty((WITNESS_M, n), dtype=int)
    for j in range(WITNESS_M):
        latent = loading * factor + idio * rng.standard_normal(n)
        out[j] = (latent <= threshold).astype(int)
    return out


def gaussian_moments() -> tuple[float, float]:
    """Exact (pairwise, all-success) moments of the one-factor witness law.

    Deterministic Gauss--Legendre quadrature over the common factor, NOT Monte
    Carlo. Every constant the paper quotes for this law comes from here, so the
    same number appears everywhere it is cited; reporting two independent MC
    estimates of the same quantity previously produced two slightly different
    "truths" in different sections.

    With loading ``lam``, latent ``Z_i = lam*F + sqrt(1-lam^2)*e_i`` and success
    ``Z_i <= t`` where ``t = Phi^{-1}(p)``, conditioning on ``F = f`` makes the
    components independent with common probability
    ``Phi((t - lam*f)/sqrt(1-lam^2))``.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.stats import norm

    t = norm.ppf(WITNESS_P)
    lam = WITNESS_LAMBDA
    idio = np.sqrt(1.0 - lam**2)

    # map Gauss-Legendre nodes from [-1,1] onto a wide factor range
    lo, hi = -12.0, 12.0
    x, w = leggauss(400)
    f = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    jac = 0.5 * (hi - lo)
    dens = norm.pdf(f) * w * jac
    cond = norm.cdf((t - lam * f) / idio)

    pair = float(np.sum(dens * cond**2))
    allsucc = float(np.sum(dens * cond**WITNESS_M))
    return pair, allsucc


def true_all_success(rng: np.random.Generator | None = None) -> float:
    """Exact all-success probability of the Gaussian witness law."""
    return gaussian_moments()[1]


def adversarial_law(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """The LP-minimizing joint sharing the Gaussian witness's low-order moments.

    Sampling from the Gaussian witness cannot exhibit coverage collapse: the
    one-factor model is CORRECTLY SPECIFIED there, so the identification gap is
    zero and Thm 4.5's hypothesis fails. To exhibit the collapse we need a law
    that is *pairwise-indistinguishable* from the Gaussian witness -- identical
    marginals and identical pairwise co-success -- but whose all-success sits at
    the bottom of the LP-identified set.

    That law is exactly the LP minimizer. Returns its cell distribution (in
    :func:`cell_patterns` order) and its true all-success probability.
    """
    pair, _ = gaussian_moments()
    subsets = moment_subsets(WITNESS_M, orders=(1, 2))
    values = [WITNESS_P if len(sub) == 1 else pair for sub in subsets]
    bounds = moment_lp_all_success_bounds(WITNESS_M, subsets, values)
    if not bounds.feasible or bounds.minimizer is None:
        raise DependenceError("adversarial LP infeasible")
    cells = np.asarray(bounds.minimizer, dtype=float)
    cells = np.clip(cells, 0.0, None)
    cells /= cells.sum()
    return cells, float(bounds.lower)


def sample_cells(cells: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``m x n`` pass matrix from an explicit joint over ``2^m`` cells."""
    patterns = cell_patterns(WITNESS_M)
    idx = rng.choice(len(cells), size=n, p=cells)
    return patterns[idx].T.astype(int)


def run(reps: int, n_boot: int, sizes: tuple[int, ...], eta: float, seed: int,
        witness: str = "adversarial") -> None:
    rng = np.random.default_rng(seed)
    gauss_truth = true_all_success(rng)
    if witness == "gaussian":
        cells, truth = None, gauss_truth
    else:
        cells, truth = adversarial_law(rng)
    print(f"witness law: p={WITNESS_P}, lambda={WITNESS_LAMBDA}, m={WITNESS_M}")
    print(f"sampling from: {witness}")
    print(f"Gaussian-witness all-success (exact quad): {gauss_truth:.6f}")
    print(f"TRUE all-success of sampled law:          {truth:.6f}")
    if witness != "gaussian":
        print(f"identification gap Delta:                 {gauss_truth - truth:.6f}")
    print(f"settings: reps={reps}, n_boot={n_boot}, eta={eta}, seed={seed}")
    print()
    print(f"{'n':>6}  {'model-floor cov':>16}  {'Tier-1 cov':>11}  {'elapsed':>8}")
    print("-" * 50)

    for n in sizes:
        t0 = time.perf_counter()
        model_hits = 0
        tier1_hits = 0
        for _ in range(reps):
            a = sample_witness(n, rng) if cells is None else sample_cells(cells, n, rng)
            try:
                model = series_reliability_floor(a, eta_conf=eta, n_boot=n_boot)
                model_floor = model.model_floor
            except DependenceError:
                # A genuinely degenerate resample (e.g. a saturated stage) has
                # no model floor; count it as a miss. Only this exception is
                # caught: a bare `except Exception` here previously swallowed an
                # AttributeError from a wrong field name and scored every
                # replication as a miss, reporting coverage 0.00 for a run in
                # which the estimator never executed at all.
                model_floor = 1.0
            if model_floor <= truth:
                model_hits += 1
            if pairwise_cp_box_floor(a, eta_conf=eta).floor <= truth:
                tier1_hits += 1
        elapsed = time.perf_counter() - t0
        print(
            f"{n:>6}  {model_hits / reps:>16.2f}  {tier1_hits / reps:>11.2f}  "
            f"{elapsed:>7.1f}s"
        )

    print()
    print("Expected: model-floor coverage collapses toward 0 as n grows (it")
    print("covers the fitted Gaussian functional, not the truth); Tier-1 stays")
    print(f">= nominal {1 - eta:.2f}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200, help="outer replications S")
    parser.add_argument("--n-boot", type=int, default=500, help="inner bootstrap draws")
    parser.add_argument("--eta", type=float, default=0.05, help="miscoverage")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000],
        help="mission counts to evaluate",
    )
    parser.add_argument(
        "--witness", choices=("adversarial", "gaussian"), default="adversarial",
        help="'gaussian' = correctly-specified control (no collapse expected); "
             "'adversarial' = LP minimizer, pairwise-indistinguishable, Delta > 0",
    )
    args = parser.parse_args()
    run(args.reps, args.n_boot, tuple(args.sizes), args.eta, args.seed, args.witness)


if __name__ == "__main__":
    main()
