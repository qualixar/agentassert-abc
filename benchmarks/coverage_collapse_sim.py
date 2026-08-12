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

Usage
-----
    python benchmarks/coverage_collapse_sim.py                 # paper settings
    python benchmarks/coverage_collapse_sim.py --reps 50 --n-boot 150   # fast

Paper settings (S=500, n_boot=1000) take a long time; the defaults here match
the paper so the published numbers are reproducible, and ``--reps``/``--n-boot``
let you trade fidelity for runtime. Every run prints the exact parameters used,
so a reduced-fidelity result can never be mistaken for the published one.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agentassert_abc.certification.factor_reliability import series_reliability_floor
from agentassert_abc.certification.lp_bound import pairwise_cp_box_floor

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


def true_all_success(rng: np.random.Generator, n: int = 4_000_000) -> float:
    """Monte-Carlo the witness law's true all-success probability."""
    a = sample_witness(n, rng)
    return float(a.prod(axis=0).mean())


def run(reps: int, n_boot: int, sizes: tuple[int, ...], eta: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    truth = true_all_success(rng)
    print(f"witness law: p={WITNESS_P}, lambda={WITNESS_LAMBDA}, m={WITNESS_M}")
    print(f"true all-success (4e6 MC): {truth:.5f}")
    print(f"settings: reps={reps}, n_boot={n_boot}, eta={eta}, seed={seed}")
    print()
    print(f"{'n':>6}  {'model-floor cov':>16}  {'Tier-1 cov':>11}  {'elapsed':>8}")
    print("-" * 50)

    for n in sizes:
        t0 = time.perf_counter()
        model_hits = 0
        tier1_hits = 0
        for _ in range(reps):
            a = sample_witness(n, rng)
            try:
                model = series_reliability_floor(a, eta_conf=eta, n_boot=n_boot)
                model_floor = model.floor
            except Exception:  # noqa: BLE001 — degenerate resample; count as a miss
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
    parser.add_argument("--reps", type=int, default=500, help="outer replications S")
    parser.add_argument("--n-boot", type=int, default=1000, help="inner bootstrap draws")
    parser.add_argument("--eta", type=float, default=0.05, help="miscoverage")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000],
        help="mission counts to evaluate",
    )
    args = parser.parse_args()
    run(args.reps, args.n_boot, tuple(args.sizes), args.eta, args.seed)


if __name__ == "__main__":
    main()
