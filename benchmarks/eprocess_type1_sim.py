# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents

"""Anytime-valid type-I simulation for the paper's E4 result.

The paper reports an empirical crossing rate of 0.048 over 8,000 null streams of
500 missions at ``p0=0.8, alpha=0.05``, but never records the betting fraction
``lambda`` that produced it, and no generating script shipped. A type-I rate is
not reproducible without ``lambda``, so this script both fixes that and makes
the stronger claim the theory actually supports: Ville's inequality bounds the
crossing rate for *every* admissible ``lambda``, not for one tuned value. We
therefore sweep the admissible range and report the whole curve.

Usage
-----
    python benchmarks/eprocess_type1_sim.py                  # paper settings
    python benchmarks/eprocess_type1_sim.py --streams 500    # fast smoke run

Every run prints the exact parameters, so a reduced-fidelity result can never be
mistaken for the published one.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from agentassert_abc.certification.eprocess import GraphEProcess

PAPER_P0 = 0.8
PAPER_ALPHA = 0.05
PAPER_STREAMS = 8_000
PAPER_MISSIONS = 500
PAPER_SEED = 42


def crossing_rate(
    p0: float,
    alpha: float,
    lam: float,
    n_streams: int,
    n_missions: int,
    seed: int,
) -> tuple[float, float]:
    """Fraction of null streams whose wealth ever reaches ``1/alpha``.

    Returns ``(rate, mean_first_crossing)``; the second entry is NaN when no
    stream crossed. Streams are simulated at the null boundary ``p_true = p0``,
    which is the least favourable case for the bound.
    """
    rng = np.random.default_rng(seed)
    threshold = -math.log(alpha)
    crossings = 0
    first: list[int] = []

    for _ in range(n_streams):
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        draws = rng.random(n_missions) < p0
        for i, y in enumerate(draws):
            ep.update(int(y))
            if ep.log_wealth >= threshold:
                crossings += 1
                first.append(i + 1)
                break

    rate = crossings / n_streams
    return rate, (float(np.mean(first)) if first else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p0", type=float, default=PAPER_P0)
    ap.add_argument("--alpha", type=float, default=PAPER_ALPHA)
    ap.add_argument("--streams", type=int, default=PAPER_STREAMS)
    ap.add_argument("--missions", type=int, default=PAPER_MISSIONS)
    ap.add_argument("--seed", type=int, default=PAPER_SEED)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    # lambda must lie in (0, 1/p0) for the betting factor to stay positive.
    lam_max = 1.0 / args.p0
    lambdas = [round(f * lam_max, 4) for f in (0.1, 0.25, 0.5, 0.7, 0.9, 0.99)]

    # 3-sigma Monte Carlo band on a proportion of size alpha.
    mc = 3.0 * math.sqrt(args.alpha * (1 - args.alpha) / args.streams)

    print("E4 -- anytime-valid type-I error under the null")
    print(f"  p0={args.p0}  alpha={args.alpha}  streams={args.streams}  "
          f"missions={args.missions}  seed={args.seed}")
    print(f"  lambda admissible range (0, {lam_max:.4f});  "
          f"3-sigma MC band = {mc:.4f}")
    print(f"  {'lambda':>9} {'rate':>8} {'<= alpha?':>10} {'mean 1st cross':>16}")

    rows = []
    t0 = time.time()
    for lam in lambdas:
        rate, mean_first = crossing_rate(
            args.p0, args.alpha, lam, args.streams, args.missions, args.seed
        )
        ok = rate <= args.alpha + mc
        rows.append({"lambda": lam, "rate": rate, "within_bound": ok,
                     "mean_first_crossing": mean_first})
        cross = "--" if math.isnan(mean_first) else f"{mean_first:.1f}"
        print(f"  {lam:>9.4f} {rate:>8.4f} {'yes' if ok else 'NO':>10} {cross:>16}")

    worst = max(r["rate"] for r in rows)
    print(f"\n  worst-case rate across lambda: {worst:.4f}  "
          f"(alpha={args.alpha}, +3sigma={args.alpha + mc:.4f})")
    print(f"  elapsed {time.time() - t0:.1f}s")

    if args.json_out:
        payload = {"params": vars(args), "lambda_max": lam_max,
                   "mc_3sigma": mc, "rows": rows, "worst_rate": worst}
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
