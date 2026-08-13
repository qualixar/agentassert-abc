# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents

"""E6: ablation over the independence assumptions the E1--E2 analysis rests on.

Every interval in E1 and every certified floor in E2 assumes missions are i.i.d.
That assumption is testable rather than axiomatic, and this script tests it two
ways:

1. **Serial dependence.** Lag-k autocorrelation of each agent's failure
   indicator in execution order, and the induced design effect
   ``DEFF = 1 + 2 * sum_k (1 - k/n) rho_k``. DEFF > 1 means the effective sample
   size is below ``n`` and every CI in E1 is optimistic by ``sqrt(DEFF)``.

2. **Floor sensitivity.** How far the certified E2 floor moves if we *concede* a
   design effect and inflate the Clopper--Pearson intervals accordingly by
   shrinking the effective sample size to ``n / DEFF``. Reported at a
   deliberately pessimistic DEFF bound, so the number is what the certificate
   costs under an assumption we do not believe rather than under the one we
   measured.

Usage
-----
    python benchmarks/deff_ablation.py
    python benchmarks/deff_ablation.py --deff-bound 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentassert_abc.certification.lp_bound import moment_cp_box_floor

BASE = Path(
    "/Users/v.pratap.bhardwaj/Documents/varun-world/Agentic_official"
    "/AgentAssert-private/experiments-baseline"
)

ARMS = ("same_model", "same_vendor", "different_vendor")
SERIES = {a: (f"frontier_{a}.jsonl", ["node_a", "node_b"]) for a in ARMS}
QUORUM = {a: (f"frontier_topo_quorum2of3_{a}.jsonl",
              ["worker_0", "worker_1", "worker_2"]) for a in ARMS}
QUAD = ("frontier_topo_quorum3of4_same_model.jsonl",
        "frontier_topo_quorum3of4_same_model_clean.jsonl")
QUAD_IDS = ["worker_0", "worker_1", "worker_2", "worker_3"]


def load_matrix(fn: str, ids: list[str]) -> np.ndarray:
    """m x n SUCCESS matrix in file (execution) order."""
    cols = []
    for ln in (BASE / fn).open():
        if not ln.strip():
            continue
        r = json.loads(ln)
        c = {x["component_id"]: x for x in r["components"]}
        if all(i in c for i in ids):
            cols.append([bool(c[i]["hard_ok"]) for i in ids])
    return np.array(cols, dtype=bool).T


def load_quad() -> np.ndarray:
    by_mission: dict[str, dict] = {}
    for fn in QUAD:  # 'clean' listed second so it wins the overlap
        for ln in (BASE / fn).open():
            if ln.strip():
                r = json.loads(ln)
                by_mission[r["mission_id"]] = r
    cols = [[bool({x["component_id"]: x for x in r["components"]}[i]["hard_ok"])
             for i in QUAD_IDS] for r in by_mission.values()]
    return np.array(cols, dtype=bool).T


def autocorr(x: np.ndarray, k: int) -> float:
    x = x.astype(float)
    n = len(x)
    if k >= n:
        return float("nan")
    a, b = x[:-k], x[k:]
    va, vb = a.std(), b.std()
    if va == 0 or vb == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (va * vb))


def deff(x: np.ndarray, max_lag: int = 10) -> tuple[float, float]:
    """(DEFF, rho_1) via the Newey-West style truncated sum."""
    n = len(x)
    rho1 = autocorr(x, 1)
    s = sum((1.0 - k / n) * autocorr(x, k) for k in range(1, max_lag + 1))
    return 1.0 + 2.0 * s, rho1


def rescale_to_n(mat: np.ndarray, n_eff: int) -> np.ndarray:
    """Rebuild a pass matrix with ``n_eff`` columns and the SAME joint law.

    Sensitivity to the effective sample size must isolate the width of the
    Clopper--Pearson intervals. Subsampling the columns would also perturb the
    empirical moments, mixing interval width with sampling noise and reporting
    the sum of the two as if it were the former. Instead we take the exact joint
    cell distribution, rescale the cell counts to total ``n_eff`` with largest-
    remainder rounding, and rebuild. Every moment is preserved to within one
    count, so the only thing that changes is ``n``.
    """
    m, n = mat.shape
    cols = mat.T.astype(np.uint8)
    keys = cols @ (1 << np.arange(m - 1, -1, -1)).astype(np.uint8)
    counts = np.bincount(keys, minlength=1 << m).astype(float)

    exact = counts * (n_eff / n)
    floor_counts = np.floor(exact).astype(int)
    remainder = n_eff - floor_counts.sum()
    if remainder > 0:  # largest-remainder apportionment
        order = np.argsort(-(exact - floor_counts))
        floor_counts[order[:remainder]] += 1

    out = np.empty((m, n_eff), dtype=bool)
    pos = 0
    for cell, k in enumerate(floor_counts):
        if k == 0:
            continue
        bits = [(cell >> (m - 1 - i)) & 1 for i in range(m)]
        out[:, pos:pos + k] = np.array(bits, dtype=bool)[:, None]
        pos += k
    return out[:, :pos]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deff-bound", type=float, default=1.5,
                    help="pessimistic DEFF conceded in the sensitivity check")
    ap.add_argument("--max-lag", type=int, default=10)
    args = ap.parse_args()

    print("E6 -- serial dependence and floor sensitivity\n")
    print("  (1) measured serial dependence, failure indicator, execution order")
    print(f"  {'motif/arm':<34}{'agent':<10}{'rho_1':>9}{'DEFF':>9}")

    worst_deff, worst_rho = 1.0, 0.0
    measured = []
    for label, spec in (("series2", SERIES), ("quorum2of3", QUORUM)):
        for arm, (fn, ids) in spec.items():
            m = load_matrix(fn, ids)
            for i, cid in enumerate(ids):
                fail = ~m[i]
                d, r1 = deff(fail, args.max_lag)
                worst_deff, worst_rho = max(worst_deff, d), max(worst_rho, abs(r1))
                measured.append({"motif": label, "arm": arm, "agent": cid,
                                 "rho_1": r1, "deff": d})
                print(f"  {label + '/' + arm:<34}{cid:<10}{r1:>9.4f}{d:>9.4f}")
    q = load_quad()
    for i, cid in enumerate(QUAD_IDS):
        fail = ~q[i]
        d, r1 = deff(fail, args.max_lag)
        worst_deff, worst_rho = max(worst_deff, d), max(worst_rho, abs(r1))
        measured.append({"motif": "quorum3of4", "arm": "same_model",
                         "agent": cid, "rho_1": r1, "deff": d})
        print(f"  {'quorum3of4/same_model':<34}{cid:<10}{r1:>9.4f}{d:>9.4f}")

    print(f"\n  worst |rho_1| = {worst_rho:.4f}   worst DEFF = {worst_deff:.4f}")

    print(f"\n  (2) certified floor sensitivity at a conceded DEFF = "
          f"{args.deff_bound}")
    print(f"  {'case':<34}{'m':>3}{'n':>7}{'n_eff':>7}"
          f"{'floor(n)':>10}{'floor(n_eff)':>14}{'delta pp':>10}")

    cases: list[tuple[str, np.ndarray, tuple[int, ...]]] = []
    for arm, (fn, ids) in SERIES.items():
        cases.append((f"series2/{arm}", load_matrix(fn, ids), (1, 2)))
    for arm, (fn, ids) in QUORUM.items():
        cases.append((f"quorum2of3/{arm}", load_matrix(fn, ids), (1, 2)))
    cases.append(("quorum3of4/same_model", q, (1, 2, 3)))

    sens = []
    for name, mat, orders in cases:
        m, n = mat.shape
        n_eff = max(2, int(round(n / args.deff_bound)))
        f_full = moment_cp_box_floor(mat, eta_conf=0.05, orders=orders).floor
        f_eff = moment_cp_box_floor(
            rescale_to_n(mat, n_eff), eta_conf=0.05, orders=orders
        ).floor
        delta = 100.0 * (f_full - f_eff)
        sens.append({"case": name, "m": m, "n": n, "n_eff": n_eff,
                     "floor_full": f_full, "floor_eff": f_eff, "delta_pp": delta})
        print(f"  {name:<34}{m:>3}{n:>7}{n_eff:>7}"
              f"{f_full:>10.4f}{f_eff:>14.4f}{delta:>10.2f}")

    print(f"\n  max floor movement: {max(s['delta_pp'] for s in sens):.2f} pp")
    out = Path(__file__).with_name("e6_results.json")
    out.write_text(json.dumps({"measured": measured, "sensitivity": sens,
                               "deff_bound": args.deff_bound}, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
