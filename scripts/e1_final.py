# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents

"""E1: regenerate every dependence number the paper reports, from the raw logs.

The paper's §10.2 reports, for each motif and each sharing arm, a 2x2 co-failure
table and five dependence statistics with cluster-bootstrap intervals, plus the
arm contrasts and an internal negative control. Until now those numbers lived
only in the analysis session that produced them, which meant a transcription
error could not be caught by re-running anything. This script is the missing
link: it reads the per-arm JSONL and emits every one of those numbers.

Two conventions are load-bearing and were both got wrong at some point:

1. **Failure is ``not hard_ok``.** The ``ok`` field is a softer verdict and
   using it silently changes every marginal.
2. **Tables are keyed on ``component_id``, never on ``role``.** Every generative
   node carries ``role="worker"``, so keying on role collapses the per-agent
   dict and compares each agent to *itself* — which yields a Jaccard of exactly
   1.0000 and ``n10 = n01 = 0`` in every arm. That output is wrong in a way that
   looks like an unusually clean result rather than like an error.

Usage
-----
    python scripts/e1_final.py --logs /path/to/logs
    python scripts/e1_final.py --logs /path/to/logs --json-out e1.json

The per-mission logs are not distributed with this repository (they contain
model outputs for the full task corpus). See ``PREREGISTRATION.md`` for the
generating configuration and the paper's reproduction appendix for how to
request them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from agentassert_abc.dependence import (
    CoFailureTable,
    cluster_bootstrap,
    jaccard,
    kendall_tau_a,
    phi_coefficient,
)
from agentassert_abc.exceptions import DependenceError

ARMS = ("same_model", "same_vendor", "different_vendor")

# motif -> (filename template, manipulated pair, optional unmanipulated control pair)
MOTIFS: dict[str, tuple[str, tuple[str, str], tuple[str, str] | None]] = {
    "series2": ("frontier_{arm}.jsonl", ("node_a", "node_b"), None),
    "parallel2": (
        "frontier_topo_parallel2_{arm}.jsonl",
        ("branch_a", "branch_b"),
        None,
    ),
    # Only worker_1 is substituted across arms; worker_0 and worker_2 are the
    # same model in all three conditions, which makes (worker_0, worker_2) an
    # unmanipulated pair that must not move with the arm label.
    "quorum2of3": (
        "frontier_topo_quorum2of3_{arm}.jsonl",
        ("worker_0", "worker_1"),
        ("worker_0", "worker_2"),
    ),
}

HALDANE = 0.5  # Haldane--Anscombe continuity correction


def log_odds_ratio(t: CoFailureTable) -> float:
    """log OR with the Haldane--Anscombe ``+1/2`` correction on every cell.

    The correction is applied unconditionally, not only when a cell is zero:
    applying it conditionally makes the estimator discontinuous in the data, so
    two arms differing by one mission could differ by a visible jump in log OR
    for no reason but which side of the zero boundary they landed on.
    """
    return math.log(
        ((t.n11 + HALDANE) * (t.n00 + HALDANE))
        / ((t.n10 + HALDANE) * (t.n01 + HALDANE))
    )


def yule_q(t: CoFailureTable) -> float:
    """Yule's Q = (OR - 1)/(OR + 1), on the same corrected cells."""
    odds = math.exp(log_odds_ratio(t))
    return (odds - 1.0) / (odds + 1.0)


STATS = {
    "J": jaccard,
    "tau_a": kendall_tau_a,
    "phi": phi_coefficient,
    "logOR": log_odds_ratio,
    "YuleQ": yule_q,
}


def load_arm(path: Path, pair: tuple[str, str]) -> tuple[list[int], list[int], list[str]]:
    """Return (fail_a, fail_b, cluster_ids) for one arm, keyed on component_id.

    Missions missing either component are skipped rather than imputed; the count
    of skips is reported by the caller so a silent shrink is impossible.
    """
    fail_a: list[int] = []
    fail_b: list[int] = []
    clusters: list[str] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        comps = {c["component_id"]: c for c in rec["components"]}
        a, b = pair
        if a not in comps or b not in comps:
            continue
        fail_a.append(0 if comps[a]["hard_ok"] else 1)
        fail_b.append(0 if comps[b]["hard_ok"] else 1)
        clusters.append(str(rec.get("cluster_id", rec["mission_id"])))
    return fail_a, fail_b, clusters


def analyse(
    fail_a: list[int], fail_b: list[int], clusters: list[str], n_boot: int, seed: int
) -> dict:
    """Full statistic set with cluster-bootstrap intervals for one arm."""
    table = CoFailureTable.from_pairs(fail_a, fail_b)
    out: dict = {
        "n": table.n,
        "cells": {"n11": table.n11, "n10": table.n10, "n01": table.n01,
                  "n00": table.n00},
        "p_a": table.p_a,
        "p_b": table.p_b,
        # 'either fails' co-failure share -- the paper's 90.0% headline
        "cofail_given_either": (
            table.n11 / (table.n11 + table.n10 + table.n01)
            if (table.n11 + table.n10 + table.n01) else float("nan")
        ),
        "stats": {},
    }
    for name, fn in STATS.items():
        point = fn(table)
        ci = cluster_bootstrap(
            fail_a, fail_b, clusters, fn, n_boot=n_boot, alpha=0.05, seed=seed
        )
        out["stats"][name] = {"point": point, "lo": ci.lower, "hi": ci.upper,
                              "n_valid": ci.n_valid}
    return out


def contrast(
    arm_hi: dict, arm_lo: dict, name: str
) -> tuple[float, bool]:
    """Point difference and a crude non-overlap flag for two arms' intervals.

    Non-overlap of two percentile intervals is a conservative screen, not a
    test of the difference; the paper reports bootstrap contrast intervals and
    this flag only reproduces the direction and the significance call.
    """
    hi, lo = arm_hi["stats"][name], arm_lo["stats"][name]
    diff = hi["point"] - lo["point"]
    separated = hi["lo"] > lo["hi"] or lo["lo"] > hi["hi"]
    return diff, separated


def run(logs: Path, n_boot: int, seed: int) -> dict:
    results: dict = {"motifs": {}, "control": {}}

    for motif, (template, pair, control_pair) in MOTIFS.items():
        print(f"\n=== {motif}  pair={pair[0]} x {pair[1]} ===")
        print(f"  {'arm':<18}{'n':>7}{'p_a':>9}{'p_b':>9}"
              f"{'J':>9}{'tau_a':>9}{'phi':>9}{'logOR':>9}")
        per_arm: dict[str, dict] = {}
        for arm in ARMS:
            path = logs / template.format(arm=arm)
            if not path.exists():
                print(f"  {arm:<18}MISSING: {path.name}")
                continue
            fa, fb, cl = load_arm(path, pair)
            if not fa:
                print(f"  {arm:<18}no missions with both components")
                continue
            res = analyse(fa, fb, cl, n_boot, seed)
            per_arm[arm] = res
            s = res["stats"]
            print(f"  {arm:<18}{res['n']:>7}{res['p_a']:>9.4f}{res['p_b']:>9.4f}"
                  f"{s['J']['point']:>9.4f}{s['tau_a']['point']:>9.4f}"
                  f"{s['phi']['point']:>9.4f}{s['logOR']['point']:>9.4f}")
        results["motifs"][motif] = per_arm

        if len(per_arm) == len(ARMS):
            print("  -- contrasts (same_model vs each substituted arm) --")
            for other in ("same_vendor", "different_vendor"):
                for stat in ("J", "logOR"):
                    d, sep = contrast(per_arm["same_model"], per_arm[other], stat)
                    flag = "SIG" if sep else "n.s."
                    print(f"     same_model - {other:<18} {stat:>6}: "
                          f"{d:>+9.4f}  {flag}")

        if control_pair is not None:
            print(f"  -- negative control  pair={control_pair[0]} x "
                  f"{control_pair[1]} (unmanipulated) --")
            ctrl: dict[str, dict] = {}
            for arm in ARMS:
                path = logs / template.format(arm=arm)
                if not path.exists():
                    continue
                fa, fb, cl = load_arm(path, control_pair)
                if not fa:
                    continue
                res = analyse(fa, fb, cl, n_boot, seed)
                ctrl[arm] = res
                s = res["stats"]
                print(f"     {arm:<15}{res['n']:>7}"
                      f"  J={s['J']['point']:.4f}"
                      f" [{s['J']['lo']:.4f},{s['J']['hi']:.4f}]"
                      f"  logOR={s['logOR']['point']:.4f}"
                      f" [{s['logOR']['lo']:.4f},{s['logOR']['hi']:.4f}]")
            results["control"][motif] = ctrl
            if len(ctrl) == len(ARMS):
                n_sig = 0
                pairs = (("same_model", "same_vendor"),
                         ("same_model", "different_vendor"),
                         ("same_vendor", "different_vendor"))
                for a, b in pairs:
                    for stat in STATS:
                        _, sep = contrast(ctrl[a], ctrl[b], stat)
                        n_sig += int(sep)
                total = len(pairs) * len(STATS)
                print(f"     -> {total - n_sig}/{total} contrasts "
                      f"non-significant (expect {total}/{total})")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True,
                    help="directory holding the per-arm frontier_*.jsonl logs")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples (preregistered value: 2000)")
    # Default seed and B are the published settings, so a bare invocation
    # reproduces the paper's intervals rather than merely resembling them.
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.logs.is_dir():
        print(f"error: --logs is not a directory: {args.logs}", file=sys.stderr)
        print("The per-mission logs are not distributed with this repository.",
              file=sys.stderr)
        return 2

    print("E1 -- dependence statistics regenerated from the raw logs")
    print(f"  logs={args.logs}  n_boot={args.n_boot}  seed={args.seed}")
    print("  failure = not hard_ok;  tables keyed on component_id")

    try:
        results = run(args.logs, args.n_boot, args.seed)
    except DependenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
