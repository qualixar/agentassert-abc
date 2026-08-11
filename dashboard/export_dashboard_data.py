"""Dashboard data exporter for AgentAssert experiments.

Reads a JSONL mission log and writes ``data.js`` so that ``index.html``
can display all four confirmatory analysis results entirely offline —
no CDN, no fetch, no CORS issue.

The generated file assigns the payload to ``window.DASHBOARD_DATA`` so
``index.html`` loads it via a plain ``<script src="data.js">`` tag.

Usage (from agentassert-abc repo root with .venv active):
    .venv/bin/python dashboard/export_dashboard_data.py \\
        --input /path/to/missions.jsonl \\
        --out-dir dashboard/

To open the dashboard after export:
    cd dashboard && python -m http.server 8080
    # then http://localhost:8080/index.html
    # OR open dashboard/index.html directly (file://) — both work.

CRIT addressed:
  1. No HTML-unsafe values in the payload (only numeric + whitelisted strings).
  2. data.js not data.json — avoids fetch() CORS on file:// origins.
  3. Timeline excludes component vectors — minimal per-mission fields only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Package import guard
# ---------------------------------------------------------------------------
try:
    from agentassert_abc.certification.eprocess import GraphEProcess
    from agentassert_abc.experiments.analysis import (
        certification_report,
        composition_report,
        dependence_report,
        drift_report,
    )
    from agentassert_abc.experiments.logging_schema import JsonlLogger, MissionRecord
except ImportError as exc:
    print(
        f"Cannot import agentassert_abc: {exc}\n"
        "Run from the agentassert-abc repo root with the venv active:\n"
        "  .venv/bin/python dashboard/export_dashboard_data.py --help",
        file=sys.stderr,
    )
    sys.exit(1)

if TYPE_CHECKING:
    pass

__all__ = ["build_dashboard_payload", "write_data_js"]

# ---------------------------------------------------------------------------
# Defaults (mirror experiments/config.py)
# ---------------------------------------------------------------------------
_DEFAULT_P0: float = 0.90
_DEFAULT_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_agent_pair(missions: list[MissionRecord]) -> tuple[str | None, str | None]:
    """Return the most common scored agent pair by co-appearance count.

    Mirrors the logic in run.py::_select_dependence_pair but returns
    (None, None) gracefully when no valid pair exists instead of raising.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for m in missions:
        scored_ids = sorted(c.component_id for c in m.components if c.scored)
        for i in range(len(scored_ids)):
            for j in range(i + 1, len(scored_ids)):
                pair_counts[(scored_ids[i], scored_ids[j])] += 1
    if not pair_counts:
        return None, None
    best = max(pair_counts, key=lambda k: (pair_counts[k], k))
    return best if pair_counts[best] >= 2 else (None, None)


def _build_wealth_curve(
    missions: list[MissionRecord],
    p0: float,
    alpha: float,
) -> list[float]:
    """Return per-mission log-wealth snapshots from the e-process stream.

    Uses the same GraphEProcess.mixture() configuration as analysis.py
    certification_report() so the curve matches the certified outcome
    exactly.  The returned list has length == len(missions).
    """
    epsilon = (1.0 - p0) / 3.0
    ep = GraphEProcess.mixture(p0=p0, alpha=alpha, epsilon=epsilon)
    return [ep.update(int(m.y_graph)).log_wealth for m in missions]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dashboard_payload(
    missions: list[MissionRecord],
    *,
    p0: float = _DEFAULT_P0,
    alpha: float = _DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Build the complete dashboard JSON payload from a mission batch.

    Pure function — no I/O.  Calls the four analysis functions from
    experiments/analysis.py and GraphEProcess for the per-step wealth curve.
    All four report objects are serialised to plain dicts; nested dataclasses
    (CoFailureTable, BootstrapCI) are handled field-by-field so the output
    is a stable, explicitly typed contract — not a blind dataclasses.asdict dump.

    Returns a JSON-serialisable dict with keys:
        meta, composition, certification, dependence, drift,
        timeline, motif_breakdown, condition_breakdown
    """
    # ----- four confirmatory reports -----
    comp = composition_report(missions)
    cert = certification_report(missions, p0=p0, alpha=alpha)
    dr = drift_report(missions, dt=1.0)

    agent_i, agent_j = _select_agent_pair(missions)
    dep_payload: dict[str, Any]
    if agent_i is not None and agent_j is not None:
        dep = dependence_report(missions, agent_i, agent_j)
        dep_payload = {
            "tau_a": dep.tau_a,
            "tetrachoric_rho": dep.tetrachoric_rho,
            "n_missions": dep.n_missions,
            "agent_pair": [agent_i, agent_j],
            "table": {
                "n11": dep.table.n11,
                "n10": dep.table.n10,
                "n01": dep.table.n01,
                "n00": dep.table.n00,
            },
            "tau_a_ci": {
                "point": dep.tau_a_ci.point,
                "lower": dep.tau_a_ci.lower,
                "upper": dep.tau_a_ci.upper,
                "alpha": dep.tau_a_ci.alpha,
                "n_boot": dep.tau_a_ci.n_boot,
                "n_clusters": dep.tau_a_ci.n_clusters,
            },
        }
    else:
        dep_payload = {"error": "No scored agent pair with >= 2 co-appearances found."}

    # ----- per-step e-process wealth curve -----
    wealth_curve = _build_wealth_curve(missions, p0=p0, alpha=alpha)
    threshold = math.log(1.0 / alpha)  # log(1/0.05) ≈ 2.996

    # ----- drift: per-agent results -----
    drift_payload: dict[str, Any] = {
        "n_agents": dr.n_agents,
        "n_passing": dr.n_passing,
        "n_failing_gate": dr.n_failing_gate,
        "n_fit_error": dr.n_fit_error,
        "agent_results": [
            {
                "agent_id": r.agent_id,
                "n_obs": r.n_obs,
                "gate_passed": (
                    r.gate_result.gate_passed if r.gate_result is not None else None
                ),
                "fit_error": r.fit_error,
            }
            for r in dr.agent_results
        ],
    }

    # ----- timeline: minimal per-mission fields, NO component vectors -----
    timeline = [
        {
            "mission_id": m.mission_id,
            "motif": m.motif,
            "sharing_condition": m.sharing_condition,
            "y_graph": m.y_graph,
            "timestamp": m.timestamp,
        }
        for m in missions
    ]

    # ----- breakdown by motif and condition -----
    motif_breakdown: dict[str, dict[str, int]] = {}
    condition_breakdown: dict[str, dict[str, int]] = {}
    for m in missions:
        motif_breakdown.setdefault(m.motif, {"n": 0, "passed": 0})
        motif_breakdown[m.motif]["n"] += 1
        motif_breakdown[m.motif]["passed"] += int(m.y_graph)
        condition_breakdown.setdefault(m.sharing_condition, {"n": 0, "passed": 0})
        condition_breakdown[m.sharing_condition]["n"] += 1
        condition_breakdown[m.sharing_condition]["passed"] += int(m.y_graph)

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_missions": len(missions),
            "p0": p0,
            "alpha": alpha,
        },
        "composition": {
            "observed_reliability": comp.observed_reliability,
            "independence_product": comp.independence_product,
            "gap": comp.gap,
            "n_missions": comp.n_missions,
            "n_components": comp.n_components,
            "n_handoffs": comp.n_handoffs,
        },
        "certification": {
            "certified": cert.certified,
            "first_crossing_index": cert.first_crossing_index,
            "final_wealth": cert.final_wealth,
            "n_missions": cert.n_missions,
            "p0": cert.p0,
            "alpha": cert.alpha,
            "wealth_curve": wealth_curve,
            "threshold": threshold,
        },
        "dependence": dep_payload,
        "drift": drift_payload,
        "timeline": timeline,
        "motif_breakdown": motif_breakdown,
        "condition_breakdown": condition_breakdown,
    }


def write_data_js(
    missions: list[MissionRecord],
    *,
    p0: float = _DEFAULT_P0,
    alpha: float = _DEFAULT_ALPHA,
    output_path: Path,
) -> None:
    """Build the payload and write it as a JavaScript data file.

    Assigns to ``window.DASHBOARD_DATA`` so index.html loads it via
    ``<script src="data.js"></script>`` — works on both file:// and
    http://localhost without any fetch() or CORS concern.
    """
    payload = build_dashboard_payload(missions, p0=p0, alpha=alpha)
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    output_path.write_text(f"window.DASHBOARD_DATA = {json_str};\n", encoding="utf-8")
    size_kb = output_path.stat().st_size // 1024
    print(f"  Wrote {output_path}  ({size_kb} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Argparse CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export AgentAssert mission log to dashboard/data.js"
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to missions.jsonl (e.g. experiments-baseline/missions.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default="dashboard",
        metavar="DIR",
        help="Directory to write data.js (default: dashboard/)",
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=_DEFAULT_P0,
        help=f"Null reliability threshold (default: {_DEFAULT_P0})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_DEFAULT_ALPHA,
        help=f"Significance level (default: {_DEFAULT_ALPHA})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path} …")
    logger = JsonlLogger(input_path)
    missions = logger.read_all()
    print(f"  Loaded {len(missions)} missions")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "data.js"

    print(f"Computing analysis reports (p0={args.p0}, alpha={args.alpha}) …")
    write_data_js(missions, p0=args.p0, alpha=args.alpha, output_path=output_path)

    print("\nTo view the dashboard:")
    print(f"  cd {out_dir} && python -m http.server 8080")
    print("  then open http://localhost:8080/index.html")
    print("  (or open index.html directly via file://)")


if __name__ == "__main__":
    main()
