# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Experiment orchestrator for the $20-capped empirical validation (LLD-E §5, §12).

Wires the motif runner, logging schema, budget ledger, and analysis pipeline
into a single experiment loop.  Provides:

  - :class:`DryRunClient` — in-process zero-cost fake; satisfies the
    :class:`~.motifs.ModelClient` protocol; never touches any network.
  - :class:`ExperimentSummary` — frozen result dataclass with all four
    confirmatory analysis reports.
  - :func:`run_experiment` — deterministic orchestrator (side effects limited
    to JSONL file I/O); returns a frozen :class:`ExperimentSummary`.
  - :func:`main` — argparse CLI entry point:

        ``--dry-run``  (DEFAULT)   DryRunClient, $0, no network
        ``--local``                LocalClient (Ollama, $0 API cost)
        ``--frontier``             NO-OP; prints approval message and exits

Safety invariants (enforced, never relaxed)
-------------------------------------------
- :data:`config.FRONTIER_ENABLED` is ``False`` and is **never mutated here**.
- :class:`DryRunClient` contains zero network/subprocess/file-system calls.
- ``budget_spent == 0.0`` for all :class:`DryRunClient` runs; this is
  asserted in :func:`main` and in the test suite.
- The ``--frontier`` CLI path exits before constructing any client.

References
----------
LLD-E-experiment-design-v2.md  §5  (execution), §12 (outputs).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from typing import TYPE_CHECKING

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments import config
from agentassert_abc.experiments.analysis import (
    CertificationReport,
    CompositionReport,
    DependenceReport,
    DriftReport,
    certification_report,
    composition_report,
    dependence_report,
    drift_report,
)
from agentassert_abc.experiments.budget import BudgetExceeded, BudgetLedger
from agentassert_abc.experiments.logging_schema import JsonlLogger, MissionRecord
from agentassert_abc.experiments.models import ModelResponse
from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, ModelClient, Motif, run_mission
from agentassert_abc.experiments.tasks import TASK_LIBRARY

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "DryRunClient",
    "ExperimentSummary",
    "RunError",
    "_FRONTIER_MODEL_PAIRS",
    "build_client",
    "main",
    "run_experiment",
]


# ---------------------------------------------------------------------------
# Module-local exception
# ---------------------------------------------------------------------------


class RunError(AgentAssertError):
    """Raised for experiment runner failures.

    Examples include insufficient missions to compute a dependence estimate,
    or an invalid parameter combination.

    Subclasses :class:`~agentassert_abc.exceptions.AgentAssertError` so
    callers can catch all AgentAssert failures uniformly.
    """


# ---------------------------------------------------------------------------
# DryRunClient — zero-cost in-process fake
# ---------------------------------------------------------------------------

# Gold answer for TASK_LIBRARY[0] (arith_add: 1234 + 5678 = 6912).
# run_experiment always uses TASK_LIBRARY[0], so returning this canned answer
# produces hard_ok=True for every node in every dry-run mission.
_DRY_CANNED_ANSWER: str = TASK_LIBRARY[0].ground_truth


class DryRunClient:
    """Zero-cost model client for local dry-run pipeline validation.

    Returns a pre-built :class:`~.models.ModelResponse` with the gold answer
    for :data:`~.tasks.TASK_LIBRARY[0]` so that every node produces
    ``hard_ok=True`` when :func:`run_experiment` uses that task.

    **No network, subprocess, or file-system activity occurs.**

    Structurally satisfies the :class:`~.motifs.ModelClient` protocol via
    duck typing — no inheritance required.

    Cost is always 0.0 USD.  The ``ledger.spent == 0.0`` post-condition is
    testable and asserted in :func:`main`.
    """

    def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
        """Return a canned :class:`~.models.ModelResponse` at zero cost.

        Args:
            model:  Model identifier (echoed verbatim into the response).
            prompt: Prompt string (ignored; not used to vary the response).

        Returns:
            :class:`~.models.ModelResponse` with ``cost_usd=0.0``,
            ``input_tokens=0``, ``output_tokens=0``, and
            ``text=TASK_LIBRARY[0].ground_truth``.
        """
        return ModelResponse(
            text=_DRY_CANNED_ANSWER,
            input_tokens=0,
            output_tokens=0,
            model=model,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# ExperimentSummary — frozen result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Immutable summary returned by :func:`run_experiment`.

    Contains all four confirmatory analysis reports plus budget metadata
    and the path to the JSONL mission log.

    All fields are set exactly once at construction; the frozen dataclass
    ensures no subsequent mutation is possible.

    Attributes
    ----------
    dependence:
        Co-failure dependence report for the most common scored agent pair.
    composition:
        Observed graph reliability versus the multiplicative independence
        product.
    certification:
        Anytime-valid e-process certification result over the full mission
        stream.
    drift:
        Per-agent Jacobi drift fit and identifiability gate outcomes.
    budget_spent:
        Total API spend in USD recorded by the :class:`~.budget.BudgetLedger`.
        Always 0.0 for :class:`DryRunClient` and
        :class:`~.models.LocalClient` runs.
    n_missions:
        Total number of missions executed and logged.
    out_path:
        Path to the JSONL mission log (as a string).
    """

    dependence: DependenceReport
    composition: CompositionReport
    certification: CertificationReport
    drift: DriftReport
    budget_spent: float
    n_missions: int
    out_path: str


# ---------------------------------------------------------------------------
# Module-level constants: model pairs by sharing condition
# ---------------------------------------------------------------------------

# For DryRunClient — model strings are arbitrary (client ignores them).
_DRY_MODEL_PAIRS: dict[str, tuple[str, str]] = {
    "same_model":       ("dry-run",   "dry-run"),
    "same_vendor":      ("dry-run-a", "dry-run-b"),
    "different_vendor": ("dry-run-x", "dry-run-y"),
}

# For LocalClient — uses config.LOCAL_MODELS = ("qwen2.5:7b", "gemma3:4b", "llama3.2").
_LOCAL_MODEL_PAIRS: dict[str, tuple[str, str]] = {
    "same_model":       (config.LOCAL_MODELS[0], config.LOCAL_MODELS[0]),
    "same_vendor":      (config.LOCAL_MODELS[0], config.LOCAL_MODELS[2]),
    "different_vendor": (config.LOCAL_MODELS[0], config.LOCAL_MODELS[1]),
}

# ---------------------------------------------------------------------------
# Frontier model pairs (permanently inert unless FRONTIER_ENABLED=True)
# ---------------------------------------------------------------------------
#
# Maps each sharing condition to (model_a, model_b) for use by frontier
# adapters.  These identifiers are locked before the first confirmatory run;
# any substitution requires a dated preregistration amendment BEFORE outcomes
# are visible.
#
# Nested design anchored on OPENROUTER_DEFAULT_MODEL (mistral-small-24b); each
# condition flips exactly one factor.
# Primary conditions (all through OpenRouterClient):
#   same_model:       mistral-small-24b × mistral-small-24b (maximal shared config)
#   same_vendor:      mistral-small-24b × ministral-8b (same vendor Mistral, diff size)
#   different_vendor: mistral-small-24b × gemma-3-12b-it (cross-vendor, Mistral vs Google)
#
# Breadth arms for different_vendor replication (use the other two backends):
#   different_vendor_meta: MetaSparkClient (Spark, reasoning) × OpenRouter anchor
#   different_vendor_grok: GrokBridgeClient (Grok, subscription) × OpenRouter anchor
#
# SAFETY: reading this constant never constructs an adapter.  Adapters are
# built only by build_client(..., "frontier") and require FRONTIER_ENABLED=True.
_FRONTIER_MODEL_PAIRS: dict[str, tuple[str, str]] = {
    # Primary arms (LLD-E §4.1, Table 1 rows 1-3)
    "same_model": (
        config.OPENROUTER_DEFAULT_MODEL,
        config.OPENROUTER_DEFAULT_MODEL,
    ),
    "same_vendor": (
        config.OPENROUTER_DEFAULT_MODEL,
        config.OPENROUTER_SAME_VENDOR_MODEL,
    ),
    "different_vendor": (
        config.OPENROUTER_DEFAULT_MODEL,
        config.OPENROUTER_DIFF_VENDOR_MODEL,
    ),
    # Breadth arms — additional different-vendor replication (LLD-E §4.1)
    "different_vendor_meta": (
        config.META_CONTRIBUTOR_MODEL,
        config.OPENROUTER_DEFAULT_MODEL,
    ),
    "different_vendor_grok": (
        config.GROK_MODEL,
        config.OPENROUTER_DEFAULT_MODEL,
    ),
}

# Deterministic (non-generative) node IDs — never passed to client.generate.
_DETERMINISTIC_NODES: frozenset[str] = frozenset({"aggregator", "merge"})


# ---------------------------------------------------------------------------
# Public factory: build_client
# ---------------------------------------------------------------------------

# Valid frontier conditions — used for ValueError message construction.
_FRONTIER_CONDITIONS: frozenset[str] = frozenset(_FRONTIER_MODEL_PAIRS)


def build_client(condition: str, tier: str) -> ModelClient:
    """Return the appropriate :class:`ModelClient` for *(condition, tier)*.

    This factory is the single-call entry point for experiment callers who
    need the right client without hard-coding adapter classes.  The frontier
    gate is enforced *inside* each adapter's ``__init__``; this function does
    not duplicate it.

    Parameters
    ----------
    condition:
        Model-sharing label.  For ``tier="frontier"`` must be one of the keys
        in :data:`_FRONTIER_MODEL_PAIRS`.  Ignored for ``"dry"`` / ``"local"``.
    tier:
        One of:

        ``"dry"``
            Returns a :class:`DryRunClient` ($0, no network, no Ollama).
        ``"local"``
            Returns a :class:`~.models.LocalClient` (Ollama, $0 API cost).
        ``"frontier"``
            Returns the frontier adapter for *condition*.  **Raises
            :class:`~.models.FrontierDisabledError` when
            config.FRONTIER_ENABLED is False** (the safe default).  Also
            raises when the required API key env var is missing.

    Returns
    -------
    ModelClient
        A client satisfying the :class:`~.motifs.ModelClient` protocol.

    Raises
    ------
    FrontierDisabledError
        *tier* is ``"frontier"`` and the gate is closed (config.FRONTIER_ENABLED
        is False), or a required API key env var is absent.  Raised by the
        adapter ``__init__`` before any network contact.
    ValueError
        Unknown *tier*, or *tier* is ``"frontier"`` with an unknown *condition*.

    Notes
    -----
    Frontier condition routing:

    +------------------------+-------------------------------+
    | condition              | adapter                        |
    +========================+===============================+
    | same_model             | :class:`~.providers.OpenRouterClient` |
    | same_vendor            | :class:`~.providers.OpenRouterClient` |
    | different_vendor       | :class:`~.providers.OpenRouterClient` |
    | different_vendor_meta  | :class:`~.providers.MetaSparkClient`  |
    | different_vendor_grok  | :class:`~.providers.GrokBridgeClient` |
    +------------------------+-------------------------------+

    When FRONTIER_ENABLED is False (the permanent default tonight), every
    frontier adapter raises at construction — so $0 is guaranteed regardless
    of which condition is passed.
    """
    if tier == "dry":
        return DryRunClient()

    if tier == "local":
        from agentassert_abc.experiments.models import LocalClient  # noqa: PLC0415
        return LocalClient()

    if tier == "frontier":
        from agentassert_abc.experiments import providers  # noqa: PLC0415
        # Gate is enforced inside each adapter __init__ (FrontierDisabledError).
        # Unknown condition → ValueError BEFORE any adapter construction.
        if condition in ("same_model", "same_vendor", "different_vendor"):
            return providers.OpenRouterClient()
        if condition == "different_vendor_meta":
            return providers.MetaSparkClient()
        if condition == "different_vendor_grok":
            return providers.GrokBridgeClient()
        valid = sorted(_FRONTIER_CONDITIONS)
        raise ValueError(
            f"Unknown frontier condition {condition!r}. "
            f"Valid conditions: {valid!r}."
        )

    raise ValueError(
        f"Unknown tier {tier!r}. Valid tiers: 'dry', 'local', 'frontier'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_model_assignment(
    motif: Motif,
    model_a: str,
    model_b: str,
) -> dict[str, str]:
    """Map every node in *motif* to a model identifier.

    Generative nodes alternate between *model_a* (even-indexed) and *model_b*
    (odd-indexed).  Deterministic aggregator/merge nodes are excluded because
    they never invoke ``client.generate``.

    For the ``same_model`` sharing condition, pass identical values for
    *model_a* and *model_b*.

    Args:
        motif:   A :class:`~.motifs.Motif` from :data:`~.motifs.MOTIF_LIBRARY`.
        model_a: Model identifier for even-indexed generative nodes.
        model_b: Model identifier for odd-indexed generative nodes.

    Returns:
        ``dict[node_id → model_identifier]`` covering all non-deterministic
        nodes.
    """
    assignment: dict[str, str] = {}
    gen_nodes = [n for n in motif.nodes if n not in _DETERMINISTIC_NODES]
    for idx, node_id in enumerate(gen_nodes):
        assignment[node_id] = model_a if idx % 2 == 0 else model_b
    return assignment


def _select_dependence_pair(
    missions: Sequence[MissionRecord],
) -> tuple[str, str]:
    """Return the scored agent pair with the highest mission co-appearance count.

    Iterates over all missions and counts how many times each
    ``(component_id_i, component_id_j)`` pair co-appears as **scored**
    components.  Returns the lexicographically smallest pair among those
    with the maximum count.

    Args:
        missions: All missions from the experiment run.

    Returns:
        ``(agent_i, agent_j)`` with ``agent_i < agent_j`` lexicographically.

    Raises:
        RunError: If no scored pair appears together in >= 2 missions.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for m in missions:
        scored_ids = sorted(c.component_id for c in m.components if c.scored)
        for i in range(len(scored_ids)):
            for j in range(i + 1, len(scored_ids)):
                pair_counts[(scored_ids[i], scored_ids[j])] += 1

    if not pair_counts:
        raise RunError("No scored component pairs found across all missions.")

    # Max by (count, pair) for deterministic tie-breaking
    best = max(pair_counts, key=lambda k: (pair_counts[k], k))
    if pair_counts[best] < 2:
        raise RunError(
            f"No scored agent pair co-appears in >= 2 missions. "
            f"Best pair {best!r} appeared only {pair_counts[best]} time(s). "
            f"Increase n_per_cell or add more motifs."
        )
    return best


# ---------------------------------------------------------------------------
# Private helpers: mission batch execution + analysis assembly
# ---------------------------------------------------------------------------


def _execute_mission_batch(
    client: ModelClient,
    motifs: Sequence[Motif],
    sharing_conditions: Sequence[str],
    n_per_cell: int,
    out_path: Path | str,
    ledger: BudgetLedger,
    model_pairs: dict[str, tuple[str, str]],
    per_call_ceiling: float = 0.0,
) -> list[MissionRecord]:
    """Execute all missions and append each to the JSONL log.

    Iterates over every ``(motif, condition)`` cell and runs *n_per_cell*
    missions, logging each via :class:`~.logging_schema.JsonlLogger` and
    accumulating API cost in *ledger*.  All missions use
    :data:`~.tasks.TASK_LIBRARY[0]` as the task.

    Returns the complete list of executed :class:`~.logging_schema.MissionRecord`
    objects in execution order.
    """
    logger = JsonlLogger(out_path)
    all_missions: list[MissionRecord] = []
    task = TASK_LIBRARY[0]  # arith_add — DryRunClient returns its ground truth

    # LLD-E §6.3: prospective worst-case budget gate before each batch of <=25.
    # per_call_ceiling is 0.0 for $0 dry/local clients; a paid FrontierClient
    # MUST pass config.PER_CALL_CEILING_USD so the $19.50 hard stop is live.
    batch_size = 25
    total = len(motifs) * len(sharing_conditions) * n_per_cell
    issued = 0

    for motif in motifs:
        for condition in sharing_conditions:
            model_a, model_b = model_pairs.get(
                condition,
                model_pairs.get("same_model", ("dry-run", "dry-run")),
            )
            assignment = _build_model_assignment(motif, model_a, model_b)
            for i in range(n_per_cell):
                if issued % batch_size == 0:
                    upcoming = min(batch_size, total - issued)
                    if not ledger.plan_batch(per_call_ceiling, upcoming):
                        raise BudgetExceeded(
                            f"Budget stop before batch: spent={ledger.spent:.6f} USD; "
                            f"worst-case next {upcoming} calls at "
                            f"{per_call_ceiling:.6f}/call would exceed the "
                            f"{config.BUDGET_STOP_USD} USD hard stop."
                        )
                mission_id = f"mission-{motif.name}-{condition}-{i}"
                cluster_id = f"cluster-{condition}-{i}"
                record = run_mission(
                    motif, task, assignment, client,
                    sharing_condition=condition,
                    cluster_id=cluster_id,
                    mission_id=mission_id,
                )
                logger.append(record)
                ledger.record(record.cost_usd)
                issued += 1
                all_missions.append(record)
    return all_missions


def _build_summary(
    missions: list[MissionRecord],
    p0: float,
    alpha: float,
    ledger: BudgetLedger,
    out_path: Path | str,
) -> ExperimentSummary:
    """Compute all four analysis reports and assemble a frozen ExperimentSummary."""
    agent_i, agent_j = _select_dependence_pair(missions)
    return ExperimentSummary(
        dependence=dependence_report(missions, agent_i, agent_j),
        composition=composition_report(missions),
        certification=certification_report(missions, p0, alpha),
        drift=drift_report(missions, dt=1.0),
        budget_spent=ledger.spent,
        n_missions=len(missions),
        out_path=str(out_path),
    )


def _resolve_run_tier(
    client: ModelClient,
) -> tuple[dict[str, tuple[str, str]], float]:
    """Auto-select the model-pair table and §6.3 batch-gate ceiling for *client*.

    - :class:`DryRunClient`  → ``_DRY_MODEL_PAIRS``,   ceiling ``0.0`` ($0).
    - :class:`~.models.LocalClient` → ``_LOCAL_MODEL_PAIRS``, ceiling ``0.0``
      (Ollama; $0 API cost).
    - anything else (a PAID frontier adapter) → ``_FRONTIER_MODEL_PAIRS`` and
      ``config.PER_CALL_CEILING_USD`` — this is what ARMS the $19.50 prospective
      batch gate.  A ceiling of 0.0 here would silently disable the gate, so a
      non-dry/non-local client MUST get the real per-call ceiling.
    """
    if isinstance(client, DryRunClient):
        return _DRY_MODEL_PAIRS, 0.0
    from agentassert_abc.experiments.models import LocalClient  # noqa: PLC0415

    if isinstance(client, LocalClient):
        return _LOCAL_MODEL_PAIRS, 0.0
    return _FRONTIER_MODEL_PAIRS, config.PER_CALL_CEILING_USD


# ---------------------------------------------------------------------------
# Public API: run_experiment
# ---------------------------------------------------------------------------


def run_experiment(
    client: ModelClient,
    *,
    motifs: Sequence[Motif],
    sharing_conditions: Sequence[str],
    n_per_cell: int,
    p0: float,
    alpha: float,
    out_path: Path | str,
    ledger: BudgetLedger,
    model_pairs: dict[str, tuple[str, str]] | None = None,
    per_call_ceiling: float | None = None,
) -> ExperimentSummary:
    """Orchestrate the $20-capped validation experiment and return all reports.

    Runs every ``(motif, condition)`` cell *n_per_cell* times via *client*,
    logs each :class:`~.logging_schema.MissionRecord` to *out_path*, records
    spend in *ledger*, then computes and returns all four confirmatory analysis
    reports as a frozen :class:`ExperimentSummary`.  All missions use
    :data:`~.tasks.TASK_LIBRARY[0]` (``arith_add``).

    Model-pair selection and the §6.3 batch-gate ceiling are auto-detected from
    the client's tier via :func:`_resolve_run_tier` (dry/local → $0, no gate;
    a paid frontier adapter → :data:`_FRONTIER_MODEL_PAIRS` and
    ``config.PER_CALL_CEILING_USD``, which ARMS the $19.50 prospective stop).
    Pass *model_pairs* and/or *per_call_ceiling* explicitly to override the
    auto-detection (e.g. a preregistration amendment); an explicit
    ``per_call_ceiling`` is used verbatim, so never pass 0.0 for a paid client.

    Raises:
        RunError: No scored agent pair co-appears in >= 2 missions.
        BudgetExceeded: The prospective §6.3 batch gate would breach the
            $19.50 hard stop (only fires when the ceiling is armed).
    """
    resolved_pairs, resolved_ceiling = _resolve_run_tier(client)
    if model_pairs is None:
        model_pairs = resolved_pairs
    if per_call_ceiling is None:
        per_call_ceiling = resolved_ceiling
    all_missions = _execute_mission_batch(
        client, motifs, sharing_conditions, n_per_cell, out_path, ledger, model_pairs,
        per_call_ceiling=per_call_ceiling,
    )
    return _build_summary(all_missions, p0, alpha, ledger, out_path)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m agentassert_abc.experiments.run",
        description=(
            "AgentAssert $20-capped validation experiment runner. "
            "Default mode is dry-run ($0, no network)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="In-process DryRunClient ($0, no network). This is the default mode.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="LocalClient: real Ollama inference ($0 API cost). Requires localhost:11434.",
    )
    parser.add_argument(
        "--frontier",
        action="store_true",
        default=False,
        help="Frontier mode (always NO-OP): prints approval requirement and exits.",
    )
    parser.add_argument(
        "--out",
        default="missions.jsonl",
        metavar="PATH",
        help="Output JSONL log path (default: missions.jsonl).",
    )
    return parser


def _select_client(args: argparse.Namespace) -> tuple[ModelClient, bool]:
    """Return ``(client, dry_run_mode)`` based on parsed CLI flags.

    ``dry_run_mode`` is ``True`` when :class:`DryRunClient` is used so that
    callers can assert ``budget_spent == 0.0`` afterward.
    """
    if args.local:
        from agentassert_abc.experiments.models import LocalClient  # noqa: PLC0415
        print("Mode: LocalClient (Ollama, $0 API cost).", flush=True)
        return LocalClient(), False
    print("Mode: DryRunClient ($0, no network).", flush=True)
    return DryRunClient(), True


def _print_experiment_summary(summary: ExperimentSummary) -> None:
    """Print all four analysis reports to stdout."""
    ci = summary.dependence.tau_a_ci
    print("\n=== Experiment Summary ===")
    print(f"Missions logged : {summary.n_missions}")
    print(f"Budget spent    : ${summary.budget_spent:.4f}")
    print(f"Output file     : {summary.out_path}")

    print("\n--- Dependence (auto-selected agent pair) ---")
    print(f"  tau_a         : {summary.dependence.tau_a:.4f}")
    print(f"  tetrachoric_rho: {summary.dependence.tetrachoric_rho}")
    print(f"  n_missions    : {summary.dependence.n_missions}")
    print(f"  CI (95%)      : [{ci.lower:.4f}, {ci.upper:.4f}]")

    print("\n--- Composition ---")
    print(f"  observed_rel  : {summary.composition.observed_reliability:.4f}")
    print(f"  indep_product : {summary.composition.independence_product:.4f}")
    print(f"  gap           : {summary.composition.gap:.4f}")
    print(f"  n_missions    : {summary.composition.n_missions}")

    cert = summary.certification
    print(f"\n--- Certification (p0={cert.p0}, alpha={cert.alpha}) ---")
    print(f"  certified     : {cert.certified}")
    print(f"  final_wealth  : {cert.final_wealth:.4f}")
    print(f"  first_crossing: {cert.first_crossing_index}")
    print(f"  n_missions    : {cert.n_missions}")

    print("\n--- Drift ---")
    print(f"  n_agents      : {summary.drift.n_agents}")
    print(f"  n_passing     : {summary.drift.n_passing}")
    print(f"  n_failing_gate: {summary.drift.n_failing_gate}")
    print(f"  n_fit_error   : {summary.drift.n_fit_error}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Argparse CLI entry point.  Default: DryRunClient ($0, no network).
    ``--local`` uses LocalClient (Ollama).  ``--frontier`` exits with code 1.
    """
    args = _build_arg_parser().parse_args()

    # --frontier is ALWAYS a NO-OP; config.FRONTIER_ENABLED stays False.
    if args.frontier:
        print(
            "Frontier mode requires explicit approval: config.FRONTIER_ENABLED "
            "must be set True by an authorised operator before any frontier call. "
            "It is currently disabled (False). "
            "No API calls have been made. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    client, dry_run_mode = _select_client(args)
    ledger = BudgetLedger()
    motifs_list = [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]]
    conditions = ["same_model", "different_vendor"]
    n_per_cell = 3

    print(
        f"Running: {len(motifs_list)} motifs x {len(conditions)} conditions x "
        f"{n_per_cell} per cell = {len(motifs_list) * len(conditions) * n_per_cell} missions.",
        flush=True,
    )

    summary = run_experiment(
        client,
        motifs=motifs_list,
        sharing_conditions=conditions,
        n_per_cell=n_per_cell,
        p0=config.P0_RELIABILITY,
        alpha=config.ALPHA,
        out_path=args.out,
        ledger=ledger,
    )

    if dry_run_mode and summary.budget_spent != 0.0:
        raise RunError(
            f"DRY-RUN BUDGET VIOLATION: spent={summary.budget_spent} USD "
            "(expected 0.0). FrontierClient.generate must NOT have been called."
        )

    _print_experiment_summary(summary)


if __name__ == "__main__":
    main()
