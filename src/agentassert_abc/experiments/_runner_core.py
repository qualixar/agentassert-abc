# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Crash-proof mission batch executor (LLD-F §B, §C).

Extracted from :mod:`run` to keep that module under the 800-line cap.

Contains:

- :func:`_build_model_assignment` — maps motif nodes to model identifiers.
- :func:`_read_prior_run` — reads a prior JSONL run for resume (LLD-F §C.1).
- :func:`_write_progress` — heartbeat JSON writer (LLD-F §C.5).
- :func:`_append_failure` — per-mission failure log writer (LLD-F §C.3).
- :func:`_execute_mission_batch` — the main execution loop with resume,
  per-mission isolation, and heartbeat.

All helpers are private (underscore-prefixed).  The public interface of the
:mod:`run` module is unchanged.
"""

from __future__ import annotations

import contextlib
import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from agentassert_abc.experiments import config
from agentassert_abc.experiments.budget import BudgetExceeded, BudgetLedger
from agentassert_abc.experiments.logging_schema import JsonlLogger, MissionRecord
from agentassert_abc.experiments.motifs import ModelClient, Motif, run_mission

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agentassert_abc.experiments.tasks import Task

__all__: list[str] = []  # internal module — nothing exported to the public API


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Deterministic (non-generative) node IDs — never passed to client.generate.
_DETERMINISTIC_NODES: frozenset[str] = frozenset({"aggregator", "merge"})

# Write progress.json every N missions (LLD-F §C.5).
_HEARTBEAT_INTERVAL: int = 100


# ---------------------------------------------------------------------------
# _build_model_assignment
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


# ---------------------------------------------------------------------------
# Resume helper: read prior run
# ---------------------------------------------------------------------------


def _read_prior_run(out_path: Path) -> tuple[set[str], float]:
    """Read a prior JSONL run to collect completed mission IDs and total cost.

    Used by the resume path (LLD-F §C.1): if *out_path* exists, parse every
    line to extract ``mission_id`` and ``cost_usd``, then return a set of
    completed IDs and the sum of prior spend.

    A line that fails to parse (e.g., truncated write) is silently skipped so
    a partially-written file does not crash resume.

    Args:
        out_path: Path to the JSONL log from a previous run.

    Returns:
        ``(completed_ids, prior_cost_usd)`` — empty set and 0.0 if the file
        does not exist or is empty.
    """
    if not out_path.exists():
        return set(), 0.0
    completed_ids: set[str] = set()
    prior_cost = 0.0
    try:
        logger = JsonlLogger(out_path)
        for rec in logger.read_all():
            completed_ids.add(rec.mission_id)
            prior_cost += rec.cost_usd
    except Exception:  # noqa: BLE001
        pass  # best-effort; partial file → keep what we have
    return completed_ids, prior_cost


# ---------------------------------------------------------------------------
# Heartbeat helper
# ---------------------------------------------------------------------------


def _write_progress(
    out_path: Path | str,
    *,
    completed: int,
    total: int,
    spent_usd: float,
) -> None:
    """Write a progress heartbeat JSON file (LLD-F §C.5).

    Writes ``<out_path>.progress.json`` with ``{completed, total,
    spent_usd, ts}``.  Errors are swallowed so a write failure never
    aborts the run.

    Args:
        out_path:   Base output path (without the ``.progress.json`` suffix).
        completed:  Number of missions finished so far (including skipped).
        total:      Total planned missions in the batch.
        spent_usd:  Total ledger spend at heartbeat time.
    """
    progress_path = Path(str(out_path) + ".progress.json")
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "completed": completed,
        "total": total,
        "spent_usd": spent_usd,
        "ts": ts,
    }
    print(
        f"[heartbeat] completed={completed}/{total} spent=${spent_usd:.4f} ts={ts}",
        flush=True,
    )
    with contextlib.suppress(OSError):
        progress_path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Per-mission failure log helper
# ---------------------------------------------------------------------------


def _append_failure(
    out_path: Path | str,
    *,
    mission_id: str,
    condition: str,
    motif: str,
    error: str,
) -> None:
    """Append a failure record to ``<out_path>.failures.jsonl`` (LLD-F §C.3).

    Args:
        out_path:   Base output path (without the ``.failures.jsonl`` suffix).
        mission_id: Identifier of the failed mission.
        condition:  Sharing condition label.
        motif:      Motif name.
        error:      String representation of the exception.
    """
    failures_path = Path(str(out_path) + ".failures.jsonl")
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "mission_id": mission_id,
        "condition": condition,
        "motif": motif,
        "error": error,
        "ts": ts,
    }
    with contextlib.suppress(OSError), failures_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# _execute_mission_batch — main execution loop
# ---------------------------------------------------------------------------


def _execute_mission_batch(
    client: ModelClient,
    motifs: Sequence[Motif],
    sharing_conditions: Sequence[str],
    n_per_cell: int,
    out_path: Path | str,
    ledger: BudgetLedger,
    model_pairs: dict[str, tuple[str, str]],
    task_sampler: Callable[[str], Task],
    per_call_ceiling: float = 0.0,
) -> list[MissionRecord]:
    """Execute all missions and append each to the JSONL log.

    Iterates over every ``(motif, condition)`` cell and runs *n_per_cell*
    missions via *task_sampler* (one task per mission ID), logging each via
    :class:`~.logging_schema.JsonlLogger` and accumulating API cost in *ledger*.

    Resume / idempotency (LLD-F §C.1)
    -----------------------------------
    If *out_path* already exists, completed mission IDs and their prior cost
    are read first.  Completed missions are skipped (no re-run, no re-charge)
    and the ledger is pre-seeded with the prior cost so the $19.50 batch gate
    accounts for total study spend across restarts.

    Per-mission isolation (LLD-F §C.3)
    ------------------------------------
    Each ``run_mission`` call is wrapped in a try/except.  Unrecoverable errors
    are logged to ``<out_path>.failures.jsonl`` and the run continues.

    Heartbeat (LLD-F §C.5)
    -----------------------
    Every :data:`_HEARTBEAT_INTERVAL` missions a progress JSON is written.

    Returns the complete list of successfully executed
    :class:`~.logging_schema.MissionRecord` objects in execution order.
    """
    out_path = Path(str(out_path))
    logger = JsonlLogger(out_path)

    # --- C.1 Resume: read prior run and seed ledger -------------------------
    completed_ids, prior_cost = _read_prior_run(out_path)
    if prior_cost > 0.0:
        ledger.record(prior_cost)

    # Pre-populate with prior records so callers (e.g. _build_summary) receive
    # the FULL mission set regardless of how many missions are skipped this run.
    # On a complete resume (every mission already logged) the returned list is
    # still correct — callers never see an empty list for a finished experiment.
    if completed_ids:
        try:
            all_missions: list[MissionRecord] = list(JsonlLogger(out_path).read_all())
        except Exception:  # noqa: BLE001
            all_missions = []
    else:
        all_missions = []

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
                mission_id = f"mission-{motif.name}-{condition}-{i}"
                cluster_id = f"cluster-{condition}-{i}"

                # --- C.1 Resume: skip completed missions ---------------------
                if mission_id in completed_ids:
                    issued += 1
                    if issued % _HEARTBEAT_INTERVAL == 0:
                        _write_progress(
                            out_path, completed=issued, total=total,
                            spent_usd=ledger.spent,
                        )
                    continue

                # Budget gate before each new batch of ≤25 fresh missions.
                if issued % batch_size == 0:
                    upcoming = min(batch_size, total - issued)
                    if not ledger.plan_batch(per_call_ceiling, upcoming):
                        raise BudgetExceeded(
                            f"Budget stop before batch: spent={ledger.spent:.6f} USD; "
                            f"worst-case next {upcoming} calls at "
                            f"{per_call_ceiling:.6f}/call would exceed the "
                            f"{config.BUDGET_STOP_USD} USD hard stop."
                        )

                # --- C.2 Per-mission task resolution ------------------------
                task = task_sampler(mission_id)

                # --- C.3 Per-mission isolation: catch all errors ------------
                try:
                    record = run_mission(
                        motif, task, assignment, client,
                        sharing_condition=condition,
                        cluster_id=cluster_id,
                        mission_id=mission_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    _append_failure(
                        out_path,
                        mission_id=mission_id,
                        condition=condition,
                        motif=motif.name,
                        error=repr(exc),
                    )
                    issued += 1
                    if issued % _HEARTBEAT_INTERVAL == 0:
                        _write_progress(
                            out_path, completed=issued, total=total,
                            spent_usd=ledger.spent,
                        )
                    continue

                logger.append(record)
                ledger.record(record.cost_usd)
                issued += 1
                all_missions.append(record)

                # --- C.5 Progress heartbeat ---------------------------------
                if issued % _HEARTBEAT_INTERVAL == 0:
                    _write_progress(
                        out_path, completed=issued, total=total,
                        spent_usd=ledger.spent,
                    )

    return all_missions
