# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Crash-proof mission batch executor with bounded concurrency (LLD-F §B, §C, §G).

Extracted from :mod:`run` to keep that module under the 800-line cap.

Contains:

- :func:`_build_model_assignment` — maps motif nodes to model identifiers.
- :func:`_read_prior_run` — reads a prior JSONL run for resume (LLD-F §C.1).
- :func:`_write_progress` — heartbeat JSON writer (LLD-F §C.5).
- :func:`_append_failure` — per-mission failure log writer (LLD-F §C.3).
- :func:`_n_gen_nodes` — count generative nodes per motif (for the budget gate).
- :func:`_run_batch_serial` — serial execution helper.
- :func:`_run_batch_concurrent` — concurrent execution helper (ThreadPoolExecutor).
- :func:`_execute_mission_batch` — the main execution loop with resume,
  per-mission isolation, bounded concurrency, and heartbeat (LLD-F §G).

Concurrency design (LLD-F §G.1) — lock-free "compute concurrently, write serially,
sorted by canonical index":

1. All mission specs are enumerated in canonical ``(motif, condition, i)`` order
   and tagged with a 0-based ``idx``.
2. Resume: drop specs already in ``completed_ids``; seed ledger from prior cost.
3. Remaining specs are batched in chunks of ``batch_size=25``.
4. Per batch, IN ORDER:
   a. Prospective budget gate — checked SINGLE-THREADED before any dispatch.
      Worst-case spend = Σ ``n_gen_nodes(motif) × per_call_ceiling``.
      Raises :exc:`~.budget.BudgetExceeded` and dispatches NOTHING if gate fires.
   b. Concurrent compute — :class:`~concurrent.futures.ThreadPoolExecutor` submits
      ``run_mission`` for each spec.  ``run_mission`` mutates no shared state;
      the provider client is stateless and thread-safe (urllib per call).
      Results are collected as ``(idx, result_or_exception)`` pairs.
   c. Serial write — results sorted by ``idx`` are written one at a time:
      success → logger.append + ledger.record + all_missions.append;
      exception → ``_append_failure``.  Single-threaded → no locks needed.
   d. Heartbeat — ``<out_path>.progress.json`` written after each batch.

``concurrency=1`` is the serial fallback (byte-identical to the pre-concurrency
runner) used for dry/local tiers.

All helpers are private (underscore-prefixed).  The public interface of the
:mod:`run` module is unchanged.
"""

from __future__ import annotations

import contextlib
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from agentassert_abc.experiments import config
from agentassert_abc.experiments.budget import BudgetExceeded, BudgetLedger
from agentassert_abc.experiments.logging_schema import JsonlLogger, MissionRecord
from agentassert_abc.experiments.motifs import ModelClient, Motif, run_mission

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from concurrent.futures import Future

    from agentassert_abc.experiments.tasks import Task

__all__: list[str] = []  # internal module — nothing exported to the public API


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Deterministic (non-generative) node IDs — never passed to client.generate.
_DETERMINISTIC_NODES: frozenset[str] = frozenset({"aggregator", "merge"})

# Write progress.json every N missions (LLD-F §C.5).
# Retained for backward-compatibility with tests that patch this constant; the
# new per-batch heartbeat fires independently of this value.
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
# _n_gen_nodes — generative node count for the budget gate
# ---------------------------------------------------------------------------


def _n_gen_nodes(motif: Motif) -> int:
    """Return the number of generative (non-deterministic) nodes in *motif*.

    Deterministic aggregator/merge nodes are excluded because they never call
    ``client.generate``.  This count is used as the realistic worst-case call
    count per mission execution in the prospective budget gate (LLD-F §G.1).

    Conservative by design: inactive hierarchy workers are counted even though
    only one branch is active per run — the gate must bound worst-case spend.

    Args:
        motif: A :class:`~.motifs.Motif` from :data:`~.motifs.MOTIF_LIBRARY`.

    Returns:
        Count of nodes in ``motif.nodes`` not in :data:`_DETERMINISTIC_NODES`.
    """
    return sum(1 for n in motif.nodes if n not in _DETERMINISTIC_NODES)


# ---------------------------------------------------------------------------
# Batch execution helpers
# ---------------------------------------------------------------------------

# A mission specification: (canonical_idx, mission_id, cluster_id, motif, condition, assignment)
# canonical_idx is the global 0-based position across all (motif, condition, i) cells.
_MissionSpec = tuple[int, str, str, Motif, str, dict[str, str]]

# Result from one batch: (canonical_idx, MissionRecord) or (canonical_idx, exception)
_BatchResult = tuple[int, MissionRecord | BaseException]


def _run_batch_serial(
    batch: list[_MissionSpec],
    client: ModelClient,
    task_sampler: Callable[[str], Task],
) -> list[_BatchResult]:
    """Execute a batch of mission specs serially.

    Runs each spec in submission order.  Exceptions from ``run_mission`` are
    caught per-spec and returned as ``(idx, exception)`` pairs — the caller
    is responsible for routing them to ``_append_failure``.

    Args:
        batch:        Ordered list of mission specs.
        client:       Model client (no network calls for DryRunClient).
        task_sampler: Deterministic ``mission_id → Task`` callable.

    Returns:
        ``list[(canonical_idx, MissionRecord | exception)]`` in submission order.
    """
    results: list[_BatchResult] = []
    for spec_idx, mission_id, cluster_id, motif, condition, assignment in batch:
        task = task_sampler(mission_id)
        try:
            record = run_mission(
                motif, task, assignment, client,
                sharing_condition=condition,
                cluster_id=cluster_id,
                mission_id=mission_id,
            )
            results.append((spec_idx, record))
        except Exception as exc:  # noqa: BLE001
            results.append((spec_idx, exc))
    return results


def _run_batch_concurrent(
    batch: list[_MissionSpec],
    client: ModelClient,
    task_sampler: Callable[[str], Task],
    max_workers: int,
) -> list[_BatchResult]:
    """Execute a batch of mission specs concurrently via ThreadPoolExecutor.

    Tasks are resolved in the main thread (``task_sampler`` is pure/deterministic
    and must not be called from worker threads).  Each resolved spec is submitted
    to the executor as ``run_mission(...)``.  ``run_mission`` mutates NO shared
    state and the provider client is stateless (urllib per call) — thread-safe.

    Results are collected AFTER the executor shuts down (all futures complete).
    Exceptions from ``run_mission`` are captured per-future and returned as
    ``(idx, exception)`` pairs so the caller can route them to ``_append_failure``.

    Args:
        batch:       Ordered list of mission specs.
        client:      Stateless model client.
        task_sampler: Deterministic ``mission_id → Task`` callable.
        max_workers: Maximum concurrent worker threads.

    Returns:
        ``list[(canonical_idx, MissionRecord | exception)]`` in submission order.
        The returned list is already in canonical order (submission = canonical);
        ``_execute_mission_batch`` sorts it defensively before writing.
    """
    futures_and_idx: list[tuple[int, Future[MissionRecord]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for spec_idx, mission_id, cluster_id, motif, condition, assignment in batch:
            task = task_sampler(mission_id)
            fut = executor.submit(
                run_mission,
                motif, task, assignment, client,
                sharing_condition=condition,
                cluster_id=cluster_id,
                mission_id=mission_id,
            )
            futures_and_idx.append((spec_idx, fut))
    # executor.__exit__ calls shutdown(wait=True) — all futures are done here.

    results: list[_BatchResult] = []
    for spec_idx, fut in futures_and_idx:
        try:
            results.append((spec_idx, fut.result()))
        except Exception as exc:  # noqa: BLE001
            results.append((spec_idx, exc))
    return results


# ---------------------------------------------------------------------------
# _execute_mission_batch — main execution loop (LLD-F §G)
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
    concurrency: int = 1,
) -> list[MissionRecord]:
    """Execute all missions with optional bounded concurrency and return results.

    Implements the LLD-F §G.1 "compute concurrently, write serially, sorted"
    design.  Key invariants:

    * **Budget gate** is checked single-threaded BEFORE any dispatch for each
      batch.  The worst-case cost is ``Σ n_gen_nodes(motif) × per_call_ceiling``
      which correctly counts real API calls (fixing the prior per-mission==1-call
      under-count).  A gate trip raises :exc:`~.budget.BudgetExceeded` and
      dispatches NOTHING.
    * **JSONL order** is canonical (global 0-based ``idx`` across all
      ``(motif, condition, i)`` cells) regardless of thread completion order,
      guaranteeing reproducible analysis (drift / e-process first-crossing).
    * **Ledger mutations** (``record``) happen only in the single-threaded write
      phase — no lock needed.
    * ``concurrency=1`` is byte-identical to the pre-concurrency serial runner.

    Resume / idempotency (LLD-F §C.1)
    -----------------------------------
    If *out_path* already exists, completed mission IDs and their prior cost
    are read first.  Completed missions are skipped (no re-run, no re-charge)
    and the ledger is pre-seeded with the prior cost so the $19.50 gate
    accounts for total study spend across restarts.

    Per-mission isolation (LLD-F §C.3)
    ------------------------------------
    Each ``run_mission`` call is wrapped in a try/except.  Unrecoverable errors
    are logged to ``<out_path>.failures.jsonl`` and the run continues.

    Heartbeat (LLD-F §C.5 / §G.1.d)
    ----------------------------------
    A progress JSON is written after every batch (≤ 25 missions).

    Args:
        client:            Model client injected for all generate calls.
        motifs:            Motif sequence (defines the canonical enumeration order).
        sharing_conditions: Condition labels (define the canonical enumeration order).
        n_per_cell:        Missions per ``(motif, condition)`` cell.
        out_path:          JSONL log path (also base for ``.progress.json`` /
                           ``.failures.jsonl``).
        ledger:            Budget ledger (pre-seeded with prior spend on resume).
        model_pairs:       Condition → ``(model_a, model_b)`` mapping.
        task_sampler:      ``mission_id → Task`` callable (deterministic / pure).
        per_call_ceiling:  Worst-case cost per API call in USD.  ``0.0`` for
                           dry/local tiers (gate disabled); must be
                           ``config.PER_CALL_CEILING_USD`` for paid frontier runs.
        concurrency:       Worker threads for concurrent execution.  ``1`` → serial
                           (byte-identical to pre-concurrency runner).  Frontier
                           tier passes ``config.FRONTIER_CONCURRENCY``.

    Returns:
        Complete list of successfully executed :class:`~.logging_schema.MissionRecord`
        objects (includes records pre-populated from a prior run on resume).

    Raises:
        BudgetExceeded: Prospective gate for a batch would breach
            ``config.BUDGET_STOP_USD``.  No missions in the offending batch
            are dispatched.
    """
    out_path = Path(str(out_path))
    logger = JsonlLogger(out_path)

    # --- Step 1: Enumerate all specs in canonical order ----------------------
    specs: list[_MissionSpec] = []
    canonical_idx = 0
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
                specs.append(
                    (canonical_idx, mission_id, cluster_id, motif, condition, assignment)
                )
                canonical_idx += 1

    total = len(specs)

    # --- Step 2: Resume — read prior run, seed ledger, pre-populate ----------
    completed_ids, prior_cost = _read_prior_run(out_path)
    if prior_cost > 0.0:
        ledger.record(prior_cost)

    # Pre-populate with prior records so callers receive the FULL set on resume.
    if completed_ids:
        try:
            all_missions: list[MissionRecord] = list(JsonlLogger(out_path).read_all())
        except Exception:  # noqa: BLE001
            all_missions = []
    else:
        all_missions = []

    # Drop already-completed specs; count them toward the issued total.
    remaining: list[_MissionSpec] = [
        spec for spec in specs if spec[1] not in completed_ids
    ]
    issued = total - len(remaining)  # already-completed missions counted

    # --- Step 3: Batch remaining specs into chunks of batch_size -------------
    batch_size = 25
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]

        # --- Step 4a: Prospective gate (SINGLE-THREADED, before dispatch) ----
        # Fix: count real API calls (Σ n_gen_nodes), not 1 per mission.
        # Gate is skipped when ceiling==0.0 (dry/local — $0 spend is guaranteed).
        if per_call_ceiling > 0.0:
            total_calls = sum(_n_gen_nodes(spec[3]) for spec in batch)
            if not ledger.plan_batch(per_call_ceiling, total_calls):
                raise BudgetExceeded(
                    f"Budget stop before batch: spent={ledger.spent:.6f} USD; "
                    f"worst-case {total_calls} calls at "
                    f"{per_call_ceiling:.6f}/call would exceed the "
                    f"{config.BUDGET_STOP_USD} USD hard stop."
                )

        # --- Step 4b: Concurrent (or serial) compute -------------------------
        if concurrency == 1:
            results = _run_batch_serial(batch, client, task_sampler)
        else:
            results = _run_batch_concurrent(batch, client, task_sampler, concurrency)

        # --- Step 4c: Serial write, sorted by canonical idx ------------------
        # Sorting guarantees canonical JSONL order regardless of completion order.
        results.sort(key=lambda r: r[0])
        idx_to_spec: dict[int, _MissionSpec] = {s[0]: s for s in batch}
        for spec_idx, outcome in results:
            _, mission_id, _, motif_obj, condition, _ = idx_to_spec[spec_idx]
            if isinstance(outcome, BaseException):
                _append_failure(
                    out_path,
                    mission_id=mission_id,
                    condition=condition,
                    motif=motif_obj.name,
                    error=repr(outcome),
                )
            else:
                logger.append(outcome)
                ledger.record(outcome.cost_usd)
                all_missions.append(outcome)
            issued += 1

        # --- Step 4d: Heartbeat after each batch -----------------------------
        _write_progress(
            out_path, completed=issued, total=total, spent_usd=ledger.spent,
        )

    return all_missions
