# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Agent motif definitions and mission runner (LLD-E §3.2).

A *motif* is a fixed multi-agent graph topology.  Five motifs are
pre-registered in :data:`MOTIF_LIBRARY`.  :func:`run_mission` executes
one mission through a motif, scoring each active node against the task's
gold judge and assembling a route-consistent :class:`MissionRecord`.

No real model calls occur unless a production client is injected.  Tests
pass a ``FakeClient`` that returns canned :class:`~.models.ModelResponse`
objects without touching any network.

Motif descriptions (LLD-E §3.2)
---------------------------------
series2
    A→B, one handoff.  Both nodes must satisfy the contract for Y_G=True.
series3
    A→B→C, two handoffs.  All three must pass.
parallel2
    Two independent branches + deterministic merge.  Y_G=True iff at
    least one branch passes AND the merge is valid.
    ``quorum_threshold=1``.
quorum2of3
    Three branches + deterministic 2-of-3 aggregator.  Y_G=True iff at
    least two branches pass AND the aggregation record is valid.
    ``quorum_threshold=2``.
hierarchy
    Supervisor selects worker_0 (fixed default); selected worker produces
    an answer; verifier accepts/corrects.  Only the activated route
    (supervisor→worker_0→verifier) counts.  worker_1 is logged as
    inactive with ``hard_ok=False, soft_ok=False, scored=False``.

Hard/soft/handoff contract rules (documented)
---------------------------------------------
hard_ok
    ``task.scorer(extract_answer(response.text))`` — True iff the raw
    output contains the correct answer.  For deterministic merge/aggregator
    nodes: True iff the quorum condition is met.  For inactive nodes: False.
soft_ok
    True iff ``response.text.strip()`` is non-empty (a response was
    received).  For deterministic aggregator nodes: always True.
    For inactive nodes: False.
handoff_ok
    True iff the *sending* component had ``hard_ok=True`` — the handoff
    carries a correct result downstream.  For parallel/quorum motifs only
    the successful branches emit handoffs; those are unconditionally True.
scored
    True for all actively executed (generative) nodes; False for
    deterministic aggregator nodes and inactive hierarchy workers.

Route-consistency rule
----------------------
For series and hierarchy motifs the *realized route* equals
``motif.route`` (all nodes, in order).

For parallel/quorum motifs the realized route is computed from outcomes:
  * If the quorum condition is met: ``tuple(passing_branch_ids) + (aggregator_id,)``
  * If not met: ``(aggregator_id,)``

This ensures :func:`~.logging_schema.compute_y_graph` — which checks the
AND of all on-route components — faithfully reflects "at-least-N" semantics
without the need to modify ``compute_y_graph``.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING, Final, Protocol

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments.logging_schema import (
    ComponentRecord,
    HandoffRecord,
    MissionRecord,
)
from agentassert_abc.experiments.tasks import Task, score

if TYPE_CHECKING:
    from agentassert_abc.experiments.models import ModelResponse

# ---------------------------------------------------------------------------
# Module-local exception
# ---------------------------------------------------------------------------


class MotifError(AgentAssertError):
    """Raised for unknown motif names or motif execution failures.

    Subclasses :class:`~agentassert_abc.exceptions.AgentAssertError` so
    callers can catch all AgentAssert failures uniformly.
    """


# ---------------------------------------------------------------------------
# ModelClient protocol — satisfied by LocalClient, FrontierClient, FakeClient
# ---------------------------------------------------------------------------


class ModelClient(Protocol):
    """Structural type accepted by :func:`run_mission` as the *client* argument.

    Any object that implements ``generate(model, prompt) → ModelResponse``
    satisfies this protocol.  No inheritance required.
    """

    def generate(self, model: str, prompt: str) -> ModelResponse:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Motif dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Motif:
    """Immutable specification of a multi-agent graph topology.

    Attributes
    ----------
    name:
        Canonical identifier (e.g. ``"series2"``, ``"quorum2of3"``).
    nodes:
        Tuple of all node IDs in this motif, including inactive branches.
        Node IDs encode role: ``"supervisor"``, ``"worker_0"``,
        ``"verifier"``, ``"aggregator"``, ``"merge"``, ``"node_a"``, etc.
    edges:
        Tuple of directed edges as ``(from_id, to_id)`` pairs.
    route:
        Default/canonical realized execution path.  For series motifs this
        equals the full execution order.  For hierarchy it encodes the
        selected worker.  For parallel/quorum it lists all branch IDs
        followed by the aggregator — the *realized* route computed at
        run-time may be a subset (see module docstring).
    quorum_threshold:
        For parallel/quorum motifs: minimum number of branches that must
        pass for the aggregator's ``hard_ok`` to be True.
        ``None`` for series and hierarchy.
    """

    name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    route: tuple[str, ...]
    quorum_threshold: int | None = None


# ---------------------------------------------------------------------------
# MOTIF_LIBRARY
# ---------------------------------------------------------------------------

#: Canonical five-motif library (LLD-E §3.2).
#: Annotated Final to signal that the reference should not be rebound.
#: The dict itself is mutable by Python semantics; callers must not
#: add/remove entries — use a private copy if custom motifs are needed.
MOTIF_LIBRARY: Final[dict[str, Motif]] = {
    "series2": Motif(
        name="series2",
        nodes=("node_a", "node_b"),
        edges=(("node_a", "node_b"),),
        route=("node_a", "node_b"),
    ),
    "series3": Motif(
        name="series3",
        nodes=("node_a", "node_b", "node_c"),
        edges=(("node_a", "node_b"), ("node_b", "node_c")),
        route=("node_a", "node_b", "node_c"),
    ),
    "parallel2": Motif(
        name="parallel2",
        nodes=("branch_a", "branch_b", "merge"),
        edges=(("branch_a", "merge"), ("branch_b", "merge")),
        route=("branch_a", "branch_b", "merge"),
        quorum_threshold=1,
    ),
    "quorum2of3": Motif(
        name="quorum2of3",
        nodes=("worker_0", "worker_1", "worker_2", "aggregator"),
        edges=(
            ("worker_0", "aggregator"),
            ("worker_1", "aggregator"),
            ("worker_2", "aggregator"),
        ),
        route=("worker_0", "worker_1", "worker_2", "aggregator"),
        quorum_threshold=2,
    ),
    "hierarchy": Motif(
        name="hierarchy",
        nodes=("supervisor", "worker_0", "worker_1", "verifier"),
        edges=(
            ("supervisor", "worker_0"),
            ("supervisor", "worker_1"),
            ("worker_0", "verifier"),
            ("worker_1", "verifier"),
        ),
        # Default activated route: supervisor selects worker_0
        route=("supervisor", "worker_0", "verifier"),
    ),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_INACTIVE_MODEL: str = "inactive"


def _role_from_node_id(node_id: str) -> str:
    """Derive a human-readable role label from a canonical node ID.

    Mapping rules (applied in order):
    - ``"supervisor"`` → ``"supervisor"``
    - ``"verifier"``   → ``"verifier"``
    - ``"aggregator"`` → ``"aggregator"``
    - ``"merge"``      → ``"merge"``
    - Starts with ``"worker"`` or ``"branch_"`` or ``"node_"`` → ``"worker"``
    - Anything else → ``"worker"`` (safe default)
    """
    if node_id in ("supervisor", "verifier", "aggregator", "merge"):
        return node_id
    return "worker"


def _build_prompt(task: Task, node_id: str, *, context: str) -> str:
    """Build a role-annotated prompt string for *node_id*.

    If *context* is non-empty the prompt is prefixed with the upstream
    output so downstream nodes receive a typed context token.

    Args:
        task:     The task whose prompt is used.
        node_id:  Identifies the node (used for role annotation only).
        context:  Prior output from the upstream node (empty for first node).

    Returns:
        A formatted prompt string.
    """
    role = _role_from_node_id(node_id).upper()
    if context:
        return f"[{role}] Context: {context}\n{task.prompt}"
    return f"[{role}] {task.prompt}"


def _make_comp(
    node_id: str,
    resp: ModelResponse,
    *,
    hard_ok: bool,
    soft_ok: bool,
    scored: bool,
) -> ComponentRecord:
    """Build an immutable ComponentRecord from a generate response."""
    return ComponentRecord(
        component_id=node_id,
        model=resp.model,
        role=_role_from_node_id(node_id),
        hard_ok=hard_ok,
        soft_ok=soft_ok,
        drift=None,
        raw_output=resp.text,
        scored=scored,
    )


def _inactive_comp(node_id: str) -> ComponentRecord:
    """Build a ComponentRecord for an inactive (non-executed) node."""
    return ComponentRecord(
        component_id=node_id,
        model=_INACTIVE_MODEL,
        role=_role_from_node_id(node_id),
        hard_ok=False,
        soft_ok=False,
        drift=None,
        raw_output="",
        scored=False,
    )


# ---------------------------------------------------------------------------
# Motif-specific execution helpers
# ---------------------------------------------------------------------------


def _run_series(
    motif: Motif,
    task: Task,
    model_assignment: dict[str, str],
    client: ModelClient,
) -> tuple[
    tuple[str, ...],
    tuple[ComponentRecord, ...],
    tuple[HandoffRecord, ...],
    int,
    float,
]:
    """Execute a series motif (series2 or series3).

    All nodes run in route order.  Each node receives the previous node's
    raw output as context.  The realized route equals ``motif.route``.

    Returns
    -------
    (route, components, handoffs, total_tokens, total_cost_usd)
    """
    comps: dict[str, ComponentRecord] = {}
    total_tokens = 0
    total_cost = 0.0
    prev_output = ""

    for node_id in motif.route:
        model = model_assignment[node_id]
        prompt = _build_prompt(task, node_id, context=prev_output)
        resp = client.generate(model, prompt)
        hard_ok = score(task, resp.text)
        soft_ok = len(resp.text.strip()) > 0
        comps[node_id] = _make_comp(
            node_id, resp, hard_ok=hard_ok, soft_ok=soft_ok, scored=True
        )
        prev_output = resp.text
        total_tokens += resp.input_tokens + resp.output_tokens
        total_cost += resp.cost_usd

    # Handoffs: each consecutive pair; ok iff sender had hard_ok
    handoffs: tuple[HandoffRecord, ...] = tuple(
        HandoffRecord(
            from_id=motif.route[i],
            to_id=motif.route[i + 1],
            handoff_ok=comps[motif.route[i]].hard_ok,
        )
        for i in range(len(motif.route) - 1)
    )

    components = tuple(comps[nid] for nid in motif.route)
    return motif.route, components, handoffs, total_tokens, total_cost


def _run_parallel_quorum(
    motif: Motif,
    task: Task,
    model_assignment: dict[str, str],
    client: ModelClient,
) -> tuple[
    tuple[str, ...],
    tuple[ComponentRecord, ...],
    tuple[HandoffRecord, ...],
    int,
    float,
]:
    """Execute a parallel or quorum motif.

    The last ID in ``motif.route`` is the deterministic aggregator/merge
    node; all preceding IDs are independent branches.  The aggregator is
    never passed to ``client.generate`` — its outcome is determined by the
    quorum condition.

    Route-consistency rule
    ~~~~~~~~~~~~~~~~~~~~~~
    * Count passing branches (``hard_ok AND soft_ok``).
    * If passing count ≥ quorum_threshold:
        realized_route = (all passing branch IDs) + (aggregator_id,)
        aggregator.hard_ok = True
    * Else:
        realized_route = (aggregator_id,)
        aggregator.hard_ok = False

    Handoffs connect each passing branch to the aggregator
    (handoff_ok=True for those connections).  Failing branches contribute
    to the full component vector but are absent from the realized route.

    Returns
    -------
    (realized_route, all_components, handoffs, total_tokens, total_cost_usd)
    """
    # Use explicit None check — avoid treating quorum_threshold=0 as falsy.
    quorum_threshold: int = 1 if motif.quorum_threshold is None else motif.quorum_threshold
    agg_id: str = motif.route[-1]
    branch_ids: tuple[str, ...] = motif.route[:-1]

    branch_comps: dict[str, ComponentRecord] = {}
    total_tokens = 0
    total_cost = 0.0

    for node_id in branch_ids:
        model = model_assignment[node_id]
        prompt = _build_prompt(task, node_id, context="")
        resp = client.generate(model, prompt)
        hard_ok = score(task, resp.text)
        soft_ok = len(resp.text.strip()) > 0
        branch_comps[node_id] = _make_comp(
            node_id, resp, hard_ok=hard_ok, soft_ok=soft_ok, scored=True
        )
        total_tokens += resp.input_tokens + resp.output_tokens
        total_cost += resp.cost_usd

    passing_ids: list[str] = [
        bid
        for bid in branch_ids
        if branch_comps[bid].hard_ok and branch_comps[bid].soft_ok
    ]
    quorum_met: bool = len(passing_ids) >= quorum_threshold

    # Deterministic aggregator — no generate call
    agg_comp = ComponentRecord(
        component_id=agg_id,
        model="deterministic",
        role=_role_from_node_id(agg_id),
        hard_ok=quorum_met,
        soft_ok=True,  # deterministic aggregator always produces output
        drift=None,
        raw_output="",
        scored=False,
    )

    # Realized route
    realized_route: tuple[str, ...]
    realized_route = tuple(passing_ids) + (agg_id,) if quorum_met else (agg_id,)

    # Handoffs: only passing branches emit handoffs to aggregator
    handoffs: tuple[HandoffRecord, ...] = tuple(
        HandoffRecord(from_id=pid, to_id=agg_id, handoff_ok=True)
        for pid in passing_ids
    )

    # Full component vector: all branches (in motif order) + aggregator
    all_comps: tuple[ComponentRecord, ...] = tuple(
        branch_comps[bid] for bid in branch_ids
    ) + (agg_comp,)

    return realized_route, all_comps, handoffs, total_tokens, total_cost


def _run_hierarchy(
    motif: Motif,
    task: Task,
    model_assignment: dict[str, str],
    client: ModelClient,
) -> tuple[
    tuple[str, ...],
    tuple[ComponentRecord, ...],
    tuple[HandoffRecord, ...],
    int,
    float,
]:
    """Execute the hierarchy motif.

    The activated route is encoded in ``motif.route``:
    ``(supervisor_id, selected_worker_id, verifier_id)``.

    All three active nodes call ``client.generate``.  Inactive worker
    nodes (those in ``motif.nodes`` but absent from ``motif.route``) are
    logged as inactive ComponentRecords with
    ``hard_ok=False, soft_ok=False, scored=False``.

    Handoffs:
        supervisor → selected_worker (ok iff supervisor hard_ok)
        selected_worker → verifier    (ok iff worker hard_ok)

    Returns
    -------
    (realized_route, all_components, handoffs, total_tokens, total_cost_usd)
    """
    if len(motif.route) != 3:
        raise MotifError(
            f"Hierarchy motif {motif.name!r} requires exactly 3 nodes in route "
            f"(supervisor, selected_worker, verifier); got {len(motif.route)}: "
            f"{motif.route}"
        )
    sup_id: str = motif.route[0]
    worker_id: str = motif.route[1]
    verifier_id: str = motif.route[2]
    inactive_ids: tuple[str, ...] = tuple(
        n for n in motif.nodes if n not in frozenset(motif.route)
    )

    comps: dict[str, ComponentRecord] = {}
    total_tokens = 0
    total_cost = 0.0

    # --- Supervisor ---
    resp_s = client.generate(
        model_assignment[sup_id],
        _build_prompt(task, sup_id, context=""),
    )
    hard_ok_s = score(task, resp_s.text)
    comps[sup_id] = _make_comp(
        sup_id, resp_s,
        hard_ok=hard_ok_s,
        soft_ok=len(resp_s.text.strip()) > 0,
        scored=True,
    )
    total_tokens += resp_s.input_tokens + resp_s.output_tokens
    total_cost += resp_s.cost_usd

    # --- Selected worker ---
    resp_w = client.generate(
        model_assignment[worker_id],
        _build_prompt(task, worker_id, context=resp_s.text),
    )
    hard_ok_w = score(task, resp_w.text)
    comps[worker_id] = _make_comp(
        worker_id, resp_w,
        hard_ok=hard_ok_w,
        soft_ok=len(resp_w.text.strip()) > 0,
        scored=True,
    )
    total_tokens += resp_w.input_tokens + resp_w.output_tokens
    total_cost += resp_w.cost_usd

    # --- Verifier ---
    resp_v = client.generate(
        model_assignment[verifier_id],
        _build_prompt(task, verifier_id, context=resp_w.text),
    )
    hard_ok_v = score(task, resp_v.text)
    comps[verifier_id] = _make_comp(
        verifier_id, resp_v,
        hard_ok=hard_ok_v,
        soft_ok=len(resp_v.text.strip()) > 0,
        scored=True,
    )
    total_tokens += resp_v.input_tokens + resp_v.output_tokens
    total_cost += resp_v.cost_usd

    # --- Inactive workers (logged, never executed) ---
    for iid in inactive_ids:
        comps[iid] = _inactive_comp(iid)

    # Handoffs along route (ok iff sender had hard_ok)
    handoffs: tuple[HandoffRecord, ...] = tuple(
        HandoffRecord(
            from_id=motif.route[i],
            to_id=motif.route[i + 1],
            handoff_ok=comps[motif.route[i]].hard_ok,
        )
        for i in range(len(motif.route) - 1)
    )

    # Full component vector in motif.nodes order (deterministic)
    all_comps: tuple[ComponentRecord, ...] = tuple(
        comps[nid] for nid in motif.nodes
    )

    return motif.route, all_comps, handoffs, total_tokens, total_cost


# ---------------------------------------------------------------------------
# Public API: run_mission
# ---------------------------------------------------------------------------

_UTC = datetime.UTC


def run_mission(
    motif: Motif,
    task: Task,
    model_assignment: dict[str, str],
    client: ModelClient,
    *,
    sharing_condition: str,
    cluster_id: str,
    mission_id: str,
) -> MissionRecord:
    """Execute one mission through *motif* and return a complete MissionRecord.

    For each generative node in the motif, calls
    ``client.generate(model_assignment[node_id], prompt)`` and scores the
    response against *task*'s gold judge.  Deterministic merge/aggregator
    nodes and inactive hierarchy workers do **not** call generate.

    The injected *client* decouples the runner from any real API.  Tests
    pass a ``FakeClient`` that returns canned responses; production callers
    pass a :class:`~.models.LocalClient` or
    :class:`~.models.FrontierClient`.

    Parameters
    ----------
    motif:
        A :class:`Motif` from :data:`MOTIF_LIBRARY` (or a custom one).
    task:
        The task whose prompt and gold scorer are used for every node.
    model_assignment:
        Mapping from generative node ID → model identifier string.
        Deterministic aggregator and inactive nodes need not be included.
    client:
        Any object with ``generate(model, prompt) → ModelResponse``.
    sharing_condition:
        Model-sharing label (``"same_model"``, ``"same_vendor"``, or
        ``"different_vendor"``).  Stored verbatim in the returned record.
    cluster_id:
        Scenario ID used as the bootstrap cluster key.
    mission_id:
        Unique identifier for this mission execution.

    Returns
    -------
    MissionRecord
        Complete per-mission record with ``y_graph`` auto-computed from the
        realized route via :func:`~.logging_schema.compute_y_graph`.

    Raises
    ------
    MotifError
        If *motif.name* is not one of the five registered names.
    KeyError
        If *model_assignment* is missing a required generative node ID.
    """
    if motif.name in ("series2", "series3"):
        route, comps, handoffs, tokens, cost = _run_series(
            motif, task, model_assignment, client
        )
    elif motif.name in ("parallel2", "quorum2of3"):
        route, comps, handoffs, tokens, cost = _run_parallel_quorum(
            motif, task, model_assignment, client
        )
    elif motif.name == "hierarchy":
        route, comps, handoffs, tokens, cost = _run_hierarchy(
            motif, task, model_assignment, client
        )
    else:
        raise MotifError(
            f"Unknown motif name {motif.name!r}. "
            f"Registered names: {sorted(MOTIF_LIBRARY)}"
        )

    timestamp = (
        datetime.datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    return MissionRecord.make(
        mission_id=mission_id,
        cluster_id=cluster_id,
        motif=motif.name,
        sharing_condition=sharing_condition,
        route=route,
        components=comps,
        handoffs=handoffs,
        tokens=tokens,
        cost_usd=cost,
        timestamp=timestamp,
    )
