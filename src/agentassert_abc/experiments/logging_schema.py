# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Per-mission experiment logging schema (LLD-C §9, LLD-E §7).

Defines immutable data records capturing everything needed to reconstruct
the route-consistent whole-graph satisfaction Y_graph and all downstream
estimators.  Component vectors are stored in full — NOT collapsed to tier
summaries H, S, Y — because route-consistency and dependence diagnostics
cannot be reconstructed from tier summaries alone (LLD-C §9).

Core objects
------------
ComponentRecord
    Per-node state: X^H_{jr}, X^S_{jr}, drift D_i, raw output.
HandoffRecord
    Per-edge outcome: H_e ∈ {0, 1}.
MissionRecord
    Per-mission aggregate with realized route, full component/handoff
    vectors, explicit y_graph, token count, cost, and timestamp.
JsonlLogger
    Append-only UTF-8 JSONL writer/reader with exact round-trip guarantee.
compute_y_graph
    Route-consistent Y_{G,r} = φ_{G,r}(x^H ⊙ x^S), LLD-C Eq 1.1.

References
----------
LLD-C-graph-eprocess-v2.md  §9  (logging interface)
LLD-E-experiment-design-v2.md   §7  (mission/component records)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from agentassert_abc.exceptions import AgentAssertError

# ---------------------------------------------------------------------------
# Module-local exception
# ---------------------------------------------------------------------------


class LoggingSchemaError(AgentAssertError):
    """Raised for malformed JSONL lines or schema deserialization failures.

    Subclasses AgentAssertError so callers can catch all AgentAssert failures
    uniformly (repo convention).
    """


# ---------------------------------------------------------------------------
# Immutable records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentRecord:
    """Per-component outcome record (LLD-E §7.2).

    Attributes
    ----------
    component_id:
        Unique identifier for this node in the mission graph.
    model:
        Exact model identifier / digest (e.g. ``"qwen2.5:7b"``).
    role:
        Role this component played (e.g. ``"worker"``, ``"supervisor"``).
    hard_ok:
        X^H_{jr} = 1 iff no applicable hard violation occurred in this
        mission.  Permanently absorbing: once False it cannot become True
        within the mission (LLD-C §6.4).
    soft_ok:
        X^S_{jr}(δ_j, k_j) = 1 iff every soft excursion recovered within
        its k_j-step deadline (LLD-C §1.3).
    drift:
        D_i = 1 − S_i ∈ [0, 1], or None when not measured / not applicable.
    raw_output:
        Verbatim model response or parsed-and-re-serialised candidate.
        Stored for scorer audit and extraction sensitivity analysis.
    scored:
        True iff this component contributed to a primary hard score.
    """

    component_id: str
    model: str
    role: str
    hard_ok: bool
    soft_ok: bool
    drift: float | None
    raw_output: str
    scored: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HandoffRecord:
    """Per-edge handoff outcome record (LLD-E §7.2).

    Attributes
    ----------
    from_id:
        component_id of the sending node.
    to_id:
        component_id of the receiving node.
    handoff_ok:
        H_e ∈ {0, 1}: True iff the schema version matched, all required
        fields were present, the payload checksum verified, and the
        semantic-gold check (where applicable) passed.
    """

    from_id: str
    to_id: str
    handoff_ok: bool


@dataclasses.dataclass(frozen=True, slots=True)
class MissionRecord:
    """Complete per-mission record (LLD-E §7.1).

    Stores the full component vector so that route-consistency and
    dependence diagnostics can be reconstructed exactly.  y_graph is the
    explicit route-consistent terminal label Y_{G,r}; it is computed from
    components and handoffs (see :func:`compute_y_graph`) and stored for
    direct downstream consumption.

    Use :meth:`MissionRecord.make` to construct a record and auto-compute
    y_graph in one call.

    Attributes
    ----------
    mission_id:
        Unique identifier for this mission execution.
    cluster_id:
        Scenario ID used as the bootstrap cluster key (LLD-E §8.1).
    motif:
        Graph topology: ``"series2"``, ``"series3"``, ``"parallel2"``,
        ``"quorum3"``, or ``"hierarchy"``.
    sharing_condition:
        Model-sharing condition: ``"same_model"``, ``"same_vendor"``,
        or ``"different_vendor"``.
    route:
        Ordered tuple of component_ids on the realized execution path.
        Inactive branches (e.g. the unselected hierarchy worker) are
        absent; they are logged as inactive in their ComponentRecord.
    components:
        Full vector of ComponentRecord objects for every attempted node,
        including inactive-branch nodes (logged, never promoted to success).
    handoffs:
        Tuple of HandoffRecord objects for every attempted edge, including
        off-route edges.
    y_graph:
        Y_{G,r} = φ_{G,r}(x^H ⊙ x^S) — route-consistent whole-graph
        satisfaction (LLD-C Eq 1.1).  Stored explicitly; equals the result
        of compute_y_graph(route, components, handoffs).
    tokens:
        Total token count across all active agents in this mission.
    cost_usd:
        Reconstructed dollar cost for this mission's API calls.
    timestamp:
        ISO 8601 UTC creation timestamp (e.g. ``"2026-07-26T00:00:00Z"``).
    """

    mission_id: str
    cluster_id: str
    motif: str
    sharing_condition: str
    route: tuple[str, ...]
    components: tuple[ComponentRecord, ...]
    handoffs: tuple[HandoffRecord, ...]
    y_graph: bool
    tokens: int
    cost_usd: float
    timestamp: str

    @classmethod
    def make(
        cls,
        *,
        mission_id: str,
        cluster_id: str,
        motif: str,
        sharing_condition: str,
        route: tuple[str, ...],
        components: tuple[ComponentRecord, ...],
        handoffs: tuple[HandoffRecord, ...],
        tokens: int,
        cost_usd: float,
        timestamp: str,
    ) -> MissionRecord:
        """Construct a MissionRecord with y_graph computed automatically.

        y_graph is derived from *route*, *components*, and *handoffs* via
        :func:`compute_y_graph` (route-consistent LLD-C Eq 1.1 semantics).
        The caller does not pass y_graph; it is always computed, never
        supplied externally, preventing accidental staleness.
        """
        y_graph = compute_y_graph(route, components, handoffs)
        return cls(
            mission_id=mission_id,
            cluster_id=cluster_id,
            motif=motif,
            sharing_condition=sharing_condition,
            route=route,
            components=components,
            handoffs=handoffs,
            y_graph=y_graph,
            tokens=tokens,
            cost_usd=cost_usd,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Route-consistent Y_{G,r} computation
# ---------------------------------------------------------------------------


def compute_y_graph(
    route: tuple[str, ...],
    components: tuple[ComponentRecord, ...],
    handoffs: tuple[HandoffRecord, ...],
) -> bool:
    """Compute route-consistent Y_{G,r} from LLD-C Equation 1.1.

    Y_{G,r} = φ_{G,r}(x^H ⊙ x^S) is True iff both conditions hold:

    1. **Component condition** — every ComponentRecord whose
       ``component_id`` appears in *route* satisfies
       ``hard_ok AND soft_ok``.  Off-route components (inactive
       branches in a parallel or hierarchy motif) are not checked;
       their failure does not block certification.

    2. **Handoff condition** — every HandoffRecord whose ``from_id``
       AND ``to_id`` both appear in *route* satisfies ``handoff_ok``.
       A handoff with one endpoint outside the route is not on the
       realized path and is not checked.

    This is a faithful implementation of the union-over-minimal-
    successful-sets formula (LLD-C Eq 6.3): for the realized route the
    required minimal successful set is exactly the route nodes, and
    the required edges are those connecting route nodes.

    Vacuous case: an empty *route* returns ``True`` because there are
    no required nodes or edges to fail.

    Parameters
    ----------
    route:
        Ordered realized execution path (component IDs).
    components:
        All ComponentRecord objects for this mission (including
        off-route nodes).
    handoffs:
        All HandoffRecord objects for this mission (including
        off-route edges).

    Returns
    -------
    bool
        Route-consistent Y_{G,r}.
    """
    route_set: frozenset[str] = frozenset(route)

    for comp in components:
        if comp.component_id in route_set and not (comp.hard_ok and comp.soft_ok):
            return False

    for hoff in handoffs:
        if hoff.from_id in route_set and hoff.to_id in route_set and not hoff.handoff_ok:
            return False

    return True


# ---------------------------------------------------------------------------
# JSONL serialization helpers (module-private)
# ---------------------------------------------------------------------------


def _record_to_dict(record: MissionRecord) -> dict[str, Any]:
    """Serialize a MissionRecord to a JSON-safe plain dict.

    Uses :func:`dataclasses.asdict` which recursively converts nested
    frozen dataclasses to dicts and converts tuples to lists (JSON has
    no native tuple type).  The reverse is handled by :func:`_dict_to_record`.
    """
    return dataclasses.asdict(record)


def _dict_to_record(data: dict[str, Any]) -> MissionRecord:
    """Deserialize a JSON-parsed dict back to an exact MissionRecord.

    Reconstructs all frozen dataclasses and converts JSON arrays back
    to Python tuples so that equality with the original holds.

    Raises
    ------
    LoggingSchemaError
        On any missing field, wrong type, or other structural mismatch.
    """
    try:
        components: tuple[ComponentRecord, ...] = tuple(
            ComponentRecord(
                component_id=c["component_id"],
                model=c["model"],
                role=c["role"],
                hard_ok=c["hard_ok"],
                soft_ok=c["soft_ok"],
                drift=c["drift"],
                raw_output=c["raw_output"],
                scored=c["scored"],
            )
            for c in data["components"]
        )
        handoffs: tuple[HandoffRecord, ...] = tuple(
            HandoffRecord(
                from_id=h["from_id"],
                to_id=h["to_id"],
                handoff_ok=h["handoff_ok"],
            )
            for h in data["handoffs"]
        )
        return MissionRecord(
            mission_id=data["mission_id"],
            cluster_id=data["cluster_id"],
            motif=data["motif"],
            sharing_condition=data["sharing_condition"],
            route=tuple(data["route"]),
            components=components,
            handoffs=handoffs,
            y_graph=data["y_graph"],
            tokens=data["tokens"],
            cost_usd=data["cost_usd"],
            timestamp=data["timestamp"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LoggingSchemaError(
            f"Cannot deserialize MissionRecord: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# JsonlLogger
# ---------------------------------------------------------------------------


class JsonlLogger:
    """Append-only UTF-8 JSONL logger for MissionRecord objects.

    Each line is one MissionRecord serialized as compact JSON.  The reader
    reconstructs exact MissionRecord values — including all component
    vectors — from each line.

    Round-trip guarantee: ``read_all()`` after ``append(record)`` returns a
    list containing a value equal to *record* (all fields, including nested
    ComponentRecord and HandoffRecord tuples).

    Parameters
    ----------
    path:
        Path to the JSONL file.  The file is created on first append.

    Examples
    --------
    >>> logger = JsonlLogger("missions.jsonl")
    >>> logger.append(record)
    >>> records = logger.read_all()
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Write

    def append(self, record: MissionRecord) -> None:
        """Append one MissionRecord as a single UTF-8 JSON line.

        The line is written atomically in the sense that a complete
        ``\\n``-terminated JSON object is flushed before returning.
        """
        line = json.dumps(_record_to_dict(record), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------
    # Read

    def read_all(self) -> list[MissionRecord]:
        """Read all MissionRecord lines from the JSONL file.

        Returns an empty list if the file does not exist or is empty.
        Blank lines are skipped.

        Raises
        ------
        LoggingSchemaError
            On any line that is not valid JSON or cannot be deserialized
            to a MissionRecord.
        """
        if not self._path.exists():
            return []

        records: list[MissionRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise LoggingSchemaError(
                        f"Invalid JSON at line {lineno} in {self._path}: {exc}"
                    ) from exc
                records.append(_dict_to_record(data))

        return records
