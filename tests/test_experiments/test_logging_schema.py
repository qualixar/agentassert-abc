# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for experiments/logging_schema.py (LLD-C §9, LLD-E §7).

TDD RED → GREEN: tests are written before the implementation.

Coverage targets:
  - y_graph=True for all-pass route (all components + handoffs succeed)
  - y_graph=False when on-route component fails hard_ok
  - y_graph=False when on-route component fails soft_ok
  - y_graph=False when on-route handoff fails
  - y_graph=True when off-route component fails (route-consistent semantics)
  - y_graph=True when off-route handoff fails
  - JSONL round-trip: MissionRecord → file → MissionRecord exact equality
  - Multiple records round-trip preserves order and values
  - Component vectors preserved in round-trip (not collapsed to tier summaries)
  - drift=None preserved through JSONL
  - MissionRecord.make() auto-computes y_graph from components+handoffs
  - compute_y_graph() empty route returns True
  - LoggingSchemaError raised on malformed JSONL
  - read_all() returns empty list for non-existent file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentassert_abc.experiments.logging_schema import (
    ComponentRecord,
    HandoffRecord,
    JsonlLogger,
    LoggingSchemaError,
    MissionRecord,
    compute_y_graph,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

_TIMESTAMP = "2026-07-26T00:00:00Z"
_MODEL = "qwen2.5:7b"


def _comp(
    cid: str,
    *,
    hard_ok: bool = True,
    soft_ok: bool = True,
    drift: float | None = 0.05,
    role: str = "worker",
    raw_output: str = '{"status":"ok"}',
    scored: bool = True,
) -> ComponentRecord:
    return ComponentRecord(
        component_id=cid,
        model=_MODEL,
        role=role,
        hard_ok=hard_ok,
        soft_ok=soft_ok,
        drift=drift,
        raw_output=raw_output,
        scored=scored,
    )


def _hoff(from_id: str, to_id: str, *, ok: bool = True) -> HandoffRecord:
    return HandoffRecord(from_id=from_id, to_id=to_id, handoff_ok=ok)


def _make_mission(
    *,
    mission_id: str = "m001",
    cluster_id: str = "s001",
    motif: str = "series2",
    sharing_condition: str = "same_model",
    route: tuple[str, ...] = ("A", "B"),
    components: tuple[ComponentRecord, ...] | None = None,
    handoffs: tuple[HandoffRecord, ...] | None = None,
    y_graph: bool = True,
    tokens: int = 120,
    cost_usd: float = 0.0,
    timestamp: str = _TIMESTAMP,
) -> MissionRecord:
    if components is None:
        components = (_comp("A"), _comp("B"))
    if handoffs is None:
        handoffs = (_hoff("A", "B"),)
    return MissionRecord(
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
# compute_y_graph: correctness
# ---------------------------------------------------------------------------


class TestComputeYGraph:
    """Unit tests for the standalone compute_y_graph function."""

    def test_all_pass_returns_true(self) -> None:
        """All on-route components pass hard+soft; all handoffs pass → True."""
        comps = (_comp("A"), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is True

    def test_hard_fail_on_route_returns_false(self) -> None:
        """One on-route component with hard_ok=False → False."""
        comps = (_comp("A", hard_ok=False), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is False

    def test_soft_fail_on_route_returns_false(self) -> None:
        """One on-route component with soft_ok=False → False."""
        comps = (_comp("A", soft_ok=False), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is False

    def test_both_fail_on_route_returns_false(self) -> None:
        """hard_ok=False AND soft_ok=False on same on-route component → False."""
        comps = (_comp("A", hard_ok=False, soft_ok=False), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is False

    def test_handoff_fail_on_route_returns_false(self) -> None:
        """Handoff connecting two route nodes fails → False."""
        comps = (_comp("A"), _comp("B"))
        hoffs = (_hoff("A", "B", ok=False),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is False

    def test_off_route_hard_fail_does_not_affect_ygraph(self) -> None:
        """Component NOT in route can fail without affecting y_graph (route-consistent)."""
        # C is off-route (route is only A, B)
        comps = (_comp("A"), _comp("B"), _comp("C", hard_ok=False))
        hoffs = (_hoff("A", "B"),)
        assert compute_y_graph(("A", "B"), comps, hoffs) is True

    def test_off_route_handoff_fail_does_not_affect_ygraph(self) -> None:
        """Handoff between two off-route nodes does not block y_graph."""
        # Route is A → B; handoff C→D is off-route (neither C nor D in route)
        comps = (_comp("A"), _comp("B"), _comp("C"), _comp("D"))
        hoffs = (_hoff("A", "B"), _hoff("C", "D", ok=False))
        assert compute_y_graph(("A", "B"), comps, hoffs) is True

    def test_empty_route_returns_true(self) -> None:
        """Empty route: no components to check → True (vacuously)."""
        comps = (_comp("A", hard_ok=False),)
        hoffs = (_hoff("A", "B", ok=False),)
        assert compute_y_graph((), comps, hoffs) is True

    def test_no_components_no_handoffs_returns_true(self) -> None:
        """No components or handoffs on route → True."""
        assert compute_y_graph(("A",), (), ()) is True

    def test_series3_all_pass(self) -> None:
        """Series-3 motif: A→B→C all pass."""
        comps = (_comp("A"), _comp("B"), _comp("C"))
        hoffs = (_hoff("A", "B"), _hoff("B", "C"))
        assert compute_y_graph(("A", "B", "C"), comps, hoffs) is True

    def test_series3_middle_hard_fail(self) -> None:
        """Series-3 motif: B fails hard_ok → False."""
        comps = (_comp("A"), _comp("B", hard_ok=False), _comp("C"))
        hoffs = (_hoff("A", "B"), _hoff("B", "C"))
        assert compute_y_graph(("A", "B", "C"), comps, hoffs) is False

    def test_parallel_off_branch_does_not_block(self) -> None:
        """Parallel-2: only branch1 in route; branch2 fails but is off-route → True."""
        # Route = [branch1, merge]; branch2 is inactive
        comps = (_comp("branch1"), _comp("branch2", hard_ok=False), _comp("merge"))
        hoffs = (_hoff("branch1", "merge"), _hoff("branch2", "merge"))
        assert compute_y_graph(("branch1", "merge"), comps, hoffs) is True

    def test_handoff_only_from_on_route_to_off_route_not_checked(self) -> None:
        """Handoff from on-route to off-route node is not checked (to_id absent)."""
        comps = (_comp("A"), _comp("B"))
        # B is on route, but C is not — handoff B→C should be ignored
        hoffs = (_hoff("A", "B"), _hoff("B", "C", ok=False))
        assert compute_y_graph(("A", "B"), comps, hoffs) is True


# ---------------------------------------------------------------------------
# MissionRecord.make() classmethod
# ---------------------------------------------------------------------------


class TestMissionRecordMake:
    """Classmethod .make() computes y_graph automatically."""

    def test_make_computes_y_graph_true(self) -> None:
        comps = (_comp("A"), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        rec = MissionRecord.make(
            mission_id="m1",
            cluster_id="s1",
            motif="series2",
            sharing_condition="same_model",
            route=("A", "B"),
            components=comps,
            handoffs=hoffs,
            tokens=50,
            cost_usd=0.0,
            timestamp=_TIMESTAMP,
        )
        assert rec.y_graph is True

    def test_make_computes_y_graph_false_on_hard_fail(self) -> None:
        comps = (_comp("A", hard_ok=False), _comp("B"))
        hoffs = (_hoff("A", "B"),)
        rec = MissionRecord.make(
            mission_id="m2",
            cluster_id="s1",
            motif="series2",
            sharing_condition="same_model",
            route=("A", "B"),
            components=comps,
            handoffs=hoffs,
            tokens=50,
            cost_usd=0.0,
            timestamp=_TIMESTAMP,
        )
        assert rec.y_graph is False

    def test_make_computes_y_graph_false_on_handoff_fail(self) -> None:
        comps = (_comp("A"), _comp("B"))
        hoffs = (_hoff("A", "B", ok=False),)
        rec = MissionRecord.make(
            mission_id="m3",
            cluster_id="s1",
            motif="series2",
            sharing_condition="same_model",
            route=("A", "B"),
            components=comps,
            handoffs=hoffs,
            tokens=50,
            cost_usd=0.0,
            timestamp=_TIMESTAMP,
        )
        assert rec.y_graph is False


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


class TestJsonlRoundTrip:
    """MissionRecord → JSONL → MissionRecord must be exact."""

    def test_single_record_round_trip(self, tmp_path: Path) -> None:
        """Single append + read_all returns identical record."""
        path = tmp_path / "missions.jsonl"
        record = _make_mission()
        logger = JsonlLogger(path)
        logger.append(record)
        loaded = logger.read_all()
        assert len(loaded) == 1
        assert loaded[0] == record

    def test_multiple_records_round_trip(self, tmp_path: Path) -> None:
        """Multiple appends preserve order and all field values."""
        path = tmp_path / "missions.jsonl"
        records = [
            _make_mission(mission_id="m001"),
            _make_mission(mission_id="m002", y_graph=False),
            _make_mission(mission_id="m003", motif="series3"),
        ]
        logger = JsonlLogger(path)
        for rec in records:
            logger.append(rec)
        loaded = logger.read_all()
        assert loaded == records

    def test_component_vectors_preserved_not_collapsed(self, tmp_path: Path) -> None:
        """Each component's hard_ok, soft_ok, drift preserved (not collapsed to H/S/Y)."""
        comps = (
            _comp("A", hard_ok=True, soft_ok=True, drift=0.02),
            _comp("B", hard_ok=True, soft_ok=False, drift=0.7),  # off-route
        )
        record = _make_mission(
            route=("A",),
            components=comps,
            handoffs=(),
            y_graph=True,  # B is off-route; only A matters
        )
        path = tmp_path / "missions.jsonl"
        logger = JsonlLogger(path)
        logger.append(record)
        loaded = logger.read_all()[0]

        # Both components preserved verbatim
        assert loaded.components[0].component_id == "A"
        assert loaded.components[0].hard_ok is True
        assert loaded.components[0].soft_ok is True
        assert loaded.components[0].drift == pytest.approx(0.02)

        assert loaded.components[1].component_id == "B"
        assert loaded.components[1].hard_ok is True
        assert loaded.components[1].soft_ok is False
        assert loaded.components[1].drift == pytest.approx(0.7)

    def test_drift_none_preserved(self, tmp_path: Path) -> None:
        """drift=None serializes to JSON null and deserializes back to None."""
        comps = (_comp("A", drift=None),)
        record = _make_mission(route=("A",), components=comps, handoffs=())
        path = tmp_path / "missions.jsonl"
        logger = JsonlLogger(path)
        logger.append(record)
        loaded = logger.read_all()[0]
        assert loaded.components[0].drift is None

    def test_route_tuple_preserved(self, tmp_path: Path) -> None:
        """route is a tuple, not a list, after round-trip."""
        record = _make_mission(route=("A", "B", "C"))
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert isinstance(loaded.route, tuple)
        assert loaded.route == ("A", "B", "C")

    def test_components_handoffs_are_tuples(self, tmp_path: Path) -> None:
        """components and handoffs are tuples after round-trip."""
        record = _make_mission()
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert isinstance(loaded.components, tuple)
        assert isinstance(loaded.handoffs, tuple)

    def test_y_graph_false_preserved(self, tmp_path: Path) -> None:
        """y_graph=False round-trips exactly."""
        record = _make_mission(y_graph=False)
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert loaded.y_graph is False

    def test_handoff_record_fields_preserved(self, tmp_path: Path) -> None:
        """HandoffRecord from_id, to_id, handoff_ok all preserved."""
        hoffs = (_hoff("comp1", "comp2", ok=False),)
        record = _make_mission(
            route=("comp1", "comp2"),
            components=(_comp("comp1"), _comp("comp2")),
            handoffs=hoffs,
            y_graph=False,
        )
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert loaded.handoffs[0].from_id == "comp1"
        assert loaded.handoffs[0].to_id == "comp2"
        assert loaded.handoffs[0].handoff_ok is False

    def test_scored_field_preserved(self, tmp_path: Path) -> None:
        """scored=False is preserved through JSONL."""
        comps = (_comp("A", scored=False),)
        record = _make_mission(route=("A",), components=comps, handoffs=())
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert loaded.components[0].scored is False

    def test_raw_output_preserved(self, tmp_path: Path) -> None:
        """raw_output string (including special chars) is preserved verbatim."""
        raw = '{"msg": "héllo wörld\\n", "val": 42}'
        comps = (_comp("A", raw_output=raw),)
        record = _make_mission(route=("A",), components=comps, handoffs=())
        path = tmp_path / "missions.jsonl"
        JsonlLogger(path).append(record)
        loaded = JsonlLogger(path).read_all()[0]
        assert loaded.components[0].raw_output == raw

    def test_read_all_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """read_all on a non-existent path returns an empty list."""
        logger = JsonlLogger(tmp_path / "does_not_exist.jsonl")
        assert logger.read_all() == []

    def test_each_jsonl_line_is_valid_json(self, tmp_path: Path) -> None:
        """Each line written by append() is independently valid JSON."""
        path = tmp_path / "missions.jsonl"
        records = [_make_mission(mission_id="m001"), _make_mission(mission_id="m002")]
        logger = JsonlLogger(path)
        for rec in records:
            logger.append(rec)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "mission_id" in parsed
            assert "y_graph" in parsed
            assert "components" in parsed

    def test_append_is_idempotent_on_sequential_calls(self, tmp_path: Path) -> None:
        """Two append calls produce two lines (not one overwritten line)."""
        path = tmp_path / "missions.jsonl"
        rec1 = _make_mission(mission_id="m001")
        rec2 = _make_mission(mission_id="m002")
        logger = JsonlLogger(path)
        logger.append(rec1)
        logger.append(rec2)
        loaded = logger.read_all()
        assert loaded[0].mission_id == "m001"
        assert loaded[1].mission_id == "m002"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """LoggingSchemaError raised on malformed data."""

    def test_malformed_json_raises_logging_schema_error(self, tmp_path: Path) -> None:
        """A line that is not valid JSON raises LoggingSchemaError."""
        path = tmp_path / "bad.jsonl"
        path.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(LoggingSchemaError):
            JsonlLogger(path).read_all()

    def test_missing_field_raises_logging_schema_error(self, tmp_path: Path) -> None:
        """Valid JSON but missing required field raises LoggingSchemaError."""
        path = tmp_path / "bad.jsonl"
        # Write JSON missing 'components'
        path.write_text(
            json.dumps({"mission_id": "x", "cluster_id": "c"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(LoggingSchemaError):
            JsonlLogger(path).read_all()

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path: Path) -> None:
        """Blank lines interspersed in JSONL are silently skipped."""
        path = tmp_path / "blanks.jsonl"
        rec = _make_mission(mission_id="m001")
        logger = JsonlLogger(path)
        logger.append(rec)
        # Inject a blank line between records
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n")
        logger.append(_make_mission(mission_id="m002"))
        loaded = logger.read_all()
        assert len(loaded) == 2
        assert loaded[0].mission_id == "m001"
        assert loaded[1].mission_id == "m002"


# ---------------------------------------------------------------------------
# Immutability contract
# ---------------------------------------------------------------------------


class TestImmutability:
    """Records must be frozen — mutation attempts raise FrozenInstanceError."""

    def test_component_record_is_frozen(self) -> None:
        comp = _comp("A")
        with pytest.raises(Exception):  # FrozenInstanceError inherits from AttributeError
            comp.hard_ok = False  # type: ignore[misc]

    def test_handoff_record_is_frozen(self) -> None:
        hoff = _hoff("A", "B")
        with pytest.raises(Exception):
            hoff.handoff_ok = False  # type: ignore[misc]

    def test_mission_record_is_frozen(self) -> None:
        rec = _make_mission()
        with pytest.raises(Exception):
            rec.y_graph = False  # type: ignore[misc]
