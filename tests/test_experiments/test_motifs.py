# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for experiments/motifs.py (LLD-E §3.2 agent motifs).

TDD RED → GREEN: tests are written BEFORE the implementation.

Coverage targets (task-prompt requirements):
  - series-2 both correct  → y_graph True
  - series-2 node-2 wrong  → hard_ok False, y_graph False
  - quorum-2of3 2-of-3 correct → y_graph True
  - quorum-2of3 1-of-3 correct → y_graph False
  - MissionRecord round-trips through JsonlLogger
  - MOTIF_LIBRARY has 5 expected entries
  - MotifError is an AgentAssertError subclass
  - Motif dataclass is frozen (immutable)
  - parallel-2 at-least-one-branch semantics
  - hierarchy route semantics + inactive branch logged
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments.logging_schema import JsonlLogger
from agentassert_abc.experiments.models import ModelResponse
from agentassert_abc.experiments.tasks import TASK_LIBRARY

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# FakeClient — zero network calls, canned outputs keyed by model name
# ---------------------------------------------------------------------------

_TASK = TASK_LIBRARY[0]  # arith_add: ground_truth = "6912"
_CORRECT = "6912"
_WRONG = "0000"
_SHARING = "same_model"
_CLUSTER = "scenario-001"
_MISSION = "mission-001"


class FakeClient:
    """Synchronous test double for any ModelClient.

    Returns a ModelResponse whose text is looked up by model name from the
    supplied ``outputs`` dict.  Missing keys return an empty string so callers
    that don't care about correctness don't need exhaustive coverage.
    No network or subprocess activity occurs.
    """

    def __init__(self, outputs: dict[str, str]) -> None:
        self._outputs = outputs

    def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
        text = self._outputs.get(model, "")
        return ModelResponse(
            text=text,
            input_tokens=10,
            output_tokens=5,
            model=model,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# TestMotifLibrary — structural assertions on MOTIF_LIBRARY
# ---------------------------------------------------------------------------


class TestMotifLibrary:
    def test_library_has_five_entries(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert len(MOTIF_LIBRARY) == 5

    def test_library_has_expected_keys(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert set(MOTIF_LIBRARY.keys()) == {
            "series2",
            "series3",
            "parallel2",
            "quorum2of3",
            "hierarchy",
        }

    def test_motif_is_frozen(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        motif = MOTIF_LIBRARY["series2"]
        with pytest.raises((AttributeError, TypeError)):
            motif.name = "tampered"  # type: ignore[misc]

    def test_series2_route_length(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert len(MOTIF_LIBRARY["series2"].route) == 2

    def test_series3_route_length(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert len(MOTIF_LIBRARY["series3"].route) == 3

    def test_quorum2of3_threshold_is_two(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert MOTIF_LIBRARY["quorum2of3"].quorum_threshold == 2

    def test_parallel2_threshold_is_one(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        assert MOTIF_LIBRARY["parallel2"].quorum_threshold == 1

    def test_motif_error_is_agent_assert_error(self) -> None:
        from agentassert_abc.experiments.motifs import MotifError

        assert issubclass(MotifError, AgentAssertError)

    def test_hierarchy_has_inactive_worker(self) -> None:
        """hierarchy.nodes must have more nodes than hierarchy.route."""
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        motif = MOTIF_LIBRARY["hierarchy"]
        assert len(motif.nodes) > len(motif.route)


# ---------------------------------------------------------------------------
# TestSeries2
# ---------------------------------------------------------------------------


class TestSeries2:
    """Series-2: A→B, one handoff.  Both nodes must pass for y_graph=True."""

    def _run(self, a_text: str, b_text: str):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["series2"]
        node_a, node_b = motif.route[0], motif.route[1]
        client = FakeClient({f"{node_a}_m": a_text, f"{node_b}_m": b_text})
        assignment = {node_a: f"{node_a}_m", node_b: f"{node_b}_m"}
        return run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )

    # --- Task-prompt requirement 1 ---
    def test_both_correct_y_graph_true(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.y_graph is True

    # --- Task-prompt requirement 2 ---
    def test_node2_wrong_hard_ok_false_and_y_graph_false(self) -> None:
        """node-2 wrong → its hard_ok False AND y_graph False."""
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(_CORRECT, _WRONG)
        node_b_id = MOTIF_LIBRARY["series2"].route[1]
        node_b_comp = next(
            c for c in record.components if c.component_id == node_b_id
        )
        assert node_b_comp.hard_ok is False
        assert record.y_graph is False

    def test_node1_wrong_y_graph_false(self) -> None:
        record = self._run(_WRONG, _CORRECT)
        assert record.y_graph is False

    def test_both_wrong_y_graph_false(self) -> None:
        record = self._run(_WRONG, _WRONG)
        assert record.y_graph is False

    def test_motif_field_stored(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.motif == "series2"

    def test_sharing_condition_stored(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.sharing_condition == _SHARING

    def test_cluster_id_stored(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.cluster_id == _CLUSTER

    def test_tokens_nonzero(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.tokens > 0

    def test_cost_usd_nonnegative(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.cost_usd >= 0.0

    def test_route_length_is_two(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert len(record.route) == 2

    def test_handoff_count_is_one(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert len(record.handoffs) == 1

    def test_component_count_is_two(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert len(record.components) == 2

    def test_node1_correct_hard_ok_true(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(_CORRECT, _WRONG)
        node_a_id = MOTIF_LIBRARY["series2"].route[0]
        node_a_comp = next(
            c for c in record.components if c.component_id == node_a_id
        )
        assert node_a_comp.hard_ok is True

    def test_scored_flag_true_for_route_nodes(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        for comp in record.components:
            assert comp.scored is True

    def test_handoff_ok_when_sender_correct(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert record.handoffs[0].handoff_ok is True

    def test_handoff_not_ok_when_sender_wrong(self) -> None:
        record = self._run(_WRONG, _CORRECT)
        # sender (node_a) failed → handoff not OK
        assert record.handoffs[0].handoff_ok is False

    def test_timestamp_is_string(self) -> None:
        record = self._run(_CORRECT, _CORRECT)
        assert isinstance(record.timestamp, str)
        assert len(record.timestamp) > 0


# ---------------------------------------------------------------------------
# TestSeries3
# ---------------------------------------------------------------------------


class TestSeries3:
    """Series-3: A→B→C, two handoffs.  All three must pass for y_graph=True."""

    def _run(self, a_ok: bool, b_ok: bool, c_ok: bool):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["series3"]
        nids = motif.route
        outputs = {
            f"{nid}_m": (_CORRECT if ok else _WRONG)
            for nid, ok in zip(nids, [a_ok, b_ok, c_ok], strict=False)
        }
        client = FakeClient(outputs)
        assignment = {nid: f"{nid}_m" for nid in nids}
        return run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )

    def test_all_correct_y_graph_true(self) -> None:
        assert self._run(True, True, True).y_graph is True

    def test_middle_wrong_y_graph_false(self) -> None:
        assert self._run(True, False, True).y_graph is False

    def test_last_wrong_y_graph_false(self) -> None:
        assert self._run(True, True, False).y_graph is False

    def test_handoff_count_is_two(self) -> None:
        record = self._run(True, True, True)
        assert len(record.handoffs) == 2


# ---------------------------------------------------------------------------
# TestParallel2
# ---------------------------------------------------------------------------


class TestParallel2:
    """Parallel-2: two independent branches + deterministic merge.

    LLD §3.2: 'graph success requires at least one valid branch
    and a valid merge handoff.'
    """

    def _run(self, branch_a_ok: bool, branch_b_ok: bool):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["parallel2"]
        branch_ids = motif.route[:-1]
        outputs = {
            f"{bid}_m": (_CORRECT if ok else _WRONG)
            for bid, ok in zip(branch_ids, [branch_a_ok, branch_b_ok], strict=False)
        }
        client = FakeClient(outputs)
        assignment = {bid: f"{bid}_m" for bid in branch_ids}
        return run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )

    def test_both_correct_y_graph_true(self) -> None:
        assert self._run(True, True).y_graph is True

    def test_only_branch_a_correct_y_graph_true(self) -> None:
        """At-least-one-branch semantics: one valid branch suffices."""
        assert self._run(True, False).y_graph is True

    def test_only_branch_b_correct_y_graph_true(self) -> None:
        assert self._run(False, True).y_graph is True

    def test_both_wrong_y_graph_false(self) -> None:
        assert self._run(False, False).y_graph is False

    def test_failing_branch_logged_in_components(self) -> None:
        """The failing branch must still appear in components (complete vector)."""
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(True, False)
        branch_b_id = MOTIF_LIBRARY["parallel2"].route[1]
        comp_ids = {c.component_id for c in record.components}
        assert branch_b_id in comp_ids

    def test_failing_branch_hard_ok_false(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(True, False)
        branch_b_id = MOTIF_LIBRARY["parallel2"].route[1]
        comp = next(c for c in record.components if c.component_id == branch_b_id)
        assert comp.hard_ok is False


# ---------------------------------------------------------------------------
# TestQuorum2of3
# ---------------------------------------------------------------------------


class TestQuorum2of3:
    """Quorum-3: three branches + 2-of-3 aggregator.

    LLD §3.2: 'graph success requires at least two valid branches
    and a valid aggregation record.'
    """

    def _worker_ids(self):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        motif = MOTIF_LIBRARY["quorum2of3"]
        agg_id = motif.route[-1]
        return [nid for nid in motif.route if nid != agg_id]

    def _run(self, w0_ok: bool, w1_ok: bool, w2_ok: bool):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["quorum2of3"]
        worker_ids = self._worker_ids()
        outputs = {
            f"{wid}_m": (_CORRECT if ok else _WRONG)
            for wid, ok in zip(worker_ids, [w0_ok, w1_ok, w2_ok], strict=False)
        }
        client = FakeClient(outputs)
        assignment = {wid: f"{wid}_m" for wid in worker_ids}
        return run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )

    # --- Task-prompt requirement 3 ---
    def test_two_of_three_correct_y_graph_true(self) -> None:
        """2-of-3 workers correct → y_graph True (task-prompt quorum requirement)."""
        assert self._run(True, True, False).y_graph is True

    def test_three_of_three_correct_y_graph_true(self) -> None:
        assert self._run(True, True, True).y_graph is True

    def test_first_two_correct_y_graph_true(self) -> None:
        assert self._run(True, True, False).y_graph is True

    def test_last_two_correct_y_graph_true(self) -> None:
        assert self._run(False, True, True).y_graph is True

    # --- Task-prompt requirement 4 (implicit: fewer than quorum → False) ---
    def test_one_of_three_correct_y_graph_false(self) -> None:
        assert self._run(True, False, False).y_graph is False

    def test_zero_correct_y_graph_false(self) -> None:
        assert self._run(False, False, False).y_graph is False

    def test_all_components_logged(self) -> None:
        """Complete component vector retained regardless of branch outcome."""
        record = self._run(True, True, False)
        worker_ids = self._worker_ids()
        comp_ids = {c.component_id for c in record.components}
        for wid in worker_ids:
            assert wid in comp_ids

    def test_failing_worker_hard_ok_false(self) -> None:
        record = self._run(True, True, False)
        worker_ids = self._worker_ids()
        failing_id = worker_ids[2]
        comp = next(c for c in record.components if c.component_id == failing_id)
        assert comp.hard_ok is False

    def test_passing_workers_not_in_route_when_quorum_fails(self) -> None:
        """When quorum is not met, passing branches are not in the realized route."""
        record = self._run(True, False, False)
        # Only aggregator should be in route
        assert len(record.route) == 1


# ---------------------------------------------------------------------------
# TestHierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    """Hierarchy: supervisor selects worker_0; worker_0 + verifier are active."""

    def _run(self, sup_ok: bool, worker_ok: bool, verifier_ok: bool):  # type: ignore[return]
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["hierarchy"]
        sup_id, worker_id, verifier_id = (
            motif.route[0],
            motif.route[1],
            motif.route[2],
        )
        outputs = {
            f"{sup_id}_m": (_CORRECT if sup_ok else _WRONG),
            f"{worker_id}_m": (_CORRECT if worker_ok else _WRONG),
            f"{verifier_id}_m": (_CORRECT if verifier_ok else _WRONG),
        }
        client = FakeClient(outputs)
        assignment = {
            sup_id: f"{sup_id}_m",
            worker_id: f"{worker_id}_m",
            verifier_id: f"{verifier_id}_m",
        }
        return run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )

    def test_all_correct_y_graph_true(self) -> None:
        assert self._run(True, True, True).y_graph is True

    def test_worker_wrong_y_graph_false(self) -> None:
        assert self._run(True, False, True).y_graph is False

    def test_supervisor_wrong_y_graph_false(self) -> None:
        assert self._run(False, True, True).y_graph is False

    def test_verifier_wrong_y_graph_false(self) -> None:
        assert self._run(True, True, False).y_graph is False

    def test_route_has_three_nodes(self) -> None:
        record = self._run(True, True, True)
        assert len(record.route) == 3

    def test_inactive_worker_in_components(self) -> None:
        """Inactive hierarchy branch is logged but absent from route."""
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(True, True, True)
        motif = MOTIF_LIBRARY["hierarchy"]
        inactive_ids = [n for n in motif.nodes if n not in set(motif.route)]
        assert len(inactive_ids) >= 1
        comp_ids = {c.component_id for c in record.components}
        for iid in inactive_ids:
            assert iid in comp_ids

    def test_inactive_worker_not_in_route(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(True, True, True)
        motif = MOTIF_LIBRARY["hierarchy"]
        inactive_ids = set(motif.nodes) - set(motif.route)
        route_set = set(record.route)
        assert inactive_ids.isdisjoint(route_set)

    def test_inactive_worker_hard_ok_false(self) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

        record = self._run(True, True, True)
        motif = MOTIF_LIBRARY["hierarchy"]
        inactive_ids = set(motif.nodes) - set(motif.route)
        for comp in record.components:
            if comp.component_id in inactive_ids:
                assert comp.hard_ok is False

    def test_handoffs_count(self) -> None:
        record = self._run(True, True, True)
        # supervisor→worker, worker→verifier
        assert len(record.handoffs) == 2


# ---------------------------------------------------------------------------
# TestMissionRecordRoundtrip — task-prompt requirement
# ---------------------------------------------------------------------------


class TestMissionRecordRoundtrip:
    """MissionRecord must round-trip through JsonlLogger with exact equality."""

    def test_series2_both_correct_roundtrip(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["series2"]
        node_a, node_b = motif.route[0], motif.route[1]
        client = FakeClient({f"{node_a}_m": _CORRECT, f"{node_b}_m": _CORRECT})
        assignment = {node_a: f"{node_a}_m", node_b: f"{node_b}_m"}
        record = run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )
        logger = JsonlLogger(tmp_path / "missions.jsonl")
        logger.append(record)
        recovered = logger.read_all()
        assert len(recovered) == 1
        assert recovered[0] == record

    def test_roundtrip_preserves_y_graph_true(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["series2"]
        node_a, node_b = motif.route[0], motif.route[1]
        client = FakeClient({f"{node_a}_m": _CORRECT, f"{node_b}_m": _CORRECT})
        assignment = {node_a: f"{node_a}_m", node_b: f"{node_b}_m"}
        record = run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )
        logger = JsonlLogger(tmp_path / "missions.jsonl")
        logger.append(record)
        assert logger.read_all()[0].y_graph is True

    def test_roundtrip_preserves_y_graph_false(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["series2"]
        node_a, node_b = motif.route[0], motif.route[1]
        client = FakeClient({f"{node_a}_m": _CORRECT, f"{node_b}_m": _WRONG})
        assignment = {node_a: f"{node_a}_m", node_b: f"{node_b}_m"}
        record = run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id=_MISSION,
        )
        logger = JsonlLogger(tmp_path / "missions.jsonl")
        logger.append(record)
        assert logger.read_all()[0].y_graph is False

    def test_quorum_record_roundtrip(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.motifs import MOTIF_LIBRARY, run_mission

        motif = MOTIF_LIBRARY["quorum2of3"]
        worker_ids = [nid for nid in motif.route if nid != motif.route[-1]]
        outputs = {
            f"{wid}_m": _CORRECT for wid in worker_ids[:2]
        }
        outputs[f"{worker_ids[2]}_m"] = _WRONG
        client = FakeClient(outputs)
        assignment = {wid: f"{wid}_m" for wid in worker_ids}
        record = run_mission(
            motif,
            _TASK,
            assignment,
            client,
            sharing_condition=_SHARING,
            cluster_id=_CLUSTER,
            mission_id="quorum-001",
        )
        logger = JsonlLogger(tmp_path / "quorum.jsonl")
        logger.append(record)
        recovered = logger.read_all()
        assert len(recovered) == 1
        assert recovered[0] == record
        assert recovered[0].y_graph is True


# ---------------------------------------------------------------------------
# TestUnknownMotif — MotifError raised for unrecognized names
# ---------------------------------------------------------------------------


class TestMalformedHierarchy:
    """Guards for the hierarchy route-length validation."""

    def test_short_hierarchy_route_raises_motif_error(self) -> None:
        """Hierarchy motif with fewer than 3 route nodes must raise MotifError."""
        from agentassert_abc.experiments.motifs import Motif, MotifError, run_mission

        bad = Motif(
            name="hierarchy",
            nodes=("supervisor", "worker_0", "verifier"),
            edges=(("supervisor", "worker_0"),),
            route=("supervisor", "worker_0"),  # only 2 — missing verifier
        )
        client = FakeClient({"sup_m": _CORRECT, "w0_m": _CORRECT})
        with pytest.raises(MotifError, match="exactly 3 nodes"):
            run_mission(
                bad,
                _TASK,
                {"supervisor": "sup_m", "worker_0": "w0_m"},
                client,
                sharing_condition=_SHARING,
                cluster_id=_CLUSTER,
                mission_id=_MISSION,
            )


class TestUnknownMotif:
    def test_unknown_motif_raises_motif_error(self) -> None:
        from agentassert_abc.experiments.motifs import Motif, MotifError, run_mission

        bad = Motif(
            name="not_a_real_motif",
            nodes=("a",),
            edges=(),
            route=("a",),
        )
        client = FakeClient({"m_a": _CORRECT})
        with pytest.raises(MotifError):
            run_mission(
                bad,
                _TASK,
                {"a": "m_a"},
                client,
                sharing_condition=_SHARING,
                cluster_id=_CLUSTER,
                mission_id=_MISSION,
            )
