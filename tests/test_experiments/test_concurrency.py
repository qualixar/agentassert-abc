# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Part G.3 tests — bounded concurrency for the experiment runner (LLD-F §G).

Test families
-------------
TestDeterminismVsSerial
    concurrency=1 vs concurrency=16 with identical inputs/seed produce the
    same JSONL mission_ids in the same order (canonical idx order).

TestGateUnderConcurrency
    Ledger pre-seeded near the $19.50 stop → BudgetExceeded is raised BEFORE
    any generate call is dispatched (spy client call count == 0).

TestIsolationUnderConcurrency
    A mission that raises inside run_mission lands in .failures.jsonl while
    all other missions complete normally.  Run does not crash.

TestThreadSafetySmoke
    concurrency=16 over ≥ 200 dry missions → exact ledger.spent (0.0) and
    exact record count (no lost or double writes).

Invariants enforced across ALL tests
--------------------------------------
- config.FRONTIER_ENABLED is never set True.
- Every test uses DryRunClient or a fake/spy client (zero cost, zero network).
- per_call_ceiling > 0 appears ONLY in gate tests that assert BudgetExceeded
  before any generate call is made.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from agentassert_abc.experiments import config
from agentassert_abc.experiments._runner_core import _execute_mission_batch
from agentassert_abc.experiments.budget import BudgetExceeded, BudgetLedger
from agentassert_abc.experiments.logging_schema import JsonlLogger
from agentassert_abc.experiments.models import ModelResponse
from agentassert_abc.experiments.motifs import MOTIF_LIBRARY
from agentassert_abc.experiments.run import _DRY_MODEL_PAIRS, DryRunClient, _dry_task_sampler
from agentassert_abc.experiments.tasks import TASK_LIBRARY

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zero_cost_response(model: str) -> ModelResponse:
    """Return a canned ModelResponse that scores hard_ok=True at zero cost."""
    return ModelResponse(
        text=TASK_LIBRARY[0].ground_truth,
        input_tokens=0,
        output_tokens=0,
        model=model,
        cost_usd=0.0,
    )


class _SpyClient:
    """ModelClient that counts generate calls and returns zero-cost canned answers."""

    def __init__(self) -> None:
        self.call_count: int = 0

    def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
        self.call_count += 1
        return _zero_cost_response(model)


# ---------------------------------------------------------------------------
# TestDeterminismVsSerial
# ---------------------------------------------------------------------------


class TestDeterminismVsSerial:
    """concurrency=1 and concurrency=16 produce JSONL in the same canonical order."""

    def test_mission_ids_identical_order(self, tmp_path: Path) -> None:
        """Mission IDs in the JSONL must be in identical canonical order."""
        motif_list = [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]]
        conditions = ["same_model", "same_vendor"]
        n_per_cell = 5

        out_serial = tmp_path / "serial.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_serial,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=1,
        )

        out_concurrent = tmp_path / "concurrent.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_concurrent,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        serial_ids = [r.mission_id for r in JsonlLogger(out_serial).read_all()]
        concurrent_ids = [r.mission_id for r in JsonlLogger(out_concurrent).read_all()]

        assert serial_ids == concurrent_ids, (
            "JSONL mission_ids must be in identical canonical order for "
            "concurrency=1 and concurrency=16.\n"
            f"serial:     {serial_ids}\n"
            f"concurrent: {concurrent_ids}"
        )

    def test_hard_ok_values_identical(self, tmp_path: Path) -> None:
        """hard_ok values at each position must be identical (same DryRunClient answers)."""
        motif_list = [MOTIF_LIBRARY["series2"]]
        conditions = ["same_model"]
        n_per_cell = 4

        out_serial = tmp_path / "s.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_serial,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=1,
        )

        out_concurrent = tmp_path / "c.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_concurrent,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        s_records = list(JsonlLogger(out_serial).read_all())
        c_records = list(JsonlLogger(out_concurrent).read_all())

        assert len(s_records) == len(c_records)
        for s, c in zip(s_records, c_records, strict=True):
            assert s.mission_id == c.mission_id
            assert s.y_graph == c.y_graph

    def test_record_count_identical(self, tmp_path: Path) -> None:
        """Both runs must produce the same number of JSONL records."""
        motif_list = [MOTIF_LIBRARY["quorum2of3"]]
        conditions = ["same_model", "different_vendor"]
        n_per_cell = 3

        out_s = tmp_path / "qs.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_s,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=1,
        )

        out_c = tmp_path / "qc.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_c,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        assert (
            len(list(JsonlLogger(out_s).read_all()))
            == len(list(JsonlLogger(out_c).read_all()))
            == len(motif_list) * len(conditions) * n_per_cell
        )


# ---------------------------------------------------------------------------
# TestGateUnderConcurrency
# ---------------------------------------------------------------------------


class TestGateUnderConcurrency:
    """Budget gate fires BEFORE dispatch — spy client receives zero calls."""

    def test_budget_exceeded_raised_before_dispatch(self, tmp_path: Path) -> None:
        """BudgetExceeded must be raised before any generate call in the batch."""
        spy = _SpyClient()
        ledger = BudgetLedger()
        # series2 has 2 gen nodes.  25 missions × 2 nodes × $0.0072 = $0.36.
        # $19.40 + $0.36 = $19.76 > $19.50 → gate must trip.
        ledger.record(19.40)

        with pytest.raises(BudgetExceeded):
            _execute_mission_batch(
                spy,
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                25,  # one full batch of 25
                tmp_path / "gate_test.jsonl",
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                per_call_ceiling=config.PER_CALL_CEILING_USD,
                concurrency=16,
            )

        assert spy.call_count == 0, (
            f"Gate must trip before any generate call; got {spy.call_count} call(s)."
        )

    def test_budget_exceeded_raised_serial_also(self, tmp_path: Path) -> None:
        """Gate fires for concurrency=1 (serial path) too — not a concurrency quirk."""
        spy = _SpyClient()
        ledger = BudgetLedger()
        ledger.record(19.40)

        with pytest.raises(BudgetExceeded):
            _execute_mission_batch(
                spy,
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                25,
                tmp_path / "gate_serial.jsonl",
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                per_call_ceiling=config.PER_CALL_CEILING_USD,
                concurrency=1,
            )

        assert spy.call_count == 0

    def test_gate_skipped_for_zero_ceiling(self, tmp_path: Path) -> None:
        """per_call_ceiling=0.0 (dry/local) disables the gate so run completes."""
        spy = _SpyClient()
        ledger = BudgetLedger()
        ledger.record(19.40)  # near stop, but ceiling=0 → gate disabled

        # Must NOT raise BudgetExceeded; run completes normally.
        result = _execute_mission_batch(
            spy,
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            2,
            tmp_path / "gate_off.jsonl",
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            per_call_ceiling=0.0,
            concurrency=16,
        )
        assert len(result) == 2
        assert spy.call_count > 0  # calls DID happen (gate was not tripped)

    def test_gate_counts_real_calls_not_missions(self, tmp_path: Path) -> None:
        """Gate math uses n_gen_nodes × ceiling, not 1-call-per-mission."""
        # series2: 2 gen nodes.  1 mission × 2 nodes × $0.0072 = $0.0144.
        # Ledger pre-seeded at $19.49: $19.49 + $0.0144 = $19.5044 > $19.50 → trip.
        spy = _SpyClient()
        ledger = BudgetLedger()
        ledger.record(19.49)

        # With old 1-call-per-mission under-count: $19.49 + $0.0072 = $19.4972 ≤ $19.50 (no trip).
        # With correct count: $19.49 + $0.0144 > $19.50 (trip) — proves the fix.
        with pytest.raises(BudgetExceeded):
            _execute_mission_batch(
                spy,
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                1,
                tmp_path / "gate_count.jsonl",
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                per_call_ceiling=config.PER_CALL_CEILING_USD,
                concurrency=1,
            )

        assert spy.call_count == 0

    def test_frontier_enabled_stays_false(self, tmp_path: Path) -> None:
        """config.FRONTIER_ENABLED must remain False throughout the gate test."""
        spy = _SpyClient()
        ledger = BudgetLedger()
        ledger.record(19.40)

        with pytest.raises(BudgetExceeded):
            _execute_mission_batch(
                spy,
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                25,
                tmp_path / "gate_flag.jsonl",
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                per_call_ceiling=config.PER_CALL_CEILING_USD,
                concurrency=16,
            )

        assert config.FRONTIER_ENABLED is False


# ---------------------------------------------------------------------------
# TestIsolationUnderConcurrency
# ---------------------------------------------------------------------------


class TestIsolationUnderConcurrency:
    """One failing mission lands in .failures.jsonl; all others complete; no crash."""

    def _make_selective_run_mission(self, bad_mission_id: str):
        """Return a run_mission wrapper that raises RuntimeError for one specific id."""
        import agentassert_abc.experiments.motifs as _motifs_mod

        real_run_mission = _motifs_mod.run_mission

        def selective(  # noqa: PLR0913
            motif, task, assignment, client,
            *, sharing_condition, cluster_id, mission_id,
        ):
            if mission_id == bad_mission_id:
                raise RuntimeError(f"injected failure for {mission_id!r}")
            return real_run_mission(
                motif, task, assignment, client,
                sharing_condition=sharing_condition,
                cluster_id=cluster_id,
                mission_id=mission_id,
            )

        return selective

    def test_failing_mission_logged_to_failures_concurrent(self, tmp_path: Path) -> None:
        """Failed mission appears in .failures.jsonl under concurrency=16."""
        bad_id = "mission-series2-same_model-2"
        out = tmp_path / "iso_conc.jsonl"

        with patch(
            "agentassert_abc.experiments._runner_core.run_mission",
            self._make_selective_run_mission(bad_id),
        ):
            _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                5,  # 5 missions; mission at index 2 will fail
                out,
                BudgetLedger(),
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                concurrency=16,
            )

        failures_path = tmp_path / "iso_conc.jsonl.failures.jsonl"
        assert failures_path.exists(), "failures.jsonl must exist after an injected failure."
        content = failures_path.read_text()
        assert bad_id in content, (
            f"Failed mission {bad_id!r} must appear in failures.jsonl.\n"
            f"Content: {content}"
        )

    def test_other_missions_complete_after_failure(self, tmp_path: Path) -> None:
        """4 of 5 missions succeed when 1 raises (concurrency=16)."""
        bad_id = "mission-series2-same_model-2"
        out = tmp_path / "iso_conc2.jsonl"

        with patch(
            "agentassert_abc.experiments._runner_core.run_mission",
            self._make_selective_run_mission(bad_id),
        ):
            result = _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                5,
                out,
                BudgetLedger(),
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                concurrency=16,
            )

        assert len(result) == 4, (
            f"Expected 4 successful records (5 - 1 failed); got {len(result)}."
        )

    def test_run_does_not_crash_concurrent(self, tmp_path: Path) -> None:
        """_execute_mission_batch must not propagate the injected exception."""
        bad_id = "mission-series2-same_model-0"
        out = tmp_path / "iso_no_crash.jsonl"

        # If this raises anything other than BudgetExceeded the test fails.
        with patch(
            "agentassert_abc.experiments._runner_core.run_mission",
            self._make_selective_run_mission(bad_id),
        ):
            # Should complete without raising.
            _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                3,
                out,
                BudgetLedger(),
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                concurrency=16,
            )

    def test_failure_content_has_required_fields(self, tmp_path: Path) -> None:
        """failures.jsonl records must contain mission_id, error, condition, motif."""
        bad_id = "mission-series2-same_model-1"
        out = tmp_path / "iso_fields.jsonl"

        with patch(
            "agentassert_abc.experiments._runner_core.run_mission",
            self._make_selective_run_mission(bad_id),
        ):
            _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                3,
                out,
                BudgetLedger(),
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
                concurrency=16,
            )

        failures_path = tmp_path / "iso_fields.jsonl.failures.jsonl"
        assert failures_path.exists()
        for line in failures_path.read_text().splitlines():
            rec = json.loads(line)
            for key in ("mission_id", "error", "condition", "motif", "ts"):
                assert key in rec, f"failures.jsonl record missing key {key!r}."


# ---------------------------------------------------------------------------
# TestThreadSafetySmoke
# ---------------------------------------------------------------------------


class TestThreadSafetySmoke:
    """concurrency=16 over ≥200 dry missions → exact counts, no lost/double writes."""

    def test_record_count_exact_200_missions(self, tmp_path: Path) -> None:
        """All 200 dry missions written exactly once to JSONL (no lost or double)."""
        motif_list = [
            MOTIF_LIBRARY["series2"],
            MOTIF_LIBRARY["parallel2"],
            MOTIF_LIBRARY["quorum2of3"],
            MOTIF_LIBRARY["hierarchy"],
        ]
        conditions = ["same_model", "same_vendor", "different_vendor"]
        # 4 motifs × 3 conditions × 17 = 204 missions (>= 200 per spec)
        n_per_cell = 17
        expected = len(motif_list) * len(conditions) * n_per_cell

        out = tmp_path / "smoke.jsonl"
        result = _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        assert len(result) == expected, (
            f"Expected {expected} records in return value; got {len(result)}."
        )

        logged = list(JsonlLogger(out).read_all())
        assert len(logged) == expected, (
            f"Expected {expected} records in JSONL; got {len(logged)}. "
            "Possible lost or double write under concurrency."
        )

    def test_no_duplicate_mission_ids(self, tmp_path: Path) -> None:
        """Each mission_id must appear exactly once (no double writes)."""
        motif_list = [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["series3"]]
        conditions = ["same_model", "same_vendor", "different_vendor"]
        n_per_cell = 15  # 2 × 3 × 15 = 90 missions
        expected = len(motif_list) * len(conditions) * n_per_cell

        out = tmp_path / "dedup.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        logged = list(JsonlLogger(out).read_all())
        ids = [r.mission_id for r in logged]
        assert len(ids) == len(set(ids)) == expected, (
            f"Duplicate mission_ids detected: {len(ids)} total, {len(set(ids))} unique."
        )

    def test_ledger_spent_zero_dry_run(self, tmp_path: Path) -> None:
        """DryRunClient produces zero cost; ledger.spent must be exactly 0.0."""
        ledger = BudgetLedger()
        out = tmp_path / "zero_cost.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]],
            ["same_model", "different_vendor"],
            25,  # 2 × 2 × 25 = 100 missions
            out,
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )
        assert ledger.spent == 0.0, (
            f"DryRunClient must produce zero spend; ledger.spent={ledger.spent}."
        )

    def test_canonical_order_preserved_under_concurrency(self, tmp_path: Path) -> None:
        """JSONL mission_ids must be in the same canonical order as concurrency=1."""
        motif_list = [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["series3"]]
        conditions = ["same_model", "same_vendor"]
        n_per_cell = 13  # 2 × 2 × 13 = 52 missions

        out_ref = tmp_path / "ref.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_ref,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=1,
        )

        out_conc = tmp_path / "conc.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            motif_list, conditions, n_per_cell,
            out_conc,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )

        ref_ids = [r.mission_id for r in JsonlLogger(out_ref).read_all()]
        conc_ids = [r.mission_id for r in JsonlLogger(out_conc).read_all()]
        assert ref_ids == conc_ids, (
            "Canonical order must be identical for concurrency=1 and concurrency=16.\n"
            f"ref:  {ref_ids[:5]}...\nconc: {conc_ids[:5]}..."
        )

    def test_progress_file_written_after_batch(self, tmp_path: Path) -> None:
        """progress.json must exist after a concurrent run (heartbeat per batch)."""
        out = tmp_path / "prog.jsonl"
        _execute_mission_batch(
            DryRunClient(),
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            5,
            out,
            BudgetLedger(),
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
            concurrency=16,
        )
        progress = tmp_path / "prog.jsonl.progress.json"
        assert progress.exists(), "progress.json must be written after each batch."
        data = json.loads(progress.read_text())
        assert data["completed"] == 5
        assert data["total"] == 5
        assert data["spent_usd"] == 0.0
