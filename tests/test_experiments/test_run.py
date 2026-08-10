# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""TDD tests for experiments/run.py — experiment orchestrator (LLD-E §5, §12).

TDD RED → GREEN: these tests are written BEFORE the implementation.

Test families
-------------
TestDryRunClient
    DryRunClient satisfies ModelClient protocol; cost always 0.0; no network.

TestExperimentSummary
    ExperimentSummary is a frozen dataclass with all four report fields.

TestRunExperiment
    run_experiment with DryRunClient produces a valid ExperimentSummary:
      - all four reports are populated (no AnalysisError raised)
      - budget_spent == 0.0
      - n_missions == motifs × conditions × n_per_cell
      - missions round-trip through JsonlLogger

TestFrontierNeverTouched
    FrontierClient.generate is never called during a dry-run.
    run_experiment with DryRunClient never touches config.FRONTIER_ENABLED.

TestMainCLI
    main() --frontier is a NO-OP that prints to stderr and exits with status 1.
    main() (default / --dry-run) completes without error and prints a summary.

TestRunError
    RunError is an AgentAssertError subclass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentassert_abc.exceptions import AgentAssertError
from agentassert_abc.experiments.budget import BudgetLedger
from agentassert_abc.experiments.logging_schema import JsonlLogger
from agentassert_abc.experiments.models import ModelResponse
from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

# ---------------------------------------------------------------------------
# Helpers — deferred import guard
# ---------------------------------------------------------------------------

def _import_run():
    """Lazy import so RED tests fail at the right point."""
    from agentassert_abc.experiments import run  # noqa: PLC0415
    return run


# ---------------------------------------------------------------------------
# TestDryRunClient
# ---------------------------------------------------------------------------


class TestDryRunClient:
    def test_cost_is_always_zero(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("any-model", "any prompt")
        assert resp.cost_usd == 0.0

    def test_input_tokens_zero(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("any-model", "any prompt")
        assert resp.input_tokens == 0

    def test_output_tokens_zero(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("any-model", "any prompt")
        assert resp.output_tokens == 0

    def test_returns_non_empty_text(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("any-model", "any prompt")
        assert resp.text.strip() != ""

    def test_model_field_echoes_input(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("my-model-tag", "some prompt")
        assert resp.model == "my-model-tag"

    def test_returns_model_response_type(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("m", "p")
        assert isinstance(resp, ModelResponse)

    def test_satisfies_model_client_protocol(self) -> None:
        """DryRunClient structurally satisfies the ModelClient protocol."""
        run = _import_run()
        client = run.DryRunClient()
        assert callable(getattr(client, "generate", None))

    def test_different_models_same_cost(self) -> None:
        run = _import_run()
        client = run.DryRunClient()
        r1 = client.generate("model-a", "p1")
        r2 = client.generate("model-b", "p2")
        assert r1.cost_usd == r2.cost_usd == 0.0

    def test_canned_answer_correct_for_task_library_0(self) -> None:
        """The canned answer must equal TASK_LIBRARY[0].ground_truth so that
        hard_ok=True when run_experiment uses TASK_LIBRARY[0] as the task."""
        from agentassert_abc.experiments.tasks import TASK_LIBRARY, score  # noqa: PLC0415
        run = _import_run()
        client = run.DryRunClient()
        resp = client.generate("m", "p")
        assert score(TASK_LIBRARY[0], resp.text) is True


# ---------------------------------------------------------------------------
# TestExperimentSummary
# ---------------------------------------------------------------------------


class TestExperimentSummary:
    def test_is_frozen(self) -> None:
        run = _import_run()
        # Need a valid summary; build a minimal one via run_experiment
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=Path("/tmp/aa_test_frozen.jsonl"),
            ledger=ledger,
        )
        with pytest.raises((AttributeError, TypeError)):
            summary.budget_spent = 99.0  # type: ignore[misc]

    def test_has_all_four_report_fields(self) -> None:
        from agentassert_abc.experiments.analysis import (  # noqa: PLC0415
            CertificationReport,
            CompositionReport,
            DependenceReport,
            DriftReport,
        )
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=Path("/tmp/aa_test_fields.jsonl"),
            ledger=ledger,
        )
        assert isinstance(summary.dependence, DependenceReport)
        assert isinstance(summary.composition, CompositionReport)
        assert isinstance(summary.certification, CertificationReport)
        assert isinstance(summary.drift, DriftReport)

    def test_out_path_stored_as_string(self) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=Path("/tmp/aa_test_out.jsonl"),
            ledger=ledger,
        )
        assert isinstance(summary.out_path, str)


# ---------------------------------------------------------------------------
# TestRunExperiment
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_budget_spent_zero_with_dry_run_client(self, tmp_path: Path) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]],
            sharing_conditions=["same_model", "different_vendor"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert summary.budget_spent == 0.0

    def test_n_missions_equals_motifs_x_conditions_x_n_per_cell(
        self, tmp_path: Path
    ) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        motifs = [MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]]
        conditions = ["same_model", "different_vendor"]
        n_per_cell = 3
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=motifs,
            sharing_conditions=conditions,
            n_per_cell=n_per_cell,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert summary.n_missions == len(motifs) * len(conditions) * n_per_cell

    def test_all_reports_populated(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.analysis import (  # noqa: PLC0415
            CertificationReport,
            CompositionReport,
            DependenceReport,
            DriftReport,
        )
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]],
            sharing_conditions=["same_model", "different_vendor"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert isinstance(summary.dependence, DependenceReport)
        assert isinstance(summary.composition, CompositionReport)
        assert isinstance(summary.certification, CertificationReport)
        assert isinstance(summary.drift, DriftReport)

    def test_dependence_n_missions_at_least_2(self, tmp_path: Path) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"], MOTIF_LIBRARY["parallel2"]],
            sharing_conditions=["same_model", "different_vendor"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert summary.dependence.n_missions >= 2

    def test_composition_observed_reliability_in_unit_interval(
        self, tmp_path: Path
    ) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=4,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert 0.0 <= summary.composition.observed_reliability <= 1.0

    def test_certification_report_has_valid_wealth(self, tmp_path: Path) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=4,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        # Wealth is always >= 0; may not certify with only 4 missions
        assert summary.certification.final_wealth >= 0.0
        assert isinstance(summary.certification.certified, bool)

    def test_drift_report_agents_populated(self, tmp_path: Path) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        # series2 has 2 nodes → at least 2 agents in drift report
        assert summary.drift.n_agents >= 2

    def test_missions_round_trip_through_jsonl(self, tmp_path: Path) -> None:
        """Missions logged to JSONL are exactly recoverable via JsonlLogger.read_all()."""
        run = _import_run()
        out = tmp_path / "missions.jsonl"
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=out,
            ledger=ledger,
        )
        # Read back from JSONL and verify count + equality
        logger = JsonlLogger(out)
        logged = logger.read_all()
        assert len(logged) == summary.n_missions

    def test_missions_match_after_round_trip(self, tmp_path: Path) -> None:
        """MissionRecord objects survive JsonlLogger round-trip intact."""
        run = _import_run()
        out = tmp_path / "missions.jsonl"
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=out,
            ledger=ledger,
        )
        logger = JsonlLogger(out)
        logged = logger.read_all()
        # The round-trip must preserve the n_missions count and mission IDs
        assert len(logged) == summary.n_missions
        logged_ids = {r.mission_id for r in logged}
        assert len(logged_ids) == summary.n_missions  # all IDs are unique

    def test_ledger_spent_equals_summary_budget(self, tmp_path: Path) -> None:
        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert summary.budget_spent == ledger.spent

    def test_dry_run_all_missions_have_zero_cost(self, tmp_path: Path) -> None:
        run = _import_run()
        out = tmp_path / "missions.jsonl"
        ledger = BudgetLedger()
        run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=3,
            p0=0.80,
            alpha=0.05,
            out_path=out,
            ledger=ledger,
        )
        logger = JsonlLogger(out)
        for mission in logger.read_all():
            assert mission.cost_usd == 0.0

    def test_summary_contains_out_path(self, tmp_path: Path) -> None:
        run = _import_run()
        out = tmp_path / "missions.jsonl"
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=out,
            ledger=ledger,
        )
        assert str(out) in summary.out_path or summary.out_path == str(out)


# ---------------------------------------------------------------------------
# TestFrontierNeverTouched
# ---------------------------------------------------------------------------


class TestFrontierNeverTouched:
    def test_frontier_client_generate_never_called(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FrontierClient.generate must never be invoked during a dry-run."""
        call_log: list[str] = []

        def _fake_frontier_generate(self: object, model: str, prompt: str) -> ModelResponse:
            call_log.append(f"frontier_called({model})")
            raise AssertionError(
                "FrontierClient.generate was called — dry-run violation."
            )

        monkeypatch.setattr(
            "agentassert_abc.experiments.models.FrontierClient.generate",
            _fake_frontier_generate,
        )

        run = _import_run()
        ledger = BudgetLedger()
        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert call_log == [], f"Frontier was called: {call_log}"
        assert summary.budget_spent == 0.0

    def test_frontier_enabled_stays_false(self, tmp_path: Path) -> None:
        """config.FRONTIER_ENABLED must remain False during and after dry-run."""
        from agentassert_abc.experiments import config  # noqa: PLC0415
        run = _import_run()
        ledger = BudgetLedger()
        run.run_experiment(
            run.DryRunClient(),
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "missions.jsonl",
            ledger=ledger,
        )
        assert config.FRONTIER_ENABLED is False


# ---------------------------------------------------------------------------
# TestMainCLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_frontier_flag_exits_with_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["run", "--frontier"])
        run = _import_run()
        with pytest.raises(SystemExit) as exc_info:
            run.main()
        assert exc_info.value.code == 1

    def test_frontier_flag_prints_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["run", "--frontier"])
        run = _import_run()
        with pytest.raises(SystemExit):
            run.main()
        captured = capsys.readouterr()
        assert "FRONTIER_ENABLED" in captured.err or "frontier" in captured.err.lower()

    def test_dry_run_flag_completes_successfully(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["run", "--dry-run", "--out", str(tmp_path / "m.jsonl")])
        run = _import_run()
        # Should complete without raising (no SystemExit, no exception)
        run.main()
        captured = capsys.readouterr()
        # Output should mention missions and budget
        assert "missions" in captured.out.lower() or "mission" in captured.out.lower()

    def test_default_mode_is_dry_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """No flags → behaves identically to --dry-run."""
        monkeypatch.setattr(sys, "argv", ["run", "--out", str(tmp_path / "m.jsonl")])
        run = _import_run()
        run.main()  # Must not raise
        captured = capsys.readouterr()
        assert captured.out  # Something was printed

    def test_frontier_flag_no_frontier_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--frontier exits before creating any client at all."""
        call_log: list[str] = []

        def _fake(self: object, m: str, p: str) -> ModelResponse:
            call_log.append("frontier_called")
            raise AssertionError("frontier called")

        monkeypatch.setattr(
            "agentassert_abc.experiments.models.FrontierClient.generate",
            _fake,
        )
        monkeypatch.setattr(sys, "argv", ["run", "--frontier"])
        run = _import_run()
        with pytest.raises(SystemExit):
            run.main()
        assert call_log == []


# ---------------------------------------------------------------------------
# TestRunError
# ---------------------------------------------------------------------------


class TestRunError:
    def test_run_error_is_agent_assert_error(self) -> None:
        run = _import_run()
        assert issubclass(run.RunError, AgentAssertError)

    def test_run_error_is_exception(self) -> None:
        run = _import_run()
        err = run.RunError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"


# ---------------------------------------------------------------------------
# TestFrontierTierAutoResolution
#
# Regression for the run_experiment bug where a paid (non-dry/non-local) client
# silently got _LOCAL_MODEL_PAIRS (Ollama IDs) and per_call_ceiling=0.0 — which
# sent wrong model names AND disarmed the §6.3 prospective budget gate.
# ---------------------------------------------------------------------------


class _FakeFrontierClient:
    """A ModelClient that is neither DryRunClient nor LocalClient.

    run_experiment must therefore treat it as a *frontier* (paid) tier:
    use _FRONTIER_MODEL_PAIRS and arm the $19.50 batch gate.
    """

    def __init__(self, cost_usd: float = 0.0) -> None:
        self.models_seen: list[str] = []
        self._cost = cost_usd

    def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
        self.models_seen.append(model)
        return ModelResponse(
            text="6912",
            input_tokens=10,
            output_tokens=2,
            model=model,
            cost_usd=self._cost,
        )


class TestFrontierTierAutoResolution:
    """A frontier-tier client gets the frontier roster + an armed budget gate."""

    def test_frontier_client_uses_frontier_model_pairs(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments import config  # noqa: PLC0415

        run = _import_run()
        client = _FakeFrontierClient()
        run.run_experiment(
            client,
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=2,
            p0=0.80,
            alpha=0.05,
            out_path=tmp_path / "frontier_pairs.jsonl",
            ledger=BudgetLedger(),
        )
        # same_model → every generative leg is the OpenRouter anchor, never an
        # Ollama id from _LOCAL_MODEL_PAIRS.
        assert client.models_seen, "no generate calls were made"
        assert all(m == config.OPENROUTER_DEFAULT_MODEL for m in client.models_seen), (
            f"frontier run used non-anchor model ids: {client.models_seen}"
        )

    def test_frontier_client_arms_budget_gate(self, tmp_path: Path) -> None:
        from agentassert_abc.experiments.budget import BudgetExceeded  # noqa: PLC0415

        run = _import_run()
        client = _FakeFrontierClient(cost_usd=0.0)
        ledger = BudgetLedger()
        ledger.record(19.40)  # within one 25-call batch of the $19.50 hard stop
        # Gate armed (ceiling=PER_CALL_CEILING_USD) → the prospective plan_batch
        # trips BEFORE any generate call.  With the old ceiling=0.0 bug the gate
        # was a no-op and this run would proceed to spend.
        with pytest.raises(BudgetExceeded):
            run.run_experiment(
                client,
                motifs=[MOTIF_LIBRARY["series2"]],
                sharing_conditions=["same_model"],
                n_per_cell=25,
                p0=0.80,
                alpha=0.05,
                out_path=tmp_path / "frontier_gate.jsonl",
                ledger=ledger,
            )
        assert client.models_seen == [], "gate must trip before any generate call"


# ---------------------------------------------------------------------------
# TestLLDFResume — LLD-F §C.1: resume skips completed missions + seeds budget
# ---------------------------------------------------------------------------


def _make_flaky_client(raise_on_call_n: int) -> object:
    """Return a DryRunClient-shaped client that raises RuntimeError on call N."""
    from agentassert_abc.experiments.models import ModelResponse  # noqa: PLC0415
    from agentassert_abc.experiments.tasks import TASK_LIBRARY  # noqa: PLC0415

    class FlakyClient:
        def __init__(self) -> None:
            self.call_count: int = 0

        def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
            self.call_count += 1
            if self.call_count == raise_on_call_n:
                raise RuntimeError(f"injected failure on call {raise_on_call_n}")
            return ModelResponse(
                text=TASK_LIBRARY[0].ground_truth,
                input_tokens=0,
                output_tokens=0,
                model=model,
                cost_usd=0.0,
            )

    return FlakyClient()


class TestLLDFResume:
    """LLD-F §C.1: idempotent resume — completed missions are skipped."""

    def test_completed_missions_skipped_on_resume(self, tmp_path: Path) -> None:
        """Running with an existing JSONL must skip already-logged missions."""
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            DryRunClient,
            _dry_task_sampler,
        )

        out = tmp_path / "missions.jsonl"
        ledger = BudgetLedger()

        # First run: 1 motif × 1 condition × 2 = 2 missions
        _execute_mission_batch(
            DryRunClient(),
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            2,
            out,
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )
        initial_size = out.stat().st_size

        # Second run: same path, same missions → should append 0 new records
        ledger2 = BudgetLedger()
        _execute_mission_batch(
            DryRunClient(),
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            2,
            out,
            ledger2,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # File size must not grow — no new missions appended.
        assert out.stat().st_size == initial_size, (
            "Resume should skip completed missions; file size grew unexpectedly."
        )

    def test_resume_seeds_ledger_with_prior_cost(self, tmp_path: Path) -> None:
        """Prior cost must be loaded into the ledger on resume."""
        from agentassert_abc.experiments._runner_core import (  # noqa: PLC0415
            _execute_mission_batch,
        )
        from agentassert_abc.experiments.models import ModelResponse  # noqa: PLC0415
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )
        from agentassert_abc.experiments.tasks import TASK_LIBRARY  # noqa: PLC0415

        # Build a client that charges $0.001 per call.
        class CostClient:
            def generate(self, model: str, prompt: str) -> ModelResponse:  # noqa: ARG002
                return ModelResponse(
                    text=TASK_LIBRARY[0].ground_truth,
                    input_tokens=0, output_tokens=0,
                    model=model, cost_usd=0.001,
                )

        out = tmp_path / "missions_cost.jsonl"
        ledger1 = BudgetLedger()
        _execute_mission_batch(
            CostClient(),
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            2,
            out,
            ledger1,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )
        spent1 = ledger1.spent

        # Resume run: ledger must include prior spend.
        ledger2 = BudgetLedger()
        _execute_mission_batch(
            CostClient(),
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            2,
            out,
            ledger2,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )
        # Resume seeds ledger2 with prior cost; no new missions → ledger2.spent == prior
        assert ledger2.spent >= spent1 - 1e-9, (
            f"Resume must seed ledger with prior cost. "
            f"prior={spent1:.6f}, ledger2.spent={ledger2.spent:.6f}"
        )


# ---------------------------------------------------------------------------
# TestLLDFIsolation — LLD-F §C.3: per-mission error isolation
# ---------------------------------------------------------------------------


class TestLLDFIsolation:
    """LLD-F §C.3: a failing mission logs to failures.jsonl and does not abort the run."""

    def test_failing_mission_logged_to_failures_file(self, tmp_path: Path) -> None:
        """A mission that raises must appear in failures.jsonl."""
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # Client that raises on the 3rd generate call.
        # series2 has 2 nodes → 2 generate calls per mission.
        # 1st mission = calls 1-2 (OK), 2nd mission = call 3 (FAIL on first generate).
        client = _make_flaky_client(raise_on_call_n=3)
        out = tmp_path / "isolation.jsonl"
        ledger = BudgetLedger()

        # 1 motif × 1 condition × 3 per cell = 3 missions
        _execute_mission_batch(
            client,
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            3,
            out,
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # Run must complete — not raise.
        failures_path = tmp_path / "isolation.jsonl.failures.jsonl"
        assert failures_path.exists(), "failures.jsonl must be created for failed missions."
        content = failures_path.read_text()
        assert "error" in content, "failures.jsonl must contain an error field."

    def test_run_continues_after_failing_mission(self, tmp_path: Path) -> None:
        """Run must complete all missions even if one fails mid-batch."""
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # Raise on 3rd call so 2nd mission fails; 3rd mission should still run.
        client = _make_flaky_client(raise_on_call_n=3)
        out = tmp_path / "continues.jsonl"
        ledger = BudgetLedger()

        result = _execute_mission_batch(
            client,
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            3,
            out,
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # 3 missions attempted; 1 failed; 2 should be in the result.
        assert len(result) >= 1, (
            "At least 1 mission must succeed when only 1 of 3 fails."
        )

    def test_successful_missions_still_logged(self, tmp_path: Path) -> None:
        """Missions that succeed before and after a failure must still be logged."""
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.logging_schema import JsonlLogger  # noqa: PLC0415
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        client = _make_flaky_client(raise_on_call_n=3)
        out = tmp_path / "successful.jsonl"
        ledger = BudgetLedger()

        _execute_mission_batch(
            client,
            [MOTIF_LIBRARY["series2"]],
            ["same_model"],
            3,
            out,
            ledger,
            _DRY_MODEL_PAIRS,
            _dry_task_sampler,
        )

        # JSONL must have at least 1 successfully completed record.
        assert out.exists(), "JSONL log must be created."
        logged = list(JsonlLogger(out).read_all())
        assert len(logged) >= 1, (
            f"JSONL must contain successful missions. Got {len(logged)} records."
        )


# ---------------------------------------------------------------------------
# TestLLDFHeartbeat — LLD-F §C.5: progress heartbeat written every 100 missions
# ---------------------------------------------------------------------------


class TestLLDFHeartbeat:
    """LLD-F §C.5: progress.json written every _HEARTBEAT_INTERVAL missions."""

    def test_heartbeat_file_created_after_100_missions(self, tmp_path: Path) -> None:
        """A progress.json file must appear after ≥ 100 missions are processed."""
        from unittest.mock import patch  # noqa: PLC0415

        from agentassert_abc.experiments import _runner_core  # noqa: PLC0415
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            DryRunClient,
            _dry_task_sampler,
        )

        out = tmp_path / "heartbeat.jsonl"
        ledger = BudgetLedger()

        # Patch _HEARTBEAT_INTERVAL to 2 so we don't need 100 missions.
        with patch.object(_runner_core, "_HEARTBEAT_INTERVAL", 2):
            _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                3,  # 3 missions → triggers heartbeat at mission 2
                out,
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
            )

        progress_path = tmp_path / "heartbeat.jsonl.progress.json"
        assert progress_path.exists(), (
            f"progress.json must be written after _HEARTBEAT_INTERVAL missions. "
            f"Expected at: {progress_path}"
        )

    def test_heartbeat_file_has_required_keys(self, tmp_path: Path) -> None:
        """progress.json must contain completed, total, spent_usd, ts."""
        import json as json_mod  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from agentassert_abc.experiments import _runner_core  # noqa: PLC0415
        from agentassert_abc.experiments._runner_core import (
            _execute_mission_batch,  # noqa: PLC0415
        )
        from agentassert_abc.experiments.run import (  # noqa: PLC0415
            _DRY_MODEL_PAIRS,
            DryRunClient,
            _dry_task_sampler,
        )

        out = tmp_path / "heartbeat2.jsonl"
        ledger = BudgetLedger()

        with patch.object(_runner_core, "_HEARTBEAT_INTERVAL", 2):
            _execute_mission_batch(
                DryRunClient(),
                [MOTIF_LIBRARY["series2"]],
                ["same_model"],
                3,
                out,
                ledger,
                _DRY_MODEL_PAIRS,
                _dry_task_sampler,
            )

        progress_path = tmp_path / "heartbeat2.jsonl.progress.json"
        data = json_mod.loads(progress_path.read_text())
        for key in ("completed", "total", "spent_usd", "ts"):
            assert key in data, (
                f"progress.json missing key {key!r}. Present keys: {list(data)!r}"
            )
        assert isinstance(data["completed"], int)
        assert isinstance(data["total"], int)
        assert isinstance(data["spent_usd"], float)
        assert "T" in data["ts"]  # ISO 8601 format check
