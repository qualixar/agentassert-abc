"""Tests for dashboard/export_dashboard_data.py

Run from agentassert-abc repo root:
    .venv/bin/python -m pytest dashboard/tests/ -v

These tests use real agentassert_abc package functions with synthetic
MissionRecord objects — no network, no file-system beyond tempdir.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agentassert_abc.experiments.logging_schema import (
    ComponentRecord,
    HandoffRecord,
    MissionRecord,
)

import export_dashboard_data as exporter  # noqa: E402 — conftest adds dashboard/ to path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mission(
    idx: int,
    *,
    motif: str = "series2",
    condition: str = "same_model",
    hard_ok: bool = True,
    soft_ok: bool = True,
    drift_a: float | None = None,
    drift_b: float | None = None,
) -> MissionRecord:
    """Create a minimal two-node series2 MissionRecord for testing."""
    components = (
        ComponentRecord(
            component_id="node_a",
            model="test-model",
            role="worker",
            hard_ok=hard_ok,
            soft_ok=soft_ok,
            drift=drift_a,
            raw_output="42",
            scored=True,
        ),
        ComponentRecord(
            component_id="node_b",
            model="test-model",
            role="worker",
            hard_ok=hard_ok,
            soft_ok=soft_ok,
            drift=drift_b,
            raw_output="42",
            scored=True,
        ),
    )
    handoffs = (HandoffRecord(from_id="node_a", to_id="node_b", handoff_ok=True),)
    return MissionRecord(
        mission_id=f"mission-{motif}-{condition}-{idx}",
        cluster_id=f"cluster-{condition}-{idx}",
        motif=motif,
        sharing_condition=condition,
        route=("node_a", "node_b"),
        components=components,
        handoffs=handoffs,
        y_graph=hard_ok and soft_ok,
        tokens=0,
        cost_usd=0.0,
        timestamp="2026-01-01T00:00:00Z",
    )


def _all_passing(n: int = 30) -> list[MissionRecord]:
    return [_make_mission(i) for i in range(n)]


def _mixed(n: int = 30, fail_every: int = 5) -> list[MissionRecord]:
    """Missions that fail every `fail_every` steps to give non-degenerate table."""
    return [
        _make_mission(i, hard_ok=(i % fail_every != 0), soft_ok=(i % fail_every != 0))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestBuildDashboardPayload
# ---------------------------------------------------------------------------


class TestBuildDashboardPayload:

    def test_returns_all_required_top_level_keys(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        for key in ("meta", "composition", "certification", "dependence", "drift",
                    "timeline", "motif_breakdown", "condition_breakdown"):
            assert key in payload, f"Missing key: {key}"

    def test_meta_has_n_missions(self):
        payload = exporter.build_dashboard_payload(_all_passing(20))
        assert payload["meta"]["n_missions"] == 20

    def test_meta_has_generated_at(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        assert "generated_at" in payload["meta"]
        assert payload["meta"]["generated_at"]  # non-empty string

    def test_meta_records_p0_alpha(self):
        payload = exporter.build_dashboard_payload(_all_passing(10), p0=0.85, alpha=0.10)
        assert payload["meta"]["p0"] == pytest.approx(0.85)
        assert payload["meta"]["alpha"] == pytest.approx(0.10)

    # --- composition ---

    def test_composition_has_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        comp = payload["composition"]
        for field in ("observed_reliability", "independence_product", "gap",
                      "n_missions", "n_components", "n_handoffs"):
            assert field in comp, f"Missing composition field: {field}"

    def test_composition_observed_reliability_one_for_all_pass(self):
        payload = exporter.build_dashboard_payload(_all_passing(50))
        assert payload["composition"]["observed_reliability"] == pytest.approx(1.0)

    def test_composition_n_missions_correct(self):
        payload = exporter.build_dashboard_payload(_all_passing(40))
        assert payload["composition"]["n_missions"] == 40

    # --- certification ---

    def test_certification_has_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        cert = payload["certification"]
        for field in ("certified", "first_crossing_index", "final_wealth", "n_missions",
                      "p0", "alpha", "wealth_curve", "threshold"):
            assert field in cert, f"Missing certification field: {field}"

    def test_wealth_curve_length_matches_n_missions(self):
        n = 35
        payload = exporter.build_dashboard_payload(_all_passing(n))
        assert len(payload["certification"]["wealth_curve"]) == n

    def test_wealth_curve_monotone_nondecreasing_all_pass(self):
        """All y=1 missions → log-wealth must never decrease."""
        payload = exporter.build_dashboard_payload(_all_passing(60))
        curve = payload["certification"]["wealth_curve"]
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1] - 1e-12, (
                f"Wealth decreased at step {i}: {curve[i-1]:.4f} → {curve[i]:.4f}"
            )

    def test_certified_all_passing(self):
        # 100 all-passing missions at p0=0.90 must certify
        payload = exporter.build_dashboard_payload(_all_passing(100))
        assert payload["certification"]["certified"] is True

    def test_threshold_equals_log_inverse_alpha(self):
        import math
        payload = exporter.build_dashboard_payload(_all_passing(30), alpha=0.05)
        expected = math.log(1.0 / 0.05)
        assert payload["certification"]["threshold"] == pytest.approx(expected)

    def test_first_crossing_index_within_bounds(self):
        payload = exporter.build_dashboard_payload(_all_passing(100))
        cert = payload["certification"]
        if cert["certified"]:
            idx = cert["first_crossing_index"]
            assert 1 <= idx <= cert["n_missions"]

    # --- dependence ---

    def test_dependence_has_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        dep = payload["dependence"]
        for field in ("tau_a", "table", "n_missions", "agent_pair", "tau_a_ci"):
            assert field in dep, f"Missing dependence field: {field}"

    def test_dependence_table_has_four_cells(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        table = payload["dependence"]["table"]
        for cell in ("n11", "n10", "n01", "n00"):
            assert cell in table, f"Missing table cell: {cell}"

    def test_dependence_table_sums_to_n_missions(self):
        n = 30
        payload = exporter.build_dashboard_payload(_all_passing(n))
        table = payload["dependence"]["table"]
        total = table["n11"] + table["n10"] + table["n01"] + table["n00"]
        assert total == n

    def test_dependence_ci_has_lower_upper(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        ci = payload["dependence"]["tau_a_ci"]
        assert "lower" in ci
        assert "upper" in ci

    def test_dependence_ci_lower_le_upper(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        ci = payload["dependence"]["tau_a_ci"]
        assert ci["lower"] <= ci["upper"]

    # --- drift ---

    def test_drift_has_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        drift = payload["drift"]
        for field in ("n_agents", "n_passing", "n_failing_gate", "n_fit_error", "agent_results"):
            assert field in drift, f"Missing drift field: {field}"

    def test_drift_counts_partition_n_agents(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        d = payload["drift"]
        assert d["n_passing"] + d["n_failing_gate"] + d["n_fit_error"] == d["n_agents"]

    def test_drift_agent_results_have_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        for agent in payload["drift"]["agent_results"]:
            for field in ("agent_id", "n_obs", "gate_passed", "fit_error"):
                assert field in agent, f"Missing agent result field: {field}"

    def test_drift_no_obs_when_all_drift_null(self):
        payload = exporter.build_dashboard_payload(_all_passing(30))
        for agent in payload["drift"]["agent_results"]:
            assert agent["n_obs"] == 0

    # --- timeline ---

    def test_timeline_length_matches_n_missions(self):
        n = 25
        payload = exporter.build_dashboard_payload(_all_passing(n))
        assert len(payload["timeline"]) == n

    def test_timeline_entries_have_required_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        for entry in payload["timeline"]:
            for field in ("mission_id", "motif", "sharing_condition", "y_graph", "timestamp"):
                assert field in entry, f"Missing timeline field: {field}"

    def test_timeline_no_component_vectors(self):
        """Component vectors must NOT appear in timeline — they bloat the payload."""
        payload = exporter.build_dashboard_payload(_all_passing(10))
        for entry in payload["timeline"]:
            assert "components" not in entry
            assert "handoffs" not in entry

    # --- breakdown ---

    def test_motif_breakdown_present(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        assert "series2" in payload["motif_breakdown"]

    def test_condition_breakdown_present(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        assert "same_model" in payload["condition_breakdown"]

    def test_breakdown_n_and_passed_fields(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        for motif_data in payload["motif_breakdown"].values():
            assert "n" in motif_data
            assert "passed" in motif_data

    def test_breakdown_passed_le_n(self):
        payload = exporter.build_dashboard_payload(_all_passing(10))
        for motif_data in payload["motif_breakdown"].values():
            assert motif_data["passed"] <= motif_data["n"]


# ---------------------------------------------------------------------------
# TestWriteDataJs
# ---------------------------------------------------------------------------


class TestWriteDataJs:

    def _missions(self) -> list[MissionRecord]:
        return _all_passing(30)

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "data.js"
            exporter.write_data_js(self._missions(), output_path=out)
            assert out.exists()

    def test_starts_with_window_assignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "data.js"
            exporter.write_data_js(self._missions(), output_path=out)
            content = out.read_text()
            assert content.startswith("window.DASHBOARD_DATA = ")

    def test_ends_with_semicolon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "data.js"
            exporter.write_data_js(self._missions(), output_path=out)
            content = out.read_text().rstrip()
            assert content.endswith(";")

    def test_payload_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "data.js"
            exporter.write_data_js(self._missions(), output_path=out)
            content = out.read_text()
            # Strip JS wrapper
            json_str = content.removeprefix("window.DASHBOARD_DATA = ")
            json_str = json_str.rstrip().removesuffix(";")
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            assert "certification" in parsed

    def test_write_accepts_p0_alpha_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "data.js"
            exporter.write_data_js(
                self._missions(), p0=0.80, alpha=0.10, output_path=out
            )
            content = out.read_text()
            json_str = content.removeprefix("window.DASHBOARD_DATA = ").rstrip().removesuffix(";")
            parsed = json.loads(json_str)
            assert parsed["meta"]["p0"] == pytest.approx(0.80)
            assert parsed["meta"]["alpha"] == pytest.approx(0.10)
