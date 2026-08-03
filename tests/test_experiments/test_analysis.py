# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""TDD tests for experiments/analysis.py — four measurement families.

All tests use synthetic MissionRecords; NO model calls, NO I/O.

Test families:
    TestDependenceReport    — tau_a > 0, CI excludes 0 on strong co-failure.
    TestCompositionReport   — independence_product != observed when co-failing.
    TestCertificationReport — certifies on high-reliability stream; not on low.
    TestDriftReport         — fits exploring series; captures refusal on near-constant.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from agentassert_abc.experiments.logging_schema import (
    ComponentRecord,
    HandoffRecord,
    MissionRecord,
)
from agentassert_abc.experiments.analysis import (
    AnalysisError,
    AgentDriftResult,
    CertificationReport,
    CompositionReport,
    DependenceReport,
    DriftReport,
    certification_report,
    composition_report,
    dependence_report,
    drift_report,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _comp(
    component_id: str,
    *,
    hard_ok: bool,
    soft_ok: bool,
    drift: float | None = None,
) -> ComponentRecord:
    return ComponentRecord(
        component_id=component_id,
        model="qwen2.5:7b",
        role="worker",
        hard_ok=hard_ok,
        soft_ok=soft_ok,
        drift=drift,
        raw_output="{}",
        scored=True,
    )


def _make_mission(
    mission_id: str,
    cluster_id: str,
    components: tuple[ComponentRecord, ...],
    route: tuple[str, ...] = (),
    handoffs: tuple[HandoffRecord, ...] = (),
    motif: str = "series2",
    sharing_condition: str = "same_model",
) -> MissionRecord:
    return MissionRecord.make(
        mission_id=mission_id,
        cluster_id=cluster_id,
        motif=motif,
        sharing_condition=sharing_condition,
        route=route,
        components=components,
        handoffs=handoffs,
        tokens=0,
        cost_usd=0.0,
        timestamp="2026-07-26T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Dependence fixtures
# ---------------------------------------------------------------------------


def _build_dependence_missions(
    n_both_fail: int,
    n_both_succeed: int,
    n_a_only_fail: int,
    n_b_only_fail: int,
) -> list[MissionRecord]:
    """Matched-pair missions with agent_a / agent_b as paired components.

    Both agents appear in every mission's components.  Route is empty
    (matched-pair design: agents run independently, no handoffs).
    failure = not (hard_ok and soft_ok).
    """
    missions: list[MissionRecord] = []
    idx = 0

    def _mission(a_ok: bool, b_ok: bool) -> MissionRecord:
        nonlocal idx
        idx += 1
        return _make_mission(
            mission_id=f"m{idx:04d}",
            cluster_id=f"sc{idx:04d}",  # each mission is its own scenario cluster
            components=(
                _comp("agent_a", hard_ok=a_ok, soft_ok=a_ok),
                _comp("agent_b", hard_ok=b_ok, soft_ok=b_ok),
            ),
            route=("agent_a", "agent_b"),
            handoffs=(),
        )

    for _ in range(n_both_fail):
        missions.append(_mission(False, False))
    for _ in range(n_both_succeed):
        missions.append(_mission(True, True))
    for _ in range(n_a_only_fail):
        missions.append(_mission(False, True))
    for _ in range(n_b_only_fail):
        missions.append(_mission(True, False))

    return missions


# ---------------------------------------------------------------------------
# Composition fixtures
# ---------------------------------------------------------------------------


def _build_composition_missions_cofailing(
    n_both_fail: int = 100,
    n_both_succeed: int = 100,
) -> list[MissionRecord]:
    """Series-2 motif where both components ALWAYS co-fail or co-succeed.

    Independence product = 0.5 * 0.5 * 1.0 = 0.25.
    Observed reliability = 0.50 (they always fail together → graph succeeds
    iff both succeed → P(y_graph=True) = 0.50).
    Gap = 0.25 - 0.50 = -0.25.
    """
    missions: list[MissionRecord] = []
    for i in range(n_both_fail + n_both_succeed):
        succeed = i >= n_both_fail
        missions.append(
            _make_mission(
                mission_id=f"cm{i:04d}",
                cluster_id=f"csc{i:04d}",
                components=(
                    _comp("agent_a", hard_ok=succeed, soft_ok=succeed),
                    _comp("agent_b", hard_ok=succeed, soft_ok=succeed),
                ),
                route=("agent_a", "agent_b"),
                handoffs=(HandoffRecord(from_id="agent_a", to_id="agent_b", handoff_ok=True),),
            )
        )
    return missions


# ---------------------------------------------------------------------------
# Certification fixtures
# ---------------------------------------------------------------------------


def _build_cert_missions(
    n_success: int,
    n_fail: int,
    *,
    interleave: bool = False,
    motif: str = "series2",
    sharing_condition: str = "same_model",
) -> list[MissionRecord]:
    """Missions for certification test; y_graph drives the stream.

    Parameters
    ----------
    interleave:
        When True, interleave successes and failures uniformly so that the
        running success rate stays close to n_success/(n_success+n_fail)
        throughout the stream.  Use this when testing that the adaptive bet
        never spuriously spikes due to an all-success prefix.
    """
    if interleave:
        # Distribute successes evenly: for n_success successes in total,
        # spread them uniformly across the n_success+n_fail positions.
        total = n_success + n_fail
        outcomes: list[bool] = [False] * total
        step = total / n_success if n_success > 0 else float("inf")
        for k in range(n_success):
            outcomes[int(k * step)] = True
    else:
        outcomes = [True] * n_success + [False] * n_fail

    missions: list[MissionRecord] = []
    for i, succeed in enumerate(outcomes):
        missions.append(
            _make_mission(
                mission_id=f"cert{i:04d}",
                cluster_id=f"certsc{i:04d}",
                components=(
                    _comp("agent_a", hard_ok=succeed, soft_ok=succeed),
                ),
                route=("agent_a",),
                motif=motif,
                sharing_condition=sharing_condition,
            )
        )
    return missions


# ---------------------------------------------------------------------------
# Drift fixtures
# ---------------------------------------------------------------------------


def _build_drift_missions_mixed() -> list[MissionRecord]:
    """100 missions with two components:

    * ``"exploring"``  — drift follows a sinusoidal arc from ~0.1 to ~0.9.
      Positive lag-1 autocorrelation; explores both boundaries.
    * ``"constant"``   — drift fixed at 0.05 for every mission.
      Identically-valued → _lag1_autocorr returns 0.0 → JacobiFitError.
    """
    n = 100
    t = np.linspace(0.0, 2.0 * np.pi, n)
    exploring_drifts = (0.5 + 0.40 * np.sin(t)).tolist()

    missions: list[MissionRecord] = []
    for i in range(n):
        missions.append(
            _make_mission(
                mission_id=f"dr{i:04d}",
                cluster_id=f"drsc{i:04d}",
                components=(
                    _comp("exploring", hard_ok=True, soft_ok=True, drift=exploring_drifts[i]),
                    _comp("constant", hard_ok=True, soft_ok=True, drift=0.05),
                ),
                route=("exploring", "constant"),
            )
        )
    return missions


# ===========================================================================
# TestDependenceReport
# ===========================================================================


class TestDependenceReport:
    """H-D: co-failure dependence via Kendall tau_a + tetrachoric + bootstrap CI."""

    def test_tau_a_positive_on_strong_cofailure(self) -> None:
        """tau_a > 0 when agents fail together far more than independently."""
        missions = _build_dependence_missions(
            n_both_fail=90,
            n_both_succeed=90,
            n_a_only_fail=10,
            n_b_only_fail=10,
        )
        report = dependence_report(missions, "agent_a", "agent_b")
        assert isinstance(report, DependenceReport)
        assert report.tau_a > 0.0, f"Expected tau_a > 0, got {report.tau_a}"

    def test_bootstrap_ci_excludes_zero(self) -> None:
        """With 200 missions and tau_a ≈ 0.40, 95% CI lower bound must be > 0."""
        missions = _build_dependence_missions(90, 90, 10, 10)
        report = dependence_report(missions, "agent_a", "agent_b")
        ci = report.tau_a_ci
        assert ci.lower > 0.0, (
            f"CI lower bound {ci.lower:.4f} should be > 0 for strong co-failure"
        )

    def test_table_counts_match_construction(self) -> None:
        """CoFailureTable cell counts must match the synthetic construction."""
        missions = _build_dependence_missions(90, 90, 10, 10)
        report = dependence_report(missions, "agent_a", "agent_b")
        t = report.table
        # failure = not (hard_ok AND soft_ok)
        assert t.n11 == 90, f"n11={t.n11}"   # both failed
        assert t.n00 == 90, f"n00={t.n00}"   # both succeeded
        assert t.n10 == 10, f"n10={t.n10}"   # only agent_a failed
        assert t.n01 == 10, f"n01={t.n01}"   # only agent_b failed

    def test_tau_a_value_close_to_expected(self) -> None:
        """tau_a = 2*(p11*p00 - p10*p01) should equal ≈ 0.40 for (90,90,10,10)."""
        missions = _build_dependence_missions(90, 90, 10, 10)
        report = dependence_report(missions, "agent_a", "agent_b")
        expected = 2.0 * ((90 / 200) * (90 / 200) - (10 / 200) * (10 / 200))
        assert abs(report.tau_a - expected) < 1e-9

    def test_tetrachoric_positive_and_finite(self) -> None:
        """Tetrachoric rho should be in (0, 1) for positive co-failure."""
        missions = _build_dependence_missions(90, 90, 10, 10)
        report = dependence_report(missions, "agent_a", "agent_b")
        assert report.tetrachoric_rho is not None
        rho = report.tetrachoric_rho
        assert math.isfinite(rho), f"tetrachoric_rho={rho} should be finite"
        assert rho > 0.0, f"tetrachoric_rho={rho} should be positive"

    def test_n_missions_count(self) -> None:
        missions = _build_dependence_missions(90, 90, 10, 10)
        report = dependence_report(missions, "agent_a", "agent_b")
        assert report.n_missions == 200

    def test_raises_on_missing_agent(self) -> None:
        missions = _build_dependence_missions(5, 5, 0, 0)
        with pytest.raises(AnalysisError):
            dependence_report(missions, "agent_a", "nonexistent_agent")

    def test_raises_on_empty_missions(self) -> None:
        with pytest.raises(AnalysisError):
            dependence_report([], "agent_a", "agent_b")

    def test_tetrachoric_none_on_degenerate_marginal(self) -> None:
        """tetrachoric_rho is None when agent_b never fails (degenerate marginal)."""
        # agent_b always succeeds → p_b = 0, tetrachoric undefined
        missions = _build_dependence_missions(
            n_both_fail=0,
            n_both_succeed=50,
            n_a_only_fail=10,
            n_b_only_fail=0,
        )
        report = dependence_report(missions, "agent_a", "agent_b")
        assert report.tetrachoric_rho is None


# ===========================================================================
# TestCompositionReport
# ===========================================================================


class TestCompositionReport:
    """H-C: independence product vs. observed reliability gap."""

    def test_gap_nonzero_on_cofailing_series2(self) -> None:
        """independence_product - observed_reliability should be far from 0."""
        missions = _build_composition_missions_cofailing(100, 100)
        report = composition_report(missions)
        assert isinstance(report, CompositionReport)
        assert abs(report.gap) > 0.01, (
            f"Expected |gap| > 0.01 for co-failing agents, got gap={report.gap:.4f}"
        )

    def test_gap_direction_independence_underpredicts(self) -> None:
        """Independence product (0.25) < observed (0.50): gap should be negative."""
        missions = _build_composition_missions_cofailing(100, 100)
        report = composition_report(missions)
        # independence_product = 0.5 * 0.5 * 1.0 = 0.25; observed = 0.50
        assert report.gap < 0.0, (
            f"Expected gap < 0 (product underpredicts), got {report.gap:.4f}"
        )

    def test_observed_reliability_matches_y_graph_mean(self) -> None:
        """observed_reliability must equal mean(m.y_graph) over all missions."""
        missions = _build_composition_missions_cofailing(60, 40)
        report = composition_report(missions)
        expected_obs = sum(m.y_graph for m in missions) / len(missions)
        assert abs(report.observed_reliability - expected_obs) < 1e-9

    def test_independence_product_matches_marginals(self) -> None:
        """Independence product = P(A) * P(B) * P(handoff A→B)."""
        missions = _build_composition_missions_cofailing(100, 100)
        report = composition_report(missions)
        # P(A succeeds) = P(B succeeds) = 0.5; P(handoff) = 1.0
        expected_product = 0.5 * 0.5 * 1.0
        assert abs(report.independence_product - expected_product) < 1e-9

    def test_n_missions_count(self) -> None:
        missions = _build_composition_missions_cofailing(30, 70)
        report = composition_report(missions)
        assert report.n_missions == 100

    def test_raises_on_empty_missions(self) -> None:
        with pytest.raises(AnalysisError):
            composition_report([])

    def test_no_components_no_route_gives_trivial_product(self) -> None:
        """Missions with empty routes produce independence_product=1.0 (no components)."""
        missions = [
            _make_mission(f"m{i}", f"sc{i}", components=(), route=())
            for i in range(5)
        ]
        report = composition_report(missions)
        assert report.independence_product == 1.0


# ===========================================================================
# TestCertificationReport
# ===========================================================================


class TestCertificationReport:
    """H-E: anytime-valid graph e-process certification."""

    def test_certifies_on_high_reliability_stream(self) -> None:
        """95% success rate against p0=0.70 must cross the e-process threshold."""
        missions = _build_cert_missions(n_success=95, n_fail=5)
        report = certification_report(missions, p0=0.70, alpha=0.05)
        assert isinstance(report, CertificationReport)
        assert report.certified is True, (
            f"Expected certified=True, got certified={report.certified}, "
            f"wealth={report.final_wealth:.3f}"
        )

    def test_first_crossing_index_set_when_certified(self) -> None:
        """first_crossing_index must be a positive int when certified."""
        missions = _build_cert_missions(n_success=95, n_fail=5)
        report = certification_report(missions, p0=0.70, alpha=0.05)
        assert report.first_crossing_index is not None
        assert isinstance(report.first_crossing_index, int)
        assert report.first_crossing_index >= 1

    def test_no_certificate_below_null(self) -> None:
        """65% success rate against p0=0.80 must NOT certify (below null).

        Interleaved ordering keeps the running success rate at ~65% so the
        terminal-only expert stays clamped to p0 and the wealth never grows.
        """
        missions = _build_cert_missions(n_success=65, n_fail=35, interleave=True)
        report = certification_report(missions, p0=0.80, alpha=0.05)
        assert report.certified is False, (
            f"Expected certified=False for below-null stream, "
            f"wealth={report.final_wealth:.3f}"
        )

    def test_no_crossing_index_when_not_certified(self) -> None:
        # Interleaved 40% success rate stream against p0=0.80.  Interleaving
        # keeps the running success rate ~40% throughout, so the terminal-only
        # expert is always clamped to p0 → zero bet → wealth never rises.
        missions = _build_cert_missions(n_success=40, n_fail=60, interleave=True)
        report = certification_report(missions, p0=0.80, alpha=0.05)
        assert report.first_crossing_index is None

    def test_n_missions_count(self) -> None:
        missions = _build_cert_missions(n_success=50, n_fail=50)
        report = certification_report(missions, p0=0.60, alpha=0.05)
        assert report.n_missions == 100

    def test_p0_and_alpha_stored(self) -> None:
        missions = _build_cert_missions(n_success=40, n_fail=10)
        report = certification_report(missions, p0=0.75, alpha=0.10)
        assert report.p0 == 0.75
        assert report.alpha == 0.10

    def test_raises_on_empty_missions(self) -> None:
        with pytest.raises(AnalysisError):
            certification_report([], p0=0.80, alpha=0.05)

    def test_raises_on_invalid_p0(self) -> None:
        missions = _build_cert_missions(n_success=10, n_fail=10)
        with pytest.raises(AnalysisError, match="p0 must be in"):
            certification_report(missions, p0=1.5, alpha=0.05)

    def test_raises_on_invalid_alpha(self) -> None:
        missions = _build_cert_missions(n_success=10, n_fail=10)
        with pytest.raises(AnalysisError, match="alpha must be in"):
            certification_report(missions, p0=0.80, alpha=0.0)


# ===========================================================================
# TestDriftReport
# ===========================================================================


class TestDriftReport:
    """H-J: Jacobi drift fit and identifiability gate."""

    def test_exploring_series_gate_runs(self) -> None:
        """Sinusoidal drift series should produce a gate_result (not JacobiFitError)."""
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        assert isinstance(report, DriftReport)
        exploring = next(
            (r for r in report.agent_results if r.agent_id == "exploring"), None
        )
        assert exploring is not None, "No result for 'exploring' agent"
        assert exploring.fit_error is None, (
            f"'exploring' agent raised JacobiFitError: {exploring.fit_error}"
        )
        assert exploring.gate_result is not None, (
            "'exploring' agent must have a gate_result"
        )

    def test_near_constant_series_graceful_refusal(self) -> None:
        """Near-constant drift (all=0.05) raises JacobiFitError; report captures it."""
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        constant = next(
            (r for r in report.agent_results if r.agent_id == "constant"), None
        )
        assert constant is not None, "No result for 'constant' agent"
        assert constant.fit_error is not None, (
            "Expected fit_error for near-constant series; whole report must not crash"
        )
        assert constant.gate_result is None, (
            "gate_result should be None when fit failed"
        )

    def test_n_agents_correct(self) -> None:
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        assert report.n_agents == 2  # "exploring" and "constant"

    def test_n_fit_error_counts_constant_agent(self) -> None:
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        assert report.n_fit_error == 1

    def test_n_passing_counts_correctly(self) -> None:
        """n_passing counts agents where gate ran AND gate_passed is True."""
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        # n_passing + n_failing_gate + n_fit_error == n_agents
        total = report.n_passing + report.n_failing_gate + report.n_fit_error
        assert total == report.n_agents

    def test_n_obs_matches_drift_count(self) -> None:
        """n_obs for each agent should equal the number of non-None drift values."""
        missions = _build_drift_missions_mixed()
        report = drift_report(missions, dt=1.0)
        constant_result = next(r for r in report.agent_results if r.agent_id == "constant")
        assert constant_result.n_obs == 100  # 100 missions, all drift=0.05

    def test_raises_on_empty_missions(self) -> None:
        with pytest.raises(AnalysisError):
            drift_report([], dt=1.0)

    def test_raises_on_invalid_dt(self) -> None:
        missions = _build_drift_missions_mixed()
        with pytest.raises(AnalysisError):
            drift_report(missions, dt=0.0)

    def test_drift_none_values_skipped(self) -> None:
        """Components with drift=None are silently skipped; agent must still appear."""
        n = 50
        missions = [
            _make_mission(
                f"drnone{i}",
                f"sc{i}",
                components=(
                    _comp("partial_agent", hard_ok=True, soft_ok=True, drift=None),
                ),
                route=("partial_agent",),
            )
            for i in range(n)
        ]
        # All drift=None; agent should appear with n_obs=0 and a fit_error
        report = drift_report(missions, dt=1.0)
        result = next(r for r in report.agent_results if r.agent_id == "partial_agent")
        assert result.n_obs == 0
        assert result.fit_error is not None
