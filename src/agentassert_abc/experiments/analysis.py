# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Measurement families for the LLD-E zero-budget empirical validation.

Given a ``list[MissionRecord]``, computes the four confirmatory estimand
families defined in LLD-E §8–§11:

1. **Dependence** (§8)   — Kendall τ_a, tetrachoric ρ, cluster-bootstrap CI.
2. **Composition** (§9)  — observed graph reliability vs independence product.
3. **Certification** (§10) — anytime-valid e-process; certified flag + wealth.
4. **Drift** (§11)       — per-agent Jacobi fit + identifiability gate.

All four public functions are **pure** — they perform no I/O, no model calls,
and return immutable frozen dataclasses.  Degenerate or missing inputs raise
:class:`AnalysisError` rather than silently producing garbage.

References
----------
LLD-E-experiment-design-v2.md §8 (dependence), §9 (composition),
§10 (e-process), §11 (Jacobi drift / power).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentassert_abc.certification.eprocess import GraphEProcess
from agentassert_abc.dependence import (
    BootstrapCI,
    CoFailureTable,
    cluster_bootstrap,
    kendall_tau_a,
    tetrachoric,
)
from agentassert_abc.exceptions import AgentAssertError, DependenceError
from agentassert_abc.metrics.jacobi import (
    GateResult,
    identifiability_gate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentassert_abc.experiments.logging_schema import MissionRecord

__all__ = [
    "AnalysisError",
    "AgentDriftResult",
    "CertificationReport",
    "CompositionReport",
    "DependenceReport",
    "DriftReport",
    "certification_report",
    "composition_report",
    "dependence_report",
    "drift_report",
]

# ---------------------------------------------------------------------------
# Module-local exception
# ---------------------------------------------------------------------------


class AnalysisError(AgentAssertError):
    """Invalid input or insufficient data for an analysis function.

    Raised when missions are empty, a required agent is absent, or a numeric
    parameter is out of range.  Never raised for a series that merely fails
    the Jacobi identifiability gate — that outcome is captured in the result.
    """


# ---------------------------------------------------------------------------
# Result dataclasses (all frozen + slots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DependenceReport:
    """Co-failure dependence results for a pair of agents (LLD-E §8).

    Attributes
    ----------
    table:
        2×2 co-failure contingency (n11, n10, n01, n00).
    tau_a:
        Point estimate of binary Kendall τ_a = 2(p11·p00 − p10·p01).
    tetrachoric_rho:
        Latent bivariate-normal correlation.  ``None`` when a marginal is
        degenerate (0 or 1) and the model is unidentified.
    tau_a_ci:
        Scenario-cluster-bootstrap 95% CI for τ_a.
    n_missions:
        Number of missions in which both agents appeared.
    """

    table: CoFailureTable
    tau_a: float
    tetrachoric_rho: float | None
    tau_a_ci: BootstrapCI
    n_missions: int


@dataclass(frozen=True, slots=True)
class CompositionReport:
    """Observed vs. independence-product graph reliability (LLD-E §9).

    Attributes
    ----------
    observed_reliability:
        Mean Y_{G,r} across all missions.
    independence_product:
        Product of per-component and per-handoff marginal reliabilities,
        estimated from the same missions.
    gap:
        independence_product − observed_reliability.  Negative when positive
        dependence causes the product to underpredict observed reliability.
    n_missions:
        Total missions included in the computation.
    n_components:
        Number of distinct component IDs found on realized routes.
    n_handoffs:
        Number of distinct (from_id, to_id) handoff pairs found on routes.
    """

    observed_reliability: float
    independence_product: float
    gap: float
    n_missions: int
    n_components: int
    n_handoffs: int


@dataclass(frozen=True, slots=True)
class CertificationReport:
    """E-process certification result for a mission stream (LLD-E §10).

    Attributes
    ----------
    certified:
        True iff E_r ≥ 1/α at some mission r in the stream.
    first_crossing_index:
        First mission index r where E_r ≥ 1/α, or ``None`` if never crossed.
    final_wealth:
        E-process wealth E_r after all missions.
    n_missions:
        Number of missions in the stream.
    p0:
        Null reliability threshold used for certification.
    alpha:
        Significance level used for certification.
    """

    certified: bool
    first_crossing_index: int | None
    final_wealth: float
    n_missions: int
    p0: float
    alpha: float


@dataclass(frozen=True, slots=True)
class AgentDriftResult:
    """Per-agent drift fit result (LLD-E §11, LLD-D §D.4–D.5).

    Attributes
    ----------
    agent_id:
        Component ID identifying the agent slot across missions.
    n_obs:
        Number of non-``None`` drift observations collected.
    gate_result:
        Full :class:`~agentassert_abc.metrics.jacobi.GateResult` from
        :func:`~agentassert_abc.metrics.jacobi.identifiability_gate`,
        or ``None`` when the base :func:`fit_jacobi_mom` raised
        :class:`~agentassert_abc.metrics.jacobi.JacobiFitError`.
    fit_error:
        Human-readable error message when the series was too degenerate
        for the base MoM fit (e.g. zero variance, zero lag-1 autocorr).
        ``None`` when the fit succeeded (regardless of gate pass/fail).
    """

    agent_id: str
    n_obs: int
    gate_result: GateResult | None
    fit_error: str | None


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Per-agent Jacobi drift results for all agents in a mission batch (LLD-E §11).

    Attributes
    ----------
    agent_results:
        One :class:`AgentDriftResult` per distinct component ID found across
        all missions' ``components`` tuples.
    n_agents:
        Total number of distinct agents analysed.
    n_passing:
        Agents where gate ran AND ``gate_result.gate_passed`` is ``True``.
    n_failing_gate:
        Agents where gate ran but ``gate_result.gate_passed`` is ``False``.
    n_fit_error:
        Agents where the base MoM fit raised :class:`JacobiFitError`.
    """

    agent_results: tuple[AgentDriftResult, ...]
    n_agents: int
    n_passing: int
    n_failing_gate: int
    n_fit_error: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_missions(missions: Sequence[MissionRecord], caller: str) -> None:
    """Raise AnalysisError if *missions* is empty."""
    if not missions:
        raise AnalysisError(
            f"{caller}: mission list is empty — nothing to analyse"
        )


def _agent_failure(hard_ok: bool, soft_ok: bool) -> int:
    """Return 1 (failed) iff the component's system-level outcome is a failure.

    Failure = NOT (hard_ok AND soft_ok), matching the LLD-E §8.1 definition
    F_{a,r} = 1 − Y_{a,r} where Y_{a,r} = hard_ok ∧ soft_ok.
    """
    return 0 if (hard_ok and soft_ok) else 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dependence_report(
    missions: Sequence[MissionRecord],
    agent_i: str,
    agent_j: str,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260725,
) -> DependenceReport:
    """Compute co-failure dependence for a pair of agents (LLD-E §8).

    Pairs the two agents' contract-failure indicators — a component fails when
    ``NOT (hard_ok AND soft_ok)``, matching the route-consistent Y = hard ∧ soft
    convention — across all missions that contain both agents as components.
    Builds a
    :class:`~agentassert_abc.dependence.CoFailureTable`, estimates binary
    Kendall τ_a, tetrachoric ρ, and a scenario-cluster-bootstrap CI for τ_a.

    Parameters
    ----------
    missions:
        Full mission batch.  Missions that lack one or both agents are silently
        skipped; if fewer than 2 paired missions remain, :class:`AnalysisError`
        is raised.
    agent_i:
        ``component_id`` of the first agent.
    agent_j:
        ``component_id`` of the second agent.
    n_boot:
        Number of cluster-bootstrap resamples (default 2000).
    alpha:
        Two-sided CI level (default 0.05).
    seed:
        RNG seed for bootstrap reproducibility.

    Returns
    -------
    DependenceReport

    Raises
    ------
    AnalysisError
        Empty mission list, or fewer than 2 missions contain both agents.
    """
    _require_missions(missions, "dependence_report")

    fail_a: list[int] = []
    fail_b: list[int] = []
    cluster_ids: list[str] = []

    for m in missions:
        comp_by_id = {c.component_id: c for c in m.components}
        if agent_i not in comp_by_id or agent_j not in comp_by_id:
            continue
        ci = comp_by_id[agent_i]
        cj = comp_by_id[agent_j]
        fail_a.append(_agent_failure(ci.hard_ok, ci.soft_ok))
        fail_b.append(_agent_failure(cj.hard_ok, cj.soft_ok))
        cluster_ids.append(m.cluster_id)

    if len(fail_a) < 2:
        raise AnalysisError(
            f"dependence_report: fewer than 2 missions contain both "
            f"'{agent_i}' and '{agent_j}' (found {len(fail_a)})"
        )

    table = CoFailureTable.from_pairs(fail_a, fail_b)
    tau = kendall_tau_a(table)

    # Tetrachoric rho — may be undefined for degenerate marginals
    rho: float | None
    try:
        rho = tetrachoric(table)
    except DependenceError:
        rho = None

    ci = cluster_bootstrap(
        fail_a,
        fail_b,
        cluster_ids,
        kendall_tau_a,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    return DependenceReport(
        table=table,
        tau_a=tau,
        tetrachoric_rho=rho,
        tau_a_ci=ci,
        n_missions=len(fail_a),
    )


def composition_report(
    missions: Sequence[MissionRecord],
) -> CompositionReport:
    """Observed graph reliability vs. the multiplicative independence product (LLD-E §9).

    Computes per-component and per-handoff marginal reliabilities from the
    missions (restricted to on-route appearances), then multiplies them for the
    independence product.  The gap = independence_product − observed_reliability
    is negative when positive dependence makes the independent product
    underestimate actual reliability.

    Parameters
    ----------
    missions:
        Full mission batch.  All missions contribute to observed_reliability;
        route-consistent component and handoff marginals are pooled across the
        batch.

    Caveat
    ------
    The independence product is a *pooled-global* product over the union of
    on-route component ids and edges across the batch, not ``E[prod_r p_c]``
    per mission.  This is exact for a homogeneous batch (a single fixed motif);
    for a heterogeneous multi-motif batch the pooled product can mis-state the
    independence baseline — call this per-motif for heterogeneous designs.

    Returns
    -------
    CompositionReport

    Raises
    ------
    AnalysisError
        Empty mission list.
    """
    _require_missions(missions, "composition_report")

    # Collect per-component success observations — only when on route
    comp_obs: dict[str, list[bool]] = defaultdict(list)
    # Collect per-handoff success observations — only when both endpoints on route
    handoff_obs: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for m in missions:
        route_set: frozenset[str] = frozenset(m.route)
        for comp in m.components:
            if comp.component_id in route_set:
                comp_obs[comp.component_id].append(comp.hard_ok and comp.soft_ok)
        for h in m.handoffs:
            if h.from_id in route_set and h.to_id in route_set:
                handoff_obs[(h.from_id, h.to_id)].append(h.handoff_ok)

    # Marginal reliabilities
    comp_rel: dict[str, float] = {
        cid: sum(obs) / len(obs)
        for cid, obs in comp_obs.items()
        if obs
    }
    handoff_rel: dict[tuple[str, str], float] = {
        k: sum(obs) / len(obs)
        for k, obs in handoff_obs.items()
        if obs
    }

    # Independence product (1.0 if no components/handoffs found)
    independence_product = 1.0
    for p in comp_rel.values():
        independence_product *= p
    for p in handoff_rel.values():
        independence_product *= p

    # Observed reliability
    observed_reliability = sum(m.y_graph for m in missions) / len(missions)

    gap = independence_product - observed_reliability

    return CompositionReport(
        observed_reliability=observed_reliability,
        independence_product=independence_product,
        gap=gap,
        n_missions=len(missions),
        n_components=len(comp_rel),
        n_handoffs=len(handoff_rel),
    )


def certification_report(
    missions: Sequence[MissionRecord],
    p0: float,
    alpha: float,
) -> CertificationReport:
    """Anytime-valid e-process certification over a mission stream (LLD-E §10).

    Feeds the per-mission Y_{G,r} stream (in supplied order) into a mixture
    e-process with the terminal-only adaptive bet (LLD-E §10 Eq 5.6).  Returns
    the certified flag, first crossing index, and final wealth.

    Parameters
    ----------
    missions:
        Ordered mission stream.  Y_{G,r} = int(m.y_graph) for each mission.
    p0:
        Null reliability threshold, e.g. 0.80.
    alpha:
        Anytime-valid significance level, e.g. 0.05.  Certification requires
        E_r ≥ 1/α.

    Returns
    -------
    CertificationReport

    Raises
    ------
    AnalysisError
        Empty mission list, or p0/alpha outside (0, 1).
    """
    _require_missions(missions, "certification_report")
    if not (0.0 < p0 < 1.0):
        raise AnalysisError(f"certification_report: p0 must be in (0, 1), got {p0}")
    if not (0.0 < alpha < 1.0):
        raise AnalysisError(
            f"certification_report: alpha must be in (0, 1), got {alpha}"
        )

    # Forecast-clipping constant: epsilon < 1 − p0, chosen as (1−p0)/3
    epsilon = (1.0 - p0) / 3.0

    ep = GraphEProcess.mixture(p0=p0, alpha=alpha, epsilon=epsilon)

    for m in missions:
        ep.update(int(m.y_graph))

    return CertificationReport(
        certified=ep.certified(),
        first_crossing_index=ep.first_crossing_index,
        final_wealth=ep.wealth,
        n_missions=ep.mission_count,
        p0=p0,
        alpha=alpha,
    )


def drift_report(
    missions: Sequence[MissionRecord],
    dt: float,
) -> DriftReport:
    """Per-agent Jacobi drift fit and identifiability gate (LLD-E §11, LLD-D).

    Collects each agent's drift series from
    :attr:`~agentassert_abc.experiments.logging_schema.ComponentRecord.drift`
    across all missions (``None`` values skipped).  Attempts a MoM fit and
    identifiability gate for each agent.  When the base fit raises
    :class:`~agentassert_abc.metrics.jacobi.JacobiFitError` (e.g. a
    near-constant series with zero variance), the error is captured in
    :attr:`AgentDriftResult.fit_error` rather than propagating — the whole
    report is still returned.

    Parameters
    ----------
    missions:
        Full mission batch.
    dt:
        Positive time step between consecutive observations (e.g. 1.0).

    Returns
    -------
    DriftReport

    Raises
    ------
    AnalysisError
        Empty mission list or ``dt ≤ 0``.
    """
    _require_missions(missions, "drift_report")
    if dt <= 0.0:
        raise AnalysisError(f"drift_report: dt must be > 0, got {dt!r}")

    # Collect drift series per agent, preserving chronological order
    agent_drifts: dict[str, list[float]] = defaultdict(list)

    for m in missions:
        for comp in m.components:
            if comp.drift is not None:
                agent_drifts[comp.component_id].append(comp.drift)

    # Also track agents that appear but have no drift observations
    all_agent_ids: set[str] = set()
    for m in missions:
        for comp in m.components:
            all_agent_ids.add(comp.component_id)

    results: list[AgentDriftResult] = []

    for agent_id in sorted(all_agent_ids):  # deterministic ordering
        series = agent_drifts.get(agent_id, [])
        n_obs = len(series)

        if n_obs < 3:
            # Cannot fit with fewer than 3 observations
            results.append(
                AgentDriftResult(
                    agent_id=agent_id,
                    n_obs=n_obs,
                    gate_result=None,
                    fit_error=(
                        f"Insufficient drift observations for agent '{agent_id}': "
                        f"need >= 3, got {n_obs}"
                    ),
                )
            )
            continue

        # identifiability_gate is graceful: it never raises — a degenerate series
        # returns GateResult(params=None, gate_passed=False).  Detect that case
        # and surface the reason as fit_error while keeping gate_result=None so
        # the n_passing / n_failing_gate / n_fit_error partition stays disjoint.
        gate = identifiability_gate(series, dt)
        if gate.params is None:
            # Base MoM fit failed (degenerate / inadmissible series)
            results.append(
                AgentDriftResult(
                    agent_id=agent_id,
                    n_obs=n_obs,
                    gate_result=None,
                    fit_error=gate.reason,
                )
            )
        else:
            results.append(
                AgentDriftResult(
                    agent_id=agent_id,
                    n_obs=n_obs,
                    gate_result=gate,
                    fit_error=None,
                )
            )

    n_passing = sum(
        1 for r in results if r.gate_result is not None and r.gate_result.gate_passed
    )
    n_failing_gate = sum(
        1 for r in results if r.gate_result is not None and not r.gate_result.gate_passed
    )
    n_fit_error = sum(1 for r in results if r.fit_error is not None)

    return DriftReport(
        agent_results=tuple(results),
        n_agents=len(results),
        n_passing=n_passing,
        n_failing_gate=n_failing_gate,
        n_fit_error=n_fit_error,
    )
