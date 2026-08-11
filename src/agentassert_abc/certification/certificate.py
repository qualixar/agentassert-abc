# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""The AgentAssert reliability certificate — the single entry point.

Assembles the tiered all-success floors and selects the value that may be
**printed on a certificate**, with the assumptions it rests on:

* **Tier 0** — :func:`~agentassert_abc.certification.observed_floor.observed_all_success_floor`.
  Exact Clopper–Pearson lower bound on the directly-observed all-success rate.
  The guarantee **when the composition was executed end-to-end** (every stage
  scored on the same missions). No copula, no model, finite-sample.
* **Tier 1** — :func:`~agentassert_abc.certification.lp_bound.pairwise_cp_box_floor`.
  Finite-sample copula-agnostic LP floor over a Bonferroni-CP moment box. The
  guarantee **when the composition was not run end-to-end** (only per-stage /
  pairwise data), so the joint all-success rate must be bounded, not observed.
* **Tier 2** — :func:`~agentassert_abc.certification.slepian_floor.slepian_model_floor`.
  The correct finite-sample Gaussian one-factor *model* floor (Slepian corner).
  A **diagnostic**, reported for transparency — **never** the guarantee, because
  its coverage of the true reliability collapses under a pairwise-indistinguishable
  misspecification (audit F1).

The dependence-aware Gaussian *point* estimate (``series_all_success_point``) is
also a diagnostic. The naive v1 independence product is reported for contrast.
"""

from __future__ import annotations

import dataclasses

from agentassert_abc.certification.factor_reliability import _as_pass_matrix
from agentassert_abc.certification.lp_bound import (
    PairwiseRobustFloor,
    pairwise_cp_box_floor,
)
from agentassert_abc.certification.observed_floor import (
    ObservedFloor,
    observed_all_success_floor,
)
from agentassert_abc.certification.slepian_floor import (
    SlepianModelFloor,
    slepian_model_floor,
)
from agentassert_abc.exceptions import DependenceError


@dataclasses.dataclass(frozen=True, slots=True)
class Certificate:
    """A composed-pipeline reliability certificate with an auditable basis.

    Attributes:
        guarantee: the certified (1 − η_conf) lower bound on all-success
            reliability — **the value that may be printed on the certificate**.
        guarantee_tier: 0 (executed end-to-end → exact Clopper–Pearson) or
            1 (extrapolation → copula-agnostic LP).
        guarantee_basis: human-readable basis for the guarantee.
        assumptions: the assumptions the guarantee rests on.
        observed: empirical all-success rate.
        eta_conf: one-sided miscoverage.
        executed_end_to_end: whether Tier 0 applies.
        tier0: the exact Clopper–Pearson floor (always computed).
        tier1: the copula-agnostic LP floor (always computed).
        tier2: the Gaussian model floor — a DIAGNOSTIC, not a guarantee.
        m: stages. n: missions.
        scope_note: the scope/limits statement for the certificate.
    """

    guarantee: float
    guarantee_tier: int
    guarantee_basis: str
    assumptions: tuple[str, ...]
    observed: float
    eta_conf: float
    executed_end_to_end: bool
    tier0: ObservedFloor
    tier1: PairwiseRobustFloor
    tier2: SlepianModelFloor
    m: int
    n: int
    scope_note: str


def certify(
    passes: object,
    eta_conf: float = 0.05,
    executed_end_to_end: bool = False,
) -> Certificate:
    """Assemble the tiered floors and select the certifiable guarantee.

    Args:
        passes: ``m × n`` binary pass matrix (rows = stages, cols = missions).
        eta_conf: one-sided miscoverage (default 0.05 → 95% guarantee).
        executed_end_to_end: set ``True`` **only** when every stage was scored on
            the same missions AND the composed pipeline is the one being certified
            — then all-success is directly observed and Tier 0 (exact
            Clopper–Pearson) is the guarantee. Otherwise (the **fail-safe
            default**, ``False``) only per-stage / pairwise data are assumed and
            the guarantee is the copula-agnostic Tier 1 LP floor. The default is
            ``False`` by design (Opus 5 audit, 2026-08-11): forgetting the flag
            yields the *weaker* Tier-1 guarantee, never a silent Tier-0 over-claim
            whose validity requires the joint to have been observed — a violation
            no input can detect.

    Returns:
        A :class:`Certificate`.
    """
    if not 0.0 < eta_conf < 1.0:
        raise DependenceError("eta_conf must be in (0, 1)")
    a = _as_pass_matrix(passes)
    m, n = a.shape
    tier0 = observed_all_success_floor(a, eta_conf)
    tier1 = pairwise_cp_box_floor(a, eta_conf)
    tier2 = slepian_model_floor(a, eta_conf)

    if executed_end_to_end:
        guarantee, tier, basis, assumptions = (
            tier0.floor, 0, tier0.basis, tier0.assumptions,
        )
    else:
        guarantee, tier, basis, assumptions = (
            tier1.floor, 1, tier1.basis, tier1.assumptions,
        )

    scope_note = (
        "Guarantee holds over the stated mission distribution and the pipeline "
        "as executed only — not other mission mixes, model versions, or "
        "topologies. The Gaussian model floor (tier2) and the dependence-aware "
        "point estimate are DIAGNOSTICS, not guarantees: under a dependence "
        "structure indistinguishable from the model on marginals + pairwise "
        "data, the model can over-state the true reliability (audit F1)."
    )
    return Certificate(
        guarantee=guarantee,
        guarantee_tier=tier,
        guarantee_basis=basis,
        assumptions=assumptions,
        observed=tier0.observed,
        eta_conf=eta_conf,
        executed_end_to_end=executed_end_to_end,
        tier0=tier0,
        tier1=tier1,
        tier2=tier2,
        m=m,
        n=n,
        scope_note=scope_note,
    )
