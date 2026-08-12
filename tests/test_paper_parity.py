# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Paper↔code parity: every symbol the paper names must actually exist.

WHY THIS FILE EXISTS. The v2 preprint's Appendix C prints an inventory of
"shipped modules", and the CHANGELOG makes matching claims. Three of those
claims were false at publication time: ``Jaccard`` was listed as living in
``dependence/estimators.py`` but no such function existed; ``lp_bound.py`` was
described as an "LP over an arbitrary moment set" while being hard-coded to
marginals + pairwise; and Appendix A.4 described a two-gate stability /
admissibility verdict that ``metrics/dynamics.py`` never implemented.

None of those were caught by the maths audits, because they are not maths
errors — they are *inventory* errors, and nothing tied prose to importable
symbols. This module is that tie. It is deliberately dumb: it imports things
and checks they are there.

A failure here means paper and package have diverged. Fix whichever side is
wrong — sometimes that is the paper.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

_PKG = "agentassert_abc"

# --------------------------------------------------------------------------
# Every module Appendix C names as shipped.
# --------------------------------------------------------------------------
PAPER_MODULES = (
    "certification.observed_floor",
    "certification.lp_bound",
    "certification.slepian_floor",
    "certification.factor_reliability",
    "certification.certificate",
    "certification.sprt",
    "certification.composition",
    "dependence.estimators",
    "metrics.theta",
    "metrics.drift",
    "metrics.compliance",
    "metrics.dynamics",
    "metrics.jacobi",
)

# Symbols the paper names explicitly, as (module, attribute).
PAPER_SYMBOLS = (
    ("certification.observed_floor", "design_effect_adjusted_floor"),
    ("certification.observed_floor", "clopper_pearson_lower"),
    ("certification.slepian_floor", "_dominated_psd"),
    ("certification.certificate", "certify"),
    ("certification.lp_bound", "pairwise_lp_all_success_bounds"),
    ("dependence.estimators", "CoFailureTable"),
    ("dependence.estimators", "kendall_tau_a"),
    ("dependence.estimators", "tetrachoric"),
    ("dependence.estimators", "jaccard"),
    ("dependence.estimators", "phi_coefficient"),
)

# Known ERRATA in the published paper: the paper prints the left-hand name,
# but the shipped symbol is the right-hand one. Each entry is a correction owed
# to the next paper revision. When a line is fixed in the paper, delete it here
# and (if it is a symbol) add it to PAPER_SYMBOLS.
#
# Keeping these as assertions rather than comments means the list cannot quietly
# rot: if someone "helpfully" adds an `r_phi` alias, this test fails and forces
# the decision to be explicit.
KNOWN_PAPER_ERRATA = {
    # Appendix C, shipped-modules inventory
    ("dependence.estimators", "r_phi"): "phi_coefficient",
}


def _mod(dotted: str):
    return importlib.import_module(f"{_PKG}.{dotted}")


@pytest.mark.parametrize("dotted", PAPER_MODULES)
def test_every_module_the_paper_names_imports(dotted: str) -> None:
    assert _mod(dotted) is not None


@pytest.mark.parametrize(("dotted", "attr"), PAPER_SYMBOLS)
def test_every_symbol_the_paper_names_exists(dotted: str, attr: str) -> None:
    assert hasattr(_mod(dotted), attr), (
        f"the paper names {dotted}.{attr} but it does not exist — "
        "either ship it or correct the paper"
    )


@pytest.mark.parametrize(("ref", "actual"), list(KNOWN_PAPER_ERRATA.items()))
def test_known_paper_errata_are_still_errata(ref: tuple[str, str], actual: str) -> None:
    """Pin the paper's wrong names so the correction list stays honest."""
    dotted, wrong = ref
    module = _mod(dotted)
    assert not hasattr(module, wrong), (
        f"{dotted}.{wrong} now exists — the paper was right after all; "
        "remove this entry from KNOWN_PAPER_ERRATA"
    )
    assert hasattr(module, actual)


def test_certify_signature_matches_what_the_paper_prints() -> None:
    """Appendix C prints ``certify(passes, eta, executed_end_to_end)``.

    The middle parameter is really ``eta_conf``. This test pins the ACTUAL
    signature; the paper text is what needs correcting.
    """
    from agentassert_abc.certification.certificate import certify

    params = list(inspect.signature(certify).parameters)
    assert params == ["passes", "eta_conf", "executed_end_to_end"]
    assert "eta" not in params, "paper prints `eta`; code says `eta_conf`"


# --------------------------------------------------------------------------
# Capability claims — prose that asserts the code CAN do something.
# --------------------------------------------------------------------------


def test_lp_bound_really_accepts_an_arbitrary_moment_set() -> None:
    """Appendix C: "CP-box LP over an arbitrary moment set".

    Marginals + pairwise alone does not satisfy this sentence; the moment set
    must be caller-supplied and must admit orders above 2.
    """
    from agentassert_abc.certification.lp_bound import (
        empirical_subset_moments,
        moment_cp_box_floor,
        moment_lp_all_success_bounds,
        moment_subsets,
    )

    rng = np.random.default_rng(0)
    z = rng.normal(size=600)
    a = np.array([(0.8 * z + 0.6 * rng.normal(size=600) > -0.6).astype(int)
                  for _ in range(4)])

    subs = moment_subsets(4, (1, 2, 3))
    assert len(subs) == 14  # the paper's J = 14
    assert any(len(s) == 3 for s in subs), "no triple moments — not 'arbitrary'"

    bounds = moment_lp_all_success_bounds(4, subs, empirical_subset_moments(a, subs))
    assert bounds.feasible
    assert bounds.j_functionals == 14
    # an entirely ad-hoc moment set must also be accepted
    custom = ((0,), (1,), (2,), (3,), (0, 2), (1, 2, 3))
    assert moment_lp_all_success_bounds(
        4, custom, empirical_subset_moments(a, custom)
    ).feasible
    assert moment_cp_box_floor(a, 0.05, (1, 2, 3)).j_functionals == 14


def test_stability_verdict_implements_the_two_gate_scheme() -> None:
    """Appendix A.4: two separate gates, reporting INADMISSIBLE (not "divergent").

    A.4 explicitly disavows the v1 ``gamma > alpha`` criterion, so the shipped
    verdict must not depend on alpha for the stability decision.
    """
    from agentassert_abc.metrics.dynamics import (
        DEFAULT_D_CRIT,
        LyapunovStabilityCheck,
        OUParameters,
        StabilityVerdict,
    )

    assert hasattr(StabilityVerdict, "INADMISSIBLE")
    assert DEFAULT_D_CRIT == 0.6  # A.4's stated threshold

    checker = LyapunovStabilityCheck()
    seq = [0.2 + 0.01 * (i % 3) for i in range(40)]

    # alpha >> gamma: mean-reverting, but the attractor is unacceptable.
    report = checker.verdict(
        seq,
        OUParameters(alpha=0.5, gamma=0.1, sigma=0.05,
                     log_likelihood=-1.0, stationary_drift=5.0),
    )
    assert report.verdict == StabilityVerdict.INADMISSIBLE
    assert report.stable is True and report.admissible is False

    # The stability gate must ignore alpha entirely (scale-invariance).
    for alpha in (0.001, 1000.0):
        r = checker.verdict(
            seq,
            OUParameters(alpha=alpha, gamma=0.5, sigma=0.05,
                         log_likelihood=-1.0, stationary_drift=alpha / 0.5),
        )
        assert r.stable is True


def test_jaccard_reproduces_the_papers_table_1_column() -> None:
    """Table 1's Jaccard column must come out of the shipped function."""
    from agentassert_abc.dependence.estimators import CoFailureTable, jaccard

    rows = {
        "same_model": ((2177, 189, 52, 3582), 0.90),
        "different_vendor": ((1987, 289, 225, 3499), 0.79),
        "same_vendor": ((2289, 66, 757, 2888), 0.74),
        "diff_vendor_grok": ((33, 0, 80, 1788), 0.29),
        "diff_vendor_meta": ((1, 0, 9, 625), 0.10),
    }
    for name, ((n11, n10, n01, n00), expected) in rows.items():
        table = CoFailureTable(n11=n11, n10=n10, n01=n01, n00=n00)
        assert jaccard(table) == pytest.approx(expected, abs=5e-3), name
