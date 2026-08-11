# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Tests for the top-level tiered certificate assembler."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from agentassert_abc.certification.certificate import certify
from agentassert_abc.exceptions import DependenceError


def _one_factor_data(marginals, loadings, n, seed):
    rng = np.random.default_rng(seed)
    a = norm.ppf(marginals)
    lam = np.asarray(loadings, float)
    d = np.sqrt(1 - lam**2)
    factor = rng.standard_normal(n)
    m = len(marginals)
    passes = np.empty((m, n), dtype=int)
    for j in range(m):
        u = lam[j] * factor + d[j] * rng.standard_normal(n)
        passes[j] = (u <= a[j]).astype(int)
    return passes


def test_executed_uses_tier0_clopper_pearson():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=4000, seed=1)
    c = certify(passes, eta_conf=0.05, executed_end_to_end=True)
    assert c.guarantee_tier == 0
    assert c.guarantee == c.tier0.floor
    assert c.guarantee <= c.observed + 1e-9            # the safety invariant
    assert "Clopper" in c.guarantee_basis


def test_default_is_failsafe_tier1():
    # audit (Opus 5): forgetting the kwarg must NOT silently grant the stronger
    # Tier-0 guarantee (which requires the joint to have been observed).
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=3000, seed=1)
    c = certify(passes)  # no executed_end_to_end kwarg
    assert c.guarantee_tier == 1
    assert c.guarantee == c.tier1.floor


def test_extrapolation_uses_tier1_copula_agnostic_lp():
    passes = _one_factor_data([0.7, 0.65, 0.72], [0.6, 0.55, 0.62], n=4000, seed=1)
    c = certify(passes, eta_conf=0.05, executed_end_to_end=False)
    assert c.guarantee_tier == 1
    assert c.guarantee == c.tier1.floor
    assert c.guarantee <= c.observed + 1e-9
    assert "NO copula" in " ".join(c.assumptions)


def test_all_tiers_present_and_ordered_by_information():
    # more information => tighter: Tier 0 (full joint) >= Tier 1 (pairwise only).
    # Tier 2 is a model diagnostic, flagged as such.
    passes = _one_factor_data([0.7, 0.7, 0.7], [0.7, 0.7, 0.7], n=6000, seed=4)
    c = certify(passes)
    assert c.tier0.floor >= c.tier1.floor - 1e-9       # executed >= extrapolation
    assert c.tier2.is_model_bound is True
    for t in (c.tier0.floor, c.tier1.floor, c.tier2.floor):
        assert 0.0 <= t <= c.observed + 1e-9


def test_scope_note_flags_model_as_diagnostic():
    passes = _one_factor_data([0.7, 0.7], [0.5, 0.5], n=1000, seed=2)
    c = certify(passes)
    assert "DIAGNOSTIC" in c.scope_note.upper()
    assert "audit F1" in c.scope_note


def test_certificate_validation():
    passes = _one_factor_data([0.7, 0.7], [0.5, 0.5], n=500, seed=1)
    with pytest.raises(DependenceError):
        certify(passes, eta_conf=0.0)
    with pytest.raises(DependenceError):
        certify(np.array([0, 1, 1]))                   # 1-D not allowed
