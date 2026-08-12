# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Dependence measurement for correlated contract failures (LLD-B / LLD-E).

Estimators for the shared-LLM co-failure question: do agents that share a
model fail their contracts together? Everything here is pure math on paired
binary failure outcomes — no model calls, fully offline-testable.
"""

from agentassert_abc.dependence.bootstrap import BootstrapCI, cluster_bootstrap
from agentassert_abc.dependence.estimators import (
    CoFailureTable,
    jaccard,
    kendall_tau_a,
    one_factor_loadings,
    phi_coefficient,
    tau_a_min_samples,
    tetrachoric,
)

__all__ = [
    "BootstrapCI",
    "CoFailureTable",
    "cluster_bootstrap",
    "jaccard",
    "kendall_tau_a",
    "one_factor_loadings",
    "phi_coefficient",
    "tau_a_min_samples",
    "tetrachoric",
]
