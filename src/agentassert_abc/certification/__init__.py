# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Certification module — SPRT engine, e-process, and compositional guarantees.

Patent §5.5 (Composition), §5.6 (SPRT Certification), and LLD-C v2
(graph-level anytime-valid e-process certification).
"""

from agentassert_abc.certification.composition import (
    pipeline_composition_bound,
    sequential_composition_bound,
)
from agentassert_abc.certification.eprocess import (
    EProcessError,
    EProcessUpdate,
    GraphEProcess,
    kl_bernoulli,
    simulate_type1_crossing_rate,
)
from agentassert_abc.certification.sprt import (
    SPRTCertifier,
    SPRTDecision,
    SPRTResult,
    hoeffding_sample_size,
)

__all__ = [
    "EProcessError",
    "EProcessUpdate",
    "GraphEProcess",
    "SPRTCertifier",
    "SPRTDecision",
    "SPRTResult",
    "hoeffding_sample_size",
    "kl_bernoulli",
    "pipeline_composition_bound",
    "sequential_composition_bound",
    "simulate_type1_crossing_rate",
]
